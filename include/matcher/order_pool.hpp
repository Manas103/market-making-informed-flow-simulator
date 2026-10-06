// Lock-free order-node pool: a Treiber stack over 32-bit slot indices.
//
// Ownership model
// ---------------
// Gateways allocate the book node and fill it in *before* handing the
// request to the matcher; the matcher returns the node to the pool when the
// order is fully filled, cancelled, or discarded. That split is deliberate:
//
//   - No allocation ever happens on the matching thread's hot path.
//   - Pool exhaustion is the system's natural backpressure signal. A
//     gateway that runs ahead of the matcher simply cannot get a node and
//     has to wait, instead of growing an unbounded queue.
//   - It makes the free list genuinely concurrent (N gateways popping, the
//     matcher pushing), which is the point -- a pool only the matcher
//     touched would not need to be lock-free at all.
//
// ABA
// ---
// The classic Treiber failure: thread A reads head == X and stalls. X is
// popped, reused, and pushed back by other threads. A wakes, sees head == X
// still, and its CAS succeeds -- but A then installs the *stale* `next` it
// read before the stall, corrupting the list.
//
// The fix here is a 32-bit monotonically incrementing tag packed alongside
// the 32-bit index in a single uint64_t. "Same slot, later generation" is a
// different 64-bit value, so the CAS fails and the operation retries. A
// pointer-based Treiber stack cannot do this in 64 bits without a
// double-width CAS or hazard pointers; using indices makes it free, which
// is the main reason this whole project addresses nodes by index.
//
// Why free_next_ must be atomic
// -----------------------------
// acquire() reads free_next_[idx] for an idx it has *not yet* claimed. That
// slot may be concurrently popped and rewritten by another thread. The
// tagged CAS makes the algorithm correct regardless of what value is read
// (a stale read just loses the CAS and retries) -- but the read itself is
// still a data race unless the location is atomic. Relaxed atomics make it
// well-defined without emitting a single extra instruction on x86, and are
// what keeps ThreadSanitizer quiet here. Storing them in a separate array
// from the node payload also keeps the hot node fields out of the
// free-list's cache traffic.
#pragma once
#include <atomic>
#include <cstdint>
#include <vector>

#include "order_types.hpp"

namespace lfmatch {

struct OrderNode {
    uint64_t order_id;
    uint64_t rest_seq;  // arrival rank; lets tests verify time priority directly
    uint32_t qty;       // remaining quantity
    uint32_t orig_qty;  // as submitted, for accounting
    int32_t price_idx;
    uint32_t next;  // intrusive FIFO link within a price level
    uint32_t prev;
    Side side;
    uint8_t resting;
};

class OrderPool {
public:
    explicit OrderPool(uint32_t capacity)
        : capacity_(capacity), nodes_(capacity), free_next_(capacity) {
        // Build the initial free list: 0 -> 1 -> ... -> capacity-1 -> null.
        for (uint32_t i = 0; i + 1 < capacity; ++i) {
            free_next_[i].store(i + 1, std::memory_order_relaxed);
        }
        if (capacity > 0) {
            free_next_[capacity - 1].store(kNullIdx, std::memory_order_relaxed);
        }
        head_.store(pack(capacity > 0 ? 0 : kNullIdx, 0), std::memory_order_relaxed);
    }

    OrderPool(const OrderPool&) = delete;
    OrderPool& operator=(const OrderPool&) = delete;

    // Safe from any number of concurrent threads. Returns kNullIdx when the
    // pool is exhausted (the caller's cue to apply backpressure).
    uint32_t acquire() noexcept {
        uint64_t old = head_.load(std::memory_order_acquire);
        for (;;) {
            const uint32_t idx = index_of(old);
            if (idx == kNullIdx) return kNullIdx;
            // May read a value belonging to a different generation of this
            // slot; the tagged CAS below is what makes that harmless.
            const uint32_t next = free_next_[idx].load(std::memory_order_relaxed);
            const uint64_t desired = pack(next, tag_of(old) + 1);
            // Acquire on success: the node payload written by whoever
            // released this slot must be visible to us before we reuse it.
            if (head_.compare_exchange_weak(old, desired, std::memory_order_acquire,
                                            std::memory_order_acquire)) {
                return idx;
            }
        }
    }

    // Safe from any number of concurrent threads.
    void release(uint32_t idx) noexcept {
        uint64_t old = head_.load(std::memory_order_relaxed);
        for (;;) {
            free_next_[idx].store(index_of(old), std::memory_order_relaxed);
            const uint64_t desired = pack(idx, tag_of(old) + 1);
            // Release on success: our writes to this node (and the fact that
            // we are done with it) must be visible to the next acquirer.
            if (head_.compare_exchange_weak(old, desired, std::memory_order_release,
                                            std::memory_order_relaxed)) {
                return;
            }
        }
    }

    OrderNode& node(uint32_t idx) noexcept { return nodes_[idx]; }
    const OrderNode& node(uint32_t idx) const noexcept { return nodes_[idx]; }
    uint32_t capacity() const noexcept { return capacity_; }

    // Test-only: walks the free list single-threadedly. Never call this
    // while other threads are using the pool.
    uint32_t free_count_unsafe() const noexcept {
        uint32_t n = 0;
        uint32_t idx = index_of(head_.load(std::memory_order_relaxed));
        while (idx != kNullIdx && n <= capacity_) {
            ++n;
            idx = free_next_[idx].load(std::memory_order_relaxed);
        }
        return n;
    }

private:
    static constexpr uint64_t pack(uint32_t idx, uint32_t tag) {
        return (static_cast<uint64_t>(tag) << 32) | idx;
    }
    static constexpr uint32_t index_of(uint64_t v) { return static_cast<uint32_t>(v); }
    static constexpr uint32_t tag_of(uint64_t v) { return static_cast<uint32_t>(v >> 32); }

    const uint32_t capacity_;
    std::vector<OrderNode> nodes_;
    std::vector<std::atomic<uint32_t>> free_next_;
    alignas(64) std::atomic<uint64_t> head_{0};
    alignas(64) char pad_[64]{};
};

}  // namespace lfmatch
