// Seqlock publication of top-of-book and top-N depth.
//
// The problem this solves: the matching thread must never be slowed down or
// blocked by market-data consumers, and consumers must never observe a
// half-updated book. A mutex would do it, but every reader would then be
// able to stall the matcher. A seqlock inverts that -- the writer is
// *wait-free* (it never waits for a reader, ever) and readers are lock-free
// (they retry when they catch a write in progress). Readers can starve in
// theory under a continuously-publishing writer; they cannot block anyone.
//
// Protocol: an even sequence number means "consistent", odd means "write in
// progress". The writer bumps to odd, writes the payload, bumps to even. A
// reader snapshots the sequence, copies the payload, and re-reads the
// sequence; if it changed or started odd, the copy may be torn and is
// discarded.
//
// Why the payload fields are std::atomic with relaxed access
// ----------------------------------------------------------
// The textbook seqlock stores the payload in plain (non-atomic) memory. On
// x86 that works and is what the Linux kernel does -- but under the C++
// memory model it is a genuine data race (reader and writer touching the
// same non-atomic object with no happens-before between them), which is
// undefined behaviour and which ThreadSanitizer correctly reports. See
// docs/tsan_seqlock_race.txt for the actual report this produced before the
// fix, and the README section "A real race, caught".
//
// The standards-conforming form -- relaxed atomic payload accesses plus
// explicit fences -- costs nothing extra on x86 (a relaxed atomic load or
// store of a naturally-aligned scalar is a plain mov) but is well-defined,
// which is what lets the whole engine build TSan-clean.
//
// Fence reasoning, precisely:
//   (A) A release fence is a StoreStore barrier, so the odd-sequence store
//       that precedes it cannot be reordered after the payload stores that
//       follow it. Readers therefore always see "write in progress" before
//       they could see any partial payload.
//   (B) Same barrier in the other position: the payload stores cannot be
//       reordered after the even-sequence store that publishes them.
//   Reader: the initial acquire load pairs with (B). The acquire fence is a
//       LoadLoad barrier, so the payload loads cannot be reordered after the
//       validating sequence re-read -- without it the compiler could hoist
//       that re-read above the payload copy and validate nothing.
#pragma once
#include <atomic>
#include <cstdint>

namespace lfmatch {

inline constexpr int kSnapshotDepth = 5;

struct BookSnapshot {
    uint64_t version = 0;  // publish counter; lets a reader see what it missed
    uint32_t bid_levels = 0;
    uint32_t ask_levels = 0;
    int32_t bid_px[kSnapshotDepth]{};
    uint64_t bid_qty[kSnapshotDepth]{};
    int32_t ask_px[kSnapshotDepth]{};
    uint64_t ask_qty[kSnapshotDepth]{};
    uint64_t checksum = 0;
};

// Checksum over every field except `checksum` itself. The reader recomputes
// this and compares. Torn reads that happen to leave the price ordering
// invariants intact would slip past a structural check; a checksum catches
// any inconsistent combination of fields at all, which is what makes the
// concurrent test a real test of the seqlock rather than a smoke test.
inline uint64_t snapshot_checksum(const BookSnapshot& s) noexcept {
    uint64_t h = 0xcbf29ce484222325ull;
    auto mix = [&h](uint64_t v) {
        h ^= v;
        h *= 0x100000001b3ull;
    };
    mix(s.version);
    mix(s.bid_levels);
    mix(s.ask_levels);
    for (int i = 0; i < kSnapshotDepth; ++i) {
        mix(static_cast<uint64_t>(static_cast<int64_t>(s.bid_px[i])));
        mix(s.bid_qty[i]);
        mix(static_cast<uint64_t>(static_cast<int64_t>(s.ask_px[i])));
        mix(s.ask_qty[i]);
    }
    return h;
}

class SeqlockPublisher {
public:
    // Publish a valid empty book immediately. Without this the zero-
    // initialised payload carries checksum == 0, which is *not* the checksum
    // of an all-zero snapshot -- so a reader that attaches before the first
    // real event would accept a state that fails validation. Found by the
    // checksum property test in tests/structures_test.cpp; see the README.
    // It is also the right behaviour on its own terms: a market-data client
    // connecting to an idle venue should see a well-formed empty book.
    SeqlockPublisher() { publish(BookSnapshot{}); }

    // Single writer (the matching thread). Wait-free: no loops, no waiting.
    void publish(const BookSnapshot& in) noexcept {
        BookSnapshot s = in;
        s.checksum = snapshot_checksum(s);

        const uint32_t s0 = seq_.load(std::memory_order_relaxed);
        seq_.store(s0 + 1, std::memory_order_relaxed);        // odd: in progress
        std::atomic_thread_fence(std::memory_order_release);  // (A)

        version_.store(s.version, std::memory_order_relaxed);
        bid_levels_.store(s.bid_levels, std::memory_order_relaxed);
        ask_levels_.store(s.ask_levels, std::memory_order_relaxed);
        for (int i = 0; i < kSnapshotDepth; ++i) {
            bid_px_[i].store(s.bid_px[i], std::memory_order_relaxed);
            bid_qty_[i].store(s.bid_qty[i], std::memory_order_relaxed);
            ask_px_[i].store(s.ask_px[i], std::memory_order_relaxed);
            ask_qty_[i].store(s.ask_qty[i], std::memory_order_relaxed);
        }
        checksum_.store(s.checksum, std::memory_order_relaxed);

        std::atomic_thread_fence(std::memory_order_release);  // (B)
        seq_.store(s0 + 2, std::memory_order_relaxed);        // even: consistent
    }

    // Any number of concurrent readers. Returns false if a write was in
    // flight; the caller decides whether to retry.
    bool try_read(BookSnapshot& out) const noexcept {
        const uint32_t s0 = seq_.load(std::memory_order_acquire);
        if (s0 & 1u) return false;

        out.version = version_.load(std::memory_order_relaxed);
        out.bid_levels = bid_levels_.load(std::memory_order_relaxed);
        out.ask_levels = ask_levels_.load(std::memory_order_relaxed);
        for (int i = 0; i < kSnapshotDepth; ++i) {
            out.bid_px[i] = bid_px_[i].load(std::memory_order_relaxed);
            out.bid_qty[i] = bid_qty_[i].load(std::memory_order_relaxed);
            out.ask_px[i] = ask_px_[i].load(std::memory_order_relaxed);
            out.ask_qty[i] = ask_qty_[i].load(std::memory_order_relaxed);
        }
        out.checksum = checksum_.load(std::memory_order_relaxed);

        std::atomic_thread_fence(std::memory_order_acquire);
        return seq_.load(std::memory_order_relaxed) == s0;
    }

    bool read_retry(BookSnapshot& out, int max_attempts = 64) const noexcept {
        for (int i = 0; i < max_attempts; ++i) {
            if (try_read(out)) return true;
        }
        return false;
    }

private:
    alignas(64) std::atomic<uint32_t> seq_{0};

    alignas(64) std::atomic<uint64_t> version_{0};
    std::atomic<uint32_t> bid_levels_{0};
    std::atomic<uint32_t> ask_levels_{0};
    std::atomic<int32_t> bid_px_[kSnapshotDepth]{};
    std::atomic<uint64_t> bid_qty_[kSnapshotDepth]{};
    std::atomic<int32_t> ask_px_[kSnapshotDepth]{};
    std::atomic<uint64_t> ask_qty_[kSnapshotDepth]{};
    std::atomic<uint64_t> checksum_{0};
    alignas(64) char pad_[64]{};
};

}  // namespace lfmatch
