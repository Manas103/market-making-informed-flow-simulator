// order_id -> pool slot index, as an open-addressed table with linear
// probing. Owned and used exclusively by the matching thread.
//
// std::unordered_map would work, but it allocates a node per insert, which
// puts the allocator on the matching hot path -- exactly what the rest of
// this project is built to avoid. This table is sized once at construction
// and never allocates again.
//
// Deletion uses Knuth's backward-shift algorithm rather than tombstones.
// Tombstones are simpler but they accumulate: a matching engine does one
// insert and one erase per resting order, forever, so a tombstoned table
// degrades to a full linear scan over a long session and needs periodic
// rehashing to recover. Backward shifting keeps the probe sequences exactly
// as compact as a freshly built table, at the cost of a slightly more
// involved erase. Given the workload (as many erases as inserts) that is
// the right trade.
#pragma once
#include <cstdint>
#include <vector>

#include "order_types.hpp"

namespace lfmatch {

class OrderIndex {
public:
    explicit OrderIndex(size_t capacity_pow2)
        : mask_(capacity_pow2 - 1), slots_(capacity_pow2) {}

    void insert(uint64_t key, uint32_t val) noexcept {
        size_t i = hash(key) & mask_;
        while (slots_[i].used) {
            if (slots_[i].key == key) {  // overwrite
                slots_[i].val = val;
                return;
            }
            i = (i + 1) & mask_;
        }
        slots_[i] = {key, val, true};
        ++size_;
    }

    bool find(uint64_t key, uint32_t& val) const noexcept {
        size_t i = hash(key) & mask_;
        while (slots_[i].used) {
            if (slots_[i].key == key) {
                val = slots_[i].val;
                return true;
            }
            i = (i + 1) & mask_;
        }
        return false;
    }

    bool erase(uint64_t key) noexcept {
        size_t i = hash(key) & mask_;
        while (slots_[i].used) {
            if (slots_[i].key == key) break;
            i = (i + 1) & mask_;
        }
        if (!slots_[i].used) return false;

        slots_[i].used = false;
        --size_;

        // Backward shift: walk forward through the cluster and pull back any
        // entry whose ideal slot is not cyclically inside (i, j], since such
        // an entry would become unreachable once slot i reads as empty.
        size_t j = i;
        for (;;) {
            j = (j + 1) & mask_;
            if (!slots_[j].used) return true;
            const size_t k = hash(slots_[j].key) & mask_;
            const bool k_in_i_j = (i <= j) ? (i < k && k <= j) : (i < k || k <= j);
            if (k_in_i_j) continue;  // must stay where it is
            slots_[i] = slots_[j];
            slots_[j].used = false;
            i = j;
        }
    }

    size_t size() const noexcept { return size_; }
    size_t capacity() const noexcept { return mask_ + 1; }

private:
    struct Slot {
        uint64_t key = 0;
        uint32_t val = kNullIdx;
        bool used = false;
    };

    // splitmix64 finalizer -- order ids are typically dense and sequential,
    // which linear probing handles badly without a mixing step.
    static size_t hash(uint64_t x) noexcept {
        x += 0x9E3779B97F4A7C15ull;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
        return static_cast<size_t>(x ^ (x >> 31));
    }

    size_t mask_;
    size_t size_ = 0;
    std::vector<Slot> slots_;
};

}  // namespace lfmatch
