// Two-level occupancy bitmap over the price-level array.
//
// The sibling engine in this account's other repo keeps price levels in a
// std::map, so "what is the best bid?" is a tree walk to begin() -- cheap,
// but every insertion of a new price level is an allocation and a rebalance,
// and the levels are scattered across the heap.
//
// Here the levels live in one flat array and occupancy is tracked in a
// bitmap: level i occupied <=> bit i set. Best ask is then "lowest set bit",
// best bid is "highest set bit". A single-level bitmap would make that a
// scan of up to kLevels/64 words, so there is a second summary level whose
// bit w says "word w of the first level is non-zero". Finding the best price
// is then at most kSummaryWords word loads plus two hardware bit-scan
// instructions -- 3 loads at kLevels = 8192.
//
// The trade: the price band is fixed at construction. Real books do stay in
// a bounded band around the last trade (exchanges halt or widen limits long
// before price moves thousands of ticks), but an order priced outside the
// band has to be rejected rather than silently mis-placed -- see
// IntrusiveBook::submit.
#pragma once
#include <bit>
#include <cstdint>

namespace lfmatch {

template <int32_t kLevels>
class PriceBitmap {
    static_assert(kLevels % 4096 == 0, "kLevels must be a multiple of 64*64");
    static constexpr int32_t kWords = kLevels / 64;
    static constexpr int32_t kSummaryWords = kWords / 64;

public:
    void set(int32_t i) noexcept {
        const int32_t w = i >> 6;
        l1_[w] |= (uint64_t{1} << (i & 63));
        l2_[w >> 6] |= (uint64_t{1} << (w & 63));
    }

    void clear(int32_t i) noexcept {
        const int32_t w = i >> 6;
        l1_[w] &= ~(uint64_t{1} << (i & 63));
        if (l1_[w] == 0) l2_[w >> 6] &= ~(uint64_t{1} << (w & 63));
    }

    bool test(int32_t i) const noexcept { return (l1_[i >> 6] >> (i & 63)) & 1; }

    // Lowest set bit -- the best ask.
    bool find_first(int32_t& out) const noexcept {
        for (int32_t s = 0; s < kSummaryWords; ++s) {
            const uint64_t sw = l2_[s];
            if (sw == 0) continue;
            const int32_t w = s * 64 + std::countr_zero(sw);
            out = w * 64 + std::countr_zero(l1_[w]);
            return true;
        }
        return false;
    }

    // Highest set bit -- the best bid.
    bool find_last(int32_t& out) const noexcept {
        for (int32_t s = kSummaryWords - 1; s >= 0; --s) {
            const uint64_t sw = l2_[s];
            if (sw == 0) continue;
            const int32_t w = s * 64 + (63 - std::countl_zero(sw));
            out = w * 64 + (63 - std::countl_zero(l1_[w]));
            return true;
        }
        return false;
    }

    // Next set bit strictly above `from`, used to walk depth for the
    // market-data snapshot without rescanning from the start each time.
    bool find_next_above(int32_t from, int32_t& out) const noexcept {
        if (from + 1 >= kLevels) return false;
        int32_t i = from + 1;
        int32_t w = i >> 6;
        uint64_t word = l1_[w] & (~uint64_t{0} << (i & 63));
        if (word) {
            out = w * 64 + std::countr_zero(word);
            return true;
        }
        for (++w; w < kWords; ++w) {
            if (l1_[w] == 0) continue;
            out = w * 64 + std::countr_zero(l1_[w]);
            return true;
        }
        return false;
    }

    // Next set bit strictly below `from`.
    bool find_next_below(int32_t from, int32_t& out) const noexcept {
        if (from <= 0) return false;
        int32_t i = from - 1;
        int32_t w = i >> 6;
        const int32_t bit = i & 63;
        uint64_t word = l1_[w] & (bit == 63 ? ~uint64_t{0} : ((uint64_t{1} << (bit + 1)) - 1));
        if (word) {
            out = w * 64 + (63 - std::countl_zero(word));
            return true;
        }
        for (--w; w >= 0; --w) {
            if (l1_[w] == 0) continue;
            out = w * 64 + (63 - std::countl_zero(l1_[w]));
            return true;
        }
        return false;
    }

    bool empty() const noexcept {
        for (int32_t s = 0; s < kSummaryWords; ++s) {
            if (l2_[s] != 0) return false;
        }
        return true;
    }

private:
    uint64_t l1_[kWords]{};
    uint64_t l2_[kSummaryWords]{};
};

}  // namespace lfmatch
