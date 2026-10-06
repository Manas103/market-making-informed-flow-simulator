// Wire types shared by the gateways, the matching core, and the market-data
// publisher.
//
// Prices are *level indices*, not raw ticks: the book is a fixed band of
// kPriceLevels slots and a gateway converts its tick price into an index
// before submitting. That keeps the hot path free of any price->slot
// arithmetic and makes the price bitmap (see price_bitmap.hpp) a direct
// mirror of the level array.
#pragma once
#include <cstdint>

namespace lfmatch {

enum class Side : uint8_t { Buy = 0, Sell = 1 };

enum class Action : uint8_t {
    New = 0,
    Cancel = 1,
    ReduceQty = 2,  // reduce-only amend; keeps time priority
};

// Sentinel for "no slot" in every index-based structure in this project.
// Indices rather than pointers everywhere: they are half the width, which
// is what lets the free list pack an index and an ABA tag into one 64-bit
// atomic (see order_pool.hpp).
inline constexpr uint32_t kNullIdx = 0xFFFFFFFFu;

struct OrderRequest {
    uint64_t order_id;
    uint32_t node_idx;  // pool slot the gateway already filled in (New only)
    int32_t price_idx;
    uint32_t qty;  // New: order qty. ReduceQty: the new (smaller) qty.
    Action action;
    Side side;
    bool is_market;
    uint8_t _pad;
};

struct Trade {
    uint64_t resting_order_id;
    uint64_t aggressor_order_id;
    int32_t price_idx;
    uint32_t qty;

    bool operator==(const Trade& o) const {
        return resting_order_id == o.resting_order_id &&
               aggressor_order_id == o.aggressor_order_id &&
               price_idx == o.price_idx && qty == o.qty;
    }
};

}  // namespace lfmatch
