// Price-time priority limit order book over a flat level array, intrusive
// FIFO lists, and a two-level occupancy bitmap.
//
// Owned by exactly one thread (the matcher). Nothing in this class is
// atomic and nothing needs to be -- see the README on where the lock-free
// boundary actually sits. What makes the design interesting is not
// concurrency but the data structure:
//
//   level lookup   O(1)  -- direct index into a flat array, no tree walk
//   best price     O(1)  -- two-level bitmap + hardware bit scan
//   rest an order  O(1)  -- push to the tail of an intrusive list
//   cancel/amend   O(1)  -- hash index to the node, unlink via prev/next
//   fill           O(1)  per resting order consumed
//
// and that none of it allocates. std::map + std::list (the approach in this
// account's sibling matching-engine repo) gets the same asymptotics for
// cancel but allocates a tree node per new price level and a list node per
// order, scattering the book across the heap. The trade here is memory:
// the level arrays are sized for the whole price band up front
// (2 * kLevels * sizeof(Level), ~400 KB at kLevels = 8192) whether or not
// the band is occupied, and an order priced outside the band must be
// rejected rather than mis-placed.
//
// Time priority is not implicit -- every resting order carries the sequence
// number it rested at, so `check_invariants` can assert that each level's
// list really is in strictly increasing arrival order rather than assuming
// the linking code got it right.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "order_index.hpp"
#include "order_pool.hpp"
#include "order_types.hpp"
#include "price_bitmap.hpp"
#include "seqlock_snapshot.hpp"

namespace lfmatch {

inline constexpr int32_t kDefaultPriceLevels = 8192;

template <int32_t kLevels = kDefaultPriceLevels>
class IntrusiveBook {
public:
    struct Level {
        uint32_t head = kNullIdx;
        uint32_t tail = kNullIdx;
        uint64_t total_qty = 0;
        uint32_t count = 0;
    };

    IntrusiveBook(OrderPool& pool, size_t index_capacity_pow2)
        : pool_(pool), index_(index_capacity_pow2), bids_(kLevels), asks_(kLevels) {}

    // Matches `req` against the opposite side; any unfilled remainder rests.
    // Consumes req.node_idx: the node is either linked into the book or
    // returned to the pool before this returns.
    void submit(const OrderRequest& req, std::vector<Trade>& out) {
        OrderNode& n = pool_.node(req.node_idx);
        submitted_qty_ += n.orig_qty;

        if (req.price_idx < 0 || req.price_idx >= kLevels) {
            rejected_qty_ += n.orig_qty;
            ++rejected_count_;
            pool_.release(req.node_idx);
            return;
        }

        uint32_t remaining = n.qty;
        if (req.side == Side::Buy) {
            int32_t best;
            while (remaining > 0 && ask_mask_.find_first(best)) {
                if (!req.is_market && best > req.price_idx) break;
                Level& lvl = asks_[best];
                while (remaining > 0 && lvl.head != kNullIdx) {
                    const uint32_t ridx = lvl.head;
                    OrderNode& r = pool_.node(ridx);
                    const uint32_t q = remaining < r.qty ? remaining : r.qty;
                    out.push_back(Trade{r.order_id, n.order_id, best, q});
                    r.qty -= q;
                    remaining -= q;
                    lvl.total_qty -= q;
                    book_qty_ -= q;
                    traded_qty_ += q;
                    ++trade_count_;
                    if (r.qty == 0) {
                        unlink(ridx);
                        pool_.release(ridx);
                    }
                }
                if (lvl.head == kNullIdx) ask_mask_.clear(best);
            }
        } else {
            int32_t best;
            while (remaining > 0 && bid_mask_.find_last(best)) {
                if (!req.is_market && best < req.price_idx) break;
                Level& lvl = bids_[best];
                while (remaining > 0 && lvl.head != kNullIdx) {
                    const uint32_t ridx = lvl.head;
                    OrderNode& r = pool_.node(ridx);
                    const uint32_t q = remaining < r.qty ? remaining : r.qty;
                    out.push_back(Trade{r.order_id, n.order_id, best, q});
                    r.qty -= q;
                    remaining -= q;
                    lvl.total_qty -= q;
                    book_qty_ -= q;
                    traded_qty_ += q;
                    ++trade_count_;
                    if (r.qty == 0) {
                        unlink(ridx);
                        pool_.release(ridx);
                    }
                }
                if (lvl.head == kNullIdx) bid_mask_.clear(best);
            }
        }

        if (remaining == 0) {
            pool_.release(req.node_idx);
        } else if (req.is_market) {
            // A market order's unfilled remainder is discarded, not rested --
            // it has no price to rest at.
            discarded_qty_ += remaining;
            pool_.release(req.node_idx);
        } else {
            n.qty = remaining;
            rest(req.node_idx, req.side, req.price_idx);
        }
    }

    // A cancel for an order the book has already fully filled is a normal
    // event on every real exchange (the cancel and the fill crossed in
    // flight), not an error -- so it is a silent no-op.
    void cancel(uint64_t order_id) {
        uint32_t idx;
        if (!index_.find(order_id, idx)) return;
        OrderNode& n = pool_.node(idx);
        cancelled_qty_ += n.qty;
        unlink(idx);
        pool_.release(idx);
    }

    // Reduce-only amend: shrinks in place and therefore keeps time priority,
    // which is what most exchanges allow for a quantity-down amend. Growing
    // the quantity or moving the price would have to be cancel + new, losing
    // priority, so it is simply not offered here.
    void reduce(uint64_t order_id, uint32_t new_qty) {
        uint32_t idx;
        if (!index_.find(order_id, idx)) return;
        OrderNode& n = pool_.node(idx);
        if (new_qty == 0) {
            cancel(order_id);
            return;
        }
        if (new_qty >= n.qty) return;  // reduce-only: never grows
        const uint32_t delta = n.qty - new_qty;
        n.qty = new_qty;
        level_for(n.side, n.price_idx).total_qty -= delta;
        book_qty_ -= delta;
        cancelled_qty_ += delta;
    }

    bool best_bid(int32_t& px) const { return bid_mask_.find_last(px); }
    bool best_ask(int32_t& px) const { return ask_mask_.find_first(px); }

    void build_snapshot(BookSnapshot& s) const {
        s.version = ++publish_version_;
        s.bid_levels = 0;
        s.ask_levels = 0;
        for (int i = 0; i < kSnapshotDepth; ++i) {
            s.bid_px[i] = 0;
            s.bid_qty[i] = 0;
            s.ask_px[i] = 0;
            s.ask_qty[i] = 0;
        }
        int32_t px;
        if (bid_mask_.find_last(px)) {
            for (int i = 0; i < kSnapshotDepth; ++i) {
                s.bid_px[i] = px;
                s.bid_qty[i] = bids_[px].total_qty;
                s.bid_levels = static_cast<uint32_t>(i + 1);
                if (!bid_mask_.find_next_below(px, px)) break;
            }
        }
        if (ask_mask_.find_first(px)) {
            for (int i = 0; i < kSnapshotDepth; ++i) {
                s.ask_px[i] = px;
                s.ask_qty[i] = asks_[px].total_qty;
                s.ask_levels = static_cast<uint32_t>(i + 1);
                if (!ask_mask_.find_next_above(px, px)) break;
            }
        }
    }

    // --- accounting, used by the conservation checks -----------------------
    uint64_t submitted_qty() const { return submitted_qty_; }
    uint64_t traded_qty() const { return traded_qty_; }
    uint64_t rested_qty() const { return rested_qty_; }
    uint64_t cancelled_qty() const { return cancelled_qty_; }
    uint64_t discarded_qty() const { return discarded_qty_; }
    uint64_t rejected_qty() const { return rejected_qty_; }
    uint64_t book_qty() const { return book_qty_; }
    uint64_t trade_count() const { return trade_count_; }
    size_t resting_count() const { return resting_count_; }

    uint64_t qty_at(Side side, int32_t px) const {
        if (px < 0 || px >= kLevels) return 0;
        return level_for(side, px).total_qty;
    }

    // Full structural + accounting audit. O(kLevels + resting orders), so it
    // is a test-path check, not something the hot path runs.
    bool check_invariants(std::string& err) const {
        int32_t bb, ba;
        const bool has_bid = bid_mask_.find_last(bb);
        const bool has_ask = ask_mask_.find_first(ba);
        if (has_bid && has_ask && bb >= ba) {
            err = "crossed book: best_bid " + std::to_string(bb) + " >= best_ask " +
                  std::to_string(ba);
            return false;
        }

        uint64_t scanned_qty = 0;
        size_t scanned_orders = 0;
        for (int side_i = 0; side_i < 2; ++side_i) {
            const Side side = side_i == 0 ? Side::Buy : Side::Sell;
            const auto& levels = side_i == 0 ? bids_ : asks_;
            const auto& mask = side_i == 0 ? bid_mask_ : ask_mask_;
            for (int32_t p = 0; p < kLevels; ++p) {
                const Level& lvl = levels[p];
                const bool occupied = lvl.head != kNullIdx;
                if (occupied != mask.test(p)) {
                    err = "bitmap/level disagree at price " + std::to_string(p);
                    return false;
                }
                if (!occupied) {
                    if (lvl.total_qty != 0 || lvl.count != 0 || lvl.tail != kNullIdx) {
                        err = "empty level with residue at price " + std::to_string(p);
                        return false;
                    }
                    continue;
                }
                uint64_t sum = 0;
                uint32_t n_seen = 0;
                uint64_t last_seq = 0;
                uint32_t prev = kNullIdx;
                for (uint32_t idx = lvl.head; idx != kNullIdx;) {
                    const OrderNode& n = pool_.node(idx);
                    if (n.side != side || n.price_idx != p) {
                        err = "node " + std::to_string(n.order_id) + " linked at wrong level";
                        return false;
                    }
                    if (!n.resting) {
                        err = "non-resting node " + std::to_string(n.order_id) + " in book";
                        return false;
                    }
                    if (n.prev != prev) {
                        err = "broken prev link at price " + std::to_string(p);
                        return false;
                    }
                    // Price-time priority: arrival order must be strictly
                    // increasing from the head of the level.
                    if (n_seen > 0 && n.rest_seq <= last_seq) {
                        err = "time priority violated at price " + std::to_string(p);
                        return false;
                    }
                    last_seq = n.rest_seq;
                    sum += n.qty;
                    ++n_seen;
                    prev = idx;
                    idx = n.next;
                }
                if (prev != lvl.tail) {
                    err = "tail mismatch at price " + std::to_string(p);
                    return false;
                }
                if (sum != lvl.total_qty || n_seen != lvl.count) {
                    err = "level totals wrong at price " + std::to_string(p);
                    return false;
                }
                scanned_qty += sum;
                scanned_orders += n_seen;
            }
        }

        if (scanned_qty != book_qty_) {
            err = "book_qty counter " + std::to_string(book_qty_) + " != scanned " +
                  std::to_string(scanned_qty);
            return false;
        }
        if (scanned_orders != resting_count_ || index_.size() != resting_count_) {
            err = "resting order count mismatch";
            return false;
        }

        // Conservation 1: every unit submitted is traded, resting, discarded
        // (unfilled market remainder), or rejected.
        if (submitted_qty_ != traded_qty_ + rested_qty_ + discarded_qty_ + rejected_qty_) {
            err = "conservation(submitted) failed";
            return false;
        }
        // Conservation 2: every unit that came to rest is later traded away,
        // cancelled/reduced away, or still sitting in the book.
        if (rested_qty_ != traded_qty_ + cancelled_qty_ + book_qty_) {
            err = "conservation(rested) failed";
            return false;
        }
        return true;
    }

private:
    Level& level_for(Side s, int32_t px) { return s == Side::Buy ? bids_[px] : asks_[px]; }
    const Level& level_for(Side s, int32_t px) const {
        return s == Side::Buy ? bids_[px] : asks_[px];
    }

    void rest(uint32_t idx, Side side, int32_t px) {
        OrderNode& n = pool_.node(idx);
        n.side = side;
        n.price_idx = px;
        n.resting = 1;
        n.next = kNullIdx;
        n.rest_seq = ++rest_seq_;

        Level& lvl = level_for(side, px);
        n.prev = lvl.tail;
        if (lvl.tail != kNullIdx) {
            pool_.node(lvl.tail).next = idx;
        } else {
            lvl.head = idx;
        }
        lvl.tail = idx;
        lvl.total_qty += n.qty;
        ++lvl.count;
        (side == Side::Buy ? bid_mask_ : ask_mask_).set(px);

        index_.insert(n.order_id, idx);
        rested_qty_ += n.qty;
        book_qty_ += n.qty;
        ++resting_count_;
    }

    // Unlinks from its level and drops it from the index. Subtracts whatever
    // quantity is *still* on the node -- fills decrement as they go, so a
    // fully-filled order contributes 0 here and a cancel contributes the
    // whole remainder.
    void unlink(uint32_t idx) {
        OrderNode& n = pool_.node(idx);
        Level& lvl = level_for(n.side, n.price_idx);
        if (n.prev != kNullIdx) {
            pool_.node(n.prev).next = n.next;
        } else {
            lvl.head = n.next;
        }
        if (n.next != kNullIdx) {
            pool_.node(n.next).prev = n.prev;
        } else {
            lvl.tail = n.prev;
        }
        lvl.total_qty -= n.qty;
        book_qty_ -= n.qty;
        --lvl.count;
        if (lvl.head == kNullIdx) {
            (n.side == Side::Buy ? bid_mask_ : ask_mask_).clear(n.price_idx);
        }
        n.resting = 0;
        n.next = kNullIdx;
        n.prev = kNullIdx;
        index_.erase(n.order_id);
        --resting_count_;
    }

    OrderPool& pool_;
    OrderIndex index_;
    std::vector<Level> bids_;
    std::vector<Level> asks_;
    PriceBitmap<kLevels> bid_mask_;
    PriceBitmap<kLevels> ask_mask_;

    uint64_t rest_seq_ = 0;
    mutable uint64_t publish_version_ = 0;
    uint64_t submitted_qty_ = 0;
    uint64_t traded_qty_ = 0;
    uint64_t rested_qty_ = 0;
    uint64_t cancelled_qty_ = 0;
    uint64_t discarded_qty_ = 0;
    uint64_t rejected_qty_ = 0;
    uint64_t book_qty_ = 0;
    uint64_t trade_count_ = 0;
    uint64_t rejected_count_ = 0;
    size_t resting_count_ = 0;
};

}  // namespace lfmatch
