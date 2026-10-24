// One market-making session: a quoter rests two-sided size against a mix of
// informed and uninformed flow on the vendored price-time-priority book,
// and every unit of P&L is accounted for twice (an obviously-correct direct
// mark-to-market, and a three-bucket decomposition) so the two can be diffed
// exactly. See README "Validation" for why that is the real test here, not
// "the simulation ran".
//
// Exogenous fair value: the random walk in `v` does not react to the
// quoter's trades. That is a real simplification, stated here rather than
// hidden: it means skewing can change *what price* and *how much size* the
// quoter offers, but never changes which side a trader chooses to hit,
// because this flow model is not price-sensitive (informed traders act on
// their signal, uninformed traders act on noise, and neither consults the
// quote). The inventory-skew policy below works entirely through size, not
// price, for exactly that reason -- see README "Findings" for the version
// that skewed price alone and measured a flat zero effect on inventory
// variance before this one was written.
#pragma once
#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include "matcher/intrusive_book.hpp"
#include "matcher/order_pool.hpp"
#include "matcher/order_types.hpp"

namespace mmsim {

using lfmatch::Action;
using lfmatch::IntrusiveBook;
using lfmatch::kNullIdx;
using lfmatch::OrderPool;
using lfmatch::OrderRequest;
using lfmatch::Side;
using lfmatch::Trade;

inline constexpr int32_t kPriceLevels = 8192;
inline constexpr int32_t kCenterOffset = kPriceLevels / 2;
using Book = IntrusiveBook<kPriceLevels>;

enum class Policy { FixedSize, InventorySkew };

struct SessionParams {
    uint64_t base_seed = 42;
    int ticks = 4000;
    int requote_interval = 60;
    double p_jump = 0.05;
    double informed_frac = 0.20;
    int64_t half_spread = 2;
    int64_t base_size = 30;
    int64_t skew_coef = 1;   // size units removed per unit of inventory
    Policy policy = Policy::FixedSize;
};

struct SessionResult {
    uint64_t session_id = 0;
    int64_t gross_capture = 0;
    int64_t adverse_cost = 0;
    int64_t inventory_mark = 0;
    int64_t pnl_decomposed = 0;  // gross_capture - adverse_cost + inventory_mark
    int64_t pnl_oracle = 0;      // cash_total + final_inventory * final_value
    int64_t final_inventory = 0;
    int64_t final_value = 0;
    uint64_t fills = 0;
    uint64_t informed_fills = 0;
    uint64_t discarded_takers = 0;
    bool identity_ok = false;
};

// One market-maker fill, attributed to its own slice of the three-bucket
// P&L decomposition (see "Extension: per-fill post-trade attribution" in
// the README). `inventory_mark_share` is filled in incrementally as later
// ticks' carry accrues against the position *this* fill set, up until the
// next fill replaces it -- so it is only final once the session has
// finished running, which is why it lives behind a pointer the caller owns
// rather than being returned complete.
struct FillRecord {
    uint64_t session_id = 0;
    int tick = 0;
    bool mm_bought = false;
    int64_t qty = 0;
    int64_t fill_price = 0;    // v_quote +/- half_spread, the actual traded tick
    int64_t v_quote = 0;       // center_at_requote: the stale quote this fill hit
    int64_t inv_before = 0;
    bool informed = false;
    int64_t spread_captured = 0;
    int64_t adverse_selection = 0;
    int64_t inventory_mark_share = 0;

    // Signed so that positive always means favorable to the market maker;
    // see "Extension: per-fill post-trade attribution" for the exact
    // definition. *_clamped is true when the horizon ran past session end,
    // in which case the value above is measured to the last tick instead.
    int64_t markout_1 = 0;
    int64_t markout_10 = 0;
    int64_t markout_60 = 0;
    bool markout_1_clamped = false;
    bool markout_10_clamped = false;
    bool markout_60_clamped = false;
};

namespace detail {

// Submits a New order and reports any resulting trades. The node is
// acquired from `pool` and filled exactly as the vendored matcher's own
// gateway simulator would fill it (see gateway_simulator.hpp in the sibling
// repo); this is the one piece of adapter code this project writes against
// that book.
inline void submit_new(Book& book, OrderPool& pool, std::vector<Trade>& out, uint64_t order_id,
                        Side side, int64_t value_px, int64_t qty, bool is_market) {
    const uint32_t idx = pool.acquire();
    auto& node = pool.node(idx);
    node.order_id = order_id;
    node.qty = static_cast<uint32_t>(qty);
    node.orig_qty = static_cast<uint32_t>(qty);
    node.next = kNullIdx;
    node.prev = kNullIdx;
    node.resting = 0;

    OrderRequest req{};
    req.order_id = order_id;
    req.node_idx = idx;
    req.price_idx = static_cast<int32_t>(value_px + kCenterOffset);
    req.qty = static_cast<uint32_t>(qty);
    req.action = Action::New;
    req.side = side;
    req.is_market = is_market;
    book.submit(req, out);
}

}  // namespace detail

// A single market-making session over the vendored book. Not thread-safe,
// and not meant to be: one session is one deterministic replay, and the
// claim count (500 of them) comes from running this once per seed, not from
// parallelism inside a session.
class SessionRunner {
public:
    explicit SessionRunner(const SessionParams& p) : p_(p) {}

    // `fill_log`, when non-null, is appended with one FillRecord per fill
    // in this session (existing entries are left alone, so a caller can
    // accumulate fills across many sessions into one vector). Passing it
    // changes no arithmetic on the aggregate SessionResult: every quantity
    // computed below is identical to the no-argument path, just also
    // mirrored into the per-fill ledger as it goes.
    SessionResult run(uint64_t session_id, std::vector<FillRecord>* fill_log = nullptr) const {
        std::seed_seq seed{p_.base_seed, session_id};
        std::mt19937_64 rng(seed);
        std::bernoulli_distribution jump_dist(p_.p_jump);
        std::bernoulli_distribution dir_dist(0.5);
        std::bernoulli_distribution informed_dist(p_.informed_frac);
        std::bernoulli_distribution side_dist(0.5);

        OrderPool pool(1u << 16);
        Book book(pool, size_t{1} << 12);

        SessionResult res;
        res.session_id = session_id;

        int64_t v = 0;    // fair value, in ticks, centered at 0
        int64_t inv = 0;  // MM inventory
        int64_t cash = 0;
        uint64_t next_order_id = 1;
        int64_t center_at_requote = 0;
        uint64_t bid_id = 0, ask_id = 0;
        bool have_bid = false, have_ask = false;
        std::vector<Trade> scratch;
        scratch.reserve(4);

        auto requote = [&] {
            if (have_bid) {
                book.cancel(bid_id);
                have_bid = false;
            }
            if (have_ask) {
                book.cancel(ask_id);
                have_ask = false;
            }
            int64_t bid_size = p_.base_size;
            int64_t ask_size = p_.base_size;
            if (p_.policy == Policy::InventorySkew) {
                const int64_t cap = 2 * p_.base_size;
                bid_size = std::clamp<int64_t>(p_.base_size - p_.skew_coef * inv, 0, cap);
                ask_size = std::clamp<int64_t>(p_.base_size + p_.skew_coef * inv, 0, cap);
            }
            center_at_requote = v;
            scratch.clear();
            if (bid_size > 0) {
                bid_id = next_order_id++;
                detail::submit_new(book, pool, scratch, bid_id, Side::Buy, v - p_.half_spread,
                                    bid_size, false);
                have_bid = true;
            }
            if (ask_size > 0) {
                ask_id = next_order_id++;
                detail::submit_new(book, pool, scratch, ask_id, Side::Sell, v + p_.half_spread,
                                    ask_size, false);
                have_ask = true;
            }
        };

        // Per-fill ledger bookkeeping (no-op cost when fill_log == nullptr).
        // `last_fill_idx` names the fill that currently "owns" the inventory
        // position -- i.e. the most recent fill -- so that each tick's carry
        // (computed below exactly as before) can also be attributed to the
        // fill that created the position being carried, not the fill (if
        // any) that happens to land on this same tick: a fill's own carry
        // contribution starts accruing on the *next* tick, once its qty is
        // actually sitting in inventory.
        const size_t fill_log_start = fill_log ? fill_log->size() : 0;
        std::vector<int64_t> vpath;
        if (fill_log) vpath.reserve(static_cast<size_t>(p_.ticks));
        int last_fill_idx = -1;

        for (int t = 0; t < p_.ticks; ++t) {
            if (t % p_.requote_interval == 0) requote();

            const bool jump_happens = jump_dist(rng);
            const int64_t jump_dir = jump_happens ? (dir_dist(rng) ? 1 : -1) : 0;
            const bool is_informed = informed_dist(rng);

            // The exploitable gap is the true value the quote has already
            // fallen behind, not merely "a jump landed this exact tick": a
            // quote posted 15 ticks ago at the old value is just as stale on
            // tick 16 as it was on tick 2. An informed trader who already
            // knows the current value trades on that gap for as long as it
            // persists, which is every tick until the next requote.
            const int64_t gap = v - center_at_requote;
            const bool exploiting = is_informed && gap != 0;

            Side side;
            if (exploiting) {
                side = gap > 0 ? Side::Buy : Side::Sell;
            } else {
                side = side_dist(rng) ? Side::Buy : Side::Sell;
            }

            scratch.clear();
            const uint64_t taker_id = next_order_id++;
            detail::submit_new(book, pool, scratch, taker_id, side, 0, 1, true);

            const int64_t v_before = v;
            const int64_t v_after_jump = v_before + jump_dir;

            if (scratch.empty()) {
                ++res.discarded_takers;
                res.inventory_mark += inv * jump_dir;
                if (fill_log && last_fill_idx >= 0) {
                    (*fill_log)[last_fill_idx].inventory_mark_share += inv * jump_dir;
                }
            } else {
                const Trade& tr = scratch.front();
                const bool mm_bought = (side == Side::Sell);  // taker sold -> hit MM's bid
                const int64_t qty = static_cast<int64_t>(tr.qty);
                const int64_t price = static_cast<int64_t>(tr.price_idx) - kCenterOffset;
                const int64_t delta = mm_bought ? qty : -qty;

                const int64_t inv_prev = inv;
                inv += delta;
                cash += -delta * price;
                res.gross_capture += qty * p_.half_spread;
                ++res.fills;
                if (exploiting) ++res.informed_fills;

                const int64_t adverse_this_fill = -(delta * (v_after_jump - center_at_requote));
                const int64_t carry_this_tick = inv_prev * (v_after_jump - v_before);
                res.adverse_cost += adverse_this_fill;
                res.inventory_mark += carry_this_tick;

                if (fill_log) {
                    // This tick's carry belongs to whichever fill is already
                    // holding the position (inv_prev), i.e. the *previous*
                    // fill, recorded before this one is appended.
                    if (last_fill_idx >= 0) {
                        (*fill_log)[last_fill_idx].inventory_mark_share += carry_this_tick;
                    }
                    FillRecord fr;
                    fr.session_id = session_id;
                    fr.tick = t;
                    fr.mm_bought = mm_bought;
                    fr.qty = qty;
                    fr.fill_price = price;
                    fr.v_quote = center_at_requote;
                    fr.inv_before = inv_prev;
                    fr.informed = exploiting;
                    fr.spread_captured = qty * p_.half_spread;
                    fr.adverse_selection = adverse_this_fill;
                    fr.inventory_mark_share = 0;
                    fill_log->push_back(fr);
                    last_fill_idx = static_cast<int>(fill_log->size()) - 1;
                }
            }

            if (fill_log) vpath.push_back(v_after_jump);
            v = v_after_jump;
        }

        res.final_inventory = inv;
        res.final_value = v;
        res.pnl_decomposed = res.gross_capture - res.adverse_cost + res.inventory_mark;
        res.pnl_oracle = cash + inv * v;
        res.identity_ok = (res.pnl_decomposed == res.pnl_oracle);

        // Markouts: how far, and in which direction for the market maker,
        // did fair value move after each fill, measured from the price
        // actually traded (not from fair value at fill time -- see README
        // "Extension: per-fill post-trade attribution" for why). Positive
        // means favorable to the market maker; a fill whose markout is
        // consistently negative is a fill that information kept moving
        // against. Horizons beyond the session's end are clamped to the
        // last tick and flagged, not extrapolated.
        if (fill_log) {
            for (size_t i = fill_log_start; i < fill_log->size(); ++i) {
                FillRecord& fr = (*fill_log)[i];
                const int sign = fr.mm_bought ? 1 : -1;
                const auto markout_at = [&](int horizon_ticks) {
                    const size_t idx = std::min(static_cast<size_t>(fr.tick + horizon_ticks),
                                                 vpath.size() - 1);
                    return sign * (vpath[idx] - fr.fill_price);
                };
                fr.markout_1 = markout_at(1);
                fr.markout_10 = markout_at(10);
                fr.markout_60 = markout_at(60);
                fr.markout_1_clamped = fr.tick + 1 > p_.ticks - 1;
                fr.markout_10_clamped = fr.tick + 10 > p_.ticks - 1;
                fr.markout_60_clamped = fr.tick + 60 > p_.ticks - 1;
            }
        }
        return res;
    }

private:
    SessionParams p_;
};

}  // namespace mmsim
