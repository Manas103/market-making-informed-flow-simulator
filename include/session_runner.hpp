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

    SessionResult run(uint64_t session_id) const {
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

                res.adverse_cost += -(delta * (v_after_jump - center_at_requote));
                res.inventory_mark += inv_prev * (v_after_jump - v_before);
            }

            v = v_after_jump;
        }

        res.final_inventory = inv;
        res.final_value = v;
        res.pnl_decomposed = res.gross_capture - res.adverse_cost + res.inventory_mark;
        res.pnl_oracle = cash + inv * v;
        res.identity_ok = (res.pnl_decomposed == res.pnl_oracle);
        return res;
    }

private:
    SessionParams p_;
};

}  // namespace mmsim
