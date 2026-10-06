// Extension (Oct. 2026): a 33-contract option chain (11 strikes x 3
// expiries, European calls only, see README "Honest framing") quoted into
// 33 independent vendored IntrusiveBook instances at a Black-Scholes
// theoretical value, delta-hedged against a single synthetic underlying on
// a band policy, with every session's P&L split into exactly the three
// buckets the resume claims: edge captured, adverse selection, hedge
// slippage. The proof obligation is the same shape as session_runner.hpp's
// gross_capture/adverse_cost/inventory_mark identity, just carried out once
// per contract and once more for the underlying hedge leg, then summed; see
// README "The P&L identity (options)" for the derivation this file
// implements line for line.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <random>
#include <vector>

#include "matcher/intrusive_book.hpp"
#include "matcher/order_pool.hpp"
#include "matcher/order_types.hpp"
#include "options_math.hpp"

namespace mmsim::opt {

using lfmatch::Action;
using lfmatch::IntrusiveBook;
using lfmatch::kNullIdx;
using lfmatch::OrderPool;
using lfmatch::OrderRequest;
using lfmatch::Side;
using lfmatch::Trade;

inline constexpr int32_t kOptPriceLevels = 8192;
inline constexpr int32_t kOptCenterOffset = 0;  // prices (cents) are already non-negative
using OptBook = IntrusiveBook<kOptPriceLevels>;
inline constexpr double kTickSize = 0.01;  // $0.01 per integer price unit

struct OptionsSessionParams {
    uint64_t base_seed = 42;
    int ticks = 4000;
    int requote_interval = 60;
    double informed_frac = 0.20;
    double sigma = 0.30;               // annualized vol driving the underlying AND the quoter's theo
    double dt_years = 1.0 / (252 * 390);  // one tick == one trading minute
    double underlying0 = 100.0;
    int n_strikes = 11;                // odd, centered on underlying0
    double strike_step = 2.0;
    std::vector<double> expiry_ticks = {6000, 8000, 10000};  // all > `ticks`, never expire in-session
    int64_t quote_size = 5;             // lots per side per contract, fixed
    int64_t base_half_spread_ticks = 2;  // in integer cents
    bool vega_scaled_widening = false;
    double vega_scale_coef = 0.15;      // widening strength when vega_scaled_widening is on
    double hedge_band = 0.10;           // trigger hedge when |portfolio delta| exceeds this many shares
    double hedge_slippage_dollars = 0.02;  // cost per share traded to hedge
};

struct OptionsSessionResult {
    uint64_t session_id = 0;
    double edge_captured = 0.0;
    double adverse_selection = 0.0;
    double hedge_slippage = 0.0;
    double pnl_decomposed = 0.0;
    double pnl_oracle = 0.0;
    uint64_t fills = 0;
    uint64_t informed_fills = 0;
    uint64_t hedge_trades = 0;
    double identity_abs_diff = 0.0;
};

namespace detail {

inline void submit_new(OptBook& book, OrderPool& pool, std::vector<Trade>& out, uint64_t order_id,
                        Side side, int64_t price_ticks, int64_t qty, bool is_market) {
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
    req.price_idx = static_cast<int32_t>(price_ticks + kOptCenterOffset);
    req.qty = static_cast<uint32_t>(qty);
    req.action = Action::New;
    req.side = side;
    req.is_market = is_market;
    book.submit(req, out);
}

inline int64_t to_ticks(double dollars) {
    return std::max<int64_t>(1, static_cast<int64_t>(std::llround(dollars / kTickSize)));
}

}  // namespace detail

// One market-making + delta-hedging session over 33 vendored books. Not
// thread-safe; one session is one deterministic replay. See README for the
// full derivation of why pnl_decomposed == pnl_oracle exactly.
class OptionsSessionRunner {
public:
    explicit OptionsSessionRunner(const OptionsSessionParams& p) : p_(p) {
        const int half = p_.n_strikes / 2;
        for (int k = -half; k <= half; ++k) {
            strikes_.push_back(p_.underlying0 + k * p_.strike_step);
        }
        n_contracts_ = static_cast<int>(strikes_.size() * p_.expiry_ticks.size());
    }

    OptionsSessionResult run(uint64_t session_id) const {
        std::seed_seq seed{p_.base_seed, session_id};
        std::mt19937_64 rng(seed);
        std::normal_distribution<double> z_dist(0.0, 1.0);
        std::bernoulli_distribution informed_dist(p_.informed_frac);
        std::bernoulli_distribution dir_dist(0.5);
        std::uniform_real_distribution<double> unit_dist(0.0, 1.0);

        // std::deque, not std::vector: OrderPool and IntrusiveBook are
        // deliberately non-copyable/non-movable (see order_pool.hpp), and
        // deque never relocates existing elements when it grows.
        const int N = n_contracts_;
        std::deque<OrderPool> pools;
        for (int i = 0; i < N; ++i) pools.emplace_back(1u << 12);
        std::deque<OptBook> books;
        for (int i = 0; i < N; ++i) books.emplace_back(pools[i], size_t{1} << 10);

        std::vector<double> expiry_remaining(N);  // ticks to expiry, per contract
        std::vector<double> strike(N);
        for (int e = 0; e < static_cast<int>(p_.expiry_ticks.size()); ++e) {
            for (int s = 0; s < static_cast<int>(strikes_.size()); ++s) {
                const int c = e * static_cast<int>(strikes_.size()) + s;
                expiry_remaining[c] = p_.expiry_ticks[e];
                strike[c] = strikes_[s];
            }
        }

        std::vector<int64_t> inv(N, 0);
        std::vector<double> center_at_requote(N, 0.0);
        std::vector<int64_t> center_px_ticks(N, 0);
        // The actually-resting bid/ask price (in ticks): kept separately
        // from center_px_ticks +/- half spread because the bid is floored
        // at 1 tick, which can make the realized bid spread narrower than
        // the nominal half_spread_dollars for a near-worthless contract.
        // edge_captured is computed from these, not from the nominal
        // spread, which is what keeps the P&L identity exact at that floor.
        std::vector<int64_t> bid_px_ticks(N, 0), ask_px_ticks(N, 0);
        std::vector<uint64_t> bid_id(N, 0), ask_id(N, 0);
        std::vector<bool> have_bid(N, false), have_ask(N, false);

        double S = p_.underlying0;
        double cash = 0.0;
        int64_t underlying_inv = 0;
        uint64_t next_order_id = 1;
        std::vector<Trade> scratch;
        scratch.reserve(4);

        OptionsSessionResult res;
        res.session_id = session_id;

        double inventory_mark_total = 0.0;
        double underlying_carry_pnl = 0.0;
        double transaction_cost_total = 0.0;

        // Every price that feeds the P&L identity (center_at_requote,
        // theo_before/after, the final mark) is quantized to the same
        // integer-cent grid the book actually trades on. Without this, the
        // continuous Black-Scholes price and the discretized traded price
        // disagree by a fraction of a cent per fill, and that residue was
        // the first, genuine identity failure measured here (see README
        // Findings): decomposed and oracle differed by $0.01-$1+ per
        // session depending on spread width, not floating-point noise.
        // Greeks (delta, vega) stay continuous; they only drive decisions
        // (hedge trigger, widening), never a cash flow.
        auto theo_of = [&](int c, double underlying, double t_ticks) -> Greeks {
            const double T = std::max(t_ticks, 1.0) * p_.dt_years;
            return bs_call(underlying, strike[c], T, p_.sigma);
        };
        // Shares detail::to_ticks's floor-at-1-cent exactly, so a theo price
        // used for bookkeeping is never rounded differently than the same
        // price used to place the order that trades it.
        auto quantize = [&](double price) -> double {
            return static_cast<double>(detail::to_ticks(price)) * kTickSize;
        };

        auto requote_all = [&] {
            for (int c = 0; c < N; ++c) {
                if (have_bid[c]) {
                    books[c].cancel(bid_id[c]);
                    have_bid[c] = false;
                }
                if (have_ask[c]) {
                    books[c].cancel(ask_id[c]);
                    have_ask[c] = false;
                }
                const Greeks g = theo_of(c, S, expiry_remaining[c]);
                double hs = p_.base_half_spread_ticks * kTickSize;
                if (p_.vega_scaled_widening) {
                    // vega_ref is the chain's own ATM-ish longest-dated vega; widening is
                    // relative to that so vega_scale_coef is dimensionless across the sweep.
                    const double vega_ref = S * norm_pdf(0.0) * std::sqrt(p_.expiry_ticks.back() * p_.dt_years);
                    hs = hs * (1.0 + p_.vega_scale_coef * (g.vega / vega_ref));
                }
                const int64_t hs_ticks = detail::to_ticks(hs);
                const int64_t px_ticks = detail::to_ticks(g.price);
                center_at_requote[c] = quantize(g.price);
                center_px_ticks[c] = px_ticks;
                scratch.clear();
                if (p_.quote_size > 0) {
                    bid_px_ticks[c] = std::max<int64_t>(1, px_ticks - hs_ticks);
                    ask_px_ticks[c] = px_ticks + hs_ticks;
                    bid_id[c] = next_order_id++;
                    detail::submit_new(books[c], pools[c], scratch, bid_id[c], Side::Buy,
                                        bid_px_ticks[c], p_.quote_size, false);
                    have_bid[c] = true;
                    ask_id[c] = next_order_id++;
                    detail::submit_new(books[c], pools[c], scratch, ask_id[c], Side::Sell,
                                        ask_px_ticks[c], p_.quote_size, false);
                    have_ask[c] = true;
                }
            }
        };

        // Vega-weighted pick of which contract gets this tick's single taker
        // order: higher-vega contracts trade more, which is also exactly the
        // flow the vega-scaled-widening policy is designed to defend against.
        // Reuses the "before" Greeks already computed for this tick instead
        // of a second Black-Scholes pass over all N contracts.
        auto pick_contract = [&](const std::vector<Greeks>& g_before) -> int {
            double total = 0.0;
            for (int c = 0; c < N; ++c) total += std::max(g_before[c].vega, 1e-6);
            double r = unit_dist(rng) * total;
            for (int c = 0; c < N; ++c) {
                r -= std::max(g_before[c].vega, 1e-6);
                if (r <= 0.0) return c;
            }
            return N - 1;
        };

        for (int t = 0; t < p_.ticks; ++t) {
            if (t % p_.requote_interval == 0) requote_all();

            std::vector<Greeks> g_before(N);
            std::vector<double> theo_before(N);
            for (int c = 0; c < N; ++c) {
                g_before[c] = theo_of(c, S, expiry_remaining[c]);
                theo_before[c] = quantize(g_before[c].price);
            }
            const std::vector<int64_t> inv_prev = inv;
            const int64_t underlying_inv_prev = underlying_inv;

            const int idx = pick_contract(g_before);
            const double gap = theo_before[idx] - center_at_requote[idx];
            const bool is_informed = informed_dist(rng);
            const bool exploiting = is_informed && gap != 0.0;
            Side side = exploiting ? (gap > 0 ? Side::Buy : Side::Sell)
                                    : (dir_dist(rng) ? Side::Buy : Side::Sell);

            scratch.clear();
            const uint64_t taker_id = next_order_id++;
            detail::submit_new(books[idx], pools[idx], scratch, taker_id, side, 0, 1, true);

            const double Z = z_dist(rng);
            const double drift = -0.5 * p_.sigma * p_.sigma * p_.dt_years;
            const double diffusion = p_.sigma * std::sqrt(p_.dt_years) * Z;
            const double S_before = S;
            const double S_after = S_before * std::exp(drift + diffusion);
            for (int c = 0; c < N; ++c) expiry_remaining[c] -= 1.0;

            if (!scratch.empty()) {
                const Trade& tr = scratch.front();
                const bool mm_bought = (side == Side::Sell);
                const int64_t qty = static_cast<int64_t>(tr.qty);
                const int64_t delta_pos = mm_bought ? qty : -qty;
                const double price = static_cast<int64_t>(tr.price_idx) * kTickSize;

                inv[idx] += delta_pos;
                cash += -static_cast<double>(delta_pos) * price;
                // Realized edge, from the actually-resting bid/ask tick, not
                // the nominal half spread: exact even when the bid floor
                // (std::max(1, ...)) made the realized spread asymmetric.
                const int64_t realized_ticks = mm_bought ? (center_px_ticks[idx] - bid_px_ticks[idx])
                                                          : (ask_px_ticks[idx] - center_px_ticks[idx]);
                res.edge_captured += static_cast<double>(qty) * static_cast<double>(realized_ticks) * kTickSize;
                ++res.fills;
                if (exploiting) ++res.informed_fills;

                const double theo_after_idx = quantize(theo_of(idx, S_after, expiry_remaining[idx]).price);
                res.adverse_selection += -(static_cast<double>(delta_pos) *
                                            (theo_after_idx - center_at_requote[idx]));
            }

            double portfolio_delta = static_cast<double>(underlying_inv);
            for (int c = 0; c < N; ++c) {
                const Greeks g_after = theo_of(c, S_after, expiry_remaining[c]);
                const double theo_after_c = quantize(g_after.price);
                inventory_mark_total += static_cast<double>(inv_prev[c]) * (theo_after_c - theo_before[c]);
                portfolio_delta += static_cast<double>(inv[c]) * g_after.delta;
            }

            if (std::fabs(portfolio_delta) > p_.hedge_band) {
                const int64_t hedge_qty = -static_cast<int64_t>(std::llround(portfolio_delta));
                if (hedge_qty != 0) {
                    const double exec_price = S_after + (hedge_qty > 0 ? 1 : -1) * p_.hedge_slippage_dollars;
                    cash -= static_cast<double>(hedge_qty) * exec_price;
                    transaction_cost_total += std::fabs(static_cast<double>(hedge_qty)) * p_.hedge_slippage_dollars;
                    underlying_inv += hedge_qty;
                    ++res.hedge_trades;
                }
            }

            underlying_carry_pnl += static_cast<double>(underlying_inv_prev) * (S_after - S_before);
            S = S_after;
        }

        double final_contracts_mark = 0.0;
        for (int c = 0; c < N; ++c) {
            final_contracts_mark += static_cast<double>(inv[c]) * quantize(theo_of(c, S, expiry_remaining[c]).price);
        }

        res.hedge_slippage = transaction_cost_total - inventory_mark_total - underlying_carry_pnl;
        res.pnl_decomposed = res.edge_captured - res.adverse_selection - res.hedge_slippage;
        res.pnl_oracle = cash + final_contracts_mark + static_cast<double>(underlying_inv) * S;
        res.identity_abs_diff = std::fabs(res.pnl_decomposed - res.pnl_oracle);
        return res;
    }

    int n_contracts() const { return n_contracts_; }

private:
    OptionsSessionParams p_;
    std::vector<double> strikes_;
    int n_contracts_ = 0;
};

}  // namespace mmsim::opt
