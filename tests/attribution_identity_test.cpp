// Per-fill attribution identity: summing the per-fill ledger must reproduce
// exactly the same gross_capture, adverse_cost and inventory_mark the
// session-level oracle already computes in pnl_identity_test.cpp -- a
// second, independent check that the *attribution* (who gets credit for
// which slice of P&L) is consistent with the aggregate, not just that the
// aggregate is internally consistent with itself.
#include <vector>

#include "check.hpp"
#include "session_runner.hpp"

using namespace mmsim;

namespace {

void sweep() {
    for (Policy policy : {Policy::FixedSize, Policy::InventorySkew}) {
        for (double informed_frac : {0.0, 0.1, 0.2, 0.5}) {
            SessionParams p;
            p.policy = policy;
            p.informed_frac = informed_frac;
            p.base_seed = 777;
            SessionRunner runner(p);
            for (uint64_t s = 0; s < 10; ++s) {
                std::vector<FillRecord> log;
                const SessionResult r = runner.run(s, &log);
                CHECK(r.identity_ok);
                CHECK_EQ(log.size(), static_cast<size_t>(r.fills));

                int64_t spread_sum = 0, adverse_sum = 0, invmark_sum = 0;
                for (const auto& fr : log) {
                    spread_sum += fr.spread_captured;
                    adverse_sum += fr.adverse_selection;
                    invmark_sum += fr.inventory_mark_share;
                }
                CHECK_EQ(spread_sum, r.gross_capture);
                CHECK_EQ(adverse_sum, r.adverse_cost);
                CHECK_EQ(invmark_sum, r.inventory_mark);
                // Therefore the three-term per-fill sum also reconciles to
                // the oracle, exactly, the same identity pnl_identity_test
                // checks at the session level.
                CHECK_EQ(spread_sum - adverse_sum + invmark_sum, r.pnl_oracle);
            }
        }
    }
}

void markouts_are_well_formed() {
    SessionParams p;
    p.policy = Policy::FixedSize;
    p.base_seed = 42;
    SessionRunner runner(p);
    std::vector<FillRecord> log;
    runner.run(5, &log);
    CHECK(!log.empty());
    bool any_clamped = false;
    for (const auto& fr : log) {
        if (fr.markout_60_clamped) any_clamped = true;
        CHECK(fr.tick >= 0 && fr.tick < p.ticks);
    }
    // 4000 ticks, 60-tick horizon: fills within the last 60 ticks must be
    // clamped, and this session has fills that late.
    CHECK(any_clamped);
}

// Determinism: the fill log must reproduce exactly across two runs of the
// same session, same as the aggregate SessionResult already does.
void determinism() {
    SessionParams p;
    p.policy = Policy::InventorySkew;
    SessionRunner runner(p);
    std::vector<FillRecord> a, b;
    runner.run(7, &a);
    runner.run(7, &b);
    CHECK_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        CHECK_EQ(a[i].fill_price, b[i].fill_price);
        CHECK_EQ(a[i].inventory_mark_share, b[i].inventory_mark_share);
        CHECK_EQ(a[i].markout_10, b[i].markout_10);
    }
}

}  // namespace

int main() {
    sweep();
    markouts_are_well_formed();
    determinism();
    return check_summary("attribution_identity_test");
}
