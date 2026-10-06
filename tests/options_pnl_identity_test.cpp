// The reference-oracle check for the options extension: pnl_decomposed
// (edge captured - adverse selection - hedge slippage) must equal
// pnl_oracle (cash + mark-to-market of every option position and the
// underlying hedge position) to within floating-point tolerance, for every
// session, swept across hedge bands, widening settings and informed-flow
// rates. See README "The P&L identity (options)" for the derivation.
#include <cmath>

#include "check.hpp"
#include "options_session_runner.hpp"

using namespace mmsim::opt;

namespace {

void sweep() {
    for (double hedge_band : {0.10, 1.0, 1e18}) {
        for (bool vega_scaled : {false, true}) {
            for (double informed_frac : {0.0, 0.2, 0.5}) {
                OptionsSessionParams p;
                p.base_seed = 777;
                p.hedge_band = hedge_band;
                p.vega_scaled_widening = vega_scaled;
                p.informed_frac = informed_frac;
                p.ticks = 500;  // short sessions; this test is about the identity, not the headline numbers
                OptionsSessionRunner runner(p);
                for (uint64_t s = 0; s < 5; ++s) {
                    const OptionsSessionResult r = runner.run(s);
                    const double tol = 1e-6 * std::max(1.0, std::fabs(r.pnl_oracle));
                    CHECK(r.identity_abs_diff < tol);
                    if (r.identity_abs_diff >= tol) {
                        std::fprintf(stderr,
                                      "band=%.2f vega=%d informed=%.2f seed=%llu: "
                                      "decomposed=%.6f oracle=%.6f diff=%.9f\n",
                                      hedge_band, vega_scaled, informed_frac,
                                      static_cast<unsigned long long>(s), r.pnl_decomposed,
                                      r.pnl_oracle, r.identity_abs_diff);
                    }
                    CHECK(r.fills > 0);
                    CHECK(r.informed_fills <= r.fills);
                    CHECK(r.edge_captured >= 0.0);
                }
            }
        }
    }
}

void determinism_check() {
    OptionsSessionParams p;
    p.hedge_band = 0.10;
    OptionsSessionRunner runner(p);
    const OptionsSessionResult a = runner.run(3);
    const OptionsSessionResult b = runner.run(3);
    CHECK_EQ(a.pnl_oracle, b.pnl_oracle);
    CHECK_EQ(a.edge_captured, b.edge_captured);
    CHECK_EQ(a.fills, b.fills);
}

void no_hedge_never_trades_underlying() {
    OptionsSessionParams p;
    p.hedge_band = 1e18;  // unreachable: nothing should ever trigger a hedge trade
    OptionsSessionRunner runner(p);
    const OptionsSessionResult r = runner.run(1);
    CHECK_EQ(r.hedge_trades, static_cast<uint64_t>(0));
}

void thirty_three_contracts() {
    OptionsSessionParams p;
    OptionsSessionRunner runner(p);
    CHECK_EQ(runner.n_contracts(), 33);
}

}  // namespace

int main() {
    sweep();
    determinism_check();
    no_hedge_never_trades_underlying();
    thirty_three_contracts();
    return check_summary("options_pnl_identity_test");
}
