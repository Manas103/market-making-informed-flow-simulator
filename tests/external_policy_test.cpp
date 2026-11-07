// Cross-checks Policy::External (the hook the nanobind Python binding
// drives, see include/pymm_binding.cpp) against Policy::InventorySkew: a
// C++ lambda implementing the exact same clamp formula, run through the
// External path, must reproduce InventorySkew's session P&L bit-for-bit
// (not just approximately) over every seed in the sweep. This is the
// proof obligation "a policy delegated through this hook does not change
// the session's arithmetic" has to clear before trusting that a Python
// callable through the same hook reproduces it too (that half is proven
// in python/tests/test_pymm.py, which this test's C++-only half backs).
#include "check.hpp"
#include "session_runner.hpp"

using namespace mmsim;

int main() {
    for (double informed_frac : {0.0, 0.1, 0.2, 0.3, 0.5}) {
        for (int64_t half_spread : {1, 2, 4}) {
            SessionParams native_p;
            native_p.policy = Policy::InventorySkew;
            native_p.informed_frac = informed_frac;
            native_p.half_spread = half_spread;
            native_p.base_seed = 777;
            SessionRunner native_runner(native_p);

            SessionParams ext_p = native_p;
            ext_p.policy = Policy::External;
            ext_p.external_quote_fn = [](int64_t inv, int64_t base_size, int64_t skew_coef,
                                          int64_t cap, int64_t& bid_size, int64_t& ask_size) {
                bid_size = std::clamp<int64_t>(base_size - skew_coef * inv, 0, cap);
                ask_size = std::clamp<int64_t>(base_size + skew_coef * inv, 0, cap);
            };
            SessionRunner ext_runner(ext_p);

            for (uint64_t s = 0; s < 30; ++s) {
                const SessionResult a = native_runner.run(s);
                const SessionResult b = ext_runner.run(s);
                CHECK_EQ(a.pnl_oracle, b.pnl_oracle);
                CHECK_EQ(a.pnl_decomposed, b.pnl_decomposed);
                CHECK_EQ(a.gross_capture, b.gross_capture);
                CHECK_EQ(a.adverse_cost, b.adverse_cost);
                CHECK_EQ(a.inventory_mark, b.inventory_mark);
                CHECK_EQ(a.final_inventory, b.final_inventory);
                CHECK_EQ(a.fills, b.fills);
            }
        }
    }
    return check_summary("external_policy_test");
}
