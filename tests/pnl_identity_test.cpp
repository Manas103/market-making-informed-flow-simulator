// The reference-oracle check for this project: pnl_decomposed (the fast,
// three-bucket running sum) must equal pnl_oracle (cash + inventory * final
// value, the obviously-correct direct computation) exactly, in integer
// arithmetic, for every session. This is the project's analogue of the
// sibling matcher repo's 1M-event diff against a naive reference book: a
// slow-but-clearly-correct computation diffed exactly against the fast one.
#include "check.hpp"
#include "session_runner.hpp"

using namespace mmsim;

static void sweep(Policy policy, const char* label) {
    for (double informed_frac : {0.0, 0.1, 0.2, 0.3, 0.5}) {
        for (int64_t half_spread : {1, 2, 4}) {
            SessionParams p;
            p.policy = policy;
            p.informed_frac = informed_frac;
            p.half_spread = half_spread;
            p.base_seed = 12345;
            SessionRunner runner(p);
            for (uint64_t s = 0; s < 20; ++s) {
                const SessionResult r = runner.run(s);
                CHECK(r.identity_ok);
                CHECK_EQ(r.pnl_decomposed, r.pnl_oracle);
                CHECK(r.fills > 0);
                CHECK(r.informed_fills <= r.fills);
                CHECK(r.gross_capture >= 0);
                if (!r.identity_ok) {
                    std::fprintf(stderr,
                                  "%s informed_frac=%.2f half_spread=%lld seed=%llu: "
                                  "decomposed=%lld oracle=%lld\n",
                                  label, informed_frac, static_cast<long long>(half_spread),
                                  static_cast<unsigned long long>(s),
                                  static_cast<long long>(r.pnl_decomposed),
                                  static_cast<long long>(r.pnl_oracle));
                }
            }
        }
    }
}

static void determinism_check() {
    SessionParams p;
    p.policy = Policy::InventorySkew;
    SessionRunner runner(p);
    const SessionResult a = runner.run(7);
    const SessionResult b = runner.run(7);
    CHECK_EQ(a.pnl_oracle, b.pnl_oracle);
    CHECK_EQ(a.final_inventory, b.final_inventory);
    CHECK_EQ(a.gross_capture, b.gross_capture);
}

int main() {
    sweep(Policy::FixedSize, "fixed");
    sweep(Policy::InventorySkew, "skew");
    determinism_check();
    return check_summary("pnl_identity_test");
}
