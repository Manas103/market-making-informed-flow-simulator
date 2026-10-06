// CLI driver for the options-chain extension: runs --sessions seeded
// sessions under one hedge-band / widening configuration and writes a CSV
// row per session to stdout. analysis/aggregate_options.py turns these CSVs
// into the three claimed headline numbers.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include "options_session_runner.hpp"

namespace {

struct Args {
    mmsim::opt::OptionsSessionParams p;
    int sessions = 500;
};

bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (arg == "--sessions") {
            a.sessions = std::atoi(next().c_str());
        } else if (arg == "--seed") {
            a.p.base_seed = std::strtoull(next().c_str(), nullptr, 10);
        } else if (arg == "--ticks") {
            a.p.ticks = std::atoi(next().c_str());
        } else if (arg == "--requote-interval") {
            a.p.requote_interval = std::atoi(next().c_str());
        } else if (arg == "--informed-frac") {
            a.p.informed_frac = std::atof(next().c_str());
        } else if (arg == "--sigma") {
            a.p.sigma = std::atof(next().c_str());
        } else if (arg == "--hedge-band") {
            a.p.hedge_band = std::atof(next().c_str());
        } else if (arg == "--hedge-slippage") {
            a.p.hedge_slippage_dollars = std::atof(next().c_str());
        } else if (arg == "--vega-scaled") {
            a.p.vega_scaled_widening = true;
        } else if (arg == "--vega-scale-coef") {
            a.p.vega_scale_coef = std::atof(next().c_str());
        } else if (arg == "--base-half-spread-ticks") {
            a.p.base_half_spread_ticks = std::atoll(next().c_str());
        } else if (arg == "--quote-size") {
            a.p.quote_size = std::atoll(next().c_str());
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", arg.c_str());
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse(argc, argv, a)) {
        std::fprintf(stderr,
                      "usage: run_options_sessions [--sessions N] [--seed S] [--ticks T] "
                      "[--requote-interval R] [--informed-frac F] [--sigma V] "
                      "[--hedge-band B] [--hedge-slippage C] [--vega-scaled] "
                      "[--vega-scale-coef K] [--base-half-spread-ticks H] [--quote-size SZ]\n");
        return 2;
    }

    const mmsim::opt::OptionsSessionRunner runner(a.p);
    std::printf(
        "session_id,hedge_band,vega_scaled,edge_captured,adverse_selection,hedge_slippage,"
        "pnl_decomposed,pnl_oracle,identity_abs_diff,fills,informed_fills,hedge_trades\n");
    bool any_bad_identity = false;
    for (int s = 0; s < a.sessions; ++s) {
        const mmsim::opt::OptionsSessionResult r = runner.run(static_cast<uint64_t>(s));
        if (r.identity_abs_diff > 1e-6 * std::max(1.0, std::fabs(r.pnl_oracle))) any_bad_identity = true;
        std::printf("%llu,%.6f,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.9f,%llu,%llu,%llu\n",
                    static_cast<unsigned long long>(r.session_id), a.p.hedge_band,
                    a.p.vega_scaled_widening ? 1 : 0, r.edge_captured, r.adverse_selection,
                    r.hedge_slippage, r.pnl_decomposed, r.pnl_oracle, r.identity_abs_diff,
                    static_cast<unsigned long long>(r.fills),
                    static_cast<unsigned long long>(r.informed_fills),
                    static_cast<unsigned long long>(r.hedge_trades));
    }
    if (any_bad_identity) {
        std::fprintf(stderr, "FATAL: pnl_decomposed != pnl_oracle (beyond float tolerance) in at "
                              "least one session\n");
        return 1;
    }
    return 0;
}
