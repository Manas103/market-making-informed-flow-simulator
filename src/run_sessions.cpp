// CLI driver: runs --sessions seeded sessions under one policy and writes a
// CSV row per session to stdout. The Python analysis script in analysis/
// does the aggregation across policies and informed-flow rates; this binary
// only produces the per-session ground truth.
#include <cstdio>
#include <cstring>
#include <string>

#include "session_runner.hpp"

namespace {

struct Args {
    mmsim::SessionParams p;
    int sessions = 500;
};

bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (arg == "--policy") {
            const std::string v = next();
            if (v == "fixed") a.p.policy = mmsim::Policy::FixedSize;
            else if (v == "skew") a.p.policy = mmsim::Policy::InventorySkew;
            else return false;
        } else if (arg == "--sessions") {
            a.sessions = std::atoi(next().c_str());
        } else if (arg == "--seed") {
            a.p.base_seed = std::strtoull(next().c_str(), nullptr, 10);
        } else if (arg == "--ticks") {
            a.p.ticks = std::atoi(next().c_str());
        } else if (arg == "--requote-interval") {
            a.p.requote_interval = std::atoi(next().c_str());
        } else if (arg == "--p-jump") {
            a.p.p_jump = std::atof(next().c_str());
        } else if (arg == "--informed-frac") {
            a.p.informed_frac = std::atof(next().c_str());
        } else if (arg == "--half-spread") {
            a.p.half_spread = std::atoll(next().c_str());
        } else if (arg == "--base-size") {
            a.p.base_size = std::atoll(next().c_str());
        } else if (arg == "--skew-coef") {
            a.p.skew_coef = std::atoll(next().c_str());
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
                      "usage: run_sessions --policy fixed|skew [--sessions N] [--seed S] "
                      "[--ticks T] [--requote-interval R] [--p-jump P] [--informed-frac F] "
                      "[--half-spread H] [--base-size SZ] [--skew-coef K]\n");
        return 2;
    }

    const mmsim::SessionRunner runner(a.p);
    std::printf(
        "session_id,policy,informed_frac,gross_capture,adverse_cost,inventory_mark,"
        "pnl_decomposed,pnl_oracle,final_inventory,final_value,fills,informed_fills,"
        "discarded_takers,identity_ok\n");
    const char* policy_name = a.p.policy == mmsim::Policy::FixedSize ? "fixed" : "skew";
    bool any_bad_identity = false;
    for (int s = 0; s < a.sessions; ++s) {
        const mmsim::SessionResult r = runner.run(static_cast<uint64_t>(s));
        if (!r.identity_ok) any_bad_identity = true;
        std::printf("%llu,%s,%.4f,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%llu,%llu,%llu,%d\n",
                    static_cast<unsigned long long>(r.session_id), policy_name,
                    a.p.informed_frac, static_cast<long long>(r.gross_capture),
                    static_cast<long long>(r.adverse_cost),
                    static_cast<long long>(r.inventory_mark),
                    static_cast<long long>(r.pnl_decomposed),
                    static_cast<long long>(r.pnl_oracle),
                    static_cast<long long>(r.final_inventory),
                    static_cast<long long>(r.final_value),
                    static_cast<unsigned long long>(r.fills),
                    static_cast<unsigned long long>(r.informed_fills),
                    static_cast<unsigned long long>(r.discarded_takers), r.identity_ok ? 1 : 0);
    }
    if (any_bad_identity) {
        std::fprintf(stderr, "FATAL: pnl_decomposed != pnl_oracle in at least one session\n");
        return 1;
    }
    return 0;
}
