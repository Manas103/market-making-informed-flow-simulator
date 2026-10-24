// CLI driver for the post-trade attribution extension: runs --sessions
// seeded sessions exactly like run_sessions.cpp, but also collects the
// per-fill ledger session_runner.hpp now optionally records, and writes it
// to a second CSV alongside the usual per-session one. analysis/attribution.py
// reconciles the two exactly and summarizes markouts.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "session_runner.hpp"

namespace {

struct Args {
    mmsim::SessionParams p;
    int sessions = 500;
    std::string sessions_csv = "sessions.csv";
    std::string fills_csv = "fills.csv";
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
        } else if (arg == "--informed-frac") {
            a.p.informed_frac = std::atof(next().c_str());
        } else if (arg == "--sessions-csv") {
            a.sessions_csv = next();
        } else if (arg == "--fills-csv") {
            a.fills_csv = next();
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
                      "usage: run_attribution [--policy fixed|skew] [--sessions N] [--seed S] "
                      "[--informed-frac F] [--sessions-csv PATH] [--fills-csv PATH]\n");
        return 2;
    }

    std::FILE* sf = std::fopen(a.sessions_csv.c_str(), "w");
    std::FILE* ff = std::fopen(a.fills_csv.c_str(), "w");
    if (!sf || !ff) {
        std::fprintf(stderr, "could not open output files\n");
        return 2;
    }

    std::fprintf(sf,
                 "session_id,policy,gross_capture,adverse_cost,inventory_mark,pnl_decomposed,"
                 "pnl_oracle,fills,informed_fills,identity_ok\n");
    std::fprintf(ff,
                 "session_id,tick,mm_bought,qty,fill_price,v_quote,inv_before,informed,"
                 "spread_captured,adverse_selection,inventory_mark_share,"
                 "markout_1,markout_10,markout_60,"
                 "markout_1_clamped,markout_10_clamped,markout_60_clamped\n");

    const mmsim::SessionRunner runner(a.p);
    const char* policy_name = a.p.policy == mmsim::Policy::FixedSize ? "fixed" : "skew";
    bool any_bad_identity = false;
    uint64_t total_fills = 0;

    for (int s = 0; s < a.sessions; ++s) {
        std::vector<mmsim::FillRecord> fill_log;
        const mmsim::SessionResult r = runner.run(static_cast<uint64_t>(s), &fill_log);
        if (!r.identity_ok) any_bad_identity = true;
        total_fills += fill_log.size();

        std::fprintf(sf, "%llu,%s,%lld,%lld,%lld,%lld,%lld,%llu,%llu,%d\n",
                     static_cast<unsigned long long>(r.session_id), policy_name,
                     static_cast<long long>(r.gross_capture),
                     static_cast<long long>(r.adverse_cost),
                     static_cast<long long>(r.inventory_mark),
                     static_cast<long long>(r.pnl_decomposed),
                     static_cast<long long>(r.pnl_oracle),
                     static_cast<unsigned long long>(r.fills),
                     static_cast<unsigned long long>(r.informed_fills), r.identity_ok ? 1 : 0);

        for (const mmsim::FillRecord& fr : fill_log) {
            std::fprintf(ff, "%llu,%d,%d,%lld,%lld,%lld,%lld,%d,%lld,%lld,%lld,%lld,%lld,%lld,%d,%d,%d\n",
                         static_cast<unsigned long long>(fr.session_id), fr.tick,
                         fr.mm_bought ? 1 : 0, static_cast<long long>(fr.qty),
                         static_cast<long long>(fr.fill_price),
                         static_cast<long long>(fr.v_quote),
                         static_cast<long long>(fr.inv_before), fr.informed ? 1 : 0,
                         static_cast<long long>(fr.spread_captured),
                         static_cast<long long>(fr.adverse_selection),
                         static_cast<long long>(fr.inventory_mark_share),
                         static_cast<long long>(fr.markout_1),
                         static_cast<long long>(fr.markout_10),
                         static_cast<long long>(fr.markout_60), fr.markout_1_clamped ? 1 : 0,
                         fr.markout_10_clamped ? 1 : 0, fr.markout_60_clamped ? 1 : 0);
        }
    }

    std::fclose(sf);
    std::fclose(ff);
    std::fprintf(stderr, "%d sessions, %llu fills written\n", a.sessions,
                 static_cast<unsigned long long>(total_fills));

    if (any_bad_identity) {
        std::fprintf(stderr, "FATAL: pnl_decomposed != pnl_oracle in at least one session\n");
        return 1;
    }
    return 0;
}
