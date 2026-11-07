// Python strategy interface over SessionRunner (session_runner.hpp). The
// one C++ session loop (fills, matching, the three-bucket P&L
// decomposition, the cash+inventory oracle identity) is unchanged and
// unduplicated: Policy::External (see session_runner.hpp) calls
// `external_quote_fn` once per requote, and this binding's only job is to
// make that one call cross into a Python callable and back. See README
// "Extension: Python strategy interface" for why External's own C++ unit
// test (tests/external_policy_test.cpp) has to pass before this binding's
// claim means anything: that test proves the hook itself is lossless;
// this binding and python/tests/test_pymm.py prove a real Python callable
// through that hook is too.
#include <chrono>
#include <cstdint>
#include <cstdio>

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/pair.h>

#include "session_runner.hpp"

namespace nb = nanobind;
using mmsim::Policy;
using mmsim::SessionParams;
using mmsim::SessionResult;
using mmsim::SessionRunner;

namespace {

SessionParams make_params(uint64_t base_seed, int ticks, int requote_interval, double p_jump,
                           double informed_frac, int64_t half_spread, int64_t base_size,
                           int64_t skew_coef) {
    SessionParams p;
    p.base_seed = base_seed;
    p.ticks = ticks;
    p.requote_interval = requote_interval;
    p.p_jump = p_jump;
    p.informed_frac = informed_frac;
    p.half_spread = half_spread;
    p.base_size = base_size;
    p.skew_coef = skew_coef;
    return p;
}

} // namespace

// Runs the native (pure C++) inventory-skew policy, no Python involved.
// The baseline both the exact-reproduction claim and the per-decision
// latency claim are measured against.
static SessionResult run_native_inventory_skew(uint64_t base_seed, int ticks,
                                                 int requote_interval, double p_jump,
                                                 double informed_frac, int64_t half_spread,
                                                 int64_t base_size, int64_t skew_coef,
                                                 uint64_t session_id) {
    SessionParams p = make_params(base_seed, ticks, requote_interval, p_jump, informed_frac,
                                   half_spread, base_size, skew_coef);
    p.policy = Policy::InventorySkew;
    SessionRunner runner(p);
    return runner.run(session_id);
}

// Runs the session with the bid/ask-size decision delegated to `policy_fn`,
// a Python callable invoked once per requote with (inventory, base_size,
// skew_coef, cap) and expected to return (bid_size, ask_size).
static SessionResult run_with_python_policy(
    uint64_t base_seed, int ticks, int requote_interval, double p_jump, double informed_frac,
    int64_t half_spread, int64_t base_size, int64_t skew_coef, uint64_t session_id,
    const std::function<std::pair<int64_t, int64_t>(int64_t, int64_t, int64_t, int64_t)>&
        policy_fn) {
    SessionParams p = make_params(base_seed, ticks, requote_interval, p_jump, informed_frac,
                                   half_spread, base_size, skew_coef);
    p.policy = Policy::External;
    p.external_quote_fn = [&policy_fn](int64_t inv, int64_t bs, int64_t sc, int64_t cap,
                                        int64_t& bid_size, int64_t& ask_size) {
        const auto result = policy_fn(inv, bs, sc, cap);
        bid_size = result.first;
        ask_size = result.second;
    };
    SessionRunner runner(p);
    return runner.run(session_id);
}

// Isolates the cost of one Python-policy decision: calls `policy_fn`
// `n_calls` times back to back, in a tight C++ loop, and times the whole
// loop with std::chrono (not Python's time.perf_counter(), so no Python
// interpreter overhead leaks into the timed window beyond the actual
// policy_fn invocation itself). This is what "12us per decision" means
// in the README: the per-call cost of routing one quoting decision
// through a Python callable, not the far smaller cost diluted across a
// whole 4,000-tick session where the policy fires only every
// requote_interval ticks.
static double benchmark_policy_calls(
    const std::function<std::pair<int64_t, int64_t>(int64_t, int64_t, int64_t, int64_t)>&
        policy_fn,
    uint64_t n_calls) {
    // No volatile needed: policy_fn crosses into the Python interpreter
    // through std::function's virtual call, which the compiler cannot see
    // through or elide regardless of what happens to its return value.
    int64_t sink = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (uint64_t i = 0; i < n_calls; ++i) {
        const auto r = policy_fn(static_cast<int64_t>(i % 50), 30, 1, 60);
        sink += r.first + r.second;
    }
    const auto t1 = std::chrono::steady_clock::now();
    if (sink == 0x5bad5bad) std::fputs("", stderr); // keeps sink live without affecting timing
    return std::chrono::duration<double>(t1 - t0).count();
}

NB_MODULE(pymm, m) {
    m.doc() = "Python strategy interface over market-making-informed-flow-simulator's "
              "SessionRunner";

    nb::class_<SessionResult>(m, "SessionResult")
        .def_ro("session_id", &SessionResult::session_id)
        .def_ro("gross_capture", &SessionResult::gross_capture)
        .def_ro("adverse_cost", &SessionResult::adverse_cost)
        .def_ro("inventory_mark", &SessionResult::inventory_mark)
        .def_ro("pnl_decomposed", &SessionResult::pnl_decomposed)
        .def_ro("pnl_oracle", &SessionResult::pnl_oracle)
        .def_ro("final_inventory", &SessionResult::final_inventory)
        .def_ro("final_value", &SessionResult::final_value)
        .def_ro("fills", &SessionResult::fills)
        .def_ro("informed_fills", &SessionResult::informed_fills)
        .def_ro("identity_ok", &SessionResult::identity_ok);

    m.def("run_native_inventory_skew", &run_native_inventory_skew, nb::arg("base_seed"),
          nb::arg("ticks"), nb::arg("requote_interval"), nb::arg("p_jump"),
          nb::arg("informed_frac"), nb::arg("half_spread"), nb::arg("base_size"),
          nb::arg("skew_coef"), nb::arg("session_id"));

    m.def("run_with_python_policy", &run_with_python_policy, nb::arg("base_seed"),
          nb::arg("ticks"), nb::arg("requote_interval"), nb::arg("p_jump"),
          nb::arg("informed_frac"), nb::arg("half_spread"), nb::arg("base_size"),
          nb::arg("skew_coef"), nb::arg("session_id"), nb::arg("policy_fn"));

    m.def("benchmark_policy_calls", &benchmark_policy_calls, nb::arg("policy_fn"),
          nb::arg("n_calls"),
          "Calls policy_fn n_calls times in a tight C++ loop and returns the total elapsed "
          "seconds, timed with std::chrono.");
}
