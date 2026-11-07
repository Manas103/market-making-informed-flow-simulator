"""Measures whether a Python policy reproduces SessionRunner's native
inventory-skew P&L exactly, over 500 seeded sessions, and the per-decision
cost of routing the quoting decision through Python instead of staying in
C++. Every number below was measured on this machine, not targeted.

Run (from this repository, after building python/pymm, see README):
    PYTHONPATH=~/build/pymm python3 python/bench_pymm_policy.py
"""
import time

import pymm

N_SESSIONS = 500
BASE_SEED = 42
TICKS = 4000
REQUOTE_INTERVAL = 60
P_JUMP = 0.05
INFORMED_FRAC = 0.20
HALF_SPREAD = 2
BASE_SIZE = 30
SKEW_COEF = 1
N_DECISIONS_PER_SESSION = -(-TICKS // REQUOTE_INTERVAL)  # ceil division, matches t % interval == 0


def python_inventory_skew_policy(inv, base_size, skew_coef, cap):
    """The exact same clamp formula session_runner.hpp's InventorySkew
    branch computes in C++, re-expressed in Python; this is the policy
    being proven to reproduce the native path, not a different one."""
    bid_size = max(0, min(cap, base_size - skew_coef * inv))
    ask_size = max(0, min(cap, base_size + skew_coef * inv))
    return bid_size, ask_size


def main():
    native_results = []
    t0 = time.perf_counter()
    for session_id in range(N_SESSIONS):
        r = pymm.run_native_inventory_skew(BASE_SEED, TICKS, REQUOTE_INTERVAL, P_JUMP,
                                            INFORMED_FRAC, HALF_SPREAD, BASE_SIZE, SKEW_COEF,
                                            session_id)
        native_results.append(r)
    native_elapsed = time.perf_counter() - t0

    python_results = []
    t0 = time.perf_counter()
    for session_id in range(N_SESSIONS):
        r = pymm.run_with_python_policy(BASE_SEED, TICKS, REQUOTE_INTERVAL, P_JUMP,
                                         INFORMED_FRAC, HALF_SPREAD, BASE_SIZE, SKEW_COEF,
                                         session_id, python_inventory_skew_policy)
        python_results.append(r)
    python_elapsed = time.perf_counter() - t0

    mismatches = 0
    for n, p in zip(native_results, python_results):
        if (n.pnl_oracle != p.pnl_oracle or n.pnl_decomposed != p.pnl_decomposed or
                n.gross_capture != p.gross_capture or n.final_inventory != p.final_inventory):
            mismatches += 1

    total_decisions = N_SESSIONS * N_DECISIONS_PER_SESSION
    added_time = python_elapsed - native_elapsed

    # Isolated per-decision cost: policy_fn called back to back in a tight
    # C++ loop (benchmark_policy_calls), timed with std::chrono inside the
    # extension, not diluted across a 4,000-tick session where the policy
    # only fires every requote_interval ticks. This is the number the
    # README reports as "12us per decision" (see README Findings for what
    # it measured).
    n_calls = 100_000
    isolated_elapsed = pymm.benchmark_policy_calls(python_inventory_skew_policy, n_calls)
    us_per_decision_isolated = 1e6 * isolated_elapsed / n_calls

    print(f"sessions: {N_SESSIONS}")
    print(f"decisions per session (requotes): {N_DECISIONS_PER_SESSION}")
    print(f"total decisions: {total_decisions}")
    print(f"sessions with any P&L mismatch: {mismatches} / {N_SESSIONS}")
    print(f"native (C++ inventory-skew) total: {native_elapsed:.4f}s "
          f"({1e6 * native_elapsed / N_SESSIONS:.1f} us/session)")
    print(f"python-policy total:               {python_elapsed:.4f}s "
          f"({1e6 * python_elapsed / N_SESSIONS:.1f} us/session)")
    print(f"added time from routing through Python over a full session: {added_time:.4f}s")
    print(f"isolated policy-call benchmark: {n_calls} calls in {isolated_elapsed:.4f}s "
          f"= {us_per_decision_isolated:.3f} us/decision")

    if mismatches > 0:
        print(f"NOTE: exact-reproduction claim not met; {mismatches} sessions disagreed")
    if us_per_decision_isolated > 12.0 * 1.1:
        print(f"NOTE: 12us/decision claim not met; measured {us_per_decision_isolated:.3f} us")


if __name__ == "__main__":
    main()
