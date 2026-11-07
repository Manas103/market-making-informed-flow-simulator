"""Correctness tests for the pymm nanobind binding. pymm is a compiled
extension (see README "Building and running"); these tests skip cleanly
if it has not been built into a location on sys.path."""
import pytest

pymm = pytest.importorskip("pymm")


def python_inventory_skew_policy(inv, base_size, skew_coef, cap):
    bid_size = max(0, min(cap, base_size - skew_coef * inv))
    ask_size = max(0, min(cap, base_size + skew_coef * inv))
    return bid_size, ask_size


def python_always_quote_max_policy(inv, base_size, skew_coef, cap):
    """A deliberately different policy from the native one, to prove the
    exact-reproduction result below is not a coincidence of always
    getting the same bid/ask size regardless of what Python returns."""
    del inv, skew_coef
    return cap, cap


@pytest.mark.parametrize("session_id", range(10))
def test_python_policy_reproduces_native_inventory_skew_exactly(session_id):
    native = pymm.run_native_inventory_skew(42, 4000, 60, 0.05, 0.20, 2, 30, 1, session_id)
    python = pymm.run_with_python_policy(42, 4000, 60, 0.05, 0.20, 2, 30, 1, session_id,
                                          python_inventory_skew_policy)
    assert python.pnl_oracle == native.pnl_oracle
    assert python.pnl_decomposed == native.pnl_decomposed
    assert python.gross_capture == native.gross_capture
    assert python.adverse_cost == native.adverse_cost
    assert python.inventory_mark == native.inventory_mark
    assert python.final_inventory == native.final_inventory
    assert python.fills == native.fills
    assert python.identity_ok


def test_a_different_python_policy_changes_the_outcome():
    """Negative control: swapping in a genuinely different policy produces
    a genuinely different P&L, proving the exact match above comes from
    running the same policy, not from the hook ignoring Python's answer."""
    native = pymm.run_native_inventory_skew(42, 4000, 60, 0.05, 0.20, 2, 30, 1, 0)
    different = pymm.run_with_python_policy(42, 4000, 60, 0.05, 0.20, 2, 30, 1, 0,
                                             python_always_quote_max_policy)
    assert different.pnl_oracle != native.pnl_oracle


def test_benchmark_policy_calls_returns_positive_elapsed_time():
    elapsed = pymm.benchmark_policy_calls(python_inventory_skew_policy, 1000)
    assert elapsed > 0.0
