import numpy as np
import pytest

from risk_budget import chain, reference_oracle, simulate as sim
from risk_budget.optimizer import BookOptimizer, Budgets


# ---------------------------------------------------------------------------
# 1. Black-Scholes Greeks vs. an independent bump-and-revalue oracle
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("S,strike,T,sigma", [
    (100.0, 100.0, 0.05, 0.30),
    (100.0, 86.0, 0.10, 0.30),
    (100.0, 114.0, 0.02, 0.30),
    (50.0, 55.0, 0.25, 0.45),
    (200.0, 180.0, 0.5, 0.15),
])
def test_greeks_match_bump_and_revalue(S, strike, T, sigma):
    strikes = np.array([strike])
    Ts = np.array([T])
    g = chain.bs_call(S, strikes, Ts, sigma)
    delta_fd, gamma_fd, vega_fd = reference_oracle.bump_and_revalue_greeks(S, strikes, Ts, sigma)
    assert g.delta[0] == pytest.approx(delta_fd[0], abs=1e-4)
    assert g.gamma[0] == pytest.approx(gamma_fd[0], abs=1e-4)
    assert g.vega[0] == pytest.approx(vega_fd[0], abs=1e-3)


def test_greeks_structural_properties():
    strikes = np.array([70.0, 100.0, 130.0])
    T = np.array([0.1, 0.1, 0.1])
    g = chain.bs_call(100.0, strikes, T, 0.30)
    assert g.delta[0] > g.delta[1] > g.delta[2]  # delta decreasing in strike
    assert np.all(g.vega >= 0.0)
    assert np.all(g.gamma >= 0.0)
    deep_itm = chain.bs_call(100.0, np.array([1.0]), np.array([0.01]), 0.30)
    assert deep_itm.delta[0] == pytest.approx(1.0, abs=1e-6)
    deep_otm = chain.bs_call(100.0, np.array([10000.0]), np.array([0.01]), 0.30)
    assert deep_otm.delta[0] == pytest.approx(0.0, abs=1e-6)


def test_build_chain_is_120_contracts():
    strikes, expiries = chain.build_chain(chain.ChainConfig())
    assert strikes.shape[0] == 120
    assert expiries.shape[0] == 120
    assert len(set(expiries.tolist())) == 8
    assert len(set(strikes.tolist())) == 15


# ---------------------------------------------------------------------------
# 2. The convex program vs. an independent LP solver (the real test here)
# ---------------------------------------------------------------------------

def _random_scenario(rng, n=120):
    vega = rng.uniform(0.1, 13.0, n)
    gamma = rng.uniform(0.001, 0.08, n)
    delta = rng.uniform(0.0, 1.0, n)
    edge = np.full(n, 0.02)
    return vega, gamma, delta, edge


@pytest.mark.parametrize("seed", range(10))
def test_optimizer_matches_independent_lp_solver(seed):
    rng = np.random.default_rng(seed)
    n = 120
    vega, gamma, delta, edge = _random_scenario(rng, n)
    budgets = Budgets(vega=220.0, gamma=10.0, delta=60.0)
    net_vega0 = rng.uniform(-300, 300)
    net_gamma0 = rng.uniform(-15, 15)
    net_delta0 = rng.uniform(-80, 80)
    max_size = 10.0

    opt = BookOptimizer(n, max_size, budgets)
    bid, ask = opt.solve(vega, gamma, delta, edge, net_vega0, net_gamma0, net_delta0)
    cvxpy_objective = float(edge @ (bid + ask))

    bid_cap_vega, ask_cap_vega = BookOptimizer._cap(budgets.vega, net_vega0)
    bid_cap_gamma, ask_cap_gamma = BookOptimizer._cap(budgets.gamma, net_gamma0)
    bid_cap_delta, ask_cap_delta = BookOptimizer._cap(budgets.delta, net_delta0)
    _, _, scipy_objective = reference_oracle.solve_lp_scipy(
        vega, gamma, delta, edge, max_size,
        bid_cap_vega, ask_cap_vega, bid_cap_gamma, ask_cap_gamma, bid_cap_delta, ask_cap_delta,
    )

    assert cvxpy_objective == pytest.approx(scipy_objective, rel=1e-4, abs=1e-6)


def test_optimizer_never_breaches_its_own_caps():
    rng = np.random.default_rng(7)
    n = 120
    vega, gamma, delta, edge = _random_scenario(rng, n)
    budgets = Budgets(vega=220.0, gamma=10.0, delta=60.0)
    opt = BookOptimizer(n, 10.0, budgets)
    for net_vega0 in (-400.0, -100.0, 0.0, 100.0, 400.0):
        bid, ask = opt.solve(vega, gamma, delta, edge, net_vega0, 0.0, 0.0)
        new_net_vega = net_vega0 + vega @ bid - vega @ ask
        assert abs(new_net_vega) <= budgets.vega + 1e-6, (
            f"net_vega0={net_vega0} produced a worst-case breach"
        )
        assert np.all(bid >= -1e-9) and np.all(bid <= 10.0 + 1e-9)
        assert np.all(ask >= -1e-9) and np.all(ask <= 10.0 + 1e-9)


def test_cap_is_never_negative_even_when_budget_already_breached():
    bid_cap, ask_cap = BookOptimizer._cap(budget=100.0, net0=500.0)
    assert bid_cap == 0.0
    assert ask_cap == 600.0  # selling is still allowed; it is what unwinds the breach

    bid_cap2, ask_cap2 = BookOptimizer._cap(budget=100.0, net0=-500.0)
    assert bid_cap2 == 600.0
    assert ask_cap2 == 0.0


def test_optimizer_feasible_even_when_budget_already_breached():
    """A book that has already drifted past its budget (from fills the LP
    did not control) must still produce a quotable, feasible solution, not
    an infeasible program -- the whole reason the caps are computed in
    plain Python (see optimizer.py) instead of as `max(...)` inside CVXPY."""
    n = 120
    vega = np.full(n, 5.0)
    gamma = np.full(n, 0.02)
    delta = np.full(n, 0.5)
    edge = np.full(n, 0.02)
    budgets = Budgets(vega=50.0, gamma=1.0, delta=5.0)
    opt = BookOptimizer(n, 10.0, budgets)
    bid, ask = opt.solve(vega, gamma, delta, edge, net_vega0=500.0, net_gamma0=0.0, net_delta0=0.0)
    assert opt.problem.status in ("optimal", "optimal_inaccurate")
    assert np.allclose(bid, 0.0)  # cannot add any more vega-positive exposure
    assert np.any(ask > 0.0)      # but selling to unwind is still offered


# ---------------------------------------------------------------------------
# 3. Simulation-level invariants and determinism
# ---------------------------------------------------------------------------

def test_fills_plus_declines_equals_total_ticks():
    cfg = sim.SimConfig(ticks=200, requote_interval=50)
    res = sim.run_session(0, cfg, "contract_by_contract")
    assert res.fills + res.declined_fills == res.total_ticks


def test_contract_by_contract_never_declines_with_ample_size():
    # max_size=10 lots and <=1 arrival/tick across 120 contracts almost
    # never depletes a contract's quote within one requote interval, so
    # the uncoordinated baseline should decline a fill only extremely
    # rarely; this is a sanity check on the flow model, not a tuned target.
    cfg = sim.SimConfig(ticks=500, requote_interval=50)
    res = sim.run_session(1, cfg, "contract_by_contract")
    assert res.declined_fills / res.total_ticks < 0.05


def test_book_optimized_declines_more_than_baseline():
    cfg = sim.SimConfig(ticks=500, requote_interval=50)
    optimizer = BookOptimizer(120, cfg.max_size, cfg.budgets)
    base = sim.run_session(2, cfg, "contract_by_contract")
    opt = sim.run_session(2, cfg, "book_optimized", optimizer)
    assert opt.declined_fills >= base.declined_fills


def test_determinism_same_seed_same_result():
    cfg = sim.SimConfig(ticks=300, requote_interval=50)
    r1 = sim.run_session(5, cfg, "contract_by_contract")
    r2 = sim.run_session(5, cfg, "contract_by_contract")
    assert r1.edge_captured == r2.edge_captured
    assert r1.fills == r2.fills
    assert r1.ticks_in_vega_budget == r2.ticks_in_vega_budget


def test_different_seeds_give_different_sessions():
    # fills/edge_captured are both deterministic structural facts whenever
    # quoted size never depletes (every tick fills, same edge per lot), so
    # the randomness this test is checking for shows up in *which*
    # contracts trade and how inventory drifts, not in those two totals.
    cfg = sim.SimConfig(ticks=300, requote_interval=50)
    r1 = sim.run_session(10, cfg, "contract_by_contract")
    r2 = sim.run_session(11, cfg, "contract_by_contract")
    assert r1.max_abs_net_vega != r2.max_abs_net_vega
    assert r1.ticks_in_vega_budget != r2.ticks_in_vega_budget


def test_unreachable_budget_matches_uncoordinated_baseline():
    """With budgets wide enough that the LP never binds, book_optimized
    should quote max_size everywhere just like contract_by_contract, and
    the two policies' fill counts should agree: the optimizer is not doing
    anything strange when its constraints are slack."""
    cfg = sim.SimConfig(ticks=300, requote_interval=50,
                         budgets=Budgets(vega=1e9, gamma=1e9, delta=1e9))
    optimizer = BookOptimizer(120, cfg.max_size, cfg.budgets)
    base = sim.run_session(3, cfg, "contract_by_contract")
    opt = sim.run_session(3, cfg, "book_optimized", optimizer)
    assert opt.fills == base.fills
    assert opt.edge_captured == pytest.approx(base.edge_captured)
