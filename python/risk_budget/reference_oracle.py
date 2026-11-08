"""Two independent oracles, each cross-checking a part of this extension
that is not obviously correct on inspection: Black-Scholes Greeks via
finite differences of the closed form itself, and the convex program's
optimum via a completely separate LP solver and code path (SciPy's HiGHS,
not CVXPY).
"""
import numpy as np
from scipy.optimize import linprog

from . import chain


def bump_and_revalue_greeks(S: float, strikes: np.ndarray, T: np.ndarray, sigma: float,
                             h_S: float = 1e-3, h_sigma: float = 1e-4):
    """Central-difference delta, gamma (in S) and vega (in sigma) of
    `chain.bs_call`'s own price function, sharing no formula with the
    closed-form delta/gamma/vega in `chain.bs_call` -- the same cross-check
    `options_pricer_test.cpp` runs for the C++ pricer."""
    price_up_S = chain.bs_call(S + h_S, strikes, T, sigma).price
    price_dn_S = chain.bs_call(S - h_S, strikes, T, sigma).price
    price_mid = chain.bs_call(S, strikes, T, sigma).price
    delta = (price_up_S - price_dn_S) / (2 * h_S)
    gamma = (price_up_S - 2 * price_mid + price_dn_S) / (h_S ** 2)

    price_up_sig = chain.bs_call(S, strikes, T, sigma + h_sigma).price
    price_dn_sig = chain.bs_call(S, strikes, T, sigma - h_sigma).price
    vega = (price_up_sig - price_dn_sig) / (2 * h_sigma)

    return delta, gamma, vega


def solve_lp_scipy(vega: np.ndarray, gamma: np.ndarray, delta: np.ndarray,
                    edge_per_lot: np.ndarray, max_size: float,
                    bid_cap_vega: float, ask_cap_vega: float,
                    bid_cap_gamma: float, ask_cap_gamma: float,
                    bid_cap_delta: float, ask_cap_delta: float):
    """The identical LP `optimizer.BookOptimizer` solves, built from scratch
    against `scipy.optimize.linprog` (HiGHS), sharing no code with CVXPY.
    Variables are `[bid (N), ask (N)]`; `linprog` minimizes, so the
    objective is negated to recover a maximization."""
    n = vega.shape[0]
    c = -np.concatenate([edge_per_lot, edge_per_lot])

    zeros = np.zeros(n)
    A_ub = np.array([
        np.concatenate([vega, zeros]),
        np.concatenate([zeros, vega]),
        np.concatenate([gamma, zeros]),
        np.concatenate([zeros, gamma]),
        np.concatenate([delta, zeros]),
        np.concatenate([zeros, delta]),
    ])
    b_ub = np.array([bid_cap_vega, ask_cap_vega, bid_cap_gamma,
                      ask_cap_gamma, bid_cap_delta, ask_cap_delta])
    bounds = [(0.0, max_size)] * (2 * n)

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not result.success:
        return None, None, None
    bid = result.x[:n]
    ask = result.x[n:]
    objective = float(edge_per_lot @ (bid + ask))
    return bid, ask, objective
