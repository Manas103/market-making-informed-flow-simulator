"""The one convex program: given the book's current per-contract Greeks and
its current net inventory exposure, choose how many lots to quote on each
side of every contract to maximize expected edge captured, subject to never
letting the book's net vega, gamma or delta exceed a fixed budget even in
the worst case where every quoted lot on the exposed side fills.

This is a linear program (a convex program with an affine objective and
affine constraints), built once with `cvxpy.Parameter`s (disciplined
parametrized programming) and re-solved every requote with new Greek
values, so the ~500 x 30 re-solves in the benchmark do not each pay CVXPY's
problem-construction cost.
"""
from dataclasses import dataclass

import cvxpy as cp
import numpy as np

# A cap this large relative to max_size (10) and a contract's own vega
# (order 1-13) is already slack by 5+ orders of magnitude; one genuinely
# unreachable budget (1e9) was tried first and made CLARABEL misreport the
# LP as unbounded on at least one solve (see README Findings) despite the
# box constraint `bid <= max_size` trivially bounding it. Clipping to this
# ceiling keeps "effectively unlimited" well-conditioned instead of merely
# large.
_CAP_CEILING = 1e6


@dataclass
class Budgets:
    vega: float
    gamma: float
    delta: float


class BookOptimizer:
    def __init__(self, n_contracts: int, max_size: float, budgets: Budgets):
        self.n = n_contracts
        self.max_size = max_size
        self.budgets = budgets

        self.bid = cp.Variable(n_contracts, nonneg=True)
        self.ask = cp.Variable(n_contracts, nonneg=True)

        self.vega_p = cp.Parameter(n_contracts, nonneg=True)
        self.gamma_p = cp.Parameter(n_contracts, nonneg=True)
        self.delta_p = cp.Parameter(n_contracts, nonneg=True)
        self.edge_p = cp.Parameter(n_contracts, nonneg=True)

        # Slack caps, computed in plain Python before each solve from the
        # book's *current* net exposure (see `cap()` below): how much more
        # one-directional exposure the book can still absorb on each side,
        # in each of the three Greeks, before breaching its budget. Passing
        # these in as already-computed scalars (rather than expressing
        # "max(budget - net0, 0)" inside the CVXPY graph) keeps every
        # constraint a plain affine inequality in the decision variables,
        # which is what makes this problem DPP-compliant for fast re-solves.
        self.bid_cap_vega = cp.Parameter(nonneg=True)
        self.ask_cap_vega = cp.Parameter(nonneg=True)
        self.bid_cap_gamma = cp.Parameter(nonneg=True)
        self.ask_cap_gamma = cp.Parameter(nonneg=True)
        self.bid_cap_delta = cp.Parameter(nonneg=True)
        self.ask_cap_delta = cp.Parameter(nonneg=True)

        objective = cp.Maximize(self.edge_p @ (self.bid + self.ask))
        constraints = [
            self.bid <= max_size,
            self.ask <= max_size,
            self.vega_p @ self.bid <= self.bid_cap_vega,
            self.vega_p @ self.ask <= self.ask_cap_vega,
            self.gamma_p @ self.bid <= self.bid_cap_gamma,
            self.gamma_p @ self.ask <= self.ask_cap_gamma,
            self.delta_p @ self.bid <= self.bid_cap_delta,
            self.delta_p @ self.ask <= self.ask_cap_delta,
        ]
        self.problem = cp.Problem(objective, constraints)
        assert self.problem.is_dcp(dpp=True), "optimizer is not DPP; re-solves would not be fast"

    @staticmethod
    def _cap(budget: float, net0: float) -> tuple:
        """Worst-case headroom on the buy side and the sell side before a
        Greek budget breaches, given the book's current net exposure net0.
        Buying adds +greek*size; selling adds -greek*size. If net0 already
        exceeds the budget on one side, that side's cap is 0, not negative
        (0 lots is always feasible), and the LP is never made infeasible by
        a budget the book has already, honestly, drifted past."""
        bid_cap = min(max(budget - net0, 0.0), _CAP_CEILING)
        ask_cap = min(max(budget + net0, 0.0), _CAP_CEILING)
        return bid_cap, ask_cap

    def solve(self, vega: np.ndarray, gamma: np.ndarray, delta: np.ndarray,
              edge_per_lot: np.ndarray, net_vega0: float, net_gamma0: float,
              net_delta0: float):
        self.vega_p.value = vega
        self.gamma_p.value = gamma
        self.delta_p.value = delta
        self.edge_p.value = edge_per_lot

        self.bid_cap_vega.value, self.ask_cap_vega.value = self._cap(self.budgets.vega, net_vega0)
        self.bid_cap_gamma.value, self.ask_cap_gamma.value = self._cap(self.budgets.gamma, net_gamma0)
        self.bid_cap_delta.value, self.ask_cap_delta.value = self._cap(self.budgets.delta, net_delta0)

        self.problem.solve(solver=cp.CLARABEL)
        if self.problem.status not in ("optimal", "optimal_inaccurate"):
            # Refuse to quote rather than guess: disclosed in the README as
            # a limitation, not silently worked around.
            return np.zeros(self.n), np.zeros(self.n)

        bid = np.clip(np.asarray(self.bid.value).reshape(-1), 0.0, self.max_size)
        ask = np.clip(np.asarray(self.ask.value).reshape(-1), 0.0, self.max_size)
        return bid, ask
