"""Two quoting policies over the same 120-contract chain, same informed/
uninformed flow model as options_session_runner.hpp (vega-weighted contract
pick, a taker that exploits the gap between the current theo and the
quote's stale center whenever it is informed and the gap is nonzero), so
the only difference measured here is how each policy sizes its quotes, not
a different flow model:

- `contract_by_contract`: every contract is quoted at a fixed `max_size` on
  both sides, independent of every other contract and of the book's
  current exposure. This is literally "quoting contract by contract": each
  contract's quote decision uses no information about the rest of the book.
- `book_optimized`: `optimizer.BookOptimizer` resizes every contract's
  quoted size every `requote_interval` ticks to maximize expected edge
  subject to the book's net vega, gamma and delta staying inside a fixed
  budget even in the worst case.

This is a lighter-weight simulation than the C++ `OptionsSessionRunner`: it
does not route orders through the vendored `IntrusiveBook` matcher (a
120-contract book at this tick count would multiply that extension's
already-large sweep further, and the thing being measured here is a sizing
decision's effect on risk and edge, not price-time-priority matching,
which the sibling 33-contract extension already validates against the real
matcher). See README Limitations.
"""
from dataclasses import dataclass, field

import numpy as np

from . import chain
from .optimizer import BookOptimizer, Budgets


@dataclass
class SimConfig:
    base_seed: int = 42
    ticks: int = 1000
    requote_interval: int = 50
    informed_frac: float = 0.20
    sigma: float = 0.30
    max_size: float = 10.0
    edge_per_lot: float = 0.02
    chain_cfg: chain.ChainConfig = field(default_factory=chain.ChainConfig)
    budgets: Budgets = field(default_factory=lambda: Budgets(vega=220.0, gamma=10.0, delta=60.0))


@dataclass
class SessionResult:
    session_id: int
    edge_captured: float = 0.0
    fills: int = 0
    declined_fills: int = 0
    ticks_in_vega_budget: int = 0
    ticks_in_gamma_budget: int = 0
    ticks_in_delta_budget: int = 0
    ticks_in_all_budgets: int = 0
    total_ticks: int = 0
    max_abs_net_vega: float = 0.0


def _pick_contract(rng: np.random.Generator, vega: np.ndarray) -> int:
    weights = np.maximum(vega, 1e-6)
    probs = weights / weights.sum()
    return int(rng.choice(weights.shape[0], p=probs))


def run_session(session_id: int, cfg: SimConfig, policy: str,
                 optimizer: BookOptimizer = None) -> SessionResult:
    assert policy in ("contract_by_contract", "book_optimized")
    rng = np.random.default_rng([cfg.base_seed, session_id])

    strikes, expiry0 = chain.build_chain(cfg.chain_cfg)
    N = strikes.shape[0]
    S = cfg.chain_cfg.underlying0
    expiry_remaining = expiry0.copy()
    inv = np.zeros(N)
    bid_size = np.zeros(N)
    ask_size = np.zeros(N)
    center_at_requote = np.zeros(N)

    res = SessionResult(session_id=session_id, total_ticks=cfg.ticks)
    dt = cfg.chain_cfg.dt_years

    for t in range(cfg.ticks):
        g_before = chain.bs_call(S, strikes, expiry_remaining * dt, cfg.sigma)

        if t % cfg.requote_interval == 0:
            center_at_requote = g_before.price.copy()
            if policy == "contract_by_contract":
                bid_size = np.full(N, cfg.max_size)
                ask_size = np.full(N, cfg.max_size)
            else:
                net_vega0 = float(np.dot(inv, g_before.vega))
                net_gamma0 = float(np.dot(inv, g_before.gamma))
                net_delta0 = float(np.dot(inv, g_before.delta))
                edge_per_lot = np.full(N, cfg.edge_per_lot)
                bid_size, ask_size = optimizer.solve(
                    g_before.vega, g_before.gamma, g_before.delta, edge_per_lot,
                    net_vega0, net_gamma0, net_delta0,
                )

        idx = _pick_contract(rng, g_before.vega)
        gap = g_before.price[idx] - center_at_requote[idx]
        is_informed = rng.random() < cfg.informed_frac
        exploiting = is_informed and gap != 0.0
        if exploiting:
            buy = gap > 0.0
        else:
            buy = rng.random() < 0.5

        if buy:
            if ask_size[idx] > 0:
                ask_size[idx] -= 1.0
                inv[idx] -= 1.0
                res.edge_captured += cfg.edge_per_lot
                res.fills += 1
            else:
                res.declined_fills += 1
        else:
            if bid_size[idx] > 0:
                bid_size[idx] -= 1.0
                inv[idx] += 1.0
                res.edge_captured += cfg.edge_per_lot
                res.fills += 1
            else:
                res.declined_fills += 1

        z = rng.normal()
        drift = -0.5 * cfg.sigma * cfg.sigma * dt
        diffusion = cfg.sigma * np.sqrt(dt) * z
        S = S * np.exp(drift + diffusion)
        expiry_remaining = expiry_remaining - 1.0

        g_after = chain.bs_call(S, strikes, expiry_remaining * dt, cfg.sigma)
        net_vega = float(np.dot(inv, g_after.vega))
        net_gamma = float(np.dot(inv, g_after.gamma))
        net_delta = float(np.dot(inv, g_after.delta))
        res.max_abs_net_vega = max(res.max_abs_net_vega, abs(net_vega))

        in_vega = abs(net_vega) <= cfg.budgets.vega
        in_gamma = abs(net_gamma) <= cfg.budgets.gamma
        in_delta = abs(net_delta) <= cfg.budgets.delta
        res.ticks_in_vega_budget += int(in_vega)
        res.ticks_in_gamma_budget += int(in_gamma)
        res.ticks_in_delta_budget += int(in_delta)
        res.ticks_in_all_budgets += int(in_vega and in_gamma and in_delta)

    return res


def run_benchmark(cfg: SimConfig, n_sessions: int, policy: str):
    strikes, _ = chain.build_chain(cfg.chain_cfg)
    N = strikes.shape[0]
    optimizer = BookOptimizer(N, cfg.max_size, cfg.budgets) if policy == "book_optimized" else None
    results = [run_session(s, cfg, policy, optimizer) for s in range(n_sessions)]
    return results
