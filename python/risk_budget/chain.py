"""Black-Scholes European-call price and Greeks (r = 0), independently
re-derived here in plain Python rather than imported from options_math.hpp,
so the fast NumPy path this package uses and the C++ side cannot silently
share a bug. tests/test_risk_budget.py cross-checks this module's vega and
gamma against central-difference bump-and-revalue of this module's own
price function (the reference-oracle pattern options_pricer_test.cpp uses
for the C++ pricer).
"""
import math
from dataclasses import dataclass

import numpy as np

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / _SQRT_2PI


@dataclass
class Greeks:
    price: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray
    vega: np.ndarray


def bs_call(S: float, strikes: np.ndarray, T: np.ndarray, sigma: float) -> Greeks:
    """European call, risk-free rate 0, vectorized over contracts. `T` is
    time to expiry in years and must be > 0 element-wise; callers floor T
    before expiry, same convention as options_session_runner.hpp."""
    T = np.maximum(T, 1e-8)
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / strikes) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    Nd1 = norm_cdf(d1)
    Nd2 = norm_cdf(d2)
    pdf_d1 = norm_pdf(d1)
    price = S * Nd1 - strikes * Nd2
    delta = Nd1
    gamma = pdf_d1 / (S * sigma * sqrtT)
    vega = S * pdf_d1 * sqrtT
    return Greeks(price=price, delta=delta, gamma=gamma, vega=vega)


@dataclass
class ChainConfig:
    underlying0: float = 100.0
    n_strikes: int = 15
    strike_step: float = 2.0
    # 8 expiries, in ticks ("1 tick = 1 trading minute", same convention as
    # options_session_runner.hpp), spanning roughly 2 to 10 weeks so the
    # chain carries a real spread of vega/gamma magnitudes.
    expiry_ticks: tuple = (3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0)
    dt_years: float = 1.0 / (252 * 390)


def build_chain(cfg: ChainConfig):
    """Returns (strikes, expiry_ticks) arrays, one entry per contract, in a
    fixed (expiry, strike) order: 15 strikes x 8 expiries = 120 contracts."""
    half = cfg.n_strikes // 2
    strikes_1d = cfg.underlying0 + cfg.strike_step * np.arange(-half, half + 1)
    n_strikes = strikes_1d.shape[0]
    n_expiries = len(cfg.expiry_ticks)
    strikes = np.tile(strikes_1d, n_expiries)
    expiries = np.repeat(np.array(cfg.expiry_ticks, dtype=float), n_strikes)
    assert strikes.shape[0] == n_strikes * n_expiries
    return strikes, expiries
