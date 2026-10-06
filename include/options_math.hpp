// Closed-form Black-Scholes European call pricing and Greeks, r = 0 (no
// risk-free drift; a disclosed simplification, see README). This header has
// exactly one job: turn (S, K, T, sigma) into a theoretical price, a delta
// and a vega. tests/options_pricer_test.cpp cross-checks every formula here
// against a deliberately slow, independently-derived path (Monte Carlo for
// price, central-difference bump-and-revalue for delta and vega).
#pragma once
#include <algorithm>
#include <cmath>

namespace mmsim::opt {

inline constexpr double kSqrt2 = 1.4142135623730951;
inline constexpr double kInvSqrt2Pi = 0.3989422804014327;

inline double norm_cdf(double x) { return 0.5 * (1.0 + std::erf(x / kSqrt2)); }
inline double norm_pdf(double x) { return kInvSqrt2Pi * std::exp(-0.5 * x * x); }

struct Greeks {
    double price = 0.0;
    double delta = 0.0;
    double vega = 0.0;
};

// European call, risk-free rate 0. T is time to expiry in years and must be
// > 0; callers are responsible for flooring T before expiry (see
// options_session_runner.hpp).
inline Greeks bs_call(double S, double K, double T, double sigma) {
    Greeks g;
    const double sqrtT = std::sqrt(T);
    const double d1 = (std::log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT);
    const double d2 = d1 - sigma * sqrtT;
    const double Nd1 = norm_cdf(d1);
    const double Nd2 = norm_cdf(d2);
    g.price = S * Nd1 - K * Nd2;
    g.delta = Nd1;
    g.vega = S * norm_pdf(d1) * sqrtT;
    return g;
}

}  // namespace mmsim::opt
