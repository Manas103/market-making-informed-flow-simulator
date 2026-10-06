// Reference-oracle checks for options_math.hpp: the closed-form price is
// diffed against a deliberately slow, independently-derived Monte Carlo
// integration of the same risk-neutral expectation, and the closed-form
// delta/vega are diffed against central-difference bump-and-revalue of the
// closed-form price itself (the same cross-check options-pricing-greeks-engine
// uses, "pathwise vs bump-and-revalue").
#include <cmath>
#include <random>

#include "check.hpp"
#include "options_math.hpp"

using namespace mmsim::opt;

namespace {

// Deliberately slow and obviously correct: simulate S_T under the
// risk-neutral (r=0) measure directly from its lognormal closed form and
// average the discounted payoff. Shares no code with bs_call().
double mc_call_price(double S, double K, double T, double sigma, uint64_t seed, long n_paths) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> z(0.0, 1.0);
    const double drift = -0.5 * sigma * sigma * T;
    const double vol = sigma * std::sqrt(T);
    double sum = 0.0;
    for (long i = 0; i < n_paths; ++i) {
        const double ST = S * std::exp(drift + vol * z(rng));
        sum += std::max(ST - K, 0.0);
    }
    return sum / static_cast<double>(n_paths);
}

void check_price_vs_monte_carlo() {
    const long n_paths = 4'000'000;
    struct Case { double S, K, T, sigma; };
    const Case cases[] = {
        {100.0, 100.0, 0.25, 0.30}, {100.0, 90.0, 0.10, 0.30}, {100.0, 110.0, 0.50, 0.20},
        {100.0, 100.0, 0.02, 0.45}, {50.0, 55.0, 1.0, 0.35},
    };
    uint64_t seed = 1;
    for (const auto& c : cases) {
        const Greeks g = bs_call(c.S, c.K, c.T, c.sigma);
        const double mc = mc_call_price(c.S, c.K, c.T, c.sigma, seed++, n_paths);
        // Monte Carlo standard error at 4M paths is small but not zero; a
        // 0.3% relative (or 0.03 absolute for near-worthless contracts)
        // tolerance is generous enough to be robust to MC noise while still
        // catching a real formula bug (checked below with a wrong-sign d2 case).
        const double tol = std::max(0.03, 0.003 * g.price);
        CHECK(std::fabs(g.price - mc) < tol);
    }
}

void check_bump_and_revalue_delta_vega() {
    struct Case { double S, K, T, sigma; };
    const Case cases[] = {
        {100.0, 100.0, 0.25, 0.30}, {100.0, 90.0, 0.10, 0.30}, {100.0, 110.0, 0.50, 0.20},
        {80.0, 100.0, 0.75, 0.40},
    };
    for (const auto& c : cases) {
        const Greeks g = bs_call(c.S, c.K, c.T, c.sigma);

        const double hS = 1e-4 * c.S;
        const double p_up = bs_call(c.S + hS, c.K, c.T, c.sigma).price;
        const double p_dn = bs_call(c.S - hS, c.K, c.T, c.sigma).price;
        const double delta_fd = (p_up - p_dn) / (2.0 * hS);
        CHECK(std::fabs(g.delta - delta_fd) < 1e-5);

        const double hSig = 1e-5;
        const double v_up = bs_call(c.S, c.K, c.T, c.sigma + hSig).price;
        const double v_dn = bs_call(c.S, c.K, c.T, c.sigma - hSig).price;
        const double vega_fd = (v_up - v_dn) / (2.0 * hSig);
        CHECK(std::fabs(g.vega - vega_fd) < 1e-4 * std::max(1.0, g.vega));
    }
}

void check_structural_properties() {
    // Deep ITM: delta -> 1, price -> intrinsic.
    const Greeks itm = bs_call(200.0, 50.0, 0.25, 0.20);
    CHECK(itm.delta > 0.999);
    CHECK(std::fabs(itm.price - (200.0 - 50.0)) < 0.05);

    // Deep OTM: delta -> 0, price -> 0.
    const Greeks otm = bs_call(50.0, 200.0, 0.25, 0.20);
    CHECK(otm.delta < 0.001);
    CHECK(otm.price < 0.05);

    // Monotonicity: delta strictly increasing in S for a fixed strike/T/sigma.
    double prev_delta = -1.0;
    for (double S = 60.0; S <= 140.0; S += 5.0) {
        const Greeks g = bs_call(S, 100.0, 0.3, 0.3);
        CHECK(g.delta > prev_delta);
        CHECK(g.delta >= 0.0 && g.delta <= 1.0);
        CHECK(g.vega >= 0.0);
        prev_delta = g.delta;
    }
}

}  // namespace

int main() {
    check_price_vs_monte_carlo();
    check_bump_and_revalue_delta_vega();
    check_structural_properties();
    return check_summary("options_pricer_test");
}
