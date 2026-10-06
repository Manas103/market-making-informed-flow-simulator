// Minimal hand-rolled check suite, matching the sibling matcher repo's
// convention: no external test framework needed for a single-threaded
// executable with a controlled, deterministic set of cases.
#pragma once
#include <cstdio>
#include <cstdlib>

inline int g_checks = 0;
inline int g_failures = 0;

#define CHECK(cond)                                                         \
    do {                                                                    \
        ++g_checks;                                                         \
        if (!(cond)) {                                                      \
            ++g_failures;                                                   \
            std::fprintf(stderr, "CHECK FAILED: %s (%s:%d)\n", #cond, __FILE__, __LINE__); \
        }                                                                   \
    } while (0)

#define CHECK_EQ(a, b)                                                      \
    do {                                                                    \
        ++g_checks;                                                         \
        const auto av = (a);                                                \
        const auto bv = (b);                                                \
        if (!(av == bv)) {                                                  \
            ++g_failures;                                                   \
            std::fprintf(stderr, "CHECK_EQ FAILED: %s != %s (%s:%d)\n", #a, #b, __FILE__, \
                          __LINE__);                                        \
        }                                                                   \
    } while (0)

inline int check_summary(const char* suite) {
    std::fprintf(stdout, "%s: %d checks, %d failures\n", suite, g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
