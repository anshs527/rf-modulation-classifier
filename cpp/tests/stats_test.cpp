// Exercise 7's feedback loop: compute_stats() on inputs whose statistics
// are hand-computable.
//
// The percentile assertions are RANGES, not exact values, because the
// interpolation choice for a percentile that falls between two samples is
// yours to make (nearest rank, linear interpolation, take-the-max — all
// defensible). Any of those lands inside these ranges; only an actual bug
// (off-by-one indexing, forgot to sort, wrong fraction) lands outside.
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "stats.hpp"

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", what);
    }
}

void check_range(double got, double lo, double hi, const char* what) {
    if (got < lo || got > hi) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s — got %.6f, expected within [%.6f, %.6f]\n",
                     what, got, lo, hi);
    }
}

}  // namespace

int main() {
    try {
        // --- 1000 shuffled values 1..1000: everything is hand-computable.
        // Shuffled on purpose: a compute_stats that forgets to sort gets the
        // mean right and both percentiles wrong.
        std::vector<double> v(1000);
        std::iota(v.begin(), v.end(), 1.0);
        std::shuffle(v.begin(), v.end(), std::mt19937(42));

        const rf::LatencyStats s = rf::compute_stats(v);
        check_range(s.mean_us, 500.5, 500.5, "mean of 1..1000 is exactly 500.5");
        check_range(s.p50_us, 500.0, 501.0, "p50 of 1..1000");
        check_range(s.p99_us, 990.0, 991.0, "p99 of 1..1000");

        // --- single element: every statistic collapses to that value.
        const rf::LatencyStats one = rf::compute_stats({42.0});
        check(one.mean_us == 42.0 && one.p50_us == 42.0 && one.p99_us == 42.0,
              "single-element vector: mean = p50 = p99 = the element");

        // --- THE edge case: p99 of 10 elements. 0.99 * (10 - 1) = 8.91 is
        // not an index. Whatever interpolation you documented in stats.cpp,
        // the answer must land between the 9th value and the max — and must
        // never exceed the max (a classic off-by-one reads past the end,
        // which with this data would surface as a wrong value).
        std::vector<double> ten{10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
        std::shuffle(ten.begin(), ten.end(), std::mt19937(7));
        const rf::LatencyStats t = rf::compute_stats(ten);
        check_range(t.p99_us, 90.0, 100.0, "p99 of 10 elements: in [9th value, max]");
        check_range(t.p50_us, 50.0, 60.0, "p50 of 10 elements (even length): in [50, 60]");
        check(t.mean_us == 55.0, "mean of 10..100 by tens is exactly 55");

        // --- empty input must throw, not report zeros.
        bool threw = false;
        try {
            rf::compute_stats({});
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "empty input throws std::invalid_argument");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL (exception): %s\n", e.what());
        return 1;
    }

    if (failures == 0) {
        std::printf("PASS: all stats checks.\n");
        return 0;
    }
    std::fprintf(stderr, "%d check(s) failed.\n", failures);
    return 1;
}
