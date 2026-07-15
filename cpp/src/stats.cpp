#include "stats.hpp"

#include <algorithm>
#include <stdexcept>

namespace rf {

LatencyStats compute_stats(std::vector<double> samples) {
    // ============================================================
    // EXERCISE 7 (part 2): mean / p50 / p99
    // See docs/understanding-guide.md § 7 for the full explanation.
    //
    // What this must do:
    //   - Reject an empty input (throw std::invalid_argument) — a
    //     benchmark that measured nothing must not report zeros.
    //   - Compute the mean.
    //   - Compute p50 (the median) and p99 from the sorted samples.
    //   - Return them in a LatencyStats.
    //
    // What you have available: `samples` is your own copy — sort it,
    //   mutate it, whatever you like.
    //
    // What it must produce: LatencyStats{mean, p50, p99} in the same
    //   units as the input (microseconds).
    //
    // The edge case you must decide and DOCUMENT (comment here):
    //   what does p99 mean for a vector of 10 elements? Index
    //   0.99 * (n - 1) = 8.91 is not an integer. Nearest rank? Linear
    //   interpolation between the 9th and 10th values? Just take the max?
    //   All are defensible — pick one, write down which and why, and be
    //   ready to say it out loud in an interview.
    //
    // Hints (read only if stuck):
    //   - std::sort, then think only about how to turn a fraction into
    //     an index (or a pair of indices).
    //   - Median of an even-length vector has the same question in
    //     miniature: middle element, or average of the two middles?
    // ============================================================
    if (samples.empty()) {
        throw std::invalid_argument("compute_stats: no samples");
    }
    std::sort(samples.begin(), samples.end());

    double sum = 0.0;
    for (double s : samples) sum += s;

    // Interpolation choice: LINEAR (numpy's default). The p-th percentile
    // sits at fractional index p * (n - 1); when that is not an integer,
    // interpolate between the two neighboring samples. Chosen over
    // nearest-rank so that a Python/numpy analysis of the same latency CSV
    // reports the same numbers as this binary — one fewer discrepancy to
    // explain away. For p99 of 10 samples: index 8.91, i.e. 91% of the way
    // from the 9th value to the max.
    auto percentile = [&](double p) {
        const double idx = p * static_cast<double>(samples.size() - 1);
        const size_t lo = static_cast<size_t>(idx);
        const size_t hi = std::min(lo + 1, samples.size() - 1);
        const double frac = idx - static_cast<double>(lo);
        return samples[lo] + frac * (samples[hi] - samples[lo]);
    };

    LatencyStats s;
    s.mean_us = sum / static_cast<double>(samples.size());
    s.p50_us = percentile(0.50);
    s.p99_us = percentile(0.99);
    return s;
}

}  // namespace rf
