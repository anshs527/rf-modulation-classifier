// Latency statistics for the Phase 9 benchmark.
#pragma once

#include <vector>

namespace rf {

struct LatencyStats {
    double mean_us = 0.0;
    double p50_us = 0.0;
    double p99_us = 0.0;
};

// EXERCISE 7 (part 2 — body in stats.cpp): mean, median, 99th percentile
// of a vector of per-inference latencies in microseconds.
//
// Taken by value on purpose: percentiles need sorted data, and sorting a
// private copy means the caller's recording order (which the CSV preserves)
// is left intact.
LatencyStats compute_stats(std::vector<double> samples);

}  // namespace rf
