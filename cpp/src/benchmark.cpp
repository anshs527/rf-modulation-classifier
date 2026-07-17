// Streaming latency benchmark (Phase 9).
//
// Pushes every sample in a dump file through the classifier one at a time —
// the way a live receiver would see them — and reports:
//   - accuracy vs. the true labels, and exact-parity vs. PyTorch's preds
//   - per-inference latency: mean / p50 / p99 (Exercise 7)
//   - peak process memory (working set ~= RSS)
//   - optionally, every raw latency to a CSV for Phase 10's histogram
//
// usage: rf_benchmark <model.onnx> <test_slice.bin> [latencies.csv]
#define NOMINMAX
#include <windows.h>
#include <psapi.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <exception>
#include <fstream>
#include <mutex>
#include <thread>
#include <vector>

#include "classifier.hpp"
#include "iq_reader.hpp"
#include "stats.hpp"

namespace {

// Peak working set is the Windows analogue of peak RSS: the high-water mark
// of physical memory the process actually touched. Sampled once at the end —
// the question is "how much memory does this pipeline need", not its timeline.
size_t peak_rss_bytes() {
    PROCESS_MEMORY_COUNTERS pmc{};
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
    return pmc.PeakWorkingSetSize;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: rf_benchmark <model.onnx> <test_slice.bin> [latencies.csv] [--pipeline]\n");
        return 2;
    }
    // Exercise 8: --pipeline switches the streaming loop to a two-thread
    // producer/consumer pipeline; everything else is identical, so the two
    // modes are an apples-to-apples before/after.
    bool pipeline = false;
    const char* csv_path = nullptr;
    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--pipeline") == 0) {
            pipeline = true;
        } else {
            csv_path = argv[i];
        }
    }
    try {
        // Phase 10: file I/O is a one-time cost, timed separately from the
        // per-inference stages — lumping it in would smear a single 8 MB
        // read across 1000 latency numbers it has nothing to do with.
        const auto io0 = std::chrono::steady_clock::now();
        const rf::IqDump d = rf::load_iq_dump(argv[2]);
        const double io_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 std::chrono::steady_clock::now() - io0).count() / 1e6;
        rf::Classifier clf(argv[1]);

        // Warm-up, unrecorded: the first Run() calls pay one-time costs
        // (arena growth, lazy kernel setup, cold caches) that belong to
        // startup, not steady-state latency. 50 is arbitrary but cheap.
        const uint32_t warmup = std::min(50u, d.count);
        for (uint32_t n = 0; n < warmup; ++n) {
            (void)clf.predict(d.sample_iq(n));
        }

        std::vector<double> lat_us;
        std::vector<double> convert_us, run_us, argmax_us;  // Phase 10 stages
        std::vector<int32_t> cpp_preds(d.count, -1);

        const auto wall0 = std::chrono::steady_clock::now();
        if (!pipeline) {
        for (uint32_t n = 0; n < d.count; ++n) {
            int pred = -1;
            // ============================================================
            // EXERCISE 7 (part 1): the timing harness
            // See docs/understanding-guide.md § 7 for the full explanation.
            //
            // What this must do:
            //   - Take a timestamp, call
            //         pred = clf.predict(d.sample_iq(n));
            //     take a second timestamp.
            //   - Convert the difference to microseconds (as a double)
            //     and push it onto lat_us.
            //
            // What you have available: <chrono> is included; `clf`, `d`,
            //   `n`, `lat_us`, and `pred` are in scope.
            //
            // What it must produce: pred holds the class index, and
            //   lat_us grows by exactly one entry per loop iteration.
            //
            // Decisions you must make and be able to defend:
            //   - steady_clock, not system_clock. Why? (What happens to
            //     a latency measurement if NTP steps the wall clock?)
            //   - This placement times predict() only — layout conversion
            //     and tensor wrapping INSIDE predict are included, the
            //     dump-file read is not. Both choices are defensible;
            //     know which one you measured and say so in the report.
            //
            // Hints (read only if stuck):
            //   - std::chrono::steady_clock::now() twice, subtract.
            //   - duration_cast<std::chrono::nanoseconds>(...).count(),
            //     then /1000.0 — casting to microseconds first throws
            //     away sub-microsecond resolution.
            // ============================================================
            const auto t0 = std::chrono::steady_clock::now();
            pred = clf.predict(d.sample_iq(n));
            const auto t1 = std::chrono::steady_clock::now();
            lat_us.push_back(
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1000.0);

            // Phase 10: per-stage split of the latency just recorded.
            const auto& st = clf.last_stage_times();
            convert_us.push_back(st.convert_us);
            run_us.push_back(st.run_us);
            argmax_us.push_back(st.argmax_us);

            cpp_preds[n] = pred;
        }
        } else {
            // Exercise 8: the production line. The producer thread converts
            // samples and pushes them into a BOUNDED queue; this (consumer)
            // thread pulls converted samples and runs the model. While the
            // model chews on sample n, sample n+1 is being converted — the
            // 1.9 us convert stage hides behind the 776 us Run stage.
            //
            // The queue is bounded (capacity 8) for backpressure: the
            // producer's stage is ~400x faster than the consumer's, and an
            // unbounded queue would just convert all 1000 samples up front
            // — 8 MB of planar copies sitting in RAM pretending to be a
            // pipeline. Same reason a crawler bounds its fetch queue when
            // parsing is the slow side.
            struct Item {
                uint32_t n = 0;
                std::vector<float> planar;   // converted, channels-first
                double convert_us = 0.0;     // producer-side stage time
            };
            std::deque<Item> queue;
            std::mutex m;
            std::condition_variable can_push, can_pop;
            bool producer_done = false;
            const size_t capacity = 8;

            std::thread producer([&] {
                for (uint32_t n = 0; n < d.count; ++n) {
                    Item item;
                    item.n = n;
                    const auto c0 = std::chrono::steady_clock::now();
                    item.planar.resize(static_cast<size_t>(clf.length()) * 2);
                    clf.convert_to_planar(d.sample_iq(n), item.planar.data());
                    item.convert_us =
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - c0).count() / 1000.0;

                    std::unique_lock<std::mutex> lock(m);
                    can_push.wait(lock, [&] { return queue.size() < capacity; });
                    queue.push_back(std::move(item));
                    can_pop.notify_one();
                }
                std::lock_guard<std::mutex> lock(m);
                producer_done = true;
                can_pop.notify_one();
            });

            for (;;) {
                Item item;
                {
                    std::unique_lock<std::mutex> lock(m);
                    can_pop.wait(lock, [&] { return !queue.empty() || producer_done; });
                    if (queue.empty()) break;  // done and drained
                    item = std::move(queue.front());
                    queue.pop_front();
                    can_push.notify_one();
                }
                // NOTE the measurement change: in this mode "latency" is the
                // consumer side only (tensor wrap + Run + argmax). The
                // convert stage still happened — on the other thread — and
                // is reported in its own column from the producer's clock.
                const auto t0 = std::chrono::steady_clock::now();
                const int pred = clf.predict_planar(item.planar.data());
                const auto t1 = std::chrono::steady_clock::now();
                lat_us.push_back(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1000.0);

                const auto& st = clf.last_stage_times();
                convert_us.push_back(item.convert_us);
                run_us.push_back(st.run_us);
                argmax_us.push_back(st.argmax_us);
                cpp_preds[item.n] = pred;
            }
            producer.join();
        }
        const double wall_s =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - wall0).count() / 1e9;

        // Correctness first — latency numbers for wrong answers are noise.
        size_t correct = 0, parity = 0;
        for (uint32_t n = 0; n < d.count; ++n) {
            if (cpp_preds[n] == d.labels[n]) ++correct;
            if (cpp_preds[n] == d.preds[n]) ++parity;
        }
        std::printf("mode:               %s\n", pipeline ? "pipelined (2 threads)" : "sequential");
        std::printf("samples:            %u\n", d.count);
        std::printf("total wall time:    %.3f s  (%.0f inferences/s)\n",
                    wall_s, d.count / wall_s);
        std::printf("accuracy:           %.3f (PyTorch got %.3f on this slice)\n",
                    static_cast<double>(correct) / d.count,
                    [&] { size_t a = 0; for (uint32_t n = 0; n < d.count; ++n) if (d.preds[n] == d.labels[n]) ++a; return static_cast<double>(a) / d.count; }());
        std::printf("parity with torch:  %zu / %u %s\n", parity, d.count,
                    parity == d.count ? "(exact)" : "(MISMATCH — layout bug?)");

        const rf::LatencyStats s = rf::compute_stats(lat_us);
        std::printf("latency mean:       %.1f us\n", s.mean_us);
        std::printf("latency p50:        %.1f us\n", s.p50_us);
        std::printf("latency p99:        %.1f us\n", s.p99_us);
        std::printf("peak RSS:           %.1f MB\n", peak_rss_bytes() / 1e6);

        // Phase 10: where does the time actually go?
        const rf::LatencyStats sc = rf::compute_stats(convert_us);
        const rf::LatencyStats sr = rf::compute_stats(run_us);
        const rf::LatencyStats sa = rf::compute_stats(argmax_us);
        std::printf("stage breakdown (mean per inference):\n");
        std::printf("  convert+wrap:  %8.1f us  (%4.1f%%)\n", sc.mean_us, 100.0 * sc.mean_us / s.mean_us);
        std::printf("  session.Run:   %8.1f us  (%4.1f%%)\n", sr.mean_us, 100.0 * sr.mean_us / s.mean_us);
        std::printf("  argmax+copy:   %8.1f us  (%4.1f%%)\n", sa.mean_us, 100.0 * sa.mean_us / s.mean_us);
        std::printf("  file I/O (one-time, all %u samples): %.1f ms\n", d.count, io_ms);

        if (csv_path != nullptr) {
            std::ofstream csv(csv_path);
            csv << "index,latency_us,convert_us,run_us,argmax_us,snr_db,label,pred,correct\n";
            for (uint32_t n = 0; n < d.count; ++n) {
                csv << n << ',' << lat_us[n] << ',' << convert_us[n] << ','
                    << run_us[n] << ',' << argmax_us[n] << ',' << d.snr[n] << ','
                    << d.labels[n] << ',' << cpp_preds[n] << ','
                    << (cpp_preds[n] == d.labels[n] ? 1 : 0) << '\n';
            }
            std::printf("wrote %s\n", csv_path);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
