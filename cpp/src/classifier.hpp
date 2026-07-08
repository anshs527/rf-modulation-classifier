// ONNX Runtime wrapper for the exported modulation classifier.
//
// Split of responsibilities (instructions.md Phase 9):
//   - construction = one-time plumbing (env, session options, model load,
//     input/output names and shapes) — written for you, read it once;
//   - predict()    = the per-sample inference call — EXERCISE 6, yours.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace rf {

class Classifier {
public:
    // model_path: path to the ONNX file exported by python/src/export.py.
    // Throws Ort::Exception if the file is missing or not a valid model.
    explicit Classifier(const std::string& model_path);

    // EXERCISE 6 (body in classifier.cpp).
    //
    // iq_interleaved: one sample's IQ block exactly as stored in the dump —
    //   length()*2 floats, time-major interleaved: i0, q0, i1, q1, ...
    // logits_out: optional; if non-null, the num_classes() raw logits are
    //   copied there (the parity test passes this to diagnose mismatches).
    // Returns the predicted class index (argmax over the logits).
    int predict(const float* iq_interleaved, float* logits_out = nullptr);

    // Exercise 8 (pipelining): predict() split into its two halves, so a
    // producer thread can convert sample n+1 while this thread runs the
    // model on sample n.
    //
    // convert_to_planar is const and touches no session state — safe to
    // call from any thread. out_planar must hold length()*2 floats.
    void convert_to_planar(const float* iq_interleaved, float* out_planar) const;
    // predict_planar takes already-converted channels-first data. NOT
    // thread-safe against itself (stage_times_); one consumer thread only.
    int predict_planar(const float* iq_planar, float* logits_out = nullptr);

    int64_t length() const { return input_shape_[2]; }        // T, 1024
    int64_t num_classes() const { return num_classes_; }      // 24

    // Phase 10 instrumentation: wall time of each stage of the most recent
    // predict() call. Filled on every call; ~two extra clock reads per
    // stage, noise next to a ~800 us inference.
    struct StageTimes {
        double convert_us = 0.0;  // interleaved -> planar + tensor wrap
        double run_us = 0.0;      // session_.Run itself
        double argmax_us = 0.0;   // logits copy + argmax
    };
    const StageTimes& last_stage_times() const { return stage_times_; }

private:
    // Declaration order matters: members are destroyed bottom-up, so the
    // session (which uses the env) must be declared after it.
    Ort::Env env_;
    Ort::Session session_{nullptr};
    Ort::MemoryInfo memory_info_{nullptr};

    std::string input_name_;               // "iq"
    std::string output_name_;              // "logits"
    std::vector<int64_t> input_shape_;     // {1, 2, T} — batch fixed to 1
    int64_t num_classes_ = 0;
    StageTimes stage_times_;
};

}  // namespace rf
