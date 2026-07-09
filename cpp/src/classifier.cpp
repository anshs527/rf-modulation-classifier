#include "classifier.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace {
// Phase 10 stage timing.
double us_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now() - t0).count() / 1000.0;
}
}  // namespace

namespace rf {

Classifier::Classifier(const std::string& model_path)
    // ORT logs to stderr; WARNING keeps it quiet unless something is wrong.
    : env_(ORT_LOGGING_LEVEL_WARNING, "rf_classifier") {
    Ort::SessionOptions opts;
    // One intra-op thread: this model is tiny (a few MB of conv work), so a
    // thread pool adds dispatch overhead and run-to-run jitter without
    // buying throughput — and stable numbers are the whole point of the
    // Phase 9 benchmark. Revisit deliberately in Exercise 8 if you suspect
    // otherwise; don't let it float.
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // ORT's file-path API uses wchar_t on Windows (ORTCHAR_T). Byte-wise
    // widening is only correct for ASCII paths — fine for this repo.
    const std::wstring wide_path(model_path.begin(), model_path.end());
    session_ = Ort::Session(env_, wide_path.c_str(), opts);

    // The model was exported with names "iq"/"logits", but read them from
    // the graph instead of hardcoding: if the export ever changes, this
    // fails loudly at load time instead of mysteriously at Run() time.
    Ort::AllocatorWithDefaultOptions alloc;
    input_name_ = session_.GetInputNameAllocated(0, alloc).get();
    output_name_ = session_.GetOutputNameAllocated(0, alloc).get();

    input_shape_ = session_.GetInputTypeInfo(0)
                       .GetTensorTypeAndShapeInfo()
                       .GetShape();                     // {-1, 2, 1024}
    input_shape_[0] = 1;  // dynamic batch dim; this pipeline runs one at a time

    num_classes_ = session_.GetOutputTypeInfo(0)
                       .GetTensorTypeAndShapeInfo()
                       .GetShape()[1];                  // {-1, 24} -> 24

    // CPU tensors only; this MemoryInfo tells CreateTensor the buffer we
    // hand it lives in ordinary host memory.
    memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
}

int Classifier::predict(const float* iq_interleaved, float* logits_out) {
    // ============================================================
    // EXERCISE 6: the inference call
    // See docs/understanding-guide.md § 6 for the full explanation.
    //
    // What this must do:
    //   - Convert the input layout. The dump stores each sample
    //     time-major interleaved (i0, q0, i1, q1, ...), but the model's
    //     input shape is {1, 2, T} channels-first — a flat buffer of
    //     [ all T I-values ][ all T Q-values ]. Getting this wrong won't
    //     crash anything; it will just classify garbage. The parity test
    //     exists to catch exactly that.
    //   - Wrap your converted buffer in an Ort::Value float tensor
    //     (shape = input_shape_, i.e. {1, 2, T}).
    //   - Call session_.Run(...) with that tensor as input "iq" and ask
    //     for output "logits".
    //   - Find the argmax over the num_classes_ output floats; that index
    //     is the prediction.
    //   - If logits_out != nullptr, also copy the num_classes_ floats
    //     into it before returning.
    //
    // What you have available (members set up by the constructor):
    //   iq_interleaved  — length()*2 floats, interleaved as described
    //   session_        — the loaded model
    //   memory_info_    — CPU memory descriptor for CreateTensor
    //   input_name_, output_name_ — std::string ("iq", "logits")
    //   input_shape_    — std::vector<int64_t> {1, 2, T}
    //   num_classes_    — 24
    //
    // What it must produce: the int class index, 0..num_classes_-1.
    //
    // Hints (read only if stuck):
    //   - The non-owning CreateTensor overload does NOT copy your data —
    //     it keeps a pointer. Whatever buffer you wrap must stay alive
    //     until Run() returns. A local std::vector<float> is fine.
    //   - Run() takes arrays of `const char*` names, not std::string —
    //     and the exact signature shifts between ORT versions. The source
    //     of truth is the vendored header:
    //     third_party/onnxruntime-win-x64-1.27.1/include/onnxruntime_cxx_api.h
    //   - Run() returns a std::vector<Ort::Value>; the output tensor's
    //     float data is reachable via GetTensorData<float>().
    // ============================================================
    // Exercise 8 note: this is still the Exercise 6 solution, split in two —
    // the conversion loop moved to convert_to_planar(), the tensor wrap /
    // Run / argmax to predict_planar() — so the pipelined benchmark can run
    // the halves on different threads. Logic is line-for-line unchanged.
    auto stamp = std::chrono::steady_clock::now();
    std::vector<float> iq_channels(input_shape_[1] * input_shape_[2]);
    convert_to_planar(iq_interleaved, iq_channels.data());
    stage_times_.convert_us = us_since(stamp);
    return predict_planar(iq_channels.data(), logits_out);
}

void Classifier::convert_to_planar(const float* iq_interleaved, float* out_planar) const {
    for (int64_t t = 0; t < input_shape_[2]; ++t) {
        out_planar[t] = iq_interleaved[t * 2];
        out_planar[input_shape_[2] + t] = iq_interleaved[t * 2 + 1];
    }
}

int Classifier::predict_planar(const float* iq_planar, float* logits_out) {
    auto stamp = std::chrono::steady_clock::now();
    // const_cast: CreateTensor's non-owning overload wants float*, but ORT
    // only reads the input buffer; the promise the cast makes is kept.
    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, const_cast<float*>(iq_planar),
        static_cast<size_t>(input_shape_[1] * input_shape_[2]),
        input_shape_.data(), input_shape_.size());
    const char* input_name_cstr = input_name_.c_str();
    const char* output_name_cstr = output_name_.c_str();
    auto logits = session_.Run(Ort::RunOptions{nullptr}, &input_name_cstr, &input_tensor, 1, &output_name_cstr, 1);
    stage_times_.run_us = us_since(stamp);

    stamp = std::chrono::steady_clock::now();
    auto args = logits[0].GetTensorData<float>();
    if (logits_out != nullptr) {
        std::copy(args, args + num_classes_, logits_out);
    }
    int argmax = 0;
    for (int64_t i = 1; i < num_classes_; ++i) {
        if (args[i] > args[argmax]) {
            argmax = static_cast<int>(i);
        }
    }
    stage_times_.argmax_us = us_since(stamp);
    return argmax;
}
}  // namespace rf
