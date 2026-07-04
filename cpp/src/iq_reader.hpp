// Reader for the RFIQ test-set dump written by python/src/dump_test_bin.py.
// Format spec lives in that file's docstring; the two must move together.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rf {

struct IqDump {
    uint32_t count = 0;        // N samples
    uint32_t length = 0;       // T timesteps per sample (1024)
    uint32_t num_classes = 0;  // 24

    // INTERLEAVED, time-major: sample n's data is
    //   iq[n*length*2 + 0] = i0, [.. + 1] = q0, [.. + 2] = i1, ...
    // This is NOT the channels-first (2, T) layout the ONNX model consumes;
    // converting is the inference code's responsibility (Exercise 6).
    std::vector<float> iq;

    std::vector<int32_t> labels;  // true class per sample
    std::vector<int32_t> preds;   // PyTorch's argmax per sample — the parity target
    std::vector<float> logits;    // PyTorch's logits, count * num_classes, for debugging
    std::vector<int32_t> snr;     // SNR in dB per sample

    // Convenience: pointer to sample n's interleaved IQ block (length*2 floats).
    const float* sample_iq(uint32_t n) const { return iq.data() + static_cast<size_t>(n) * length * 2; }
};

// Loads and validates a dump file. Throws std::runtime_error with a specific
// message on any malformation (bad magic, wrong version, truncated file) —
// a benchmark fed a corrupt input should die loudly, not measure garbage.
IqDump load_iq_dump(const std::string& path);

}  // namespace rf
