// Exercise 6's feedback loop: run the C++ pipeline on the first 100 dump
// samples and demand EXACT agreement with the PyTorch predictions saved in
// the dump. Same ONNX graph, same ONNX Runtime version, same float32 inputs
// — bit-identical logits, so unlike Exercise 4 there is no tolerance here.
// If this fails, the bug is in YOUR tensor layout, not the model.
//
// usage: rf_parity_test <model.onnx> <test_slice.bin>
#include <algorithm>
#include <cstdio>
#include <exception>
#include <vector>

#include "classifier.hpp"
#include "iq_reader.hpp"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: rf_parity_test <model.onnx> <test_slice.bin>\n");
        return 2;
    }
    try {
        const rf::IqDump d = rf::load_iq_dump(argv[2]);
        rf::Classifier clf(argv[1]);

        const uint32_t n_check = std::min(100u, d.count);
        std::vector<float> logits(d.num_classes);
        uint32_t mismatches = 0;

        for (uint32_t n = 0; n < n_check; ++n) {
            const int pred = clf.predict(d.sample_iq(n), logits.data());
            if (pred == d.preds[n]) continue;

            ++mismatches;
            if (mismatches <= 3) {  // enough to diagnose, not a wall of text
                const float* torch_logits = d.logits.data() + static_cast<size_t>(n) * d.num_classes;
                std::fprintf(stderr,
                             "MISMATCH sample %u: torch pred=%d, cpp pred=%d (snr=%d dB)\n"
                             "  first 4 logits  torch: %.5f %.5f %.5f %.5f\n"
                             "                  cpp:   %.5f %.5f %.5f %.5f\n",
                             n, d.preds[n], pred, d.snr[n],
                             torch_logits[0], torch_logits[1], torch_logits[2], torch_logits[3],
                             logits[0], logits[1], logits[2], logits[3]);
            }
        }

        if (mismatches != 0) {
            std::fprintf(stderr, "FAIL: %u / %u predictions disagree with PyTorch.\n"
                                 "Logits wildly different -> input layout is wrong (interleaved vs planar).\n"
                                 "Logits identical but preds differ -> argmax bug.\n",
                         mismatches, n_check);
            return 1;
        }
        std::printf("PASS: %u / %u C++ predictions exactly match PyTorch.\n", n_check, n_check);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL (exception): %s\n", e.what());
        return 1;
    }
}
