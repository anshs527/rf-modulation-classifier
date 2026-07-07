// Prints a summary of an RFIQ dump file. Doubles as the reader's smoke test:
// if the numbers here match what dump_test_bin.py printed when writing the
// file, both sides agree on the format.
#include <cstdio>
#include <exception>

#include "iq_reader.hpp"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: rf_dump_info <test_slice.bin>\n");
        return 2;
    }
    try {
        const rf::IqDump d = rf::load_iq_dump(argv[1]);
        std::printf("samples:      %u\n", d.count);
        std::printf("timesteps:    %u\n", d.length);
        std::printf("classes:      %u\n", d.num_classes);

        size_t agree = 0;
        for (uint32_t n = 0; n < d.count; ++n) {
            if (d.labels[n] == d.preds[n]) ++agree;
        }
        std::printf("pytorch accuracy on slice: %.3f  (must match the dump script's print)\n",
                    static_cast<double>(agree) / d.count);

        const float* s0 = d.sample_iq(0);
        std::printf("sample 0: label=%d pred=%d snr=%d dB, first pairs: "
                    "(%.4f, %.4f) (%.4f, %.4f)\n",
                    d.labels[0], d.preds[0], d.snr[0], s0[0], s0[1], s0[2], s0[3]);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
