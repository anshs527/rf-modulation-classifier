#include "iq_reader.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace rf {
namespace {

// All multi-byte fields are little-endian per the format spec. x86-64 is
// little-endian, so plain reads are correct here; this static_assert-style
// helper documents the assumption rather than silently relying on it.
template <typename T>
void read_or_throw(std::ifstream& f, T* out, size_t count, const char* what) {
    f.read(reinterpret_cast<char*>(out), static_cast<std::streamsize>(count * sizeof(T)));
    if (!f) {
        throw std::runtime_error(std::string("iq_reader: truncated file while reading ") + what);
    }
}

}  // namespace

IqDump load_iq_dump(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("iq_reader: cannot open " + path);
    }

    char magic[4];
    read_or_throw(f, magic, 4, "magic");
    if (std::memcmp(magic, "RFIQ", 4) != 0) {
        throw std::runtime_error("iq_reader: bad magic in " + path + " (not an RFIQ dump)");
    }

    uint32_t version = 0;
    read_or_throw(f, &version, 1, "version");
    if (version != 1) {
        throw std::runtime_error("iq_reader: unsupported RFIQ version " + std::to_string(version));
    }

    IqDump d;
    read_or_throw(f, &d.count, 1, "count");
    read_or_throw(f, &d.length, 1, "length");
    read_or_throw(f, &d.num_classes, 1, "num_classes");
    if (d.count == 0 || d.length == 0 || d.num_classes == 0) {
        throw std::runtime_error("iq_reader: zero-sized dimension in header");
    }

    d.iq.resize(static_cast<size_t>(d.count) * d.length * 2);
    d.labels.resize(d.count);
    d.preds.resize(d.count);
    d.logits.resize(static_cast<size_t>(d.count) * d.num_classes);
    d.snr.resize(d.count);

    read_or_throw(f, d.iq.data(), d.iq.size(), "iq");
    read_or_throw(f, d.labels.data(), d.labels.size(), "labels");
    read_or_throw(f, d.preds.data(), d.preds.size(), "preds");
    read_or_throw(f, d.logits.data(), d.logits.size(), "logits");
    read_or_throw(f, d.snr.data(), d.snr.size(), "snr");

    // The file must end exactly here — trailing bytes mean the writer and
    // reader disagree about the format, which is worth failing over.
    f.peek();
    if (!f.eof()) {
        throw std::runtime_error("iq_reader: trailing bytes after payload in " + path);
    }
    return d;
}

}  // namespace rf
