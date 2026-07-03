// Phase 7 smoke test: prove we can compile against and link ONNX Runtime
// before writing any application logic. If this builds and runs, every
// toolchain problem is behind us and later errors are our own bugs.
#include <cstdio>

#include <onnxruntime_cxx_api.h>

int main() {
    // GetVersionString comes from the C API base; it does not need an
    // Ort::Env, so this exercises the link without touching any state.
    std::printf("ONNX Runtime version: %s\n", OrtGetApiBase()->GetVersionString());
    std::printf("ORT_API_VERSION (compile-time): %d\n", ORT_API_VERSION);

    // Constructing an Env proves the DLL actually initializes, not just
    // that the symbols resolve.
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "rf_hello");
    std::printf("Ort::Env constructed OK\n");
    return 0;
}
