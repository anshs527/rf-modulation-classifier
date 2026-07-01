# C++ inference pipeline for the RF modulation classifier (Phases 7-10).
#
# Dependency policy: ONNX Runtime only, vendored as a prebuilt binary under
# third_party/ rather than found via find_package — there is no standard
# system install location for ONNX Runtime on Windows, so vendoring makes
# the build reproducible on a fresh machine with zero setup steps.
# FFTW is deliberately absent: it is only needed for the spectrogram input
# representation, and this project chose raw IQ (see instructions.md Phase 8).
cmake_minimum_required(VERSION 3.20)
project(rf_classifier_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-win-x64-1.27.1")
if(NOT EXISTS "${ORT_ROOT}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR
        "ONNX Runtime not found at ${ORT_ROOT}. Download the win-x64 release "
        "zip from https://github.com/microsoft/onnxruntime/releases and "
        "extract it into third_party/.")
endif()

# IMPORTED target: wraps the prebuilt DLL so executables can just
# target_link_libraries() it like a normal CMake target. IMPORTED_IMPLIB is
# the MSVC import library (.lib) the linker needs; IMPORTED_LOCATION is the
# DLL loaded at run time.
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ORT_ROOT}/lib/onnxruntime.dll"
    IMPORTED_IMPLIB   "${ORT_ROOT}/lib/onnxruntime.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_ROOT}/include"
)

# Copies onnxruntime.dll next to a target's .exe after building it — Windows
# resolves DLLs from the executable's own directory, and without this every
# run would need PATH surgery first.
function(copy_ort_dll target)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ORT_ROOT}/lib/onnxruntime.dll"
            "$<TARGET_FILE_DIR:${target}>"
    )
endfunction()

# Phase 7 smoke test: links the library, prints its version, exits.
# Prove the toolchain works before burying it under application logic.
add_executable(rf_hello src/main.cpp)
target_link_libraries(rf_hello PRIVATE onnxruntime)
copy_ort_dll(rf_hello)

# Reader for the flat binary test-set dump produced by
# python/src/dump_test_bin.py, plus a small executable that prints a summary
# of a dump file (doubles as the reader's smoke test).
add_library(iq_reader STATIC src/iq_reader.cpp)
target_include_directories(iq_reader PUBLIC src)

add_executable(rf_dump_info src/dump_info.cpp)
target_link_libraries(rf_dump_info PRIVATE iq_reader)

# Phase 9: inference (classifier + Exercise 6) and latency stats (Exercise 7).
add_library(rf_core STATIC src/classifier.cpp src/stats.cpp)
target_include_directories(rf_core PUBLIC src)
target_link_libraries(rf_core PUBLIC onnxruntime)

# psapi provides GetProcessMemoryInfo for the peak-RSS measurement.
add_executable(rf_benchmark src/benchmark.cpp)
target_link_libraries(rf_benchmark PRIVATE rf_core iq_reader psapi)
copy_ort_dll(rf_benchmark)

# Feedback loops for the exercises: both FAIL until implemented.
add_executable(rf_parity_test tests/parity_test.cpp)
target_link_libraries(rf_parity_test PRIVATE rf_core iq_reader)
copy_ort_dll(rf_parity_test)

add_executable(rf_stats_test tests/stats_test.cpp)
target_link_libraries(rf_stats_test PRIVATE rf_core)
