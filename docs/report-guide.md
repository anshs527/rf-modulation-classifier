# How to Write the Report — Guide + Numerical Findings

This is a guide for writing `docs/report.md` (Phase 11), not the report itself.
It collects the structure, the emphasis, and every measured number in one place
so you can write the report without re-running anything.

---

## 1. Who you're writing for, and the one rule

Write for a technical interviewer who has 10 minutes and will skim. They are
not grading your accuracy — this is a well-studied task on a public benchmark.
They are checking whether you can **defend every decision and every number**.

The one rule: **never state a number without stating how it was measured.**
"775 µs mean latency" is noise; "775 µs mean over 1000 sequential single-sample
inferences, `steady_clock` around `predict()` only, after 50 unrecorded warm-up
runs, on CPU" is a result.

## 2. Suggested structure

1. **Overview (3–4 sentences).** What was built end-to-end: CNN trained on
   RadioML 2018.01a → ONNX export → C++ streaming inference benchmark. State
   the headline numbers immediately (92% high-SNR accuracy, 775 µs mean /
   1.7 ms p99, exact PyTorch parity). Don't bury them.

2. **Problem and motivation (short).** Modulation classification as a SIGINT /
   spectrum-sensing problem; classical feature-engineering approaches vs.
   learned ones. Keep it to a paragraph — `desc.md` has the full version.

3. **Dataset.** RadioML 2018.01a: 2,555,904 examples = 24 modulation classes ×
   26 SNR levels (−20 to +30 dB in 2 dB steps) × 4,096 examples each; each
   example is 1024 complex (I, Q) samples. Describe the split (below) and note
   the file is sorted by class/SNR, which is why the split shuffles first.

4. **Architecture and design decisions.** For each decision, give the
   alternative you rejected and why:
   - **Raw IQ + 1D CNN** over spectrogram + 2D CNN (simpler; matches the
     original O'Shea work; magnitude spectrograms discard phase, and phase is
     exactly where PSK/QAM encode information).
   - **The layer plan** (see `python/src/model.py` docstring): 4 ×
     (Conv1d → ReLU → MaxPool), channels 64-64-128-128, then FC 8192→256→24.
     Wide first kernel (7) ≈ one symbol at RadioML's ~8 samples/symbol.
     ~2.2M parameters, 8.9 MB as float32.
   - **ONNX over TorchScript** (lighter C++ dependency footprint than
     LibTorch; framework-agnostic).
   - Logits out, no softmax (CrossEntropyLoss applies it internally).

5. **Training methodology.** Adam lr=1e-3, batch 256, 10 epochs, seed 0,
   90/10 train/val on a shuffled permutation. Mention the overfit-one-batch
   sanity check if you ran it. State the selection-leakage caveat from
   `evaluate.py`: `best.pt` was selected on the full val block, and the test
   set is the first half of that block — negligible with ~10 checkpoints over
   ~255k examples, but name it before an interviewer does.

6. **Accuracy results — the centerpiece.** Lead with the accuracy-by-SNR curve
   (`docs/accuracy_by_snr.png`), not the aggregate. The aggregate (~55%) is
   close to meaningless over a −20..+30 dB sweep, and saying so is the point.
   Then the confusion matrix (`docs/confusion_matrix.png`): identify *which*
   classes collapse into each other at low SNR and connect it to theory —
   e.g. higher-order QAM variants becoming indistinguishable in noise is an
   information-theoretic floor, not a model failure.

7. **The C++ pipeline and how it was validated.** The RFIQ dump format
   (interleaved IQ + PyTorch's own preds/logits), and the parity test: the
   C++ pipeline reproduces PyTorch's argmax **exactly, 1000/1000** — same
   graph, same runtime family, no tolerance needed. Contrast with the
   Python-side export check, which compares logits with a float tolerance
   (different runtimes may reorder float ops). This section answers "how do
   you know the deployed model is the model you trained."

8. **Benchmark results.** The table in §4 below, plus the histogram
   (`docs/latency_histogram.png`). State the methodology in full: what's
   inside the timer, warm-up policy, clock choice, percentile definition
   (nearest-rank on the sorted vector), n=1000, CPU, Release build.

9. **The optimization experiment — report the negative result.** Hypothesis:
   overlap the layout-conversion stage with inference using a two-thread
   bounded-queue producer/consumer. Result: no improvement (mean actually
   ~2–3% worse; see §4) — because profiling shows `session.Run` is 99.7% of
   per-inference time and convert is ~1.4 µs. The honest conclusion: the only
   ways to go faster are inside the model (batching, quantization, a smaller
   architecture), not around it. This is the strongest section in the report
   if you write it plainly: hypothesis → prediction → measurement → wrong →
   why.

10. **Limitations.** Say these before being asked:
    - Benchmark dataset, not a fielded system; "streaming" is a simulation
      over a static file, no live RF capture.
    - Low-SNR accuracy is near chance — expected and physically meaningful.
    - Not state-of-the-art and doesn't claim to be; the contribution is the
      end-to-end pipeline and measurement rigor.
    - Latency measured on one desktop CPU under Windows; numbers are
      machine-specific.

11. **Future work / applications** — see §5 and §6 below.

## 3. What to focus on (in order of interview value)

1. The accuracy-vs-SNR curve and what it means physically.
2. The exact-parity validation story (most candidates have never checked).
3. Measurement methodology — every timing decision made deliberately.
4. The negative optimization result, reported without spin.
5. Design decisions stated as tradeoffs, not defaults.

## 4. Numerical findings (all measured, 2026-07-16)

**Model** — 1D CNN, ~2.2M parameters (8.9 MB float32 checkpoint / ONNX).
Input (2, 1024) float32, output 24 logits.

**Dataset & split** — 2,555,904 total; shuffled with seed 0; 90% train
(2,300,314), 10% val (255,590); test = first half of val block (127,795),
never trained on.

**Accuracy (PyTorch, held-out test)**
| Metric | Value |
|---|---|
| Overall (full −20..+30 dB sweep) | ~55% (54.9% on the 1000-sample benchmark slice) |
| Plateau above +10 dB | ~92% |
| At 0 dB | ~50% |
| At −10 dB | ~12% |
| At −20 dB | ~4% ≈ chance (1/24) |

Read exact per-bucket values off `evaluate.py`'s stdout if you want the full
table; the curve is `docs/accuracy_by_snr.png`.

**C++ benchmark (sequential mode, n=1000 single-sample inferences, CPU,
Release, 50-run warm-up, steady_clock around `predict()` only)**
| Metric | Value |
|---|---|
| Parity with PyTorch argmax | 1000/1000 exact |
| Accuracy on slice | 0.549 (identical to PyTorch: 0.549) |
| Throughput | ~1,290 inferences/s |
| Latency mean | 775 µs |
| Latency p50 | 725 µs |
| Latency p99 | ~1.7 ms (1658 µs; an earlier run of the same build measured 1343 µs — tail latency is the noisiest number here, say so) |
| Peak RSS (peak working set) | 54.7 MB |
| One-time file I/O (8.3 MB dump) | ~6.4 ms |

**Stage breakdown (mean per inference, sequential mode)**
| Stage | Time | Share |
|---|---|---|
| convert + tensor wrap | 1.4 µs | 0.2% |
| `session.Run` | 772.9 µs | 99.7% |
| argmax + copy | 0.1 µs | <0.1% |

**Pipelined mode (2 threads, bounded queue of 8) — the negative result**
| Metric | Sequential | Pipelined |
|---|---|---|
| mean | 775.4 µs | 795.7 µs |
| p50 | 725.0 µs | 764.4 µs |
| p99 | 1658 µs | 1548 µs |
| throughput | 1289/s | 1251/s |

Interpretation: there was ~1.4 µs of work available to hide behind a ~775 µs
stage; the added synchronization cost more than it saved. The measurement
caveat to state: in pipelined mode the recorded "latency" is the consumer
side only (wrap + Run + argmax); convert happens on the producer thread and
is reported from its own clock.

Reproduce any of this with:
```
cpp\build\Release\rf_benchmark.exe python\checkpoints\model.onnx cpp\data\test_slice.bin [latencies.csv] [--pipeline]
```

## 5. Potential expansions

- **Quantization / model surgery.** INT8 quantization via ONNX Runtime, or
  shrinking the FC head (fc1 is 2.1M of the 2.2M parameters) — the profiling
  already proved the model *is* the bottleneck, so this is the justified next
  optimization, and it sets up an accuracy-vs-latency tradeoff curve.
- **Batched streaming.** Batch N windows per `Run()` and measure the
  throughput-vs-per-sample-latency tradeoff — the classic real-time systems
  tension, and the benchmark harness already supports the comparison pattern.
- **The spectrogram path.** Implement the STFT front-end (FFTW + the RAII
  plan wrapper, Exercise 5) and compare raw-IQ vs. spectrogram accuracy —
  turning the input-representation decision from an argument into a
  measurement.
- **Live capture with an SDR.** An RTL-SDR (~$30) turns the simulated stream
  into a real one and exposes the train/test distribution gap
  (synthetic-channel training data vs. real hardware impairments) — a genuine
  research-flavored finding.
- **Robustness / open-set.** Confidence thresholds or open-set recognition for
  "none of the 24 classes" inputs; adversarial-perturbation sensitivity, which
  connects directly to the anti-spoofing thread in your GNSS work.
- **Alternative architectures.** ResNet-style 1D blocks, or a small
  transformer/LSTM over IQ, vs. accuracy *and* the p99 cost at inference —
  keeping the systems framing rather than becoming a pure modeling exercise.
- **Edge deployment.** Cross-compile the pipeline for an ARM board (Raspberry
  Pi / Jetson) and re-measure — the resource-constrained numbers are the ones
  a fielded system cares about.

## 6. Real-world applications

- **SIGINT / electronic warfare:** classifying intercepted, uncooperative
  emitters is the canonical use; modulation ID is the first stage before
  demodulation or threat classification.
- **Dynamic spectrum access / cognitive radio:** sensing what's occupying a
  band before transmitting; relevant to CBRS-style shared-spectrum regimes.
- **Spectrum enforcement and interference hunting:** regulators and cellular
  operators localizing and identifying rogue transmitters.
- **Drone detection:** classifying control/video links by modulation as a
  counter-UAS signal.
- **GNSS anti-spoofing (bridge to your other project):** a spoofed GNSS
  signal is an RF signal with the wrong provenance; the same
  learned-RF-fingerprinting machinery applies to authenticating signals, not
  just classifying them.
- **The deployment pattern itself** — train in Python, export a portable
  graph, serve from C++ with parity tests and latency instrumentation — is
  the standard shape of production ML in latency-sensitive systems (defense,
  telecom, HFT), independent of the RF payload.
