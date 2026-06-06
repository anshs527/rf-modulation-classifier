# Claude Code Build Instructions — RF Modulation Classifier

## Read this first (instructions TO Claude Code)

This project is a learning project. The user must be able to defend every design decision in a technical interview. Therefore:

**You must NOT write the code sections marked `EXERCISE`.** For each of these, you will:
1. Write the surrounding scaffolding, imports, function signature, and docstring.
2. Leave the body as a clearly marked block:
   ```
   # ============================================================
   # EXERCISE N: <name>
   # See docs/understanding-guide.md § N for the full explanation.
   #
   # What this must do:
   #   - <bullet>
   #   - <bullet>
   # What you have available: <variables in scope, their shapes/types>
   # What it must produce: <return value, shape/type>
   #
   # Hints (read only if stuck):
   #   - <hint>
   # ============================================================
   raise NotImplementedError("EXERCISE N — see understanding guide § N")
   ```
3. Write a test or assertion that will FAIL until the user implements it correctly, so they get immediate feedback.
4. **Stop and wait.** Do not proceed to the next phase until the user says their implementation passes.

When the user asks for help on an exercise, **do not give them the code.** Ask what they've tried, point them at the relevant concept, and give progressively stronger hints. Only write the code if they explicitly say "just show me the answer" — and if they do, explain every line afterward and ask them a comprehension question about it.

For all NON-exercise code: write it fully and well, but add comments explaining *why*, not *what*. The user is comfortable with modern C++ and is learning ML; calibrate accordingly.

At the end of each phase, print a short summary of what was built and what the user should now be able to explain.

---

## Open decisions (resolve before the noted phase)

- **Export format (before Phase 5):** ONNX (default) vs. TorchScript.
- **C++ pipeline status (before Phase 6):** hard requirement vs. stretch goal. Default: stretch goal — Phases 0–5 must be complete and working first.
- **Input representation (before Phase 2):** raw IQ + 1D CNN (default) vs. spectrogram + 2D CNN.

---

## Phase 0 — Repo setup

**Claude Code writes all of this.**

Create:
```
rf-classifier/
├── python/
│   ├── data/              # gitignored; dataset lives here locally
│   ├── src/
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── export.py
│   ├── tests/
│   └── requirements.txt
├── cpp/                   # Phase 6+
│   ├── src/
│   ├── tests/
│   └── CMakeLists.txt
├── docs/
│   ├── understanding-guide.md   # copy the guide here
│   └── report.md                # user writes in Phase 9
└── README.md
```

`requirements.txt`: `torch`, `numpy`, `h5py`, `matplotlib`, `scikit-learn`, `onnx`, `onnxruntime`, `pytest`.

**Do not download the dataset.** RadioML 2018.01a requires manually accepting terms at https://www.deepsig.ai/datasets. Ask the user for the local path once they've downloaded it.

---

## Phase 1 — Data loading

**Claude Code writes:** `dataset.py` — an `h5py`-backed `torch.utils.data.Dataset` subclass exposing IQ samples, integer class labels, and SNR per example. Lazy-load from HDF5; do not read the whole file into memory.

**Claude Code writes:** an exploration script that prints class distribution, SNR distribution, and plots raw I and Q channels for 3 modulation types at high SNR.

**Before moving on**, Claude Code must ask the user to state, in their own words, what the two channels of an IQ sample physically represent. If the answer is vague, point them at understanding-guide § 2 and the Velardo material.

---

## Phase 2 — Input representation

**User decision, not Claude Code's.** Present both options with tradeoffs (see understanding-guide § 3), recommend raw IQ + 1D CNN, and wait for the user to choose.

---

## Phase 3 — Model architecture

**Claude Code writes:** the `nn.Module` subclass skeleton in `model.py` — the `__init__` with layer definitions, and the class docstring stating the expected input shape `(batch, 2, 1024)` and output shape `(batch, 24)`.

> ### EXERCISE 1 — `forward()`
> Claude Code leaves `forward()` unimplemented.
>
> The user must write the forward pass: pass the input through the conv stack, apply activations and pooling, flatten, pass through the fully connected head, return logits.
>
> **Test provided by Claude Code:** assert that `model(torch.randn(4, 2, 1024)).shape == (4, 24)`, and assert that the output is NOT passed through a softmax (i.e. rows do not sum to 1) — because `CrossEntropyLoss` applies softmax internally.
>
> Hint tier 1: what does each layer do to the tensor's shape?
> Hint tier 2: print `x.shape` after every layer.
> Hint tier 3: the flatten between conv and linear layers is where most people get the dimensions wrong.

**Claude Code also writes:** a `summary()` helper printing per-layer output shapes and parameter counts, so the user can check their mental model against reality.

---

## Phase 4 — Training

**Claude Code writes:** `train.py` — argument parsing, dataset/DataLoader construction, train/val split, model and optimizer instantiation, per-epoch logging, checkpoint saving, and the outer epoch loop. Everything except the inner step.

> ### EXERCISE 2 — the training step
> Claude Code leaves the body of the inner batch loop unimplemented.
>
> The user must write the five-line core: zero the gradients, forward pass, compute loss, backward pass, optimizer step. Then accumulate the running loss and accuracy.
>
> **Test provided by Claude Code:** a test that runs the training step on a tiny synthetic batch and asserts (a) loss decreases over 50 steps on a single memorized batch — if your model can't overfit one batch, the loop is wrong; (b) gradients are non-`None` after `backward()`; (c) gradients are zero at the *start* of each step.
>
> Hint tier 1: what happens if you forget `zero_grad()`? (Answer: gradients accumulate across batches. The test in (c) catches this.)
> Hint tier 2: what is the difference between `loss` and `loss.item()`, and why does using the wrong one leak memory?

This is the single most-probed piece of ML code in interviews. Do not skip understanding it.

**Then:** Claude Code writes the full-dataset training run script. Train on a 1% subset first to confirm the pipeline end-to-end before committing to a full run.

---

## Phase 5 — Evaluation

**Claude Code writes:** test-set loading, overall accuracy, confusion matrix computation and plotting.

> ### EXERCISE 3 — accuracy by SNR
> Claude Code leaves `accuracy_by_snr()` unimplemented, with signature and docstring provided.
>
> The user must: group test predictions by their SNR value, compute accuracy within each bucket, and return a sorted mapping from SNR to accuracy.
>
> **Test provided by Claude Code:** on synthetic data with known per-bucket accuracy, assert the function returns the correct values. Also assert the function does not silently drop SNR buckets with few examples.
>
> Hint tier 1: you already have `y_true`, `y_pred`, and `snr` as parallel arrays.
> Hint tier 2: `np.unique` on the SNR array gives you the buckets.

This exercise exists because this plot — accuracy rising with SNR — is the single most informative figure in the whole project, and the one an interviewer will ask you to interpret. Write it yourself.

**Claude Code writes:** the plotting code for the resulting curve, saved to `docs/`.

---

## Phase 6 — Export

**Claude Code writes:** `export.py` using `torch.onnx.export`, with correct `input_names`, `output_names`, and `dynamic_axes` for a variable batch dimension.

> ### EXERCISE 4 — export verification
> Claude Code writes the ONNX export itself, but leaves the verification function unimplemented.
>
> The user must write a check that: loads the exported model with `onnxruntime.InferenceSession`, runs both the original PyTorch model and the ONNX model on the same batch of real test inputs, and asserts the outputs agree within a tolerance.
>
> **The user must choose the tolerance and justify it.** Claude Code should ask: why is exact equality the wrong assertion here?
>
> Hint tier 1: `np.testing.assert_allclose` takes `rtol` and `atol`.
> Hint tier 2: float32 arithmetic is not associative; two runtimes may reorder operations.

Never trust an export you haven't verified. A silently-wrong export produces a C++ pipeline that runs perfectly and classifies garbage.

---

## Phase 7 — C++ build setup

*(Stretch goal. Only begin once Phases 0–6 are complete, tested, and the user can explain them.)*

**Claude Code writes:** `CMakeLists.txt` targeting C++17, finding and linking FFTW3 and ONNX Runtime. A minimal `main.cpp` that links both libraries, prints their versions, and exits. Confirm this compiles and runs before any application logic — do not let build-system problems get buried under real code.

**Claude Code writes:** a small Python script that exports a slice of the test set to a flat binary file (float32, interleaved I/Q, with a simple header), plus the C++ reader for it. HDF5-in-C++ is an unnecessary dependency for a benchmark harness.

---

## Phase 8 — C++ signal processing

*(Only needed if the user chose the spectrogram representation in Phase 2. If they chose raw IQ, skip to Phase 9 — say so explicitly rather than inventing work.)*

> ### EXERCISE 5 — the FFTW RAII wrapper
> Claude Code writes the header with the class declaration and the public interface. The user implements it.
>
> The user must write a class that owns an `fftw_plan`: acquires it in the constructor, releases it via `fftw_destroy_plan` in the destructor, deletes the copy constructor and copy assignment, and implements move construction and move assignment correctly (leaving the moved-from object in a destructible state).
>
> **Test provided by Claude Code:** construct, move, and destroy in various orders under ASan/valgrind; assert no double-free and no leak. A test that constructs a vector of these and reallocates it — which will crash if move semantics are wrong.
>
> Hint tier 1: what must the moved-from object's plan pointer be set to, so its destructor is safe?
> Hint tier 2: FFTW's plan creation is not thread-safe. Does that matter for your design? Say why or why not.

This is the exercise where your existing C++ comfort is load-bearing. It is also a clean, honest answer to "give me an example of when RAII saved you from a bug."

**Claude Code writes:** the STFT windowing/framing logic around the user's wrapper, and a test comparing the C++ STFT output against `scipy.signal.stft` on the same input, within tolerance.

---

## Phase 9 — C++ inference

**Claude Code writes:** the `Ort::Env` and `Ort::SessionOptions` setup, model loading, and the input/output tensor shape plumbing.

> ### EXERCISE 6 — the inference call
> The user must write the function that takes one preprocessed sample window, wraps it in an `Ort::Value` tensor, calls `session.Run(...)`, and extracts the predicted class index from the output tensor.
>
> **Test provided by Claude Code:** run the C++ pipeline on 100 test samples for which the Python model's predictions are already known and saved to disk; assert the C++ predictions match exactly. If they don't, the bug is in the tensor layout, not the model.
>
> Hint tier 1: what memory does `Ort::Value::CreateTensor` take ownership of, and what must outlive the call?
> Hint tier 2: is your input row-major in the order ONNX Runtime expects?
>
> Check the ONNX Runtime C++ API docs (https://onnxruntime.ai/docs/api/c/) for exact signatures — they shift between versions, do not trust a signature from memory.

> ### EXERCISE 7 — latency instrumentation
> The user must write: a timing harness that records per-inference latency into a vector, and a function computing mean, p50, and p99 from it.
>
> **Test provided by Claude Code:** on a known input vector with hand-computable percentiles, assert correctness. Include an edge case: what does p99 mean for a vector of 10 elements? The user must decide and document their interpolation choice.
>
> Hint tier 1: `std::chrono::steady_clock`, not `system_clock`. Why?
> Hint tier 2: does your timing include or exclude the tensor construction? Both are defensible — but you must know which you measured and say so.

**Claude Code writes:** the streaming loop, RSS memory measurement, and CSV output of the latency distribution.

---

## Phase 10 — Bottleneck analysis

**Claude Code writes:** stage-level timing instrumentation (I/O vs. preprocessing vs. inference) and a script to produce the latency histogram plot.

> ### EXERCISE 8 — one optimization, measured
> The user must: read the stage timings, form a hypothesis about where time is going, implement **one** optimization, and measure the before/after.
>
> Candidates: reuse the input tensor buffer instead of allocating per sample; batch N windows per `Run()` call; reserve the latency vector's capacity; avoid a copy between the STFT output and the inference input.
>
> **Claude Code must not choose the optimization.** Ask the user for their hypothesis first, and ask them to predict the magnitude of improvement before they measure. Then have them compare prediction to result.
>
> If the optimization does not help: that is a real result. Report it. "I tried X, it didn't move the number, here's why I think that is" is a stronger interview answer than a fabricated win.

---

## Phase 11 — Report

**The user writes `docs/report.md`.** Claude Code may review it and ask clarifying questions, but must not draft it.

It must cover: architecture choices and why; accuracy-by-SNR findings and what they mean; C++ benchmark numbers with the measurement methodology stated; the bottleneck found and what did or didn't fix it; and an honest limitations section.

**Claude Code's final task:** interview the user on their own project. Ask ten questions drawn from the exercises above — including at least two the report doesn't answer. Point out where their explanation is thin.
