# RF Signal Modulation Classification with a C++ Inference Pipeline

## Status note
This describes the project as currently scoped. Two decisions are still open (ONNX vs. TorchScript export, and whether the C++ pipeline is a hard requirement or a stretch goal) — flagged inline below rather than presented as settled.

---

## 1. The Problem

Modern RF/SIGINT systems need to identify what kind of signal they're looking at — is this BPSK, QPSK, QAM16, an FM broadcast, noise — before any downstream decision (demodulation, threat classification, spectrum allocation) can happen. Historically this is done with hand-engineered features and expert-designed classifiers (cyclostationary feature detection, decision trees on statistical moments of the signal). That works, but it's brittle: it requires domain experts to hand-pick features for every modulation type you care about, and it degrades under low SNR or previously-unseen modulation schemes.

The modern alternative — and the one used in actual defense and telecom systems today — is to treat modulation classification as a supervised learning problem: feed raw or lightly-processed signal data into a neural network and let it learn discriminative features directly, the same way image classification moved from hand-crafted features (SIFT, HOG) to CNNs.

The second half of the problem is deployment. A model that classifies modulation types accurately in a Jupyter notebook is not the same as a system that can do it in real time on a stream of incoming samples. Real SIGINT/radar systems are latency- and resource-constrained, typically implemented in C++ (sometimes with FPGA acceleration) rather than in Python. A project that stops at "the model works in PyTorch" doesn't demonstrate anything about whether the approach is deployable — which is the part that's actually relevant to companies building real-time defense systems.

## 2. Goal / Intended Solution

**Goal:** Build a technically honest, interview-defensible project that does two things:
1. Trains a CNN to classify radio modulation types from the standard academic benchmark dataset (RadioML 2018.01a).
2. Deploys that trained model in a C++ pipeline that simulates a real-time streaming scenario and measures actual latency/throughput/memory — not just training-time accuracy.

The point isn't to build something novel at the research level (this is a well-studied classification task on a public dataset). The point is to demonstrate, end to end, the full pipeline a real system needs: data → trained model → exported artifact → real-time inference in a systems language — and to be able to explain every choice made along the way under interview questioning.

**Non-goals (explicitly out of scope):**
- Beating state-of-the-art accuracy on RadioML. This is not a research contribution.
- Real RF hardware / SDR capture. Input is the existing dataset, not live-captured signal.
- FPGA acceleration. That's related to the separate Lariviere lab work, not this project.

## 3. Dataset

**DeepSig RadioML 2018.01a** — a widely used public benchmark for automatic modulation classification. Contains synthetically generated and over-the-air-captured signal examples across 24 modulation classes at a range of SNR levels, stored as IQ (in-phase/quadrature) sample sequences.

Source: https://www.deepsig.ai/datasets (DeepSig's own dataset page — verify current download terms/license there before use, as access details can change).

## 4. How the Project Works

### Phase 1 — Model training (Python / PyTorch)
- Input representation: either raw IQ sample windows fed directly into a 1D/2D CNN, or a spectrogram representation (via STFT) fed into a 2D CNN — this choice affects both accuracy and how much signal-processing math needs to be understood and explained. The Velardo audio-signal-processing series (Fourier transform, STFT, spectrograms) is the prerequisite material for this decision, since the math is identical to what's used for audio.
- Model: a CNN architecture, likely inspired by published modulation-classification architectures (e.g., the original DeepSig/O'Shea papers), but implemented and trained from scratch — not copied.
- Output: a trained model that classifies IQ input into one of the 24 modulation classes, evaluated for accuracy across SNR levels (this is itself an interesting axis to report — accuracy degrades at low SNR, which is realistic and worth discussing honestly rather than hiding).

### Phase 2 — Export
- Trained PyTorch model exported to a portable inference format.
- **Open decision:** ONNX (via `torch.onnx.export`) vs. TorchScript. ONNX has a well-documented, actively maintained C++ runtime (ONNX Runtime) and is more portable outside the PyTorch ecosystem; TorchScript keeps you inside PyTorch's own tooling (LibTorch) but is a heavier C++ dependency to build against. Current lean is ONNX, but this hasn't been finalized.

### Phase 3 — C++ inference pipeline
- **Open decision:** whether this phase is a hard requirement for "project done" or a stretch goal attempted after Phase 1–2 are complete and working. Current lean is the latter, to protect against the project stalling on the C++ portion and never reaching a completed, honest state.
- Simulates streaming input: reads IQ sample windows from the RadioML test set sequentially, as if arriving from a live source, rather than batch-processing the whole dataset at once.
- Signal preprocessing (if using the spectrogram representation) reimplemented in C++, using FFTW for the FFT/STFT computation.
- Inference performed via the ONNX Runtime C++ API (or LibTorch's C++ API if TorchScript is chosen instead), loading the exported model and running it against each incoming sample window.
- Instrumentation: measures per-sample and end-to-end latency (mean and p99), throughput (samples/sec), and memory usage — the same category of metrics used to evaluate real-time signal processing systems.

### Phase 4 — Report / write-up
- Documents architecture decisions (why this CNN structure, why this input representation), what the accuracy-vs-SNR tradeoff looks like, what the C++ pipeline's latency/throughput numbers are, and what bottlenecks were found and addressed (e.g., allocation overhead, cache behavior, unnecessary copies between the preprocessing and inference stages).

## 5. Why This Project Fits the Stated Goal

The project is aimed at defense-tech and high-ROI engineering roles (Anduril, Palantir, SpaceX), not quant trading. Its relevance case:
- Modulation classification is a real SIGINT problem, not a contrived one.
- The train-in-Python / deploy-in-C++ split mirrors how real ML-in-production systems are actually built — this is a stronger, more honest interview story than either "pure research notebook" or "I only did the C++ part."
- It connects thematically to the parallel Lariviere research (HFT-style low-latency infrastructure applied to RF/SIGINT), without being the same project — this one is solo-owned and independently defensible.
- Every design choice (CNN architecture, ONNX vs. TorchScript, FFTW for STFT, why measure p99 not just mean latency) is something that can be explained and justified under direct questioning, which was the explicit bar set for this project from the start.

## 6. Honest Limitations to Acknowledge Up Front

- This is a benchmark-dataset classification task, not a fielded system — worth stating plainly rather than overselling.
- Real-time "streaming" in Phase 3 is simulated from a static dataset, not live RF capture.
- Accuracy at low SNR will likely be materially worse than at high SNR — this should be reported, not glossed over, since acknowledging where a model fails is generally read as a stronger signal than pretending it doesn't.
