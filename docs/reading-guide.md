# Full Reading Guide — Everything Under the Hood of This Project

**What this is:** the ground-floor companion to `understanding-guide.md`. That document explains *why the project is designed the way it is*. This one teaches *the things you need to already know for that document to make sense*: the radio science, the Python and NumPy mechanics, how PyTorch actually works, how floating-point numbers behave, and what ONNX is. Every concept is illustrated with the actual code in this repository, so read it with the files open.

**How to use it:** don't read it linearly in one sitting. Each section ends with **Check yourself** questions. If you can answer them without looking, skip ahead. If you can't, that section is for you. The sections are ordered so that each builds on the previous.

---

## Part 1 — The Radio Science

### 1.1 A signal is a wave you deliberately mess with

A radio transmitter produces a *carrier wave* — a sinusoid at some frequency, e.g. 915 MHz:

```
s(t) = A · cos(2πft + φ)
```

A pure carrier carries no information. To transmit bits, you *modulate* it — vary one of its three knobs over time:

- **A** (amplitude) → amplitude modulation. AM radio, OOK ("on-off keying": 1 = carrier on, 0 = off — the simplest scheme in your 24 classes), ASK.
- **φ** (phase) → phase modulation. BPSK (2 phases: 0° and 180°, one bit per symbol), QPSK (4 phases, two bits per symbol), 8PSK, ...
- **f** (frequency) → frequency modulation. FM radio, GMSK (used in 2G GSM phones).
- **A and φ together** → QAM ("quadrature amplitude modulation"). 16-QAM encodes 4 bits per symbol using 16 distinct amplitude+phase combinations. 256-QAM encodes 8. Wi-Fi and 5G live here.

Every one of your 24 classes is a different answer to "which knob(s), and how many distinct positions?" **Modulation classification = looking at a received waveform and figuring out which scheme produced it**, without being told. That's the whole project.

### 1.2 Why higher-order schemes are harder to classify

BPSK's two symbols are 180° apart — maximally far. 256-QAM's 256 symbols are crowded close together. The denser the constellation, the less noise it takes to push a received symbol into a neighbor's territory. This is not a weakness of your model; it's information theory. It is why your confusion matrix shows the high-order QAM/APSK classes blurring into each other even at +30 dB, while BPSK and FM are easy.

### 1.3 IQ: representing amplitude AND phase as two numbers

A receiver can't hand you "phase" directly — phase is relative to a reference. The trick (called *quadrature demodulation*) is to multiply the incoming signal by two reference waves 90° apart (a cosine and a sine), and low-pass filter. Out come two numbers per instant:

- **I** — the in-phase component ("how much of the signal aligns with the cosine")
- **Q** — the quadrature component ("how much aligns with the sine")

Together they're a 2D coordinate: the signal at that instant is a point in the I/Q plane. Amplitude is the point's distance from the origin (`√(I²+Q²)`); phase is its angle (`atan2(Q, I)`). One complex number `I + jQ` captures both. That's why the math in `understanding-guide.md` §2 writes `s(t) = A(t)e^{jφ(t)}` — Euler's formula is just polar coordinates for this plane.

Plot many received samples as points and you see the **constellation**: QPSK is four fuzzy clusters at four angles, 16-QAM a 4×4 fuzzy grid. Noise is what makes the clusters fuzzy. Your `docs/constellation_qpsk.png` from Phase 1 is exactly this plot.

**Connection to the code:** each example in the dataset is 1024 consecutive (I, Q) pairs — a 1024-step trajectory through this plane. `dataset.py` reshapes it to `(2, 1024)`: channel 0 is the I sequence, channel 1 the Q sequence. When `model.py`'s docstring says input `(batch, 2, 1024)`, that "2" is I-and-Q.

### 1.4 SNR and decibels — the units of "how buried is the signal"

**SNR** = signal power / noise power. It's expressed in **decibels**, a logarithmic scale:

```
SNR_dB = 10 · log10(P_signal / P_noise)
```

Log scales compress huge ranges into small numbers. Memorize three anchors:

| dB | power ratio | meaning |
|---|---|---|
| 0 dB | 1× | signal and noise equally strong |
| +10 dB | 10× | signal 10× stronger than noise |
| −20 dB | 1/100× | noise is **100× stronger** than the signal |

Every 3 dB ≈ a factor of 2. Your dataset sweeps −20 dB to +30 dB in 2 dB steps — from "signal is 1% of the noise" to "signal is 1000× the noise." Now reread your evaluation results: 3.8% accuracy at −20 dB (chance is 1/24 ≈ 4.2%) isn't your model failing, it's physics — the information is essentially gone. The plateau at ~92% above +12 dB is where noise stops being the limiting factor and constellation density takes over (§1.2).

**Check yourself:** (a) What physical property of the wave does QPSK vary? (b) A signal at −6 dB SNR: is it stronger or weaker than the noise, and by roughly what factor? (c) Why does one time-sample need *two* numbers?

---

## Part 2 — The Dataset and HDF5

### 2.1 RadioML 2018.01a

2,555,904 examples. Each is: 1024 IQ samples of one modulated signal, a label (one of 24 schemes), and the SNR it was generated at. 24 classes × 26 SNR levels × 4096 examples per combination.

### 2.2 HDF5 and why we load lazily

HDF5 is a binary container format for large arrays — think "a filesystem inside one file." Your file holds three arrays sharing the same first axis:

```
X: (2555904, 1024, 2) float32   # the IQ data, ~20 GB
Y: (2555904, 24)      int64     # labels, one-hot encoded
Z: (2555904, 1)       int64     # SNR in dB
```

**One-hot:** instead of storing class 4 as the integer `4`, Y stores `[0,0,0,0,1,0,...,0]` — a 1 in position 4. That's why `dataset.py:79` does `f["Y"][row].argmax()` — `argmax` returns the *index of the largest value*, converting one-hot back to the integer 4. Remember `argmax`; it reappears everywhere.

X alone is 20 GB — bigger than your RAM. The `h5py` library lets you open the file and read *single rows on demand* (`f["X"][row]` reads just that row from disk). That's "lazy loading," and it's the entire reason `dataset.py` exists as a class instead of one `np.load()` call.

**Check yourself:** (a) Why can't we just load X into a numpy array? (b) What does `argmax` do to `[0.1, 0.9, 0.0]`?

---

## Part 3 — NumPy Mechanics You're Actually Using

NumPy arrays are typed, fixed-shape grids of numbers. Everything below appears in `evaluate.py` or `export.py`.

### 3.1 Shape, dtype, axis

```python
a = np.zeros((8, 2, 1024), dtype=np.float32)
a.shape    # (8, 2, 1024) — 8 examples, 2 channels, 1024 timesteps
a.ndim     # 3 — number of axes
a.dtype    # float32 — every element is the same type
```

"Axis 0" is the first dimension. `logits.argmax(dim=1)` in `collect_predictions` means "for each row (axis 0), find the argmax *along axis 1*" — i.e., per example, which of the 24 class scores is biggest. Result shape: `(8,)`.

### 3.2 Boolean masks — the engine of your Exercise 3

```python
snr = np.array([-20, 10, -20, 10])
mask = (snr == 10)          # array([False, True, False, True]) — elementwise comparison
y_true[mask]                # keeps only elements where mask is True
mask.sum()                  # 2 — True counts as 1; this is "how many matched"
(y_true == y_pred).mean()   # fraction of positions where they match = accuracy!
```

That last line deserves a pause: comparing two arrays gives a boolean array; the *mean* of a boolean array (True=1, False=0) is the fraction of Trues. `accuracy = (y_true == y_pred).mean()` is the idiom for accuracy in one line, and `(y_true[mask] == y_pred[mask]).mean()` is accuracy *within a bucket*. That's your whole Exercise 3 solution, and it's worth understanding rather than remembering.

`np.unique(snr)` returns the distinct values, **sorted ascending** — which is why your dict came out SNR-ordered for free (Python dicts preserve insertion order).

### 3.3 Parallel arrays

`y_true`, `y_pred`, `snr` are "parallel": position *i* in each refers to the same test example. Any mask computed from one can index the others. This is a workhorse pattern in data code — cheaper and simpler than a list of (truth, prediction, snr) objects.

### 3.4 Unpacking — the syntax that bit you

Python lets you split a sequence into named variables:

```python
a, b = (1, 2)                 # fine: 2 names, 2 values
a, b, c = (1, 2)              # ValueError: not enough values to unpack (expected 3, got 2)
(x,) = [some_array]           # unpack a 1-element list; ALSO asserts it has exactly 1 element
```

Two places this mattered recently:

- `for iq, labels, snr in loader:` works because a DataLoader yields *triples* (that's what `RadioMLDataset.__getitem__` returns). But `for ... in sample_input:` iterates a raw `(8, 2, 1024)` **tensor**, which yields `(2, 1024)` rows — and unpacking a length-2 thing into 3 names is exactly the `expected 3, got 2` error you hit. Iterating a tensor ≠ iterating a DataLoader.
- `(onnx_logits,) = session.run(...)` in `verify_export` unpacks the 1-element output list *and* crashes loudly if there were ever two outputs. Deliberate defensive style.

### 3.5 Integer types: `int` vs `np.int64`

`np.unique(snr)[0]` is an `np.int64`, not a plain Python `int`. They behave almost identically in arithmetic but are different types — which is why your Exercise 3 docstring-vs-implementation mismatch (`np.sum(mask)` stored where an `int` was promised) is a real, if minor, defect. `int(x)` converts explicitly.

**Check yourself:** (a) What does `(np.array([1,2,3]) == np.array([1,9,3])).mean()` evaluate to? (b) Why does `for a, b, c in tensor:` fail when `tensor` has shape `(8, 2, 1024)`? (c) What two things does `(x,) = lst` do?

---

## Part 4 — PyTorch, From Tensors Up

### 4.1 Tensors

A `torch.Tensor` is numpy's array with two superpowers: it can live on a GPU, and it can *remember the operations that produced it* (for gradients, §4.5). Conversions at the boundary:

```python
torch.from_numpy(arr)       # numpy → tensor (dataset.py:77)
t.numpy() / t.cpu().numpy() # tensor → numpy (only on CPU; .cpu() moves it there first)
t.detach()                  # drop gradient history first, if it has any
```

`session.run` (ONNX) speaks numpy; `model(x)` speaks tensor. `verify_export` is a border town, hence the conversions on both lines.

### 4.2 `nn.Module` — the shape of every PyTorch model

```python
class ModulationClassifier(nn.Module):
    def __init__(self):        # declare the layers (which allocates their weights)
        super().__init__()
        self.conv1 = nn.Conv1d(2, 64, kernel_size=7, padding=3)
        ...
    def forward(self, x):      # say how data flows through them
        ...
```

Calling `model(x)` invokes `forward(x)` (plus bookkeeping — never call `forward` directly). Layers declared in `__init__` register their weights so that `model.parameters()` finds them for the optimizer and `model.state_dict()` finds them for saving. A layer used twice (like `self.pool`) shares nothing between uses because pooling has no weights; the four convs each need their own weights, hence four attributes.

### 4.3 What each layer in *your* model does

Input: `(batch, 2, 1024)` — batch of examples, 2 channels (I, Q), 1024 timesteps.

- **`nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=p)`** slides `out_ch` learned filters along the time axis. Each filter spans all input channels and `k` timesteps, so it detects a *local pattern* (a phase-transition signature, say) anywhere it occurs. Output length: `L_out = L_in + 2p − k + 1` (stride 1). Your convs choose `p = (k−1)/2` so `L_out = L_in` — "same" padding; only the channel count changes. Parameter count: `out_ch × (in_ch × k) + out_ch` (the `+ out_ch` is one bias per filter). conv1: 64×(2×7)+64 = **960 weights**.
- **`nn.ReLU()`** — `max(0, x)` elementwise. Shape unchanged. This is the nonlinearity; without it, stacked convs collapse mathematically into one linear operation and depth buys nothing.
- **`nn.MaxPool1d(2)`** — keep the max of every 2 adjacent timesteps. Halves the length: 1024→512→256→128→64 across your four blocks. Discards *where exactly* within each pair the activation was — buying tolerance to small time-shifts, shrinking downstream compute.
- **`torch.flatten(x, start_dim=1)`** — reshape `(batch, 128, 64)` → `(batch, 8192)`, leaving axis 0 alone. The bridge from "feature maps" to "one feature vector per example." `start_dim=1` is the crucial argument: flattening from dim 0 would smear the batch together.
- **`nn.Linear(8192, 256)`** — full matrix multiply plus bias: `y = xWᵀ + b`. This is where "which patterns were detected" becomes "evidence for each class." 8192×256+256 ≈ 2.1M weights — the bulk of your model.
- **`nn.Dropout(0.5)`** — during *training only*, randomly zeroes 50% of activations each forward pass, preventing co-adapted features (a form of regularization against overfitting). During *eval*, it does nothing. This is the entire reason `model.eval()` matters (§4.6).
- **`nn.Linear(256, 24)`** — final scores: one number per class. These are the **logits**.

Run `python src/model.py` — the `summary()` helper prints this whole story with real numbers.

### 4.4 Logits, softmax, cross-entropy

**Logits** are raw, unbounded class scores, e.g. `[2.1, -0.3, 8.0, ...]`. **Softmax** converts them to probabilities:

```
p_i = e^{z_i} / Σ_j e^{z_j}     (all positive, sum to 1; biggest logit → biggest probability)
```

**Cross-entropy loss** = `−log(p_correct)`: confident-and-right → loss near 0; confident-and-wrong → huge loss. PyTorch's `nn.CrossEntropyLoss` **applies softmax internally**, which is why your model returns logits and why `test_model.py` asserts rows do *not* sum to 1. Softmax preserves ranking, so for prediction alone `argmax(logits)` ≡ `argmax(softmax(logits))` — you never need softmax at inference.

### 4.5 Autograd — why training works and `no_grad` exists

Every tensor operation optionally records itself in a graph. `loss.backward()` walks that graph in reverse, computing `∂loss/∂w` for every weight (backpropagation — the 3Blue1Brown episodes). `optimizer.step()` then nudges each weight against its gradient. That's your Exercise 2 five-liner:

```python
optimizer.zero_grad()   # gradients ACCUMULATE by default; clear last batch's
logits = model(x)       # forward — records the graph
loss = criterion(logits, y)
loss.backward()         # fills every weight's .grad
optimizer.step()        # w -= lr * (something like) w.grad
```

`with torch.no_grad():` turns the recording off. Use it whenever you're only *reading* the model (evaluation, verification): it's faster and doesn't build a graph you'd never use. Related trap: `loss` is a tensor tethered to the whole graph; `loss.item()` extracts the plain float. Accumulating `loss` across batches keeps every graph alive — a memory leak.

### 4.6 `train()` vs `eval()` mode

`model.eval()` flips layers with mode-dependent behavior — for you, just Dropout — into inference behavior (do nothing). `model.train()` flips them back. Forgetting `eval()` before evaluation means randomly zeroing half the features of your fc1 output *at test time*: accuracy tanks, nondeterministically. It also would have made `verify_export` fail spuriously — PyTorch-with-random-dropout can't match the ONNX graph, which has dropout removed. Mode is *state on the model object*, not on the data.

### 4.7 Dataset, DataLoader, and checkpoints

- **`Dataset`**: knows how to fetch example *i* (`__getitem__`) and how many exist (`__len__`). Yours returns `(iq_tensor, label_int, snr_int)`.
- **`DataLoader`**: wraps a Dataset; handles batching (stacks 256 examples into one `(256, 2, 1024)` tensor — note it also turns the int labels into tensors), shuffling, and parallel loading with worker processes. `next(iter(loader))` in `export.py:main` = "give me just the first batch."
- **Checkpoints**: `model.state_dict()` is a dict of weight tensors; `torch.save` writes it; `model.load_state_dict(torch.load(path))` restores it into a freshly constructed model. The *architecture* comes from the class; the *file* only carries the numbers. That's why `evaluate.py` and `export.py` both do `ModulationClassifier()` **then** `load_state_dict`.

**Check yourself:** (a) Walk `(batch, 2, 1024)` through conv1+pool: what shape comes out? (b) What breaks if you delete `zero_grad()`? (c) Why must the model return logits, not probabilities? (d) Name the one layer in this model that behaves differently in train vs eval mode, and what it does in each.

---

## Part 5 — Floating Point: Why Exercise 4 Is About Tolerance

### 5.1 What a float32 is

32 bits: 1 sign bit, 8 exponent bits, 23 fraction bits — scientific notation in binary (`±1.fraction × 2^exponent`). Consequences:

- **~7 decimal digits of precision.** The gap between adjacent representable numbers near 1.0 is ε ≈ 1.19×10⁻⁷ ("machine epsilon"). Near 8.0, the gap is ~8ε ≈ 10⁻⁶ — **precision is relative to magnitude**. This single fact is why `rtol` (relative tolerance) exists.
- **Most decimals aren't representable.** 0.1 in binary is infinitely repeating, so it's stored rounded. Every arithmetic op re-rounds to the nearest representable value.
- **Big + small can lose the small entirely:** in float32, `16777216 + 1 == 16777216` (2²⁴ is where the gaps between representable integers exceed 1).

### 5.2 Non-associativity — your comprehension answer, made concrete

Because every op rounds, *grouping changes the result*:

```
float32: (16777216 + 1) + 1  = 16777216      (each +1 individually vanishes)
float32: 16777216 + (1 + 1)  = 16777218      (the 2 survives)
```

Same numbers, same "math," different answers — both correctly rounded. Now scale that up: one output logit of your conv1 is a sum of 14 products… and fc1 sums 8192 products per output. PyTorch computes those sums in one order; ONNX Runtime fuses operations, vectorizes with SIMD, and sums in another. Each is a legitimately rounded result; they differ in the last bits. Measured on your actual model: max disagreement ≈ 4.5×10⁻⁸ on logits of magnitude ~0.1 — i.e., a relative error of ~10⁻⁶, right at the float32 noise floor times a few accumulation steps. A *wrong* model differs by whole units — 6+ orders of magnitude more. Tolerance-based comparison lives in that enormous gap.

### 5.3 The `assert_allclose` contract

```python
np.testing.assert_allclose(actual, desired, rtol, atol)
# passes iff, elementwise:  |actual − desired| ≤ atol + rtol · |desired|
```

Read the formula as two regimes:

- For elements of large magnitude, `rtol·|desired|` dominates: the allowed error *scales with the value*, matching how float precision actually degrades (§5.1).
- For elements near zero, `rtol·|desired| ≈ 0`, and without `atol` you'd be demanding near-exact equality of values that are pure rounding residue. `atol` is the floor that prevents that.

Worked example at a *small* value so you still get to do the 8.0 one yourself: for a desired value of **0.05** with `rtol=1e-4, atol=1e-5`, the budget is `1e-5 + 1e-4 × 0.05 = 1.5e-5` — and notice atol contributes two-thirds of it. Redo that arithmetic at magnitude 8.0 and watch which term takes over. That's the comprehension question.

**Check yourself:** (a) Why does float precision get coarser as numbers get bigger? (b) Give the one-line reason two correct float32 programs can disagree. (c) In `atol + rtol·|desired|`, which term protects near-zero values and why is it needed?

---

## Part 6 — ONNX and the Export

### 6.1 What ONNX is and why we export

Your trained model currently exists only as Python objects + a weights file — it needs Python and PyTorch to run. **ONNX** (Open Neural Network Exchange) is a portable file format describing the computation as a **graph**: nodes are operations (Conv, Relu, MaxPool, Gemm…), edges are tensors, weights are baked in as constants. **ONNX Runtime** is a separate, lightweight engine (with a first-class C++ API — the point of Phases 7–9) that executes such graphs with no Python and no PyTorch anywhere.

`torch.onnx.export` produces the graph by **tracing**: it runs your `forward()` once on a dummy input and records every operation into the graph. Details in `export_onnx` worth owning:

- **`model.eval()` first** — tracing a train-mode model would bake in dropout behavior. Eval-mode dropout is identity, so it simply disappears from the graph. (Compare the graph to §4.3's layer list: everything survives except Dropout.)
- **`input_names=["iq"], output_names=["logits"]`** — graph edges get names; ONNX Runtime addresses inputs by name. This is the `"iq"` key in `session.run(None, {"iq": ...})`.
- **`dynamic_shapes=({0: "batch"},)`** — the trace saw batch size 1; this declares axis 0 symbolic so the graph accepts any batch size. Without it, the batch dimension freezes at the traced value — which is what `test_dynamic_batch_dimension` guards.
- **opset** — ONNX's operator-set version number; runtime and file must agree on what ops mean. You'll see harmless version-conversion warnings in the export logs.

### 6.2 The verification mindset

An export can be *silently* wrong — the file loads, runs, and returns confident garbage (wrong layout, wrong mode, wrong weights). So: run the same batch through both runtimes and compare **logits** within a float-noise-calibrated tolerance (§5.3). Not argmax — two very different models can agree on argmax for a handful of samples, so argmax-agreement is far too weak a test to *certify* an export (though checking argmax agreement *additionally* is reasonable, since it's what matters downstream). And the negative test (`test_wrong_model_is_caught`) matters as much as the positive one: a verifier that can't fail verifies nothing.

**Check yourself:** (a) What does "tracing" mean and why must the model be in eval mode during it? (b) What would break, and where, if `dynamic_shapes` were omitted? (c) Why is comparing argmaxes alone too weak?

---

## Part 7 — pytest: How the Tests Work

- **Discovery:** pytest runs every `test_*` function in `tests/test_*.py`. A test *passes* unless it raises; `assert` is the failure mechanism.
- **`conftest.py`** runs first — yours just adds `src/` to the import path.
- **Fixtures** (`@pytest.fixture`) build shared setup; a test receives one by naming it as a parameter. `exported` in `test_export.py` builds a model + ONNX file once (`scope="module"`) and hands the same pair to all three tests — that's why the export only happens once per run.
- **`pytest.raises(AssertionError)`** inverts the logic: the block *must* raise that error or the test fails. It's how you test that a checker checks.
- Flags you've used: `-q` (quiet), `-k faithful` (only tests whose name matches).

### 7.1 Reading a traceback (the skill, not the API)

Python tracebacks read **top = outermost call, bottom = actual failure**. Go to the *bottom* line first (the exception and message), then walk up to find the line *in your code* that triggered it. The `expected 3, got 2` failure you hit displayed exactly this: bottom said `ValueError` at `export.py:72` — a line *you* wrote — even though the test file appeared higher up. The error message named the true culprit; the lesson is to trust the bottom of the traceback over your assumption about what "must have" failed.

**Check yourself:** (a) How does a pytest test signal failure? (b) What does `pytest.raises` assert? (c) In a long traceback, where do you look first?

---

## Part 8 — What's Ahead (C++, Phases 7–9), In One Breath Each

You chose raw IQ, so Phase 8 (FFTW/STFT) is skipped. What remains:

- **Phase 7:** CMake project that links ONNX Runtime; a Python script dumps test samples to a flat binary file (float32, no HDF5 in C++) and C++ reads them back.
- **Phase 9, Exercise 6:** the C++ inference call. The new concepts are *ownership* (ONNX Runtime's `CreateTensor` wraps your buffer without copying — your buffer must outlive the call) and *memory layout* (row-major order must match `(batch, 2, 1024)`). Validation: C++ argmaxes must match saved Python predictions **exactly** — same runtime = same graph = same floats, so this one *is* an exact comparison, unlike Exercise 4. Understanding why those two cases differ is a genuinely good interview moment.
- **Phase 9, Exercise 7:** latency measurement — `steady_clock` (monotonic, can't jump backwards) vs `system_clock` (wall time, can), and percentiles (p50/p99) because means hide tails.

---

## Part 9 — Suggested Study Path

1. **Now (1–2 hrs):** Parts 3, 4.4–4.6, and 5 of this guide — they cover everything the current comprehension question touches. Then answer it.
2. **This week:** 3Blue1Brown's neural network series (episodes 1–4) for gradient descent/backprop intuition → reread Part 4 → rerun `python src/model.py` and predict every row of the summary before reading it.
3. **Before the report:** Part I of `understanding-guide.md` end-to-end (it will read very differently now), plus §1 here until the dB table is reflex.
4. **Before Phase 9:** Part 8 here + understanding-guide §8.
5. **Ongoing:** after finishing any exercise, reread its section in understanding-guide Part II and answer "the question behind it" out loud, in complete sentences, without notes. That's the interview rehearsal.

### The self-test that matters most

Close everything and explain, out loud: *"My model takes 1024 IQ samples and produces 24 logits. Here's what happens in between, and here's why each step is the right thing to do to a radio signal."* When that monologue is fluent, you understand this project.
