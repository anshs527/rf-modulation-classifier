"""Export the trained classifier to ONNX and verify the export is faithful.

ONNX was chosen over TorchScript (the project's default) because the Phase 7+
C++ pipeline uses ONNX Runtime, which is a lighter dependency than libtorch
and is runtime-agnostic if we ever swap the inference engine.

Usage:
    python src/export.py data/dataset/GOLD_XYZ_OSC.0001_1024.hdf5 checkpoints/best.pt
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import RadioMLDataset
from evaluate import test_indices
from model import ModulationClassifier

CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def export_onnx(model, out_path, opset=17):
    """Export `model` to ONNX at `out_path`.

    The model is forced into eval() mode first: dropout must be baked in as
    identity. Exporting a model in train() mode would either trace the
    dropout RNG into the graph or silently disagree with PyTorch eval-mode
    outputs — exactly the class of bug verification exists to catch.
    """
    model.eval()
    # Batch size 1 here is only the tracing example; dynamic_shapes below is
    # what actually makes the batch dimension variable in the graph.
    # (dynamic_shapes is the torch>=2.13 replacement for the deprecated
    # dynamic_axes argument — same intent, keyed per positional input.)
    dummy = torch.randn(1, 2, ModulationClassifier.INPUT_LENGTH)
    torch.onnx.export(
        model,
        (dummy,),
        str(out_path),
        input_names=["iq"],
        output_names=["logits"],
        dynamic_shapes=({0: "batch"},),
        opset_version=opset,
    )
    print(f"exported {out_path}")


def verify_export(model, onnx_path, sample_input):
    """Assert the ONNX model at `onnx_path` agrees with `model`.

    Runs the same batch through both the PyTorch model and an
    onnxruntime.InferenceSession and compares the outputs. Raises
    AssertionError if they disagree beyond tolerance; returns None if they
    agree.

    Parameters
    ----------
    model : ModulationClassifier
        The original PyTorch model (this function must put it in eval mode —
        dropout in train mode would make even PyTorch disagree with itself).
    onnx_path : str or Path
        Path to the exported .onnx file.
    sample_input : (B, 2, 1024) float32 torch.Tensor
        A batch to compare on. main() passes real test examples, not random
        noise: real inputs exercise the trained weights' actual dynamic
        range, where float error accumulates differently than on randn.
    """
    # ============================================================
    # EXERCISE 4: export verification
    # See docs/understanding-guide.md § Exercise 4 for the full explanation.
    #
    # What this must do:
    #   - Put `model` in eval mode and compute its logits for
    #     `sample_input` (no gradients needed)
    #   - Load the ONNX model with onnxruntime.InferenceSession and run
    #     the SAME batch through it (the input name is "iq" — see
    #     export_onnx above; session.run wants numpy, not torch)
    #   - Assert the two outputs agree within a tolerance THAT YOU CHOOSE
    #     and can justify — raise AssertionError (or let the assertion
    #     helper raise it) if they don't
    #
    # What you have available: model (nn.Module), onnx_path,
    #   sample_input (torch.Tensor, (B, 2, 1024) float32)
    # What it must produce: None on agreement; AssertionError on mismatch
    #
    # Before you write it, answer this (put your answer in a comment
    # here — I'll ask you about it): why is exact equality the WRONG
    # assertion between two float32 runtimes computing "the same" graph?

    # Answer: Since floating-point aithmetic isn't assopciative, and the ONNX Runtime can reorder operations, 
    # the order affects the final result. Even if the two runtimes are computing the same graph, they may produce 
    # slightly different results due to differences in how they handle floating-point arithmetic, leading to small 
    # discrepancies in the output. Therefore, exact equality is not a reliable measure of agreement between the two models.
    #
    # Hints (read only if stuck):
    #   - np.testing.assert_allclose(a, b, rtol=..., atol=...) raises
    #     AssertionError with a useful diff message.
    #   - session.run(None, {"iq": batch_np}) returns a list of outputs;
    #     yours has exactly one.
    #   - .detach().cpu().numpy() gets you from torch to numpy.
    # ============================================================
    import onnxruntime as ort

    model.eval()
    with torch.no_grad():
        torch_logits = model(sample_input).cpu().numpy()

    session = ort.InferenceSession(str(onnx_path))
    (onnx_logits,) = session.run(None, {"iq": sample_input.cpu().numpy()})

    # Tolerance: measured max |diff| on a faithful export is ~1e-6; a wrong
    # model disagrees by whole units. rtol=1e-4 sits ~100x above the noise
    # floor and ~10,000x below a real mismatch. atol=1e-5 only matters for
    # logits near zero, where rtol alone would demand near-exact equality.
    np.testing.assert_allclose(torch_logits, onnx_logits, rtol=1e-4, atol=1e-5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", help="path to GOLD_XYZ_OSC.0001_1024.hdf5")
    parser.add_argument("checkpoint", help="path to a saved state_dict (e.g. checkpoints/best.pt)")
    parser.add_argument("--out", default=str(CHECKPOINTS_DIR / "model.onnx"))
    parser.add_argument("--verify-batch-size", type=int, default=256)
    args = parser.parse_args()

    model = ModulationClassifier()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    export_onnx(model, args.out)

    # Verify on real held-out test examples (see verify_export docstring for
    # why real data beats random noise here).
    with h5py.File(args.data, "r") as f:
        n_total = f["X"].shape[0]
    idx = test_indices(n_total)[: args.verify_batch_size]
    loader = DataLoader(RadioMLDataset(args.data, idx), batch_size=args.verify_batch_size)
    iq, _labels, _snr = next(iter(loader))

    verify_export(model, args.out, iq)
    print(f"verification passed: PyTorch and ONNX agree on {len(iq)} real test examples")


if __name__ == "__main__":
    main()
