"""Tests for Exercise 4 — export verification (src/export.py).

These fail with NotImplementedError until verify_export is written, then
check two things: that a faithful export passes, and that a WRONG model
fails. The second test matters just as much — a verification whose tolerance
is loose enough to pass anything verifies nothing.
"""
import numpy as np
import pytest
import torch

from export import export_onnx, verify_export
from model import ModulationClassifier


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """A random-weights model and its ONNX export.

    Random weights are fine here: verification compares the two runtimes
    against each other, not against ground truth, so no training is needed.
    module scope because the export itself takes a few seconds.
    """
    torch.manual_seed(0)
    model = ModulationClassifier()
    onnx_path = tmp_path_factory.mktemp("export") / "model.onnx"
    export_onnx(model, onnx_path)
    return model, onnx_path


def test_faithful_export_passes(exported):
    model, onnx_path = exported
    batch = torch.randn(8, 2, 1024)
    # Must not raise: same model, same graph — any disagreement within a
    # sane float32 tolerance means the verification itself is miscalibrated.
    verify_export(model, onnx_path, batch)


def test_wrong_model_is_caught(exported):
    _model, onnx_path = exported
    torch.manual_seed(1)  # different seed -> genuinely different weights
    imposter = ModulationClassifier()
    batch = torch.randn(8, 2, 1024)
    with pytest.raises(AssertionError):
        verify_export(imposter, onnx_path, batch)


def test_dynamic_batch_dimension(exported):
    """The exported graph must accept batch sizes other than the traced one.

    Guards the dynamic_axes plumbing in export_onnx: if the batch dim got
    frozen at 1, the C++ pipeline could never batch (Phase 10's most likely
    optimization).
    """
    import onnxruntime as ort

    _model, onnx_path = exported
    session = ort.InferenceSession(str(onnx_path))
    for batch_size in (1, 3, 16):
        out = session.run(None, {"iq": np.random.randn(batch_size, 2, 1024).astype(np.float32)})[0]
        assert out.shape == (batch_size, 24)
