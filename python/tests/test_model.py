"""Tests for Exercise 1 — ModulationClassifier.forward().

These fail with NotImplementedError until you implement forward(),
then pass only if the implementation is correct.
"""
import torch

from model import ModulationClassifier


def test_forward_output_shape():
    model = ModulationClassifier()
    out = model(torch.randn(4, 2, 1024))
    assert out.shape == (4, 24), (
        f"Expected output shape (4, 24), got {tuple(out.shape)}. "
        "Each example must map to one logit per modulation class."
    )


def test_forward_returns_logits_not_probabilities():
    # CrossEntropyLoss applies softmax internally. If forward() also applies
    # softmax, every row sums to 1 — and training silently underperforms.
    model = ModulationClassifier()
    model.eval()  # disable dropout so the check is deterministic
    with torch.no_grad():
        out = model(torch.randn(16, 2, 1024))
    row_sums = out.sum(dim=1)
    assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3), (
        "Every output row sums to 1 — it looks like forward() applies "
        "softmax. Return raw logits; CrossEntropyLoss softmaxes internally."
    )


def test_forward_works_with_different_batch_sizes():
    # Catches hardcoded batch dimensions in the flatten step (e.g.
    # x.view(4, -1) instead of flattening relative to the batch).
    model = ModulationClassifier()
    for batch in (1, 7):
        out = model(torch.randn(batch, 2, 1024))
        assert out.shape == (batch, 24)
