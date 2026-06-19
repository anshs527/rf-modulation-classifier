"""Tests for Exercise 2 — train_step().

These fail with NotImplementedError until you implement train_step(), then
pass only if the five-line core is genuinely correct. Each test targets a
specific classic bug.
"""
import torch
import torch.nn as nn

from model import ModulationClassifier
from train import train_step


def make_batch(batch_size=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    iq = torch.randn(batch_size, 2, 1024, generator=g)
    labels = torch.randint(0, 24, (batch_size,), generator=g)
    return iq, labels


def test_return_types():
    # Returning the loss *tensor* instead of a float keeps the entire autograd
    # graph alive for every batch you accumulate it over — a silent memory leak.
    model = ModulationClassifier()
    iq, labels = make_batch()
    optimizer = torch.optim.Adam(model.parameters())
    loss_value, num_correct = train_step(model, iq, labels, nn.CrossEntropyLoss(), optimizer)

    assert isinstance(loss_value, float), (
        f"loss_value must be a plain Python float, got {type(loss_value)}. "
        "If it's a Tensor, you returned `loss` instead of `loss.item()`."
    )
    assert isinstance(num_correct, int), (
        f"num_correct must be a plain Python int, got {type(num_correct)}."
    )
    assert 0 <= num_correct <= len(labels)


def test_gradients_exist_after_step():
    # If backward() never ran, .grad stays None on every parameter.
    model = ModulationClassifier()
    iq, labels = make_batch()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    train_step(model, iq, labels, nn.CrossEntropyLoss(), optimizer)

    for name, p in model.named_parameters():
        assert p.grad is not None, (
            f"Parameter {name} has no gradient after train_step — "
            "did backward() run?"
        )


def test_gradients_zeroed_each_step():
    # The zero_grad() test. With lr=0 the weights never change, and with the
    # model in eval() mode (dropout off) two steps on the same batch are
    # identical forward passes — so their gradients must be identical too.
    # If zero_grad() is missing, the second step's gradients are exactly
    # double the first's (accumulated), and this fails.
    model = ModulationClassifier()
    model.eval()
    iq, labels = make_batch()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    criterion = nn.CrossEntropyLoss()

    train_step(model, iq, labels, criterion, optimizer)
    first_grads = {n: p.grad.clone() for n, p in model.named_parameters()}

    train_step(model, iq, labels, criterion, optimizer)
    for name, p in model.named_parameters():
        assert torch.allclose(p.grad, first_grads[name], atol=1e-6), (
            f"Gradient for {name} changed between two identical steps with "
            "lr=0 — gradients are accumulating across steps. "
            "What happens if you forget zero_grad()?"
        )


def test_loss_decreases_overfitting_one_batch():
    # The overfit-one-batch test: train repeatedly on ONE memorized batch.
    # Loss must fall substantially. If a model can't overfit 8 examples,
    # the training loop (or the model) is broken — no full-dataset run can
    # succeed where this fails.
    torch.manual_seed(0)
    model = ModulationClassifier()
    model.train()
    iq, labels = make_batch()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    losses = [train_step(model, iq, labels, criterion, optimizer)[0] for _ in range(50)]

    assert losses[-1] < losses[0] * 0.5, (
        f"Loss went from {losses[0]:.4f} to {losses[-1]:.4f} over 50 steps "
        "on a single memorized batch — it should collapse toward zero. "
        "The loop is not learning: check the order of your five lines."
    )
