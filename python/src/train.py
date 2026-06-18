"""Training script for the RadioML modulation classifier.

Usage (1% subset — always confirm the pipeline end-to-end before a full run):
    python src/train.py data/dataset/GOLD_XYZ_OSC.0001_1024.hdf5 --subset 0.01

Full run:
    python src/train.py data/dataset/GOLD_XYZ_OSC.0001_1024.hdf5
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import RadioMLDataset
from model import ModulationClassifier


def train_step(model, iq, labels, criterion, optimizer):
    """Run one optimization step on one batch. Returns (loss_value, num_correct).

    Parameters
    ----------
    model : ModulationClassifier, already in train() mode
    iq : (batch, 2, 1024) float32 tensor
    labels : (batch,) int64 tensor of class indices
    criterion : nn.CrossEntropyLoss instance
    optimizer : optimizer constructed over model.parameters()

    Returns
    -------
    loss_value : float — the batch loss as a plain Python float
    num_correct : int — how many of the batch's predictions matched labels
    """
    optimizer.zero_grad()
    outputs = model(iq)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    num_correct = (outputs.argmax(dim=1) == labels).sum().item()
    return loss.item(), num_correct
    # ============================================================
    # EXERCISE 2: the training step
    # See docs/understanding-guide.md § Exercise 2 for the full explanation.
    #
    # What this must do:
    #   - Zero the gradients
    #   - Forward pass: get logits from the model
    #   - Compute the loss with criterion
    #   - Backward pass
    #   - Optimizer step
    #   - Compute num_correct: how many argmax-predictions equal labels
    #   - Return (loss as a plain float, num_correct as an int)
    #
    # What you have available: model, iq, labels, criterion, optimizer
    #   (shapes/types documented in the docstring above)
    # What it must produce: (float, int)
    #
    # Hints (read only if stuck):
    #   - What happens if you forget zero_grad()? Gradients accumulate
    #     across batches and the model trains on a nonsense signal,
    #     silently. One of the tests catches exactly this.
    #   - What is the difference between `loss` and `loss.item()`, and
    #     why does returning/accumulating the wrong one leak memory?
    #     (Hint: one of them drags the whole autograd graph with it.)
    #   - Predictions come from logits.argmax(dim=...) — which dim?
    # ============================================================
    #raise NotImplementedError("EXERCISE 2 — see understanding guide § Exercise 2")


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Average loss and accuracy over a loader, without touching gradients."""
    model.eval()
    total_loss, total_correct, total_seen = 0.0, 0, 0
    for iq, labels, _snr in loader:
        iq, labels = iq.to(device), labels.to(device)
        logits = model(iq)
        total_loss += criterion(logits, labels).item() * len(labels)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_seen += len(labels)
    return total_loss / total_seen, total_correct / total_seen


def make_splits(n_total, subset, val_frac, seed):
    """Shuffled train/val index arrays over (a subset of) the file's rows.

    The shuffle matters: the HDF5 file is sorted by class and SNR, so a
    contiguous slice would give a train set missing entire classes.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)[: int(n_total * subset)]
    n_val = int(len(indices) * val_frac)
    return indices[n_val:], indices[:n_val]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", help="path to GOLD_XYZ_OSC.0001_1024.hdf5")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--subset", type=float, default=1.0,
                        help="fraction of the dataset to use (e.g. 0.01 for a pipeline check)")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    import h5py
    with h5py.File(args.data, "r") as f:
        n_total = f["X"].shape[0]
    train_idx, val_idx = make_splits(n_total, args.subset, args.val_frac, args.seed)
    print(f"train: {len(train_idx)}  val: {len(val_idx)}  (subset={args.subset})")

    train_loader = DataLoader(
        RadioMLDataset(args.data, train_idx),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        RadioMLDataset(args.data, val_idx),
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    model = ModulationClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, running_correct, seen = 0.0, 0, 0
        t0 = time.time()

        for iq, labels, _snr in train_loader:
            iq, labels = iq.to(device), labels.to(device)
            loss_value, num_correct = train_step(model, iq, labels, criterion, optimizer)
            running_loss += loss_value * len(labels)
            running_correct += num_correct
            seen += len(labels)

        train_loss = running_loss / seen
        train_acc = running_correct / seen
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"epoch {epoch:>3}/{args.epochs}  "
            f"train loss {train_loss:.4f} acc {train_acc:.3f}  |  "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}  |  "
            f"{time.time() - t0:.0f}s",
            flush=True,  # epoch lines are rare and small; without this they
            # sit in the stdout buffer for hours when output is redirected
        )

        torch.save(model.state_dict(), ckpt_dir / "last.pt")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            print(f"  new best val acc {val_acc:.3f} -> saved {ckpt_dir / 'best.pt'}")

    print(f"done. best val acc: {best_val_acc:.3f}")


if __name__ == "__main__":
    main()
