"""Evaluation for the RadioML modulation classifier.

Loads a trained checkpoint, runs it over the held-out test set, and reports:
overall accuracy, a confusion matrix (saved to docs/), and accuracy as a
function of SNR (Exercise 3 + saved plot) — the single most informative
figure in this project.

Usage:
    python src/evaluate.py data/dataset/GOLD_XYZ_OSC.0001_1024.hdf5 checkpoints/best.pt
"""
import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import CLASSES, RadioMLDataset
from model import ModulationClassifier
from train import make_splits

DOCS_DIR = Path(__file__).resolve().parent.parent.parent / "docs"


def test_indices(n_total, val_frac=0.1, seed=0):
    """The held-out test set: the first half of train.py's val block.

    Reproduces the exact permutation train.py used (same seed), so these rows
    were never trained on. Caveat worth stating in the report: best.pt was
    *selected* by accuracy over the full val block, which includes these rows
    — a mild form of selection leakage. With one selection among ~10
    checkpoints over ~255k examples the effect is negligible, but naming it
    is cheaper than having an interviewer find it.
    """
    _train_idx, val_idx = make_splits(n_total, subset=1.0, val_frac=val_frac, seed=seed)
    return val_idx[: len(val_idx) // 2]


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Run the model over a loader; return (y_true, y_pred, snr) as numpy arrays."""
    model.eval()
    y_true, y_pred, snrs = [], [], []
    for iq, labels, snr in loader:
        logits = model(iq.to(device))
        y_pred.append(logits.argmax(dim=1).cpu().numpy())
        y_true.append(labels.numpy())
        snrs.append(snr.numpy())
    return np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(snrs)


def accuracy_by_snr(y_true, y_pred, snr):
    """Per-SNR-bucket accuracy over the test set.

    Parameters
    ----------
    y_true : (N,) int array of true class indices
    y_pred : (N,) int array of predicted class indices
    snr : (N,) int array of SNR in dB, parallel to the other two

    Returns
    -------
    dict mapping int SNR -> (accuracy: float, count: int), with keys in
    ascending SNR order (dict insertion order carries the sorting), and one
    entry for EVERY SNR value present in `snr` — sparse buckets included,
    because an accuracy over 3 examples must be visible as such, not dropped.
    """
    for array in (y_true, y_pred, snr):
        assert isinstance(array, np.ndarray), f"expected numpy array, got {type(array)}"
        assert array.ndim == 1, f"expected 1D array, got shape {array.shape}"
        if len(array) != len(y_true):
            raise ValueError(f"expected length {len(y_true)}, got {len(array)}")
        if len(array) == 0:
            raise ValueError("expected non-empty arrays")
        if array.dtype.kind != "i":
            raise ValueError(f"expected integer array, got dtype {array.dtype}")
        if not np.all(np.isfinite(array)):
            raise ValueError("expected finite values only")
        
        acc_by_snr = {}
        for snr_value in np.unique(snr):
            mask = (snr == snr_value)
            bucket_acc = float((y_true[mask] == y_pred[mask]).mean())
            acc_by_snr[snr_value] = (bucket_acc, np.sum(mask))
    return acc_by_snr
        
    # ============================================================
    # EXERCISE 3: accuracy by SNR
    # See docs/understanding-guide.md § Exercise 3 for the full explanation.
    #
    # What this must do:
    #   - Find every distinct SNR value present in `snr`
    #   - For each, compute accuracy over exactly the examples at that SNR
    #   - Return {snr: (accuracy, count)} in ascending SNR order, no
    #     bucket silently dropped no matter how few examples it has
    #
    # What you have available: y_true, y_pred, snr — parallel numpy
    #   arrays of equal length (see docstring above)
    # What it must produce: dict[int, tuple[float, int]] as documented
    #
    # Hints (read only if stuck):
    #   - np.unique(snr) gives you the buckets, already sorted.
    #   - A boolean mask (snr == value) selects one bucket from all
    #     three arrays at once.
    #   - Accuracy within a bucket: mean of (y_true == y_pred) over it.
    # ============================================================
    raise NotImplementedError("EXERCISE 3 — see understanding guide § Exercise 3")


def plot_confusion_matrix(y_true, y_pred, out_path):
    from sklearn.metrics import confusion_matrix

    # normalize="true" makes each row sum to 1, so cell (i, j) reads as
    # "fraction of true class i predicted as j" — comparable across classes.
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(cm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=90, fontsize=7)
    ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def plot_accuracy_by_snr(acc_by_snr, out_path):
    snrs = list(acc_by_snr.keys())
    accs = [acc for acc, _count in acc_by_snr.values()]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(snrs, accs, marker="o")
    ax.axhline(1 / len(CLASSES), linestyle="--", linewidth=0.8, label="chance (1/24)")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, linewidth=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", help="path to GOLD_XYZ_OSC.0001_1024.hdf5")
    parser.add_argument("checkpoint", help="path to a saved state_dict (e.g. checkpoints/best.pt)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModulationClassifier().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    with h5py.File(args.data, "r") as f:
        n_total = f["X"].shape[0]
    idx = test_indices(n_total)
    loader = DataLoader(
        RadioMLDataset(args.data, idx),
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    print(f"evaluating {args.checkpoint} on {len(idx)} held-out test examples...")
    y_true, y_pred, snr = collect_predictions(model, loader, device)

    overall = float((y_true == y_pred).mean())
    print(f"\noverall test accuracy: {overall:.4f}")
    print("(remember: this number aggregates over the full -20..+30 dB sweep "
          "and is close to meaningless without the SNR curve)")

    plot_confusion_matrix(y_true, y_pred, DOCS_DIR / "confusion_matrix.png")

    acc = accuracy_by_snr(y_true, y_pred, snr)
    print("\naccuracy by SNR:")
    for snr_db, (bucket_acc, count) in acc.items():
        print(f"  {snr_db:>4d} dB: {bucket_acc:.3f}  (n={count})")
    plot_accuracy_by_snr(acc, DOCS_DIR / "accuracy_by_snr.png")


if __name__ == "__main__":
    main()
