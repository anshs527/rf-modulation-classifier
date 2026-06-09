"""Exploration script for RadioML 2018.01a.

Prints class distribution and SNR distribution, and plots raw I and Q
channels for a few modulation types at high SNR so you can sanity-check
what the network is actually being fed before you write a single line of
model code.

Usage:
    python src/explore_dataset.py path/to/GOLD_XYZ_OSC.0001_1024.hdf5
"""
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from dataset import CLASSES

DOCS_DIR = Path(__file__).resolve().parent.parent.parent / "docs"


def print_class_distribution(f):
    # Y is (N, 24) one-hot; summing each column gives the per-class count.
    # Y and Z are small enough to load fully (tens of MB) — it's X, at ~20 GB,
    # that dataset.py is written to never load in one shot.
    counts = f["Y"][:].sum(axis=0)
    print("\nClass distribution:")
    for name, count in zip(CLASSES, counts):
        print(f"  {name:>10s}: {count:>8d}")


def print_snr_distribution(f):
    snr = f["Z"][:, 0]
    values, counts = np.unique(snr, return_counts=True)
    print("\nSNR distribution (dB):")
    for v, c in zip(values, counts):
        print(f"  {v:>4d} dB: {c:>8d}")


def plot_iq_examples(f, class_names, snr_db, out_path):
    """Find one example of each requested class at the given SNR and plot I/Q."""
    y_argmax = f["Y"][:].argmax(axis=1)
    snr = f["Z"][:, 0]

    fig, axes = plt.subplots(len(class_names), 1, figsize=(10, 3 * len(class_names)))
    if len(class_names) == 1:
        axes = [axes]

    for ax, name in zip(axes, class_names):
        class_idx = CLASSES.index(name)
        matches = np.nonzero((y_argmax == class_idx) & (snr == snr_db))[0]
        if len(matches) == 0:
            raise ValueError(f"No examples of {name} at {snr_db} dB")
        row = matches[0]

        iq = f["X"][row]  # (1024, 2)
        ax.plot(iq[:, 0], label="I", linewidth=0.8)
        ax.plot(iq[:, 1], label="Q", linewidth=0.8)
        ax.set_title(f"{name} @ {snr_db} dB (row {row})")
        ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path)
    print(f"\nSaved IQ plot to {out_path}")


def plot_constellation(f, class_name, snr_dbs, out_path):
    """Scatter I vs. Q for one class at several SNRs, side by side.

    This is the plot that makes 'phase shows up as rotation in the I/Q plane'
    concrete: each modulation scheme has a characteristic constellation, and
    noise smears it.
    """
    y_argmax = f["Y"][:].argmax(axis=1)
    snr = f["Z"][:, 0]
    class_idx = CLASSES.index(class_name)

    fig, axes = plt.subplots(1, len(snr_dbs), figsize=(5 * len(snr_dbs), 5))
    for ax, snr_db in zip(axes, snr_dbs):
        matches = np.nonzero((y_argmax == class_idx) & (snr == snr_db))[0]
        if len(matches) == 0:
            raise ValueError(f"No examples of {class_name} at {snr_db} dB")
        iq = f["X"][matches[0]]  # (1024, 2)
        ax.scatter(iq[:, 0], iq[:, 1], s=3, alpha=0.5)
        ax.set_title(f"{class_name} @ {snr_db} dB")
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.set_aspect("equal")
        ax.grid(True, linewidth=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved constellation plot to {out_path}")


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} path/to/GOLD_XYZ_OSC.0001_1024.hdf5")
        sys.exit(1)

    hdf5_path = sys.argv[1]
    with h5py.File(hdf5_path, "r") as f:
        print(f"X: {f['X'].shape} {f['X'].dtype}")
        print(f"Y: {f['Y'].shape} {f['Y'].dtype}")
        print(f"Z: {f['Z'].shape} {f['Z'].dtype}")

        print_class_distribution(f)
        print_snr_distribution(f)
        plot_iq_examples(
            f,
            class_names=["BPSK", "QPSK", "16QAM"],
            snr_db=30,
            out_path=DOCS_DIR / "iq_exploration.png",
        )
        plot_constellation(
            f,
            class_name="QPSK",
            snr_dbs=[30, 10, -10],
            out_path=DOCS_DIR / "constellation_qpsk.png",
        )


if __name__ == "__main__":
    main()
