"""Dump a slice of the held-out test set to a flat binary file for the C++ pipeline.

HDF5-in-C++ would be an unnecessary dependency for a benchmark harness, so
the C++ side reads this trivial format instead (reader: cpp/src/iq_reader.hpp).

Besides the IQ data, the file carries the PyTorch model's own predictions and
logits for every sample. That is the ground truth for Phase 9's parity test:
the C++ pipeline must reproduce these argmaxes EXACTLY (same ONNX graph, same
runtime, same floats — unlike Exercise 4, no tolerance is needed or excusable).

RFIQ dump format v1 (all little-endian):
    offset 0   char[4]   magic  = "RFIQ"
    offset 4   uint32    version = 1
    offset 8   uint32    count        (N samples)
    offset 12  uint32    length       (T timesteps, 1024)
    offset 16  uint32    num_classes  (24)
    then, back to back:
    iq       N*T*2 float32   per sample: interleaved i0,q0,i1,q1,...
    labels   N     int32     true class index
    preds    N     int32     PyTorch eval-mode argmax
    logits   N*24  float32   PyTorch eval-mode logits
    snr      N     int32     SNR in dB

NOTE the layout: iq is INTERLEAVED (time-major, the HDF5 file's native
(1024, 2) order) — not the channels-first (2, 1024) the model consumes.
The conversion is the C++ pipeline's job, and getting it wrong is exactly
the class of bug the preds/logits in this file exist to catch.

Usage:
    python src/dump_test_bin.py data/dataset/GOLD_XYZ_OSC.0001_1024.hdf5 \
        checkpoints/best.pt --n 1000 --out ../cpp/data/test_slice.bin
"""
import argparse
import struct
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import RadioMLDataset
from evaluate import test_indices
from model import ModulationClassifier

MAGIC = b"RFIQ"
VERSION = 1


def dump(model, loader, out_path, num_classes=24):
    model.eval()
    iq_chunks, labels, preds, logits_chunks, snrs = [], [], [], [], []
    with torch.no_grad():
        for iq, label, snr in loader:
            logits = model(iq)
            # (B, 2, T) channels-first back to (B, T, 2) time-major, then a
            # flat view is automatically interleaved i0,q0,i1,q1,...
            iq_chunks.append(iq.permute(0, 2, 1).contiguous().numpy())
            logits_chunks.append(logits.numpy())
            preds.append(logits.argmax(dim=1).numpy())
            labels.append(label.numpy())
            snrs.append(snr.numpy())

    iq = np.concatenate(iq_chunks).astype(np.float32)
    logits = np.concatenate(logits_chunks).astype(np.float32)
    labels = np.concatenate(labels).astype(np.int32)
    preds = np.concatenate(preds).astype(np.int32)
    snrs = np.concatenate(snrs).astype(np.int32)
    n, t = iq.shape[0], iq.shape[1]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        # struct.pack "<": force little-endian regardless of host, so the
        # format spec above is true by construction, not by luck.
        f.write(MAGIC)
        f.write(struct.pack("<IIII", VERSION, n, t, num_classes))
        f.write(iq.tobytes())
        f.write(labels.tobytes())
        f.write(preds.tobytes())
        f.write(logits.tobytes())
        f.write(snrs.tobytes())

    acc = float((labels == preds).mean())
    print(f"wrote {out_path} — {n} samples x {t} timesteps "
          f"({out_path.stat().st_size / 1e6:.1f} MB), PyTorch accuracy on slice: {acc:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", help="path to GOLD_XYZ_OSC.0001_1024.hdf5")
    parser.add_argument("checkpoint", help="path to a saved state_dict")
    parser.add_argument("--n", type=int, default=1000, help="number of test samples to dump")
    parser.add_argument("--out", default=str(
        Path(__file__).resolve().parent.parent.parent / "cpp" / "data" / "test_slice.bin"))
    args = parser.parse_args()

    model = ModulationClassifier()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    with h5py.File(args.data, "r") as f:
        n_total = f["X"].shape[0]
    idx = test_indices(n_total)[: args.n]
    loader = DataLoader(RadioMLDataset(args.data, idx), batch_size=256)

    dump(model, loader, Path(args.out))


if __name__ == "__main__":
    main()
