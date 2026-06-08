"""Lazy-loading PyTorch Dataset for RadioML 2018.01a.

The HDF5 file (GOLD_XYZ_OSC.0001_1024.hdf5) stores three top-level datasets,
all indexed by the same first axis:

    X: (N, 1024, 2) float32  — IQ samples, channels-last (I, Q per timestep)
    Y: (N, 24)      int64    — one-hot modulation class label
    Z: (N, 1)       int64    — SNR in dB, in {-20, -18, ..., 28, 30}

N = 2,555,904 for the full file. X alone is ~20 GB, which is why this class
never reads more than one example at a time from disk.
"""
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# Canonical class order for RadioML 2018.01a, matching data/dataset/classes-fixed.json.
# Hardcoded rather than read from the JSON at runtime: this order is fixed by the
# dataset's Y one-hot encoding, not by anything the user configures, so treating it
# as a dataset-file-path-dependent lookup would be a false degree of freedom.
CLASSES = [
    "OOK", "4ASK", "8ASK",
    "BPSK", "QPSK", "8PSK",
    "16PSK", "32PSK", "16APSK",
    "32APSK", "64APSK", "128APSK",
    "16QAM", "32QAM", "64QAM",
    "128QAM", "256QAM", "AM-SSB-WC",
    "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC",
    "FM", "GMSK", "OQPSK",
]


class RadioMLDataset(Dataset):
    """One (IQ window, class label, SNR) triple per example.

    Returns IQ as a `(2, 1024)` float32 tensor — channels-first, matching what
    `nn.Conv1d` expects (channels, length) — not the file's native `(1024, 2)`.

    Parameters
    ----------
    hdf5_path : str
        Path to GOLD_XYZ_OSC.0001_1024.hdf5.
    indices : array-like of int, optional
        Subset of row indices this dataset should expose (e.g. a train/val
        split, or a class/SNR filter). Defaults to the full file.
    """

    def __init__(self, hdf5_path, indices=None):
        self.hdf5_path = hdf5_path
        # The h5py.File handle is opened lazily, per worker process, in
        # __getitem__ rather than here. h5py file handles are not safe to
        # fork/pickle across DataLoader worker processes (num_workers > 0)
        # — opening eagerly in __init__ works fine with num_workers=0 and
        # then breaks silently or crashes the moment you turn workers on.
        self._file = None

        if indices is None:
            with h5py.File(hdf5_path, "r") as f:
                n = f["X"].shape[0]
            indices = np.arange(n)
        self.indices = np.asarray(indices)

    def _file_handle(self):
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r")
        return self._file

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        f = self._file_handle()
        row = int(self.indices[idx])

        iq = f["X"][row]  # (1024, 2) float32
        iq = torch.from_numpy(iq.T.copy())  # (2, 1024); .copy() makes the transpose contiguous

        label = int(f["Y"][row].argmax())
        snr = int(f["Z"][row, 0])

        return iq, label, snr
