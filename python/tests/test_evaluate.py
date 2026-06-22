"""Tests for Exercise 3 — accuracy_by_snr().

Synthetic data with hand-computable per-bucket accuracies. Fails with
NotImplementedError until you implement it.
"""
import numpy as np

from evaluate import accuracy_by_snr


def test_known_bucket_accuracies():
    # Three buckets, deliberately given in scrambled order so sorting is
    # actually exercised:
    #   -10 dB: 4 examples, 1 correct  -> 0.25
    #    +0 dB: 4 examples, 3 correct  -> 0.75
    #   +10 dB: 2 examples, 2 correct  -> 1.00   (sparse bucket!)
    snr    = np.array([0,  -10, 10, 0, -10, -10, 0, 10, -10, 0])
    y_true = np.array([1,   2,  3,  4,  5,   6,  7,  8,  9,  10])
    y_pred = np.array([1,   2,  3,  4,  0,   0,  7,  8,  0,  0])
    #   correct?      yes  yes yes yes  no   no yes yes  no  no

    result = accuracy_by_snr(y_true, y_pred, snr)

    assert list(result.keys()) == [-10, 0, 10], (
        f"Expected keys [-10, 0, 10] in ascending order, got {list(result.keys())}. "
        "Every SNR bucket must be present, sorted, and none dropped."
    )

    acc_m10, n_m10 = result[-10]
    acc_0, n_0 = result[0]
    acc_10, n_10 = result[10]
    assert abs(acc_m10 - 0.25) < 1e-9 and n_m10 == 4
    assert abs(acc_0 - 0.75) < 1e-9 and n_0 == 4
    assert abs(acc_10 - 1.00) < 1e-9 and n_10 == 2


def test_sparse_buckets_not_dropped():
    # A bucket with a single example must still appear. 100% accuracy over
    # n=1 is not 100% accuracy — which is exactly why the count is part of
    # the return value.
    snr    = np.array([0] * 99 + [30])
    y_true = np.zeros(100, dtype=int)
    y_pred = np.zeros(100, dtype=int)

    result = accuracy_by_snr(y_true, y_pred, snr)

    assert 30 in result, "The n=1 bucket at 30 dB was silently dropped."
    acc, count = result[30]
    assert count == 1
    assert abs(acc - 1.0) < 1e-9


def test_all_wrong_bucket():
    snr    = np.array([-20, -20, -20])
    y_true = np.array([1, 2, 3])
    y_pred = np.array([4, 5, 6])

    result = accuracy_by_snr(y_true, y_pred, snr)
    acc, count = result[-20]
    assert acc == 0.0 and count == 3
