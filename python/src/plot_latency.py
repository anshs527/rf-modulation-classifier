"""Plot the latency histogram from rf_benchmark's CSV (Phase 10).

Produces two panels:
  - histogram of total per-inference latency, with p50/p99 marked — the
    right-skew and the tail are the story, a single mean would hide both;
  - mean time per stage (convert / Run / argmax), which is the evidence an
    Exercise 8 optimization hypothesis has to be consistent with.

numpy's percentile default is linear interpolation — the same rule as the
C++ compute_stats, deliberately, so both sides print the same numbers.

Usage:
    python src/plot_latency.py ../cpp/data/latencies.csv --out ../docs/latency_histogram.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="latencies.csv written by rf_benchmark")
    parser.add_argument("--out", default=str(
        Path(__file__).resolve().parent.parent.parent / "docs" / "latency_histogram.png"))
    args = parser.parse_args()

    data = np.genfromtxt(args.csv, delimiter=",", names=True)
    lat = data["latency_us"]
    p50, p99 = np.percentile(lat, [50, 99])

    fig, (ax_hist, ax_stage) = plt.subplots(
        1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [2, 1]})

    ax_hist.hist(lat, bins=80, color="steelblue", edgecolor="none")
    ax_hist.axvline(p50, color="black", linestyle="--", label=f"p50 = {p50:.0f} us")
    ax_hist.axvline(p99, color="crimson", linestyle="--", label=f"p99 = {p99:.0f} us")
    ax_hist.set_xlabel("per-inference latency (us)")
    ax_hist.set_ylabel("count")
    ax_hist.set_title(f"C++ inference latency, n={len(lat)} (mean {lat.mean():.0f} us)")
    ax_hist.legend()

    stages = ["convert_us", "run_us", "argmax_us"]
    means = [float(data[s].mean()) for s in stages]
    ax_stage.bar(["convert\n+wrap", "session\n.Run", "argmax\n+copy"], means,
                 color=["darkorange", "steelblue", "seagreen"])
    for i, m in enumerate(means):
        ax_stage.text(i, m, f" {m:.1f}", ha="center", va="bottom")
    ax_stage.set_ylabel("mean time (us)")
    ax_stage.set_title("where the time goes")

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")
    print(f"p50 {p50:.1f} us | p99 {p99:.1f} us | stage means: "
          + ", ".join(f"{s}={m:.1f}" for s, m in zip(stages, means)))


if __name__ == "__main__":
    main()
