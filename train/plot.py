"""Plot the training curve and the engine benchmarks into results/.

    python train/plot.py
"""

import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def training_curve():
    rows = list(csv.DictReader(open(os.path.join(RESULTS, "train_log.csv"))))
    it = [int(r["iter"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(it, [float(r["train_loss"]) for r in rows], label="train")
    ax.plot(it, [float(r["val_loss"]) for r in rows], label="validation")
    ax.set_xlabel("iteration")
    ax.set_ylabel("cross-entropy (nats / char)")
    ax.set_title("Training on Tiny Shakespeare")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "training_curve.png"), dpi=150)


def throughput():
    data = json.load(open(os.path.join(RESULTS, "benchmarks.json")))["results"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for r in data:
        threads = [s["threads"] for s in r["speed"]]
        ax.plot(threads, [s["tokens_per_s"] for s in r["speed"]], marker="o", label=r["precision"])
    ax.set_xlabel("threads")
    ax.set_ylabel("tokens / second")
    ax.set_xticks(threads)
    ax.set_ylim(bottom=0)
    ax.set_title("Decoding throughput (batch 1, Ryzen 7 7800X3D)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "throughput.png"), dpi=150)


if __name__ == "__main__":
    training_curve()
    throughput()
    print("wrote results/training_curve.png and results/throughput.png")
