#!/usr/bin/env python3
"""Plot the GPU benchmark results produced by scripts/gpu_bench.sh.

Left panel: force-computation time per step across particle counts — the naive
kernel against both Barnes-Hut tree walks (per-thread and warp-cooperative) —
with measured speedups marked. Right panel: relative energy error over the full
12k-particle collision on the GPU for each variant, which is the accuracy price
of the approximations shown honestly next to the speed.
"""
import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_results(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append((int(r["n"]), r["method"], float(r["theta"]),
                         float(r["ms_per_call"])))
    return rows


def read_energy(path):
    t, rel = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            t.append(float(r["time"]))
            rel.append(abs(float(r["rel_error"])))
    return np.array(t), np.array(rel)


def series(rows, method, theta=0.5):
    pts = sorted((n, ms) for n, m, th, ms in rows
                 if m == method and th == theta)
    return [p[0] for p in pts], [p[1] for p in pts]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="benchmarks/gpu_results.csv")
    ap.add_argument("--energy-naive", default="benchmarks/energy_gpu_naive.csv")
    ap.add_argument("--energy-thread",
                    default="benchmarks/energy_gpu_bh_thread.csv")
    ap.add_argument("--energy-warp", default="benchmarks/energy_gpu_bh_warp.csv")
    ap.add_argument("--gpu", default="Tesla T4")
    ap.add_argument("--out", default="docs/benchmark_t4.png")
    args = ap.parse_args()

    rows = read_results(args.results)
    nn, tn = series(rows, "naive")
    nt, tt = series(rows, "bh-thread")
    nw, tw = series(rows, "bh-warp")

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    fig.suptitle(f"Naive all-pairs vs Barnes-Hut on a {args.gpu}", fontsize=14)

    ax[0].loglog(nn, tn, "o-", color="#c81e5a", label="naive $O(n^2)$")
    ax[0].loglog(nt, tt, "o-", color="#1f77b4",
                 label=r"Barnes-Hut, per-thread walk")
    ax[0].loglog(nw, tw, "o-", color="#2a9d5c",
                 label=r"Barnes-Hut, warp-cooperative walk")
    ax[0].annotate("47x", xy=(1e6, np.sqrt(5324.170 * 112.103)), ha="center",
                   va="center", fontsize=11, color="#1f77b4")
    ax[0].annotate("176x", xy=(1e6, np.sqrt(112.103 * 30.328) * 0.28),
                   ha="center", va="center", fontsize=11, color="#2a9d5c")
    ax[0].set_xlabel("particle count n")
    ax[0].set_ylabel("force computation, ms per step")
    ax[0].set_title(r"Time per step at $\theta=0.5$ (CUDA events)")
    ax[0].legend(fontsize=9)
    ax[0].grid(True, which="both", alpha=0.3)

    for path, color, label in [
        (args.energy_naive, "#c81e5a", "naive"),
        (args.energy_thread, "#1f77b4", "BH per-thread"),
        (args.energy_warp, "#2a9d5c", "BH warp"),
    ]:
        t, rel = read_energy(path)
        ax[1].semilogy(t[1:], rel[1:], "o-", color=color, label=label,
                       markersize=4)
    ax[1].set_xlabel("simulation time")
    ax[1].set_ylabel("|relative energy error|")
    ax[1].set_title(r"Energy error, 12k-particle collision, $\theta=0.5$")
    ax[1].legend(fontsize=9)
    ax[1].grid(True, which="both", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
