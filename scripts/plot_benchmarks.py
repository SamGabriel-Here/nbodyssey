#!/usr/bin/env python3
"""Plot the GPU benchmark results produced by scripts/gpu_bench.sh.

Left panel: force-computation time per step for the naive kernel vs the
Barnes-Hut pipeline across particle counts, with the measured speedup marked.
Right panel: relative energy error over the full 12k-particle collision on the
GPU for both force modules, which is the accuracy price of theta = 0.5 shown
honestly next to the speed.
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="benchmarks/gpu_results.csv")
    ap.add_argument("--energy-naive", default="benchmarks/energy_gpu_naive.csv")
    ap.add_argument("--energy-bh", default="benchmarks/energy_gpu_bh.csv")
    ap.add_argument("--gpu", default="Tesla T4")
    ap.add_argument("--out", default="docs/benchmark_t4.png")
    args = ap.parse_args()

    rows = read_results(args.results)
    naive = sorted((n, ms) for n, m, th, ms in rows if m == "naive")
    bh = sorted((n, ms) for n, m, th, ms in rows if m == "bh" and th == 0.5)
    nn, tn = zip(*naive)
    nb, tb = zip(*bh)

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    fig.suptitle(f"Naive all-pairs vs Barnes-Hut on a {args.gpu}", fontsize=14)

    ax[0].loglog(nn, tn, "o-", color="#c81e5a", label="naive $O(n^2)$")
    ax[0].loglog(nb, tb, "o-", color="#2a9d5c",
                 label=r"Barnes-Hut, $\theta=0.5$")
    for n_, ta, tbh in [(50000, 11.613, 1.956), (1000000, 5269.903, 111.568)]:
        ax[0].annotate(f"{ta / tbh:.0f}x", xy=(n_, np.sqrt(ta * tbh)),
                       ha="center", va="center", fontsize=11, color="#444444")
    ax[0].set_xlabel("particle count n")
    ax[0].set_ylabel("force computation, ms per step")
    ax[0].set_title("Time per step (CUDA events, 11-call average)")
    ax[0].legend()
    ax[0].grid(True, which="both", alpha=0.3)

    tn_t, rn = read_energy(args.energy_naive)
    tb_t, rb = read_energy(args.energy_bh)
    ax[1].semilogy(tn_t[1:], rn[1:], "o-", color="#c81e5a", label="naive")
    ax[1].semilogy(tb_t[1:], rb[1:], "o-", color="#2a9d5c",
                   label=r"Barnes-Hut, $\theta=0.5$")
    ax[1].set_xlabel("simulation time")
    ax[1].set_ylabel("|relative energy error|")
    ax[1].set_title("Energy error, 12k-particle collision, 1500 steps")
    ax[1].legend()
    ax[1].grid(True, which="both", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
