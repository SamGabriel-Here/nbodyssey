#!/usr/bin/env python3
"""Build the multi-panel "stages of the collision" figure used in the README.

Reads five frame_*.bin dumps (same format the simulator and reference integrator
write) and lays them out as labeled panels narrating the encounter: approach,
first contact, pericenter, tidal bridge, and separated remnants. Reproducible so
the figure in docs/ is generated, not hand-assembled.

Example:
    python scripts/stages_figure.py --frames frames/ --out docs/stages.png
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (frame index, stage label). dt * dump_every sets the time under each panel.
STAGES = [
    (0,   "Approach"),
    (90,  "First contact"),
    (134, "Pericenter"),
    (210, "Tidal bridge"),
    (299, "Remnants"),
]
cA, cB = "#5ec8ff", "#ff9d3c"   # galaxy A cool blue, galaxy B warm gold


def read_frame(path):
    with open(path, "rb") as f:
        n = int(np.frombuffer(f.read(4), "<i4")[0])
        return np.frombuffer(f.read(n * 16), "<f4").reshape(n, 4)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="directory of frame_*.bin dumps")
    ap.add_argument("--out", default="docs/stages.png")
    ap.add_argument("--extent", type=float, default=7.5, help="half-width of each panel")
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--dump-every", type=int, default=5)
    ap.add_argument("--point-size", type=float, default=1.1)
    ap.add_argument("--alpha", type=float, default=0.55)
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    ncol = len(STAGES)
    fig, axes = plt.subplots(1, ncol, figsize=(3.0 * ncol, 3.6), facecolor="black")
    fig.subplots_adjust(left=0.004, right=0.996, top=0.80, bottom=0.02, wspace=0.03)

    for ax, (idx, label) in zip(axes, STAGES):
        pos = read_frame(os.path.join(args.frames, f"frame_{idx:05d}.bin"))
        split = pos.shape[0] // 2
        ax.set_facecolor("black")
        ax.scatter(pos[:split, 0], pos[:split, 1], s=args.point_size,
                   c=cA, alpha=args.alpha, linewidths=0)
        ax.scatter(pos[split:, 0], pos[split:, 1], s=args.point_size,
                   c=cB, alpha=args.alpha, linewidths=0)
        ax.set_xlim(-args.extent, args.extent)
        ax.set_ylim(-args.extent, args.extent)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#333333")
        t = idx * args.dump_every * args.dt
        ax.set_title(label, color="white", fontsize=13, pad=5)
        ax.text(0.5, 0.02, f"t = {t:.1f}", color="#999999", fontsize=9,
                ha="center", va="bottom", transform=ax.transAxes)

    fig.suptitle("Anatomy of a disk-galaxy collision  ·  12k particles, leapfrog integrator",
                 color="white", fontsize=14, y=0.965)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor="black")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
