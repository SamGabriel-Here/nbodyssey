#!/usr/bin/env python3
"""Render dumped position frames into images and an animation.

Reads the per-frame binaries written by the simulator (or the reference
integrator) and produces a scatter plot per frame, projected onto a chosen
plane, then optionally encodes them into a movie with ffmpeg. Kept separate from
the simulation so rendering choices never constrain the physics.

The two galaxies are colored by particle index (the first `--split` particles
are galaxy A, the rest galaxy B), which lets you watch material get stripped,
flung into tidal tails, and mixed during the merger.
"""
import argparse
import glob
import os
import subprocess
import sys
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AXES = {"x": 0, "y": 1, "z": 2}


def read_frame(path):
    with open(path, "rb") as f:
        n = int(np.frombuffer(f.read(4), "<i4")[0])
        pos = np.frombuffer(f.read(n * 16), "<f4").reshape(n, 4)
    return pos


def auto_extent(frames, ai, bi, percentile):
    """Symmetric plot extent covering `percentile` of particles across frames."""
    sample = frames if len(frames) <= 12 else [frames[i] for i in
             np.linspace(0, len(frames) - 1, 12).astype(int)]
    vals = []
    for p in sample:
        pos = read_frame(p)
        vals.append(np.abs(pos[:, [ai, bi]]))
    v = np.concatenate(vals, axis=0)
    r = np.percentile(v, percentile)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="directory of frame_*.bin dumps")
    ap.add_argument("--out", default="collision.mp4",
                    help="output .mp4, or a directory to leave PNGs in")
    ap.add_argument("--split", type=int, default=-1,
                    help="index splitting galaxy A from B (default: half)")
    ap.add_argument("--projection", default="xy", choices=["xy", "xz", "yz"])
    ap.add_argument("--extent", type=float, default=0.0,
                    help="half-width of the view; 0 auto-fits from the frames")
    ap.add_argument("--percentile", type=float, default=99.0,
                    help="percentile of particles to keep in view when auto-fitting")
    ap.add_argument("--point-size", type=float, default=1.2)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    frames = sorted(glob.glob(os.path.join(args.frames, "frame_*.bin")))
    if not frames:
        sys.exit(f"no frame_*.bin found in {args.frames}")

    ai, bi = AXES[args.projection[0]], AXES[args.projection[1]]
    extent = args.extent or auto_extent(frames, ai, bi, args.percentile)

    n0 = read_frame(frames[0]).shape[0]
    split = args.split if args.split >= 0 else n0 // 2
    cA, cB = "#5ec8ff", "#ff9d3c"   # cool blue, warm gold on black

    to_dir = not args.out.endswith(".mp4")
    png_dir = args.out if to_dir else tempfile.mkdtemp(prefix="gcframes_")
    os.makedirs(png_dir, exist_ok=True)

    print(f"{len(frames)} frames, n={n0}, projection={args.projection}, "
          f"extent=+/-{extent:.2f}")
    for i, path in enumerate(frames):
        pos = read_frame(path)
        fig = plt.figure(figsize=(8, 8), facecolor="black")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_facecolor("black")
        ax.scatter(pos[:split, ai], pos[:split, bi], s=args.point_size,
                   c=cA, alpha=args.alpha, linewidths=0, edgecolors="none")
        ax.scatter(pos[split:, ai], pos[split:, bi], s=args.point_size,
                   c=cB, alpha=args.alpha, linewidths=0, edgecolors="none")
        ax.set_xlim(-extent, extent); ax.set_ylim(-extent, extent)
        ax.set_xticks([]); ax.set_yticks([])
        fig.savefig(os.path.join(png_dir, f"f_{i:05d}.png"),
                    dpi=args.dpi, facecolor="black")
        plt.close(fig)
        if (i + 1) % 25 == 0:
            print(f"  rendered {i + 1}/{len(frames)}")

    if to_dir:
        print(f"wrote {len(frames)} PNGs to {png_dir}")
        return

    cmd = ["ffmpeg", "-y", "-framerate", str(args.fps),
           "-i", os.path.join(png_dir, "f_%05d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", args.out]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"encoded {args.out}")


if __name__ == "__main__":
    main()
