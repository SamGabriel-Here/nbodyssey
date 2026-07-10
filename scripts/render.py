#!/usr/bin/env python3
"""Render dumped position frames into images or an animation.

Reads the per-frame binary dumps written by the simulator and produces a
scatter plot per frame (matplotlib), optionally stitched into a movie. Kept
separate from the simulator so rendering choices never constrain the physics.

Not implemented yet. This fixes the CLI.
"""
import argparse


def load_frame(path):
    """Read one frame's positions (float4 per particle) into an (n, 3) array."""
    raise NotImplementedError  # TODO


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", required=True, help="directory of frame dumps")
    ap.add_argument("--out", default="collision.mp4", help="output movie or image dir")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    # TODO: iterate frames, scatter x/y (project along z), write images, encode movie
    raise SystemExit("render: not implemented yet")


if __name__ == "__main__":
    main()
