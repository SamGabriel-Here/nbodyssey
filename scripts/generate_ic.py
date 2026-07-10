#!/usr/bin/env python3
"""Generate initial conditions: two disk galaxies on a collision course.

Writes a flat binary of particle state that the simulator uploads to the GPU.
Layout matches the device SoA float4 arrays: for each particle, position+mass
(x, y, z, m) followed by velocity (vx, vy, vz, 0).

Not implemented yet. This fixes the CLI and the on-disk format.
"""
import argparse
import struct  # noqa: F401  (used once the writer lands)


def build_disk(n, center, velocity, mass, radius):
    """Exponential-profile disk with particles on circular orbits.

    TODO: sample radii from an exponential surface density, place particles at
    random azimuths, set circular velocity from the enclosed mass, then boost by
    the bulk `velocity` and translate to `center`.
    """
    raise NotImplementedError


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--particles", type=int, default=50000,
                    help="total particles across both galaxies")
    ap.add_argument("--out", default="ic.bin", help="output binary path")
    ap.add_argument("--separation", type=float, default=20.0,
                    help="initial separation between galaxy centers")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # TODO: build two disks, concatenate, write float4 pos + float4 vel to args.out
    raise SystemExit("generate_ic: not implemented yet")


if __name__ == "__main__":
    main()
