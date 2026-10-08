#!/usr/bin/env python3
"""Plot an energy log (from the simulator or the reference integrator).

Left panel: kinetic, potential, and total energy versus time -- during a close
encounter kinetic energy peaks at pericenter while the potential well deepens,
and the total stays flat. Right panel: relative energy error, the symplectic
integrator's signature -- it stays bounded and oscillates rather than drifting.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True, help="energy CSV")
    ap.add_argument("--out", default="energy.png")
    args = ap.parse_args()

    d = np.genfromtxt(args.log, delimiter=",", names=True)
    max_err = float(np.max(np.abs(d["rel_error"])))

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(d["time"], d["energy"], lw=2, label="total")
    ax[0].plot(d["time"], d["kinetic"], "--", alpha=0.8, label="kinetic")
    ax[0].plot(d["time"], d["potential"], "--", alpha=0.8, label="potential")
    ax[0].set_xlabel("time"); ax[0].set_ylabel("energy")
    ax[0].set_title("energy components"); ax[0].legend()

    ax[1].plot(d["time"], 100.0 * d["rel_error"])
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xlabel("time"); ax[1].set_ylabel("relative energy error (%)")
    ax[1].set_title(f"max |relative error| = {max_err:.2e}")

    plt.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"max |relative error| = {max_err:.3e} -> {args.out}")


if __name__ == "__main__":
    main()
