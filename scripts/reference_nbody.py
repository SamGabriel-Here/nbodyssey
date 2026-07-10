#!/usr/bin/env python3
"""CPU reference N-body integrator.

A direct O(n^2) softened-gravity leapfrog integrator in NumPy. It is not the
performance path -- the GPU code is -- but it pins down the physics: the same
force law, the same kick-drift-kick scheme, and the same energy diagnostic the
CUDA kernels implement. Two uses:

  1. Correctness oracle. Run a small system and confirm total energy stays
     bounded; the GPU version is later checked against the same energy behavior
     and, on identical initial conditions, the same trajectory to float
     tolerance.
  2. Output without a GPU. Produces frame dumps and an energy log in the exact
     on-disk formats the GPU path uses (docs/FORMATS.md), so the renderer and
     the whole downstream pipeline can be built and verified before touching
     CUDA hardware.

Force per particle (G=1 in code units, eps the softening length):
    a_i = sum_j m_j (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)
Energy:
    KE = sum_i 1/2 m_i |v_i|^2
    PE = -sum_{i<j} m_i m_j / sqrt(|r_i - r_j|^2 + eps^2)
"""
import argparse
import os
import time
import numpy as np


def read_ic(path):
    with open(path, "rb") as f:
        n = int(np.frombuffer(f.read(4), "<i4")[0])
        pos = np.frombuffer(f.read(n * 16), "<f4").reshape(n, 4).copy()
        vel = np.frombuffer(f.read(n * 16), "<f4").reshape(n, 4).copy()
    return pos, vel


def write_frame(path, pos4):
    n = pos4.shape[0]
    with open(path, "wb") as f:
        f.write(np.int32(n).astype("<i4").tobytes())
        pos4.astype("<f4").tofile(f)


def accelerations(pos, mass, eps2, G, chunk, dtype):
    """Softened gravitational acceleration on every particle.

    Chunked over the target particles so peak memory is (chunk x n x 3) rather
    than (n x n x 3), which lets the reference handle tens of thousands of
    particles without exhausting RAM.
    """
    n = pos.shape[0]
    acc = np.zeros((n, 3), dtype=dtype)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        d = pos[np.newaxis, :, :] - pos[lo:hi, np.newaxis, :]   # (b, n, 3)
        inv_r = np.sqrt(np.sum(d * d, axis=2) + eps2)           # (b, n)
        inv_r = dtype(1.0) / (inv_r * inv_r * inv_r)            # 1/(r^2+eps^2)^1.5
        acc[lo:hi] = np.einsum("bn,bnk->bk", mass[np.newaxis, :] * inv_r, d,
                               dtype=dtype)
    return (G * acc).astype(dtype)


def total_energy(pos, vel, mass, eps2, G, chunk):
    """Total energy in float64 for a clean readout regardless of sim precision."""
    p = pos.astype(np.float64)
    v = vel.astype(np.float64)
    m = mass.astype(np.float64)
    ke = 0.5 * np.sum(m * np.sum(v * v, axis=1))

    n = p.shape[0]
    pe = 0.0
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        d = p[np.newaxis, :, :] - p[lo:hi, np.newaxis, :]
        inv = 1.0 / np.sqrt(np.sum(d * d, axis=2) + eps2)   # (b, n)
        # full double sum for this row block, self terms removed below
        pe += np.sum((m[lo:hi, np.newaxis] * m[np.newaxis, :]) * inv)
    self_term = np.sum(m * m) / np.sqrt(eps2)
    pe = -G * 0.5 * (pe - self_term)
    return ke + pe, ke, pe


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ic", required=True, help="initial-conditions binary")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--eps", type=float, default=0.05, help="softening length")
    ap.add_argument("--G", type=float, default=1.0)
    ap.add_argument("--dump-every", type=int, default=5)
    ap.add_argument("--out", default="frames", help="frame output directory")
    ap.add_argument("--energy-log", default="benchmarks/energy_reference.csv")
    ap.add_argument("--chunk", type=int, default=2048,
                    help="row-block size for the O(n^2) passes")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                    help="integration precision (float32 mirrors the GPU path)")
    args = ap.parse_args()

    dtype = np.float32 if args.dtype == "float32" else np.float64
    eps2 = dtype(args.eps * args.eps)
    G = dtype(args.G)
    dt = dtype(args.dt)

    pos4, vel4 = read_ic(args.ic)
    pos = pos4[:, :3].astype(dtype)
    mass = pos4[:, 3].astype(dtype)
    vel = vel4[:, :3].astype(dtype)
    n = pos.shape[0]

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.energy_log) or ".", exist_ok=True)

    e0, ke0, pe0 = total_energy(pos, vel, mass, eps2, G, args.chunk)
    print(f"n={n} dt={args.dt} eps={args.eps} dtype={args.dtype}")
    print(f"E0={e0:.6e}  KE0={ke0:.6e}  PE0={pe0:.6e}")

    elog = open(args.energy_log, "w")
    elog.write("step,time,energy,kinetic,potential,rel_error\n")

    def log_energy(step):
        e, ke, pe = total_energy(pos, vel, mass, eps2, G, args.chunk)
        rel = (e - e0) / abs(e0)
        elog.write(f"{step},{step * args.dt:.5f},{e:.8e},{ke:.8e},{pe:.8e},{rel:.3e}\n")
        elog.flush()
        return rel

    frame = 0
    acc = accelerations(pos, mass, eps2, G, args.chunk, dtype)   # a(x0)
    t_start = time.time()
    for step in range(args.steps):
        vel += dtype(0.5) * dt * acc            # kick
        pos += dt * vel                          # drift
        acc = accelerations(pos, mass, eps2, G, args.chunk, dtype)  # a(x_new)
        vel += dtype(0.5) * dt * acc            # kick

        if step % args.dump_every == 0:
            out4 = np.column_stack([pos, mass]).astype("<f4")
            write_frame(os.path.join(args.out, f"frame_{frame:05d}.bin"), out4)
            frame += 1
            log_energy(step)

    rel = log_energy(args.steps)
    elog.close()
    dt_wall = time.time() - t_start
    print(f"done: {args.steps} steps, {frame} frames in {dt_wall:.1f}s "
          f"({1e3 * dt_wall / args.steps:.1f} ms/step)")
    print(f"final relative energy error: {rel:.3e}")


if __name__ == "__main__":
    main()
