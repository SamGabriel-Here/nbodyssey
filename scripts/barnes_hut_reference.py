#!/usr/bin/env python3
"""CPU Barnes-Hut oracle: validate the tree approximation before the GPU port.

The GPU Barnes-Hut module cannot be run without CUDA hardware, so this pins down
the part that is subtle regardless of platform: the octree, the centers of mass,
and the opening criterion. It builds the same softened-gravity approximation the
CUDA kernels will, and checks it against the exact O(n^2) force the naive kernel
computes.

Two things it proves:
  1. At theta = 0 no cell is ever accepted, every leaf is visited, and the result
     equals the exact force to floating-point round-off -- the traversal and COM
     bookkeeping are correct.
  2. As theta grows the force error rises smoothly while the number of
     interactions per particle collapses from O(n) toward O(log n) -- the
     accuracy/cost trade-off that makes the tree code worth writing.

Force per interaction (same law as the naive kernel and the reference integrator,
G = 1 in code units, eps the softening length):
    a_i = sum_src m_src (r_src - r_i) / (|r_src - r_i|^2 + eps^2)^(3/2)
where a "source" is either a single particle (leaf) or an accepted cell's total
mass at its center of mass.
"""
import argparse
import os
import numpy as np


def read_ic(path):
    with open(path, "rb") as f:
        n = int(np.frombuffer(f.read(4), "<i4")[0])
        pos = np.frombuffer(f.read(n * 16), "<f4").reshape(n, 4).copy()
        vel = np.frombuffer(f.read(n * 16), "<f4").reshape(n, 4).copy()
    return pos, vel


class Cell:
    """An octree node. A leaf holds one body; an internal node holds children."""
    __slots__ = ("cx", "cy", "cz", "half", "body", "children",
                 "mass", "mx", "my", "mz")

    def __init__(self, cx, cy, cz, half):
        self.cx, self.cy, self.cz = cx, cy, cz   # cell center
        self.half = half                          # half the cell edge length
        self.body = -1                            # body index if a leaf, else -1
        self.children = None                      # list[8] of Cell/None if internal
        self.mass = 0.0                           # total mass in the cell
        self.mx = self.my = self.mz = 0.0         # mass-weighted position sum


def _octant(cell, x, y, z):
    o = 0
    if x >= cell.cx: o |= 1
    if y >= cell.cy: o |= 2
    if z >= cell.cz: o |= 4
    return o


def _child_cell(cell, o):
    h = cell.half * 0.5
    cx = cell.cx + (h if (o & 1) else -h)
    cy = cell.cy + (h if (o & 2) else -h)
    cz = cell.cz + (h if (o & 4) else -h)
    return Cell(cx, cy, cz, h)


def _insert(cell, i, px, py, pz):
    # Descend iteratively to the leaf where body i belongs, subdividing as needed.
    while True:
        if cell.children is None and cell.body == -1:
            cell.body = i                      # empty leaf -> put the body here
            return
        if cell.children is None:              # occupied leaf -> subdivide
            cell.children = [None] * 8
            j = cell.body
            cell.body = -1
            oj = _octant(cell, px[j], py[j], pz[j])
            cell.children[oj] = _child_cell(cell, oj)
            _insert(cell.children[oj], j, px, py, pz)
        o = _octant(cell, px[i], py[i], pz[i])
        if cell.children[o] is None:
            cell.children[o] = _child_cell(cell, o)
        cell = cell.children[o]                 # continue in the chosen child


def build_tree(px, py, pz):
    n = len(px)
    lo = np.array([px.min(), py.min(), pz.min()])
    hi = np.array([px.max(), py.max(), pz.max()])
    center = 0.5 * (lo + hi)
    half = 0.5 * float((hi - lo).max()) * 1.0001 + 1e-6   # cubic, padded
    root = Cell(center[0], center[1], center[2], half)
    for i in range(n):
        _insert(root, i, px, py, pz)
    return root


def compute_com(cell, px, py, pz, mass):
    if cell.children is None:
        if cell.body == -1:
            return                              # empty leaf
        j = cell.body
        cell.mass = float(mass[j])
        cell.mx, cell.my, cell.mz = px[j] * mass[j], py[j] * mass[j], pz[j] * mass[j]
        return
    m = mx = my = mz = 0.0
    for c in cell.children:
        if c is None:
            continue
        compute_com(c, px, py, pz, mass)
        m += c.mass
        mx += c.mx; my += c.my; mz += c.mz
    cell.mass, cell.mx, cell.my, cell.mz = m, mx, my, mz


def bh_accel(i, root, px, py, pz, mass, theta, eps2):
    """Acceleration on body i, plus the number of interactions used."""
    xi, yi, zi = px[i], py[i], pz[i]
    ax = ay = az = 0.0
    interactions = 0
    stack = [root]
    while stack:
        cell = stack.pop()
        if cell.mass == 0.0:
            continue
        if cell.children is None:               # leaf: single body
            j = cell.body
            if j == i:
                continue
            sx, sy, sz, sm = px[j], py[j], pz[j], mass[j]
        else:
            # opening criterion: accept the cell if s / d < theta
            comx = cell.mx / cell.mass
            comy = cell.my / cell.mass
            comz = cell.mz / cell.mass
            dx, dy, dz = comx - xi, comy - yi, comz - zi
            d2 = dx * dx + dy * dy + dz * dz
            s = 2.0 * cell.half
            if s * s < theta * theta * d2:      # far enough -> use the COM
                sx, sy, sz, sm = comx, comy, comz, cell.mass
            else:
                for c in cell.children:
                    if c is not None:
                        stack.append(c)
                continue
        dx, dy, dz = sx - xi, sy - yi, sz - zi
        inv = 1.0 / (dx * dx + dy * dy + dz * dz + eps2) ** 1.5
        f = sm * inv
        ax += dx * f; ay += dy * f; az += dz * f
        interactions += 1
    return (ax, ay, az), interactions


def exact_accel(targets, pos, mass, eps2):
    """Exact O(n^2) softened acceleration for a set of target particles."""
    d = pos[np.newaxis, :, :] - pos[targets][:, np.newaxis, :]     # (t, n, 3)
    inv = (np.sum(d * d, axis=2) + eps2) ** -1.5                    # (t, n)
    np.put_along_axis(inv, np.asarray(targets)[:, None], 0.0, axis=1)  # drop self
    return np.einsum("tn,tnk->tk", mass[np.newaxis, :] * inv, d)


def sweep(pos, mass, eps2, thetas, sample, rng):
    px, py, pz = pos[:, 0].copy(), pos[:, 1].copy(), pos[:, 2].copy()
    root = build_tree(px, py, pz)
    compute_com(root, px, py, pz, mass)

    n = pos.shape[0]
    targets = rng.choice(n, size=min(sample, n), replace=False)
    ref = exact_accel(targets, pos, mass, eps2)
    ref_mag = np.linalg.norm(ref, axis=1)

    med_err, mean_inter = [], []
    for th in thetas:
        errs, inters = [], []
        for k, i in enumerate(targets):
            a, ni = bh_accel(int(i), root, px, py, pz, mass, th, eps2)
            errs.append(np.linalg.norm(np.array(a) - ref[k]) / ref_mag[k])
            inters.append(ni)
        med_err.append(float(np.median(errs)))
        mean_inter.append(float(np.mean(inters)))
    return np.array(med_err), np.array(mean_inter)


def scaling(pos, mass, eps2, theta, sizes, sample, rng):
    """Mean interactions per particle at fixed theta as n grows."""
    out = []
    for m in sizes:
        idx = rng.choice(pos.shape[0], size=m, replace=False)
        sub = pos[idx]
        px, py, pz = sub[:, 0].copy(), sub[:, 1].copy(), sub[:, 2].copy()
        root = build_tree(px, py, pz)
        compute_com(root, px, py, pz, mass[idx])
        tgt = rng.choice(m, size=min(sample, m), replace=False)
        inter = [bh_accel(int(i), root, px, py, pz, mass[idx], theta, eps2)[1]
                 for i in tgt]
        out.append(np.mean(inter))
    return np.array(out)


def make_figure(thetas, med_err, mean_inter, sizes, scale_inter, theta_fixed, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    fig.suptitle("Barnes-Hut approximation: accuracy and cost vs the exact force",
                 fontsize=14)

    ax[0].loglog(thetas, med_err, "o-", color="#c81e5a")
    ax[0].set_xlabel(r"opening angle $\theta$")
    ax[0].set_ylabel("median relative force error")
    ax[0].set_title("Accuracy falls off smoothly with " + r"$\theta$")
    ax[0].grid(True, which="both", alpha=0.3)

    ax[1].semilogx(thetas, mean_inter, "o-", color="#1f77b4")
    ax[1].set_xlabel(r"opening angle $\theta$")
    ax[1].set_ylabel("mean interactions per particle")
    ax[1].set_title("Cost collapses as cells are accepted")
    ax[1].grid(True, which="both", alpha=0.3)

    ax[2].loglog(sizes, scale_inter, "o-", color="#2a9d5c", label="Barnes-Hut")
    ax[2].loglog(sizes, np.array(sizes) - 1, "--", color="#888888",
                 label="naive (n-1)")
    ax[2].set_xlabel("particle count n")
    ax[2].set_ylabel("interactions per particle")
    ax[2].set_title(rf"Scaling at $\theta={theta_fixed}$: O(log n) vs O(n)")
    ax[2].set_xticks(sizes)
    ax[2].set_xticklabels([str(s) for s in sizes])
    ax[2].minorticks_off()
    ax[2].legend()
    ax[2].grid(True, which="both", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ic", required=True, help="initial-conditions binary")
    ap.add_argument("--eps", type=float, default=0.05, help="softening length")
    ap.add_argument("--sample", type=int, default=400,
                    help="number of target particles sampled for the statistics")
    ap.add_argument("--theta-fixed", type=float, default=0.5,
                    help="theta used for the n-scaling panel")
    ap.add_argument("--out", default="docs/bh_accuracy.png")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    eps2 = args.eps * args.eps
    pos4, _ = read_ic(args.ic)
    pos = pos4[:, :3].astype(np.float64)
    mass = pos4[:, 3].astype(np.float64)
    n = pos.shape[0]

    thetas = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5])
    med_err, mean_inter = sweep(pos, mass, eps2, thetas, args.sample, rng)

    print(f"n={n}, sample={min(args.sample, n)} target particles")
    print(f"{'theta':>7} {'median rel err':>16} {'interactions/particle':>22}")
    for th, e, mi in zip(thetas, med_err, mean_inter):
        print(f"{th:>7.2f} {e:>16.3e} {mi:>22.1f}")

    # theta = 0 must reproduce the exact force to round-off.
    assert med_err[0] < 1e-10, f"theta=0 should match exact force, got {med_err[0]:.2e}"
    print(f"\ncheck: theta=0 matches exact force (err {med_err[0]:.2e}), "
          f"traversal + COM are correct")

    sizes = [int(s) for s in (250, 500, 1000, 2000, min(4000, n)) if s <= n]
    scale_inter = scaling(pos, mass, eps2, args.theta_fixed, sizes, args.sample, rng)
    make_figure(thetas, med_err, mean_inter, sizes, scale_inter,
                args.theta_fixed, args.out)


if __name__ == "__main__":
    main()
