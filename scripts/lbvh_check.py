#!/usr/bin/env python3
"""Logic test for the GPU Barnes-Hut tree, runnable without a GPU.

Mirrors src/forces_bh.cu step for step -- the same 21-bit Morton spreading, the
same Karras delta/range/split arithmetic including the sorted-index tie-break
for duplicate keys, the same bottom-up center-of-mass and AABB combination, and
the same stack traversal with the s/d < theta acceptance test on the longest
AABB edge. CI compiles the CUDA but cannot execute it; this executes the same
logic in Python so a bug in the tree construction fails the build rather than
waiting for GPU hardware.

Checks:
  1. The radix tree is well formed: every node has exactly one parent, and a
     walk from the root reaches every leaf exactly once.
  2. theta = 0 never accepts a node, so the traversal must equal the direct
     O(n^2) softened force to round-off.
  3. theta = 0.5 stays within a sane error band while using far fewer
     interactions than n per particle.
  4. Duplicate positions (identical Morton keys) still produce a valid tree
     via the index tie-break, and the traversal stack stays far below the
     kBhStack limit hard-coded in the CUDA.

Exits nonzero on any failure. Runs in seconds; wired into CI next to the
compile job.
"""
import sys
import numpy as np

EPS = 0.05
STACK_LIMIT = 128   # kBhStack in forces_bh.cu


# --- mirrors of the device functions ---

def spread21(v):
    v &= 0x1fffff
    v = (v | v << 32) & 0x1f00000000ffff
    v = (v | v << 16) & 0x1f0000ff0000ff
    v = (v | v << 8) & 0x100f00f00f00f00f
    v = (v | v << 4) & 0x10c30c30c30c30c3
    v = (v | v << 2) & 0x1249249249249249
    return v


def morton_keys(pos):
    lo = pos.min(axis=0)
    ext = float((pos.max(axis=0) - lo).max())
    inv = 1.0 / ext if ext > 0 else 0.0
    keys = []
    for p in pos:
        q = [min(max(int((p[a] - lo[a]) * inv * 2097152.0), 0), 2097151)
             for a in range(3)]
        keys.append((spread21(q[0]) << 2) | (spread21(q[1]) << 1) | spread21(q[2]))
    return keys


def build_tree(keys):
    """Karras radix-tree build, one iteration of the loop per internal node."""
    n = len(keys)

    def delta(i, j):
        if j < 0 or j >= n:
            return -1
        x = keys[i] ^ keys[j]
        if x:
            return 64 - x.bit_length()
        return 64 + (32 - (i ^ j).bit_length())

    children = [None] * (n - 1)
    parent = [-1] * (2 * n - 1)
    for i in range(n - 1):
        d = 1 if delta(i, i + 1) - delta(i, i - 1) >= 0 else -1
        dmin = delta(i, i - d)
        lmax = 2
        while delta(i, i + lmax * d) > dmin:
            lmax <<= 1
        l = 0
        t = lmax >> 1
        while t > 0:
            if delta(i, i + (l + t) * d) > dmin:
                l += t
            t >>= 1
        j = i + l * d

        first, last = min(i, j), max(i, j)
        common = delta(first, last)
        split, step = first, last - first
        while True:
            step = (step + 1) >> 1
            if split + step < last and delta(first, split + step) > common:
                split += step
            if step <= 1:
                break

        left = (n - 1 + split) if split == first else split
        right = (n - 1 + split + 1) if split + 1 == last else split + 1
        children[i] = (left, right)
        parent[left] = i
        parent[right] = i
    return children, parent


def check_topology(children, parent, n):
    assert parent[0] == -1, "root must have no parent"
    for node in range(1, 2 * n - 1):
        assert parent[node] >= 0, f"node {node} was never attached"
    seen_leaves = 0
    stack, visited = [0], set()
    while stack:
        id_ = stack.pop()
        assert id_ not in visited, f"node {id_} reached twice"
        visited.add(id_)
        if id_ >= n - 1:
            seen_leaves += 1
        else:
            stack.extend(children[id_])
    assert seen_leaves == n, f"walk found {seen_leaves} of {n} leaves"
    assert len(visited) == 2 * n - 1, "walk missed internal nodes"


def compute_com(children, pos_sorted, mass_sorted, n):
    """Post-order pass: COM, AABB, and longest-edge size per internal node."""
    com = np.zeros((n - 1, 4))
    box_lo = np.zeros((n - 1, 3))
    box_hi = np.zeros((n - 1, 3))

    def node_data(id_):
        if id_ >= n - 1:
            i = id_ - (n - 1)
            p = pos_sorted[i]
            return np.append(p, mass_sorted[i]), p.copy(), p.copy()
        return com[id_], box_lo[id_], box_hi[id_]

    order, stack = [], [0]
    while stack:
        id_ = stack.pop()
        if id_ < n - 1:
            order.append(id_)
            stack.extend(children[id_])
    for id_ in reversed(order):
        (cl, ll, hl), (cr, lr, hr) = (node_data(c) for c in children[id_])
        m = cl[3] + cr[3]
        com[id_, :3] = (cl[:3] * cl[3] + cr[:3] * cr[3]) / m
        com[id_, 3] = m
        box_lo[id_] = np.minimum(ll, lr)
        box_hi[id_] = np.maximum(hl, hr)
    size = (box_hi - box_lo).max(axis=1)
    return com, size


def traverse(t, pos_sorted, mass_sorted, children, com, size, n, theta, eps2):
    """Stack traversal identical to traverse_kernel; returns acc, stats."""
    bi = pos_sorted[t]
    acc = np.zeros(3)
    interactions = 0
    max_depth = 0
    stack = [0]
    while stack:
        max_depth = max(max_depth, len(stack))
        id_ = stack.pop()
        if id_ >= n - 1:
            i = id_ - (n - 1)
            src, sm = pos_sorted[i], mass_sorted[i]
        else:
            c = com[id_]
            d2 = float(np.sum((c[:3] - bi) ** 2))
            s = size[id_]
            if s * s < theta * theta * d2:
                src, sm = c[:3], c[3]
            else:
                stack.extend(children[id_])
                continue
        d = src - bi
        inv = (float(np.sum(d * d)) + eps2) ** -1.5
        acc += sm * inv * d
        interactions += 1
    return acc, interactions, max_depth


def traverse_warp(base, pos_sorted, mass_sorted, children, com, size, n,
                  theta, eps2):
    """Warp-cooperative walk over lanes base..base+31, as traverse_warp_kernel:
    one shared stack, a node opens if any live lane needs it, otherwise every
    lane accepts it. Returns per-lane accs plus interaction count and depth."""
    lanes = [base + l for l in range(32) if base + l < n]
    bis = pos_sorted[lanes]
    accs = np.zeros((len(lanes), 3))
    interactions = 0
    max_depth = 0
    stack = [0]
    while stack:
        max_depth = max(max_depth, len(stack))
        id_ = stack.pop()
        if id_ >= n - 1:
            i = id_ - (n - 1)
            src, sm = pos_sorted[i], mass_sorted[i]
        else:
            c = com[id_]
            d2 = np.sum((c[:3] - bis) ** 2, axis=1)
            s = size[id_]
            if np.any(s * s >= theta * theta * d2):   # any lane votes to open
                stack.extend(children[id_])
                continue
            src, sm = c[:3], c[3]
        d = src - bis
        inv = (np.sum(d * d, axis=1) + eps2) ** -1.5
        accs += (sm * inv)[:, None] * d
        interactions += 1
    return accs, lanes, interactions, max_depth


def exact_accel(targets, pos, mass, eps2):
    d = pos[np.newaxis, :, :] - pos[targets][:, np.newaxis, :]
    inv = (np.sum(d * d, axis=2) + eps2) ** -1.5
    np.put_along_axis(inv, np.asarray(targets)[:, None], 0.0, axis=1)
    return np.einsum("tn,tnk->tk", mass[np.newaxis, :] * inv, d)


def main():
    rng = np.random.default_rng(1)
    # two offset gaussian blobs, plus a clump of exactly coincident particles
    # to force duplicate Morton keys through the tie-break path
    a = rng.normal([-2.0, 0.0, 0.0], 0.7, size=(1500, 3))
    b = rng.normal([2.0, 0.5, 0.1], 0.7, size=(1500, 3))
    dupes = np.tile([[0.123, -0.456, 0.789]], (24, 1))
    pos = np.concatenate([a, b, dupes])
    n = pos.shape[0]
    mass = np.full(n, 1.0 / n)
    eps2 = EPS * EPS

    keys = morton_keys(pos)
    order = sorted(range(n), key=lambda i: keys[i])
    keys_sorted = [keys[i] for i in order]
    pos_sorted = pos[order]
    mass_sorted = mass[order]

    dup_keys = n - len(set(keys_sorted))
    print(f"n={n}, duplicate morton keys={dup_keys}")
    assert dup_keys >= 23, "dupe clump should collide in key space"

    children, parent = build_tree(keys_sorted)
    check_topology(children, parent, n)
    print("topology: every leaf reached exactly once, parents consistent")

    com, size = compute_com(children, pos_sorted, mass_sorted, n)
    root_m = com[0, 3]
    assert abs(root_m - mass.sum()) < 1e-12, f"root mass {root_m} != total"
    print(f"root mass matches total ({root_m:.6f})")

    sample = rng.choice(n, size=192, replace=False)
    ref = exact_accel(sample, pos_sorted, mass_sorted, eps2)
    ref_mag = np.linalg.norm(ref, axis=1)

    worst_depth = 0
    max_err0 = 0.0
    for k, t in enumerate(sample):
        acc, ni, depth = traverse(int(t), pos_sorted, mass_sorted, children,
                                  com, size, n, 0.0, eps2)
        worst_depth = max(worst_depth, depth)
        assert ni == n, f"theta=0 must touch every leaf, got {ni}"
        max_err0 = max(max_err0, np.linalg.norm(acc - ref[k]) / ref_mag[k])
    assert max_err0 < 1e-12, f"theta=0 disagrees with exact force: {max_err0:.2e}"
    print(f"theta=0.0: max rel err {max_err0:.2e} (exact), "
          f"max stack depth {worst_depth}")

    errs, inters = [], []
    for k, t in enumerate(sample):
        acc, ni, depth = traverse(int(t), pos_sorted, mass_sorted, children,
                                  com, size, n, 0.5, eps2)
        worst_depth = max(worst_depth, depth)
        errs.append(np.linalg.norm(acc - ref[k]) / ref_mag[k])
        inters.append(ni)
    med = float(np.median(errs))
    mean_i = float(np.mean(inters))
    assert med < 0.05, f"theta=0.5 median error {med:.3f} out of band"
    assert mean_i < 0.5 * n, "theta=0.5 should prune most interactions"
    print(f"theta=0.5: median rel err {med:.3e}, "
          f"{mean_i:.0f} interactions/particle (vs {n - 1} exact), "
          f"max stack depth {worst_depth}")

    assert worst_depth < STACK_LIMIT, \
        f"stack depth {worst_depth} too close to kBhStack={STACK_LIMIT}"
    print(f"stack depth bounded well under kBhStack={STACK_LIMIT}")

    # warp-cooperative walk: same shared-path voting as traverse_warp_kernel
    warp_bases = [int(b) * 32 for b in rng.choice(n // 32, size=8,
                                                  replace=False)]
    warp_err0 = 0.0
    warp_errs, warp_inters, warp_depth = [], [], 0
    for base in warp_bases:
        accs, lanes, ni, depth = traverse_warp(base, pos_sorted, mass_sorted,
                                               children, com, size, n, 0.0,
                                               eps2)
        assert ni == n, f"warp theta=0 must touch every leaf, got {ni}"
        ref_w = exact_accel(lanes, pos_sorted, mass_sorted, eps2)
        rel = (np.linalg.norm(accs - ref_w, axis=1) /
               np.linalg.norm(ref_w, axis=1))
        warp_err0 = max(warp_err0, float(rel.max()))

        accs, lanes, ni, depth = traverse_warp(base, pos_sorted, mass_sorted,
                                               children, com, size, n, 0.5,
                                               eps2)
        warp_depth = max(warp_depth, depth)
        rel = (np.linalg.norm(accs - ref_w, axis=1) /
               np.linalg.norm(ref_w, axis=1))
        warp_errs.extend(rel.tolist())
        warp_inters.append(ni)
    assert warp_err0 < 1e-12, f"warp theta=0 disagrees: {warp_err0:.2e}"
    med_w = float(np.median(warp_errs))
    assert med_w < 0.05, f"warp theta=0.5 median error {med_w:.3f} out of band"
    assert warp_depth < STACK_LIMIT
    amp = float(np.mean(warp_inters)) / mean_i
    print(f"warp walk: theta=0 exact ({warp_err0:.2e}); theta=0.5 median rel "
          f"err {med_w:.3e} vs {med:.3e} per-thread, work amplification "
          f"{amp:.2f}x, max stack depth {warp_depth}")
    print("lbvh logic: all checks passed")


if __name__ == "__main__":
    sys.exit(main())
