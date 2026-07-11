# Naive all-pairs vs Barnes-Hut on the GPU

Measurements from a Tesla T4 (sm_75, CUDA 12.8), driven by `scripts/gpu_bench.sh`.
Timing is CUDA events around the whole force computation — for Barnes-Hut that
includes rebuilding the tree from scratch every step, which is the fair unit to
compare against the naive kernel. Values are averages over 11 calls, recorded in
`benchmarks/gpu_results.csv`. Two independent T4 sessions reproduced the naive
numbers within a few percent.

![benchmark](benchmark_t4.png)

## Correctness first

Before timing anything, `--compare-forces` ran the tree against the naive kernel
on the same 20k state, once per tree walk. At `theta = 0` the tree never accepts
a node and must reproduce the naive result up to float summation order: both
walks measured a mean relative difference of 1.7e-5 (max 2.8e-4 on
near-cancelling forces), which is single-precision round-off, not algorithmic
error. The same gate passes at 2e-15 in the double-precision CPU mirrors, so the
residual here is float32, not the tree.

## Time per step, theta = 0.5

| n | naive | BH per-thread | BH warp | warp vs thread | warp vs naive |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 0.17 ms | 0.42 ms | 0.44 ms | 0.95x | — |
| 12,000 | 0.85 ms | 0.51 ms | 0.61 ms | 0.83x | 1.4x |
| 50,000 | 11.7 ms | 1.92 ms | 1.20 ms | 1.6x | 9.8x |
| 100,000 | 47.8 ms | 4.66 ms | 2.59 ms | 1.8x | 18.5x |
| 200,000 | 192.6 ms | 12.1 ms | 4.94 ms | 2.4x | 39x |
| 400,000 | 787.7 ms | 29.7 ms | 9.98 ms | 3.0x | 79x |
| 1,000,000 | 5,324.2 ms | 112.1 ms | 30.3 ms | **3.7x** | **176x** |

Three observations frame the result:

**The naive kernel is a strong baseline, not a strawman.** The tiled all-pairs
kernel sustains ~204 Ginteractions/s, about 4.1 TFLOP/s of useful work at 20
flops per pair — roughly half the T4's fp32 peak, respectable for a kernel that
is `rsqrtf`-bound. Below n of about 8,000 it simply wins: the tree costs ~0.25 ms
of fixed pipeline overhead per step and its traversal diverges, while the dense
kernel is perfectly coalesced. Asymptotics do not pay rent at small n.

**The crossover is early and the gap compounds.** From 12k particles on,
Barnes-Hut leads, and every doubling of n roughly doubles its advantage, as an
O(n log n) versus O(n^2) gap should. At one million particles the warp walk
computes forces in 30 ms — the naive kernel needs 5.3 seconds. The tree
simulates a million-body galaxy merger at ~33 steps/s where the naive kernel
manages one step every five seconds.

**Divergence was the bottleneck, and eliminating it beat doing less work.**
The warp-cooperative walk makes every lane in a warp follow one shared path,
opening a node whenever any lane votes to open it. The CPU mirror measures the
price: each lane performs about 1.9x more interactions than it strictly needs.
The GPU verdict is that uniformity wins anyway, by 3.7x at one million
particles — divergent execution was costing far more than double the useful
arithmetic. The win grows with n and with smaller theta (deeper walks diverge
more), and inverts below ~50k particles, where there are too few warps to hide
the amplified work.

## Where the time goes

Per-call phase breakdown of the tree pipeline (n = 1M, theta = 0.5):

| phase | per-thread | warp | share (warp) |
|---|---:|---:|---:|
| bounding box | 0.08 ms | 0.08 ms | 0.3% |
| Morton codes | 0.11 ms | 0.11 ms | 0.4% |
| radix sort (CUB) | 1.44 ms | 1.39 ms | 4.6% |
| Karras tree build | 0.63 ms | 0.61 ms | 2.0% |
| centers of mass | 1.02 ms | 0.99 ms | 3.3% |
| **traversal** | **108.6 ms** | **27.0 ms** | **89.5%** |

Rebuilding the LBVH from scratch — sort, hierarchy, centers of mass — costs
about 3 ms for a million particles; construction is effectively free either
way. The traversal itself sped up 4.0x when the per-thread walk's divergent
scalar loads became one uniform, coalesced access per warp — the
Morton-sorted particle order gives adjacent lanes nearly identical paths,
which is exactly the locality the shared walk exploits.

## The accuracy dial

theta trades force error for time (n = 100k):

| theta | per-thread | warp | median force error (CPU oracle, per-thread) |
|---:|---:|---:|---:|
| 0.3 | 13.9 ms | 5.2 ms | ~0.3% |
| 0.5 | 4.7 ms | 2.6 ms | ~1% |
| 0.8 | 2.2 ms | 1.6 ms | ~3% |

The warp walk's forced descents also cut its instantaneous force error roughly
in half at the same theta (on-device mean vs naive at theta 0.5: 8.6e-3 against
the per-thread walk's 1.4e-2).

Energy over a full 12k-particle, 1500-step collision on the GPU: the naive run
stays within 1e-4 of the initial energy, and the per-thread tree run stays
bounded within about 5e-3 with no secular trend. The warp run is the honest
caveat: its error grows to about 1.2e-2 by the end of the encounter. The likely
mechanism is time-consistency rather than accuracy — which nodes a particle
accepts depends on its warp's shared vote, and warp membership reshuffles with
every Morton re-sort, so the effective force field a particle feels changes
discontinuously between steps in a way the per-thread walk's does not. A
practical setting that keeps both properties: the warp walk at theta 0.3 costs
about the same as the per-thread walk at theta 0.5 and computes strictly more
accurate forces.

## Reproducing

```
bash scripts/gpu_bench.sh          # any CUDA machine; a free Colab T4 works
python scripts/plot_benchmarks.py  # regenerates docs/benchmark_t4.png
```
