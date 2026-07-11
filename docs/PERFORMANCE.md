# Naive all-pairs vs Barnes-Hut on the GPU

Measurements from a Tesla T4 (sm_75, CUDA 12.8), driven by `scripts/gpu_bench.sh`.
Timing is CUDA events around the whole force computation — for Barnes-Hut that
includes rebuilding the tree from scratch every step, which is the fair unit to
compare against the naive kernel. Values are averages over 11 calls; raw logs
are in `benchmarks/logs/`-style output captured in `benchmarks/gpu_results.csv`.

![benchmark](benchmark_t4.png)

## Correctness first

Before timing anything, `--compare-forces` ran both modules on the same 20k
state. At `theta = 0` the tree never accepts a node and must reproduce the
naive result up to float summation order: measured mean relative difference
1.7e-5 (max 2.8e-4 on near-cancelling forces), which is single-precision
round-off, not algorithmic error. The same gate passes at 2e-15 in the
double-precision CPU mirrors, so the residual here is float32, not the tree.

## Time per step, theta = 0.5

| n | naive | Barnes-Hut | speedup |
|---:|---:|---:|---:|
| 4,096 | 0.21 ms | 0.41 ms | 0.5x |
| 12,000 | 0.95 ms | 0.53 ms | 1.8x |
| 50,000 | 11.6 ms | 1.96 ms | 5.9x |
| 100,000 | 48.9 ms | 4.62 ms | 10.6x |
| 200,000 | 195.7 ms | 12.0 ms | 16.3x |
| 400,000 | 786.9 ms | 30.1 ms | 26.1x |
| 1,000,000 | 5,269.9 ms | 111.6 ms | **47.2x** |

Two honest observations frame the result:

**The naive kernel is a strong baseline, not a strawman.** The tiled all-pairs
kernel sustains ~204 Ginteractions/s, about 4.1 TFLOP/s of useful work at 20
flops per pair — roughly half the T4's fp32 peak, respectable for a kernel that
is `rsqrtf`-bound. Below n of about 8,000 it simply wins: the tree costs ~0.25 ms
of fixed pipeline overhead per step and its traversal diverges, while the dense
kernel is perfectly coalesced. Asymptotics do not pay rent at small n.

**The crossover is early and the gap compounds.** From 12k particles on,
Barnes-Hut leads, and every doubling of n roughly doubles its advantage, as an
O(n log n) versus O(n^2) gap should. At one million particles the tree computes
forces in 112 ms — the naive kernel needs 5.3 seconds. Put differently: at
interactive-adjacent rates, the tree simulates a million-body galaxy merger at
~9 steps/s where the naive kernel manages one step every five seconds.

## Where the time goes

Per-call phase breakdown of the tree pipeline (n = 1M, theta = 0.5):

| phase | ms | share |
|---|---:|---:|
| bounding box | 0.08 | 0.1% |
| Morton codes | 0.11 | 0.1% |
| radix sort (CUB) | 1.44 | 1.3% |
| Karras tree build | 0.63 | 0.6% |
| centers of mass | 1.03 | 0.9% |
| **traversal** | **107.9** | **97.0%** |

Rebuilding the LBVH from scratch — sort, hierarchy, centers of mass — costs
about 3 ms for a million particles. Construction is effectively free; the
entire cost of Barnes-Hut is walking the tree. That is the expected profile
for a per-thread stack traversal, whose warp lanes diverge whenever neighboring
particles open different nodes, and it makes the next optimization unambiguous:
a warp-cooperative traversal in which the 32 lanes of a warp walk one shared
path (`__any_sync` voting on descent), converting divergent scalar loads into
coalesced ones. The Morton-sorted particle order already gives adjacent threads
nearly identical paths, which is exactly the locality that version exploits.

## The accuracy dial

theta trades force error for time (n = 100k):

| theta | ms/step | median force error (CPU oracle) |
|---:|---:|---:|
| 0.3 | 13.9 | ~0.3% |
| 0.5 | 4.6 | ~1% |
| 0.8 | 2.4 | ~3% |

Energy over a full 12k-particle, 1500-step collision on the GPU: the naive run
stays within 1e-4 of the initial energy, matching the CPU reference bit-for-bit
in character; the Barnes-Hut run stays bounded within about 5e-3 with no
secular drift — the leapfrog integrator remains symplectic, and the tree's
approximation error shows up as slightly noisier, but not growing, energy.

## Reproducing

```
bash scripts/gpu_bench.sh          # any CUDA machine; a free Colab T4 works
python scripts/plot_benchmarks.py  # regenerates docs/benchmark_t4.png
```
