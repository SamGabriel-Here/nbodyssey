# Benchmarks

Measured on a Tesla T4 (Colab, CUDA 12.8) by `scripts/gpu_bench.sh`; the
figures in `docs/` are generated from these files by
`scripts/plot_benchmarks.py`.

- `gpu_results.csv` — force-computation ms/step across particle counts for the
  naive kernel and both Barnes-Hut tree walks (per-thread and warp-cooperative)
- `energy_gpu_naive.csv`, `energy_gpu_bh_thread.csv`, `energy_gpu_bh_warp.csv`
  — total system energy vs simulation time over a 12k-particle, 1500-step
  collision on the GPU, one log per force variant
