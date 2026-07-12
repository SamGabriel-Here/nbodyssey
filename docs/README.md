# Docs

- [PERFORMANCE.md](PERFORMANCE.md) — the measured story: naive vs Barnes-Hut vs
  the warp-cooperative traversal on a Tesla T4, phase breakdowns, the theta
  accuracy/cost dial, and energy-conservation trade-offs
- [FORMATS.md](FORMATS.md) — the on-disk binary layouts (initial conditions and
  frame dumps)
- Figures referenced by the README, all regenerable from committed scripts:
  `collision.gif` and `stages.png` (renderer), `bh_accuracy.png`
  (`scripts/barnes_hut_reference.py`), `benchmark_t4.png`
  (`scripts/plot_benchmarks.py`)
