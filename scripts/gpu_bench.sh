#!/usr/bin/env bash
# One-shot GPU session: build, correctness gate, then the naive-vs-Barnes-Hut
# benchmark sweep. Run from the repo root on a CUDA machine (a Colab T4 works).
# Results land in benchmarks/gpu_results.csv with full per-run logs alongside.
set -euo pipefail

if ! command -v nvidia-smi >/dev/null; then
  echo "no nvidia-smi -- this needs a CUDA machine" >&2
  exit 1
fi
ARCH=${ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')}
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "gpu: $GPU (sm_$ARCH)"

cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$ARCH"
cmake --build build -j

mkdir -p benchmarks/logs
scratch=$(mktemp -d)
csv=benchmarks/gpu_results.csv
echo "n,method,theta,ms_per_call" > "$csv"

run_case() {  # n method theta steps
  local n=$1 m=$2 th=$3 steps=$4
  local log="benchmarks/logs/${m}_n${n}_theta${th}.txt"
  ./build/galaxy_sim --ic "$scratch/ic_$n.bin" --steps "$steps" \
      --dump-every 1000000 --out "$scratch/frames" \
      --energy-log "$scratch/e.csv" --force "$m" --theta "$th" | tee "$log"
  local ms
  ms=$(grep 'ms/call' "$log" | awk '{print $(NF-1)}')
  echo "$n,$m,$th,$ms" >> "$csv"
}

# --- correctness gate: theta = 0 must reproduce the naive forces ------------
python3 scripts/generate_ic.py --particles 20000 --out "$scratch/ic_gate.bin"
./build/galaxy_sim --ic "$scratch/ic_gate.bin" --compare-forces --theta 0 \
    | tee benchmarks/logs/gate_theta0.txt
mean=$(grep 'relative force error' benchmarks/logs/gate_theta0.txt \
       | sed 's/.*mean //')
awk -v m="$mean" 'BEGIN { exit !(m < 1e-4) }' \
    || { echo "GATE FAILED: theta=0 mean error $mean" >&2; exit 1; }
echo "gate passed: theta=0 mean error $mean"
./build/galaxy_sim --ic "$scratch/ic_gate.bin" --compare-forces --theta 0.5 \
    | tee benchmarks/logs/gate_theta05.txt

# --- benchmark sweep: naive vs bh across n at theta = 0.5 -------------------
for n in 4096 12000 50000 100000 200000 400000 1000000; do
  python3 scripts/generate_ic.py --particles "$n" --out "$scratch/ic_$n.bin"
  run_case "$n" naive 0.5 10
  run_case "$n" bh    0.5 10
done

# theta sensitivity at one size, for the accuracy/cost trade-off plot
for th in 0.3 0.8; do
  run_case 100000 bh "$th" 10
done

# --- energy conservation on the GPU, both modules ---------------------------
python3 scripts/generate_ic.py --particles 12000 --out "$scratch/ic_e.bin"
./build/galaxy_sim --ic "$scratch/ic_e.bin" --steps 1500 --dump-every 100 \
    --out "$scratch/frames_e" --energy-log benchmarks/energy_gpu_naive.csv \
    | tee benchmarks/logs/energy_naive.txt
./build/galaxy_sim --ic "$scratch/ic_e.bin" --steps 1500 --dump-every 100 \
    --out "$scratch/frames_e" --energy-log benchmarks/energy_gpu_bh.csv \
    --force bh --theta 0.5 | tee benchmarks/logs/energy_bh.txt

rm -rf "$scratch"
echo
echo "=== $GPU (sm_$ARCH) ==="
cat "$csv"
echo
echo "final energy errors:"
tail -1 benchmarks/energy_gpu_naive.csv | awk -F, '{print "  naive: " $6}'
tail -1 benchmarks/energy_gpu_bh.csv | awk -F, '{print "  bh:    " $6}'
