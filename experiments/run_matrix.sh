#!/usr/bin/env bash
# Run the v2 experiment matrix (Tic-Tac-Toe, CPU).
# Usage: bash experiments/run_matrix.sh
# Env: GAMES=5000 MAXJOBS=4
set -uo pipefail
cd "$(dirname "$0")/.."

export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

GAMES=${GAMES:-5000}
MAXJOBS=${MAXJOBS:-4}
mkdir -p runs

run_one() {
  local cfg=$1 seed=$2 tb=$3 noise=$4 tag=$5
  local dir="runs/$tag"
  if [[ -f "$dir/res.json" || -f "$dir/ablation_${cfg}.pth" ]]; then
    echo "[skip] $tag (ya existe)"
    return 0
  fi
  mkdir -p "$dir"
  echo "[start] $tag"
  timeout 3000 python -u src/ablation_ttt.py \
    --games "$GAMES" --configs "$cfg" --seed "$seed" \
    --tiebreak "$tb" --mentor-noise "$noise" \
    --logs-dir "$dir" --results-file "$dir/res.json" \
    > "$dir/train.log" 2>&1
  local rc=$?
  echo "[done rc=$rc] $tag"
  return $rc
}

pids=()
launch() {
  run_one "$@" &
  pids+=($!)
  while (( ${#pids[@]} >= MAXJOBS )); do
    wait "${pids[0]}" || true
    pids=("${pids[@]:1}")
  done
}

# Clean 2 x 3 design: architecture x {fixed, TBR} x seed.
for seed in 42 43 44; do
  launch B "$seed" first 0.05 "fixed_B_s${seed}"
done
for seed in 42 43 44; do
  launch B "$seed" random 0.05 "tbr_B_s${seed}"
done
for seed in 42 43 44; do
  launch C "$seed" first 0.05 "fixed_C_s${seed}"
done
for seed in 42 43 44; do
  launch C "$seed" random 0.05 "tbr_C_s${seed}"
done

# Control: extra exploration without tie-break randomization.
for seed in 42 43; do
  launch B "$seed" first 0.20 "noise_B_s${seed}"
done
for seed in 42 43; do
  launch C "$seed" first 0.20 "noise_C_s${seed}"
done

# Teacher-side interventions: mixed per-decision ties, episode-coherent ties.
for seed in 42 43 44; do
  launch C "$seed" mixed 0.05 "mixed_C_s${seed}"
done
for seed in 42 43 44; do
  launch C "$seed" episode 0.05 "episode_C_s${seed}"
done
launch B 42 episode 0.05 "episode_B_s42"

wait
echo "Matriz completa."
