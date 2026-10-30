#!/usr/bin/env bash
# Queue the remaining v2 matrix jobs. Safe to launch while other runs are
# active: a job is skipped if its directory already exists (in progress or
# finished). Caps concurrency by counting live ablation_ttt.py processes.
set -uo pipefail
cd "$(dirname "$0")/.."

export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

GAMES=${GAMES:-5000}
MAXJOBS=${MAXJOBS:-4}

count_running() {
  powershell -NoProfile -Command \
    "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*ablation_ttt.py*' }).Count" \
    2>/dev/null | tr -d '\r' | tail -1
}

launch() {
  local cfg=$1 seed=$2 tb=$3 noise=$4 tag=$5
  local dir="runs/$tag"
  if [[ -d "$dir" ]]; then
    echo "[skip] $tag (dir exists)"
    return 0
  fi
  while :; do
    local n
    n=$(count_running)
    [[ -z "$n" ]] && n=0
    if (( n < MAXJOBS )); then break; fi
    sleep 20
  done
  mkdir -p "$dir"
  echo "[start] $tag $(date +%H:%M:%S)"
  timeout 3000 python -u src/ablation_ttt.py \
    --games "$GAMES" --configs "$cfg" --seed "$seed" \
    --tiebreak "$tb" --mentor-noise "$noise" \
    --logs-dir "$dir" --results-file "$dir/res.json" \
    > "$dir/train.log" 2>&1 &
  echo "[pid $!] $tag"
  sleep 2
}

# C runs are fast (~10 min); start them first. The 20%-noise control uses the
# GRU backbone so that it matches TBR seeds within a practical budget.
launch C 44 first 0.05 "fixed_C_s44"
launch C 42 random 0.05 "tbr_C_s42"
launch C 43 random 0.05 "tbr_C_s43"
launch C 44 random 0.05 "tbr_C_s44"
launch C 42 first 0.20 "noise_C_s42"
launch C 43 first 0.20 "noise_C_s43"

# CfC runs are slow (~30 min each).
launch B 42 first 0.05 "fixed_B_s42"
launch B 43 random 0.05 "tbr_B_s43"
launch B 44 random 0.05 "tbr_B_s44"

echo "[queue] waiting for children..."
wait
echo "[queue] done $(date +%H:%M:%S)"
