#!/usr/bin/env bash
# Integer-payoff tolerance sweep: does epsilon act as a dial at all when the exact
# values are integers? Seeds run one at a time because another project already owns
# a core on this two-core machine, and its keep_awake.py covers the hibernation risk.
set -u
export MSYS_NO_PATHCONV=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

REPO="/c/Users/COMPUTEL/Desktop/projectos/fragility-paper"
cd "$REPO" || exit 1
DIALS="0,0.01,0.02,0.0417,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.05,1.2,1.5"
LOG="runs/_c4_eps_int.log"

for s in 0 1 2; do
  out="data/c4_eps_int_s${s}.json"
  if [ -f "$out" ]; then
    echo "[$(date +%FT%T)] seed $s already done" >> "$LOG"
    continue
  fi
  echo "[$(date +%FT%T)] seed $s start" >> "$LOG"
  python src/c4_approx_oracle.py --mode noise --dials "$DIALS" --seed "$s" \
    --threads 1 --epochs 40 --hidden 32 --emb 8 --lr 1e-3 --agg-repeat 60 \
    --max-iters 12 --out "$out" >> "$LOG" 2>&1
  echo "[$(date +%FT%T)] seed $s rc=$?" >> "$LOG"
done
echo "[$(date +%FT%T)] integer sweep finished" >> "$LOG"
