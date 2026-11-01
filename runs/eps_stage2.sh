#!/usr/bin/env bash
# Queued stage of the epsilon study. It waits for the integer-payoff sweep
# (runs/eps_int_sweep.sh) to finish so this box never runs three single-thread
# jobs at once, then:
#   1. re-runs the integer seed 0 arm so all three seeds carry the same columns;
#   2. runs the graded-payoff blend sweep on three seeds (the real epsilon dial);
#   3. records the smallest value gap and the critical epsilon as a function of
#      the discount, which is what makes the tolerance meaningful.
# Power settings are deliberately untouched: another project owns a core and its
# keep_awake.py covers hibernation.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=python
LOG=runs/_c4_eps_stage2.log
WAIT_PID="${1:-}"

say() { echo "[$(date -Is)] $*" >> "$LOG"; }

: > "$LOG"
if [ -n "$WAIT_PID" ]; then
  say "waiting for pid $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  say "pid $WAIT_PID gone"
fi

COMMON="--epochs 40 --hidden 32 --emb 8 --lr 1e-3 --agg-repeat 60 --max-iters 12 --threads 1"

# 1. integer arm, seed 0 again (same dial grid as eps_int_sweep.sh)
INT_DIALS="0,0.01,0.02,0.0417,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.05,1.2,1.5"
cp data/c4_eps_int_s0.json runs/_c4_eps_int_s0_partial.json 2>/dev/null || true
say "integer seed 0 re-run start"
$PY src/c4_approx_oracle.py --mode noise --ref integer --seed 0 \
    --dials "$INT_DIALS" $COMMON --out data/c4_eps_int_s0.json \
    >> runs/_c4_eps_int.log 2>&1
say "integer seed 0 re-run rc=$?"

# 2. graded arm, three seeds
GRADED_DIALS="0,0.3,0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95,1.0"
for s in 0 1 2; do
  say "graded blend seed $s start"
  $PY src/c4_approx_oracle.py --mode blend --ref graded --gamma 0.9 --seed "$s" \
      --dials "$GRADED_DIALS" $COMMON --out "data/c4_eps_grad_s${s}.json" \
      >> runs/_c4_eps_grad.log 2>&1
  say "graded blend seed $s rc=$?"
done

# 3. gap versus discount (exact minimax, no training)
$PY src/c4_approx_oracle.py --gaps-only --ref graded --gamma 0.9 --threads 1 \
      --out data/c4_graded_gaps.json >> runs/_c4_eps_stage2_gaps.log 2>&1
say "gaps rc=$?"
say "stage 2 done"
