#!/usr/bin/env bash
# Stage 3: RL-trained base policies, so the second domain is not distillation-only.
# Runs only after the epsilon study releases the cores (stage 2 marker), one thread at
# a time, because this box has two cores and another project is using one.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=runs/_rl_stage3.log
say() { echo "[$(date -Is)] $*" >> "$LOG"; }
: > "$LOG"

# Wait for the tolerance study to finish (up to 90 minutes), then proceed regardless.
for _ in $(seq 1 180); do
  grep -q "stage 2 done" runs/_c4_eps_stage2.log 2>/dev/null && break
  sleep 30
done
say "starting RL arms"

say "4x4 Connect Four, seed 0"
python src/train_rl_c4.py --rows 4 --cols 4 --win 4 --games 5000 --block 500 \
  --cert-every 1 --seed 0 --threads 1 --lr 1e-3 --out data/rl_c4x4_s0.json \
  >> runs/_rl_c4x4.log 2>&1
say "4x4 seed 0 rc=$?"

W0=$(python -c "import json;print(json.load(open('data/rl_c4x4_s0.json'))['base']['cert']['worst_draw_rate'])" 2>/dev/null || echo "?")
say "4x4 seed 0 base W = $W0"
if [ "$W0" == "0.0" ]; then
  say "repairing the 4x4 RL base, seed 0"
  python src/c4_repair.py --rows 4 --cols 4 --win 4 --init data/rl_c4x4_s0.pt \
    --seed 0 --threads 1 --epochs 120 --agg-repeat 10 --max-iters 40 --no-controls \
    --no-dagger --out data/c4_rl_repair_x4_s0.json >> runs/_rl_c4x4.log 2>&1
  say "4x4 seed 0 repair rc=$?"
fi

for s in 1 2; do
  say "4x4 Connect Four, seed $s"
  python src/train_rl_c4.py --rows 4 --cols 4 --win 4 --games 5000 --block 500 \
    --cert-every 1 --seed "$s" --threads 1 --lr 1e-3 --out "data/rl_c4x4_s${s}.json" \
    >> runs/_rl_c4x4.log 2>&1
  say "4x4 seed $s rc=$?"
done

for s in 0 1 2; do
  say "4x3 Connect Three, seed $s"
  python src/train_rl_c4.py --rows 4 --cols 3 --win 3 --games 5000 --block 500 \
    --cert-every 1 --seed "$s" --threads 1 --lr 1e-3 --out "data/rl_c4_s${s}.json" \
    >> runs/_rl_c4.log 2>&1
  say "4x3 seed $s rc=$?"
done

say "RL arms done"
