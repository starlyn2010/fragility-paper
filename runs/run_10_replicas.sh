#!/usr/bin/env bash
# Replicas at 10 empties on the fixed 40-root suite, then restore power settings.
set -u
export MSYS_NO_PATHCONV=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=3

REPO="/c/Users/COMPUTEL/Desktop/projectos/fragility-paper"
cd "$REPO" || exit 1
SUM="C:\\Users\\COMPUTEL\\Desktop\\projectos\\fragility-paper"
ROOTS="data/c4_endgame_roots_10_40_s0.json"

echo "[$(date -Is)] keep-awake: disabling idle sleep" | tee -a runs/_c4_replicas.log
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change monitor-timeout-ac 0
powercfg /setacvalueindex SCHEME_CURRENT 238C9FA8-0AAD-41ED-83F4-97BE242C8F20 7bc4a2f9-d8fc-4469-b07b-33eb785aaca0 0

echo "[$(date -Is)] starting keeper process" | tee -a runs/_c4_replicas.log
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath powershell.exe -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','${SUM}\\runs\\keepawake.ps1' -WindowStyle Hidden"
sleep 3

for SEED in 1 2; do
  echo "[$(date -Is)] seed $SEED start" | tee -a runs/_c4_replicas.log
  python src/c4_endgame.py \
    --rows 6 --cols 7 --win 4 --empties 10 --roots 40 \
    --roots-file "$ROOTS" --root-value 0 \
    --hidden 128 --emb 16 --epochs 200 --lr 2e-3 --agg-repeat 60 \
    --stop both --max-iters 24 --seed "$SEED" --threads 3 --no-controls \
    --out "data/c4_end10_s${SEED}.json" --ckpt-out "runs/c4_end10_s${SEED}" \
    > "runs/_c4_end10_s${SEED}.log" 2>&1
  RC=$?
  echo "[$(date -Is)] seed $SEED done rc=$RC" | tee -a runs/_c4_replicas.log
done

echo "[$(date -Is)] replicas finished, restoring power settings" | tee -a runs/_c4_replicas.log
powershell -NoProfile -ExecutionPolicy Bypass -File "${SUM}\\runs\\restore_power.ps1"
echo "[$(date -Is)] runner exit" | tee -a runs/_c4_replicas.log
