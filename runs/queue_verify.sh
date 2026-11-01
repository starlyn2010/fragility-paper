#!/usr/bin/env bash
# Runs the independent verification of the 10-empty checkpoint after the replica
# runner has finished and restored the power settings, then restores them again.
set -u
export MSYS_NO_PATHCONV=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2

REPO="/c/Users/COMPUTEL/Desktop/projectos/fragility-paper"
SUM="C:\\Users\\COMPUTEL\\Desktop\\projectos\\fragility-paper"
cd "$REPO" || exit 1
LOG="runs/_c4_verify_queue.log"
say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

say "waiting for the replica runner to exit"
while ! grep -q "runner exit" runs/_c4_replicas.log 2>/dev/null; do
  sleep 60
done
say "replicas done; re-applying no-sleep for the verification"

powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change monitor-timeout-ac 0
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath powershell.exe -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','${SUM}\\runs\\keepawake.ps1' -WindowStyle Hidden"
sleep 3

say "verification start"
python src/c4_endgame_verify.py \
  --ckpt runs/c4_end10_s0/c4_gru.pt \
  --roots data/c4_endgame_roots_10_40_s0.json \
  --hidden 128 --emb 16 --orders 120 --games 40 \
  --out data/c4_end10_verify_s0.json >> "$LOG" 2>&1
RC=$?
say "verification exit rc=$RC"

powershell -NoProfile -ExecutionPolicy Bypass -File "${SUM}\\runs\\restore_power.ps1"
say "queue exit"
