#!/usr/bin/env bash
# Independent verification of the three ten-empty-cell checkpoints on the shared
# suite, then restore the previous power configuration even on error.
set -u
export MSYS_NO_PATHCONV=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2

REPO="/c/Users/COMPUTEL/Desktop/projectos/fragility-paper"
SUM="C:\\Users\\COMPUTEL\\Desktop\\projectos\\fragility-paper"
cd "$REPO" || exit 1
LOG="runs/_c4_verify_all.log"
say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

restore() {
  powershell -NoProfile -ExecutionPolicy Bypass \
    -File "${SUM}\\runs\\restore_power.ps1" >> "$LOG" 2>&1
  say "power settings restored"
}
trap restore EXIT

powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change monitor-timeout-ac 0
powershell -NoProfile -ExecutionPolicy Bypass -Command \
  "Start-Process -FilePath powershell.exe -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','${SUM}\\runs\\keepawake.ps1' -WindowStyle Hidden"
sleep 3
say "keeper started; verification start"

for s in 0 1 2; do
  python src/c4_endgame_verify.py \
    --ckpt runs/c4_end10_s$s/c4_gru.pt \
    --roots data/c4_endgame_roots_10_40_s0.json \
    --hidden 128 --emb 16 --orders 120 --games 40 \
    --out data/c4_end10_verify_s$s.json >> "$LOG" 2>&1
  say "seed $s rc=$? -> data/c4_end10_verify_s$s.json"
done
say "verification finished"
