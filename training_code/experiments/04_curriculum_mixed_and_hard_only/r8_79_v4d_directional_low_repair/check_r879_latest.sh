#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="/scratch/lj2634/runs/go2_soccer_r8_79_v4d_directional_low_repair"
PTR="$ROOT/latest_r879.txt"
[[ -s "$PTR" ]] || { echo "R879_CHECK_FATAL=latest_pointer_missing"; exit 21; }
RUN="$(tr -d '\r\n' < "$PTR")"
JOB="$(basename "$RUN" | cut -d_ -f2)"
echo "RUN=$RUN"
echo "===== SLURM ====="
squeue -j "$JOB" 2>/dev/null || true
sacct -X -j "$JOB" --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,NodeList 2>/dev/null || true
echo "===== PROGRESS ====="
grep -E '\[R879_(PRE_SUSTAINED|LOW_ITER|POST_SUSTAINED|PARAMETER_DELTA)\]|R879_LOW_REPAIR_|\[V4D_FATAL\]|PROCESS_RC=' \
  "$RUN/worker.log" 2>/dev/null | tail -n 24 || true
echo "===== VALIDATION ====="
cat "$RUN/post_training_validation.txt" 2>/dev/null || echo "VALIDATION=NOT_READY"
echo "===== CHECKPOINTS ====="
find "$RUN/logs" -maxdepth 1 -type f -name '*.pt' \
  -printf '%TY-%Tm-%Td %TH:%TM:%TS %s bytes %f\n' 2>/dev/null | sort || true
if [[ -f "$RUN/R879_READY_FOR_PAIRED_FULL_SUPPORT_GATE" ]]; then
  echo "VERDICT=R879_DIRECTIONAL_LOW_READY_FOR_PAIRED_GATE"
elif grep -Fq '[V4D_FATAL]' "$RUN/worker.log" 2>/dev/null; then
  echo "VERDICT=R879_FATAL"
else
  echo "VERDICT=R879_NOT_READY"
fi
