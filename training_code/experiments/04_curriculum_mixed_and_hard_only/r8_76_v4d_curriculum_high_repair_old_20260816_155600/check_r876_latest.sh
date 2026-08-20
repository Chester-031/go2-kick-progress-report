#!/usr/bin/env bash
set -u

ROOT="/scratch/lj2634/runs/go2_soccer_r8_76_v4d_curriculum_high_repair"
PTR="$ROOT/latest_r876.txt"

if [[ ! -s "$PTR" ]]; then
  echo "R876_STATUS=NO_RUN_POINTER"
  if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then return 0; fi
  exit 0
fi

RUN="$(tr -d '\r\n' < "$PTR")"
JOB="$(basename "$RUN" | cut -d_ -f2)"
echo "RUN=$RUN"

echo "===== SLURM ====="
squeue -j "$JOB" 2>/dev/null || true
sacct -X -j "$JOB" \
  --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,NodeList 2>/dev/null || true

echo "===== CURRICULUM ====="
grep -E \
  'R876_CURRICULUM_STAGE|V4D_JOINT_ITER|V4D_PARAMETER_DELTA|V4D_FATAL|Traceback|RuntimeError|PROCESS_RC' \
  "$RUN/worker.log" 2>/dev/null | tail -n 35 || true

echo "===== VALIDATION ====="
cat "$RUN/post_training_validation.txt" 2>/dev/null || echo "VALIDATION=NOT_READY"

echo "===== CHECKPOINTS ====="
find "$RUN/logs" -maxdepth 1 -type f \
  \( -name 'high_model_*.pt' -o -name 'high_model_final.pt' \
     -o -name 'low_model_final.pt' \) \
  -printf '%TY-%Tm-%Td %TH:%TM:%TS %s bytes %f\n' 2>/dev/null | sort | tail -n 20

if [[ -f "$RUN/R876_READY_FOR_FULL_SUPPORT_EVALUATION" ]]; then
  echo "VERDICT=R876_READY_FOR_METRIC_EVALUATION"
else
  echo "VERDICT=R876_NOT_READY"
fi
