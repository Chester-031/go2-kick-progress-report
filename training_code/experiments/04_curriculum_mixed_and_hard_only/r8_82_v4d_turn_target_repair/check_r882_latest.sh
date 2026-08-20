#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="/scratch/lj2634/runs/go2_soccer_r8_82_v4d_turn_target_repair"
RUN="$(tr -d '\r\n' < "$ROOT/latest_r882.txt")"
JOB="$(basename "$RUN" | cut -d_ -f2)"
echo "RUN=$RUN"
squeue -j "$JOB" 2>/dev/null || true
sacct -X -j "$JOB" --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,NodeList 2>/dev/null || true
grep -E '\[R882_(PRE_SUSTAINED|LOW_ITER|POST_SUSTAINED|PARAMETER_DELTA)\]|R882_LOW_REPAIR_|\[V4D_FATAL\]|PROCESS_RC=' \
  "$RUN/worker.log" 2>/dev/null | tail -n 24 || true
cat "$RUN/validation.txt" 2>/dev/null || echo "VALIDATION=NOT_READY"
if [[ -f "$RUN/R882_READY_FOR_PAIRED_FULL_SUPPORT_GATE" ]]; then
  echo "VERDICT=R882_AXIS_LOW_READY_FOR_PAIRED_GATE"
else
  echo "VERDICT=R882_NOT_READY"
fi
