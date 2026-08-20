#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="/scratch/lj2634/runs/go2_soccer_r8_83_v4d_reference_velocity_repair"
RUN="$(tr -d '\r\n' < "$ROOT/latest_r883.txt")"
JOB="$(basename "$RUN" | cut -d_ -f2)"
echo "RUN=$RUN"
squeue -j "$JOB" 2>/dev/null || true
sacct -X -j "$JOB" --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,NodeList 2>/dev/null || true
grep -E '\[R883_(PRE_SUSTAINED|LOW_ITER|POST_SUSTAINED|PARAMETER_DELTA)\]|R883_LOW_REPAIR_|\[V4D_FATAL\]|PROCESS_RC=' \
  "$RUN/worker.log" 2>/dev/null | tail -n 24 || true
cat "$RUN/validation.txt" 2>/dev/null || echo "VALIDATION=NOT_READY"
if [[ -f "$RUN/R883_READY_FOR_PAIRED_FULL_SUPPORT_GATE" ]]; then
  echo "VERDICT=R883_AXIS_LOW_READY_FOR_PAIRED_GATE"
else
  echo "VERDICT=R883_NOT_READY"
fi
