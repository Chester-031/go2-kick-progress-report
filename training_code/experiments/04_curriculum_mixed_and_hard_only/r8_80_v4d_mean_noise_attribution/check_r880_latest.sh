#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="/scratch/lj2634/runs/go2_soccer_r8_80_v4d_mean_noise_attribution"
RUN="$(tr -d '\r\n' < "$ROOT/latest_r880.txt")"
JOB="$(basename "$RUN" | cut -d_ -f2)"
echo "RUN=$RUN"
squeue -j "$JOB" 2>/dev/null || true
sacct -X -j "$JOB" --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,NodeList 2>/dev/null || true
grep -E '\[R880_(DETERMINISTIC|STOCHASTIC|STD)\]|R880_(SUMMARY|VERDICT)|\[V4D_FATAL\]|PROCESS_RC=' \
  "$RUN/worker.log" 2>/dev/null | tail -n 10 || true
cat "$RUN/validation.txt" 2>/dev/null || echo "VALIDATION=NOT_READY"
