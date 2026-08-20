#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/scratch/lj2634/runs/go2_soccer_r8_78_v4d_sustained_low_repair"
PTR="$ROOT/latest_r878.txt"
[[ -s "$PTR" ]] || { echo "R878_CHECK_FATAL=latest_pointer_missing"; exit 21; }
RUN="$(tr -d '\r\n' < "$PTR")"
JOB="$(basename "$RUN" | cut -d_ -f3)"

echo "RUN=$RUN"
echo "===== SLURM ====="
squeue -j "$JOB" 2>/dev/null || true
sacct -X -j "$JOB" --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,NodeList 2>/dev/null || true

echo "===== ATTRIBUTION ====="
grep -E 'R878_TRACE_VERDICT|R878_FATAL' \
  "$RUN/results/r877_sustained_reassessment.log" "$RUN/worker.log" 2>/dev/null || true

echo "===== TRAINING PROGRESS ====="
grep -E '\[R878_(PRE_SUSTAINED|LOW_ITER|POST_SUSTAINED|PARAMETER_DELTA)\]|R878_LOW_REPAIR_|\[V4D_FATAL\]|PROCESS_RC=' \
  "$RUN/worker.log" 2>/dev/null | tail -n 24 || true

echo "===== ITERATIONS ====="
"${ISAAC_PYTHON:-python3}" - "$RUN/worker.log" <<'PY' 2>/dev/null || true
import re, sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace") if Path(sys.argv[1]).is_file() else ""
values = [int(v) for v in re.findall(r'\[R878_LOW_ITER\].*?"iteration":\s*(\d+)', text)]
print(f"LAST_REPORTED_ITERATION={values[-1] if values else 'NOT_READY'}")
PY

echo "===== VALIDATION ====="
cat "$RUN/post_training_validation.txt" 2>/dev/null || echo "VALIDATION=NOT_READY"

echo "===== CHECKPOINTS ====="
find "$RUN/logs" -maxdepth 1 -type f -name '*.pt' \
  -printf '%TY-%Tm-%Td %TH:%TM:%TS %s bytes %f\n' 2>/dev/null | sort || true

if [[ -f "$RUN/R878_READY_FOR_PAIRED_FULL_SUPPORT_GATE" ]]; then
  echo "VERDICT=R878_LOW_REPAIR_READY_FOR_PAIRED_GATE"
elif grep -Fq '[V4D_FATAL]' "$RUN/worker.log" 2>/dev/null; then
  echo "VERDICT=R878_FATAL"
else
  echo "VERDICT=R878_NOT_READY"
fi
