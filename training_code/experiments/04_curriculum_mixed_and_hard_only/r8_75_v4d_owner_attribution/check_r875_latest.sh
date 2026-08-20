#!/usr/bin/env bash
set -u

ROOT="/scratch/lj2634/runs/go2_soccer_r8_75_v4d_owner_attribution"
PTR="$ROOT/latest_r875.txt"

if [[ ! -s "$PTR" ]]; then
  echo "R875_STATUS=NO_RUN_POINTER"
  if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    return 0
  fi
  exit 0
fi

RUN="$(tr -d '\r\n' < "$PTR")"
echo "RUN=$RUN"

echo "===== SLURM ====="
JOB="$(basename "$RUN" | cut -d_ -f2)"
squeue -j "$JOB" 2>/dev/null || true
sacct -X -j "$JOB" \
  --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,NodeList 2>/dev/null || true

echo "===== MARKERS ====="
grep -E \
  'R875_(CONTRACT|RESET|PROGRESS|SUMMARY|TRACE|VERDICT|OWNER_ATTRIBUTION_COMPLETE|PROCESS_RC)|V4D_FATAL|Traceback|RuntimeError' \
  "$RUN/worker.log" 2>/dev/null | tail -n 80 || true

echo "===== SUMMARY ====="
python3 - "$RUN/results/r875_summary.json" <<'PY' 2>/dev/null || true
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    print("SUMMARY=NOT_READY")
    raise SystemExit(0)
data = json.loads(path.read_text(encoding="utf-8"))
print("VERDICT=" + str(data.get("verdict")))
print("RATIONALE=" + str(data.get("rationale")))
print("PRIMITIVES=" + str(data.get("primitive_pass_count")) + "/" + str(data.get("primitive_total")))
for mode, values in data.get("mode_summary", {}).items():
    print(mode + "=" + json.dumps(values, sort_keys=True))
print("PARAMETER_DELTAS=" + json.dumps(data.get("parameter_deltas"), sort_keys=True))
print("TRIPLICATE_CHECK=" + json.dumps(data.get("triplicate_initial_state_check"), sort_keys=True))
PY

echo "===== ARTIFACTS ====="
find "$RUN/results" -maxdepth 1 -type f -size +0c \
  -printf '%s bytes %p\n' 2>/dev/null | sort || true
