#!/usr/bin/env bash
set -u

ROOT="/scratch/lj2634/runs/go2_soccer_r8_77_v4d_checkpoint_gate"
PTR="$ROOT/latest_r877.txt"
[[ -s "$PTR" ]] || { echo "R877_STATUS=NO_RUN_POINTER"; exit 0; }
RUN="$(tr -d '\r\n' < "$PTR")"
JOB="$(basename "$RUN" | cut -d_ -f2)"
echo "RUN=$RUN"

echo "===== SLURM ====="
squeue -j "$JOB" 2>/dev/null || true
sacct -X -j "$JOB" --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,NodeList 2>/dev/null || true

echo "===== CASES ====="
grep -E 'R877_CASE_START|R877_CASE_END|R877_SELECTED|R877_SELECTION_STATUS|Traceback|RuntimeError|R877_FATAL' \
  "slurm-r877-${JOB}.out" "$RUN/selection.log" 2>/dev/null | tail -n 80 || true

echo "===== COMPARISON ====="
if [[ -s "$RUN/results/r877_checkpoint_selection.json" ]]; then
  python3 - "$RUN/results/r877_checkpoint_selection.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
print("accepted_high=" + json.dumps(d["accepted_high"], sort_keys=True))
for row in d["candidates"]:
    print(row["label"] + "=" + json.dumps({
        "eligible": row["eligible"], "improved": row["materially_improved"],
        "score": row["score"], **row["metrics"]}, sort_keys=True))
print("SELECTED=" + d["selected_label"])
print("STATUS=" + d["selection_status"])
print("CHECKPOINT=" + d["selected_checkpoint"])
PY
else
  echo "COMPARISON=NOT_READY"
fi

if [[ -f "$RUN/R877_CHECKPOINT_GATE_OK" ]]; then
  echo "VERDICT=R877_READY_TO_BUILD_LOCAL_VIDEO_BUNDLE"
else
  echo "VERDICT=R877_NOT_READY"
fi
