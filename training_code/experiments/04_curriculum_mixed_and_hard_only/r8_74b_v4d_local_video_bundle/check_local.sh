#!/usr/bin/env bash
set -Eeuo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEED="${R874B_VIDEO_SEED:-86501}"
VIDEO="$BASE/videos/R874b_V4D_seed_${SEED}.mp4"
SUMMARY="$BASE/videos/R874b_V4D_seed_${SEED}.json"
TRAJECTORY="$BASE/videos/R874b_V4D_seed_${SEED}.csv"
READY="$VIDEO.ready"
LOG="$BASE/logs/render_seed_${SEED}.log"
STAGE="$BASE/logs/current_stage.txt"

echo "============================================================"
echo " R8.74b V4D LOCAL VIDEO CHECKER"
echo "============================================================"
echo "TIME=$(date -Is)"
echo "BASE=$BASE"
echo "SEED=$SEED"
echo "STAGE=$(tr -d '\r\n' < "$STAGE" 2>/dev/null || echo NOT_STARTED)"

echo "==================== GPU ===================="
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used \
  --format=csv,noheader 2>/dev/null || echo "GPU_STATUS_UNAVAILABLE"

echo "==================== MARKERS ===================="
if [[ -s "$LOG" ]]; then
  grep -En \
    'PHASEA_BOOT|V4D_TASK_REGISTRATION_OK|Completed setting up the environment|V4D_EVAL|R874B|Traceback|RuntimeError|ModuleNotFoundError|V4D_FATAL' \
    "$LOG" | tail -n 100 || true
else
  echo "LOG_NOT_READY=$LOG"
fi

echo "==================== LOG TAIL ===================="
if [[ -s "$LOG" ]]; then
  tail -n 100 "$LOG"
else
  echo "NO_LOG_CONTENT"
fi

echo "==================== ARTIFACTS ===================="
for file in "$VIDEO" "$SUMMARY" "$TRAJECTORY" "$READY"; do
  if [[ -s "$file" ]]; then
    stat --printf='READY %n %s bytes %y\n' "$file"
  else
    echo "NOT_READY $file"
  fi
done

if [[ -s "$SUMMARY" ]]; then
  echo "==================== SUMMARY ===================="
  python - "$SUMMARY" <<'PY'
import json
import sys

d = json.load(open(sys.argv[1], encoding="utf-8"))
for key in (
    "seed", "steps", "goal", "own_goal", "goal_rate",
    "own_goal_rate", "find_ball_rate", "approach_success_rate",
    "fall_rate", "body_swing_metric", "lateral_oscillation_metric",
):
    if key in d:
        print(f"{key}={d[key]}")
print("SUMMARY_KEYS=" + ",".join(sorted(d)))
PY
fi

FATAL=""
if [[ -s "$LOG" ]]; then
  FATAL="$(grep -Ein 'Traceback|RuntimeError:|ModuleNotFoundError:|V4D_FATAL|R874B_LOCAL_FATAL|Found no NVIDIA driver|libcuda.so.1.*cannot open' "$LOG" || true)"
fi

echo "==================== DIAGNOSIS ===================="
if [[ -n "$FATAL" ]]; then
  printf '%s\n' "$FATAL" | tail -n 80
  echo "FATAL_SCAN=FAILED"
else
  echo "FATAL_SCAN=CLEAN"
fi

if [[ -s "$VIDEO" && -s "$SUMMARY" && -s "$TRAJECTORY" && -s "$READY" ]] \
   && grep -qF 'V4D_EVAL_VIDEO_OK' "$LOG" 2>/dev/null \
   && [[ -z "$FATAL" ]]; then
  echo "VERDICT=R874B_LOCAL_VIDEO_READY_FOR_REVIEW"
  exit 0
fi

if [[ ! -s "$LOG" ]]; then
  VERDICT="NOT_STARTED"
elif grep -qF '[V4D_TASK_REGISTRATION_OK]' "$LOG" \
     && ! grep -qF 'Completed setting up the environment' "$LOG"; then
  VERDICT="BLOCKED_DURING_SCENE_CREATION"
elif grep -qF '[PHASEA_BOOT]' "$LOG" \
     && ! grep -qF '[V4D_TASK_REGISTRATION_OK]' "$LOG"; then
  VERDICT="BLOCKED_DURING_TASK_IMPORT"
elif ! grep -qF '[PHASEA_BOOT]' "$LOG"; then
  VERDICT="BLOCKED_DURING_ISAAC_APP_STARTUP"
elif grep -qF 'Completed setting up the environment' "$LOG" \
     && ! grep -qE 'V4D_EVAL|evaluation' "$LOG"; then
  VERDICT="BLOCKED_DURING_CHECKPOINT_OR_EVAL_SETUP"
elif grep -qE 'V4D_EVAL|evaluation' "$LOG" && [[ ! -s "$VIDEO" ]]; then
  VERDICT="BLOCKED_DURING_EVALUATION_OR_VIDEO_ENCODING"
else
  VERDICT="INCOMPLETE_REVIEW_LOG"
fi

echo "VERDICT=$VERDICT"
exit 4
