#!/usr/bin/env bash
set -u

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEED="${R877_VIDEO_SEED:-87701}"
VIDEO="$BASE/videos/R877_V4D_selected_seed_${SEED}.mp4"
SUMMARY="$BASE/videos/R877_V4D_selected_seed_${SEED}.json"
TRAJECTORY="$BASE/videos/R877_V4D_selected_seed_${SEED}.csv"
LOG="$BASE/logs/render_seed_${SEED}.log"
READY="$VIDEO.ready"

echo "===== R8.77 LOCAL VIDEO ====="
echo "BASE=$BASE"
grep -E 'R876_EVAL_DISTRIBUTION|V4D_EVAL|V4D_FATAL|Traceback|RuntimeError|R877_LOCAL_FATAL' "$LOG" 2>/dev/null | tail -n 80 || true

echo "===== ARTIFACTS ====="
for FILE in "$VIDEO" "$SUMMARY" "$TRAJECTORY" "$READY"; do
  [[ -s "$FILE" ]] && stat --printf='READY %n %s bytes %y\n' "$FILE" || echo "NOT_READY $FILE"
done

if [[ -s "$SUMMARY" && -s "$TRAJECTORY" ]]; then
  python - "$SUMMARY" "$TRAJECTORY" <<'PY'
import csv,json,statistics,sys
s=json.load(open(sys.argv[1],encoding="utf-8"))
rows=list(csv.DictReader(open(sys.argv[2],encoding="utf-8")))
dist=[float(r["robot_ball_distance"]) for r in rows]
upright=[float(r["upright_error"]) for r in rows]
stall=[int(r["push_stall"]) for r in rows]
print("summary="+json.dumps(s,sort_keys=True))
print("trajectory_rows=",len(rows),sep="")
print("robot_ball_distance_min=",min(dist),sep="")
print("robot_ball_distance_median=",statistics.median(dist),sep="")
print("upright_error_max=",max(upright),sep="")
print("push_stall_fraction=",sum(stall)/len(stall),sep="")
PY
fi

FATAL="$(grep -Ein 'Traceback|RuntimeError:|V4D_FATAL|R877_LOCAL_FATAL|cuda_unavailable' "$LOG" 2>/dev/null || true)"
if [[ -s "$VIDEO" && -s "$SUMMARY" && -s "$TRAJECTORY" && -s "$READY" \
      && -z "$FATAL" ]] \
   && grep -qF '[R876_EVAL_DISTRIBUTION] phase=full_support' "$LOG" 2>/dev/null \
   && grep -qF 'V4D_EVAL_VIDEO_OK' "$LOG" 2>/dev/null; then
  echo "VERDICT=R877_LOCAL_VIDEO_READY_FOR_REVIEW"
  exit 0
fi
[[ -z "$FATAL" ]] || printf '%s\n' "$FATAL"
echo "VERDICT=R877_LOCAL_VIDEO_NOT_READY"
exit 4
