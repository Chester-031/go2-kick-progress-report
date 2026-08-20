#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="/scratch/lj2634/runs/go2_soccer_r8_65_v4d_behavior"; P="$ROOT/latest_r865.txt"
[[ -s "$P" ]] || { echo VERDICT=R865_NOT_READY; exit 2; }
RUN="$(tr -d '\r\n' < "$P")"
echo '============================================================'; echo ' R8.65 V4D FROZEN BEHAVIOR CHECKER'; echo '============================================================'
echo "TIME=$(date -Is)"; echo "RUN=$RUN"; cat "$RUN/contract.txt"
echo '==================== ARTIFACTS ===================='
find "$RUN/videos" "$RUN/summaries" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
echo '==================== FATAL SCAN =================='
if grep -Ein 'Traceback|RuntimeError|CUDA out of memory|Segmentation fault|NaN|fatal' "$RUN/worker.log"; then echo FATAL_SCAN=FAILED; else echo FATAL_SCAN=CLEAN; fi
echo '==================== SUMMARIES ===================='
for f in "$RUN"/summaries/*.json; do
  [ -s "$f" ] || continue
  echo "--- $f"
  python - "$f" <<'PY'
import json,sys
print(json.dumps(json.load(open(sys.argv[1],encoding='utf-8')),sort_keys=True))
PY
done
(cd "$RUN" && sha256sum -c SHA256SUMS)
N=$(find "$RUN/summaries" -maxdepth 1 -type f -size +0c | wc -l); V=$(find "$RUN/videos" -maxdepth 1 -type f -size +0c | wc -l)
echo "SUMMARY_COUNT=$N VIDEO_COUNT=$V"
[[ "$N" -eq 8 && "$V" -eq 8 ]] || { echo VERDICT=R865_ARTIFACTS_INCOMPLETE; exit 3; }
echo TRAINING=FALSE; echo CHECKPOINT_MUTATION=FALSE; echo STATE=BEHAVIOR_DATA_READY; echo VERDICT=R865_READY_FOR_METRIC_DECISION
