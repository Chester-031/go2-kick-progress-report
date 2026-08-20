#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="/scratch/lj2634/runs/go2_soccer_r8_64_v4d_coverage"; POINTER="$ROOT/latest_r864.txt"
[[ -s "$POINTER" ]] || { echo VERDICT=R864_NOT_READY; exit 2; }
RUN="$(tr -d '\r\n' < "$POINTER")"
echo '============================================================'; echo ' R8.64 V4D FULL-SUPPORT COVERAGE CHECKER'; echo '============================================================'
echo "TIME=$(date -Is)"; echo "RUN=$RUN"; cat "$RUN/evidence/R864_STATUS.txt"
python - "$RUN/evidence/v4d_coverage.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8')); o=d['observed']
assert o['invalid_distance_count']==0
assert all(v>0 for v in o['quadrant_counts'].values())
assert o['robot_x'][0] < -2.5 and o['robot_x'][1] > 1.9
assert o['ball_x'][0] < -2.5 and o['ball_x'][1] > 1.9
assert o['yaw'][0] < -2.5 and o['yaw'][1] > 2.5
print('SUPPORT_DISTRIBUTION=FULL_INTERIOR')
print('QUADRANT_COUNTS='+json.dumps(o['quadrant_counts'],sort_keys=True))
PY
(cd "$RUN" && sha256sum -c SHA256SUMS)
echo 'TRAINING=FALSE'; echo 'PARENT_MUTATION=FALSE'; echo 'STATE=RESET_SUPPORT_VERIFIED'; echo 'VERDICT=R864_READY_FOR_ISAAC_BEHAVIOR_SCAN'
