#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "STATE=FAILED"; echo "VERDICT=R859_FREEZE_CHECK_FAILED"; exit "$rc"' ERR

ROOT="/scratch/lj2634/runs/go2_soccer_r8_59_parent_freeze"
POINTER="$ROOT/latest_r859_freeze.txt"
[[ -s "$POINTER" ]] || { echo VERDICT=R859_FREEZE_NOT_READY; exit 2; }
FREEZE="$(tr -d '\r\n' < "$POINTER")"

echo '============================================================'
echo ' R8.59 INTEGRATION PARENT FREEZE CHECKER'
echo '============================================================'
echo "TIME=$(date -Is)"
echo "FREEZE=$FREEZE"
echo
echo '==================== PARENTS ===================='
cat "$FREEZE/PARENT_PATHS.txt"
echo
echo '==================== FREEZE STATUS =============='
cat "$FREEZE/FREEZE_STATUS.txt"
echo
echo '==================== CONTRACT ==================='
python - "$FREEZE/integration_parent_contract.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
print('VERSION='+d['version'])
print('TRAINING='+str(d['training']))
print('MUTATION='+d['mutation'])
print('ROUTER_BOUNDARY='+d['interfaces']['router_boundary'])
print('HIGH_INTERFACE='+json.dumps(d['interfaces']['high'],sort_keys=True))
print('DELTA_POSE_INTERFACE='+json.dumps(d['interfaces']['delta_pose_low'],sort_keys=True))
print('KICK_INTERFACE='+json.dumps(d['interfaces']['kick_internal'],sort_keys=True))
for name,x in d['checkpoint_inspection'].items():
 print(name,'input',x['actor_input'],'output',x['actor_output'],'bytes',x['bytes'])
print('STRENGTH_ANCHORS='+json.dumps(d['strength_anchors'],sort_keys=True))
print('FORBIDDEN='+json.dumps(d['interfaces']['forbidden']))
PY
echo
echo '==================== FROZEN SOURCES ============='
(cd "$FREEZE" && sha256sum source/* contracts/*)
echo
echo '==================== FILE MODES =================='
find "$FREEZE" -type f -printf '%m %s %p\n' | sort
echo
echo '==================== HASH ========================'
(cd "$FREEZE" && sha256sum -c SHA256SUMS)
WRITABLE="$(find "$FREEZE" -type f -perm /222 -print -quit)"
[[ -z "$WRITABLE" ]] || { echo "WRITABLE_FILE=$WRITABLE"; echo VERDICT=FAILED_NOT_READ_ONLY; exit 3; }

trap - ERR
echo 'STATE=FROZEN_READ_ONLY'
echo 'VERDICT=R859_FROZEN_PARENTS_READY_FOR_ISOLATED_ROUTER_SOURCE'
