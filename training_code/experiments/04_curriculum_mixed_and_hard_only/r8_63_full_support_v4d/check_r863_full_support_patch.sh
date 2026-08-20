#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="/scratch/lj2634/runs/go2_soccer_r8_63_full_support_v4d"
POINTER="$ROOT/latest_r863_candidate.txt"
[[ -s "$POINTER" ]] || { echo VERDICT=R863_CANDIDATE_NOT_READY; exit 2; }
RUN="$(tr -d '\r\n' < "$POINTER")"
echo '============================================================'
echo ' R8.63 FULL-SUPPORT V4D PATCH CHECKER'
echo '============================================================'
echo "TIME=$(date -Is)"
echo "RUN=$RUN"
cat "$RUN/evidence/R863_STATUS.txt"
python - "$RUN/evidence/train_v4d_full_support.patch_report.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
for k in ('progressive_difficulty_removed','difficulty_observation_fixed','reset_distribution','reward_contract','low_step_contract','next_step'):
 print(k.upper()+'='+json.dumps(d[k],sort_keys=True))
assert d['progressive_difficulty_removed'] is True
assert d['difficulty_observation_fixed'] == 1.0
assert d['reward_contract']=={'goal':1,'own_goal':-1,'other':0}
PY
grep -n "R8.63 full-support reset\|difficulty_tensor = torch.ones\|result = self.low_env.step" \
  "$RUN/source/train_v4d_full_support.py"
(cd "$RUN" && sha256sum -c SHA256SUMS)
WRITABLE="$(find "$RUN" -type f -perm /222 -print -quit)"
[[ -z "$WRITABLE" ]] || { echo "WRITABLE_FILE=$WRITABLE"; echo VERDICT=R863_CANDIDATE_WRITABLE; exit 3; }
echo 'TRAINING=FALSE'
echo 'PARENT_MUTATION=FALSE'
echo 'STATE=FULL_SUPPORT_SOURCE_CANDIDATE'
echo 'VERDICT=R863_READY_FOR_FROZEN_FULL_SUPPORT_COVERAGE_SCAN'
