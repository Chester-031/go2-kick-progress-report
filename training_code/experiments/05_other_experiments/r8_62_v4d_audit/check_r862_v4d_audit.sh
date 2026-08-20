#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="/scratch/lj2634/runs/go2_soccer_r8_62_v4d_audit"
POINTER="$ROOT/latest_v4d_audit.txt"
[[ -s "$POINTER" ]] || { echo VERDICT=R862_AUDIT_NOT_READY; exit 2; }
RUN="$(tr -d '\r\n' < "$POINTER")"
echo '============================================================'
echo ' R8.62 V4D CONTRACT AND COVERAGE AUDIT CHECKER'
echo '============================================================'
echo "TIME=$(date -Is)"
echo "RUN=$RUN"
cat "$RUN/evidence/R862_STATUS.txt"
python - "$RUN/evidence/v4d_audit.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
print('CONTRACT_CHECKS='+json.dumps(d['contract_checks'],sort_keys=True))
for key in ('low_env_step','reset','ball_init','goal_init','curriculum','reward','agent_row','command_mapping'):
    xs=d['sites'][key]
    print(key.upper()+'_COUNT='+str(len(xs)))
    for x in xs[:12]: print(' ',x['line'],x['text'])
print('REQUIRED_FOLLOWUP='+json.dumps(d['required_followup']))
PY
(cd "$RUN" && sha256sum -c SHA256SUMS)
WRITABLE="$(find "$RUN" -type f -perm /222 -print -quit)"
[[ -z "$WRITABLE" ]] || { echo "WRITABLE_FILE=$WRITABLE"; echo VERDICT=R862_AUDIT_WRITABLE; exit 3; }
echo 'TRAINING=FALSE'
echo 'PARENT_MUTATION=FALSE'
echo 'STATE=V4D_CONTRACT_AUDITED'
echo 'VERDICT=R862_READY_FOR_FULL_SUPPORT_COVERAGE_SCAN'
