#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
echo '============================================================'
echo ' R8.60 ISOLATED ACTION-OWNER SMOKE CHECKER'
echo '============================================================'
echo "TIME=$(date -Is)"
echo "ROOT=$ROOT"
python "$ROOT/run_r860_router_smoke.py"
test -s "$ROOT/r860_smoke_summary.json"
python - "$ROOT/r860_smoke_summary.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
assert d['training'] is False
assert d['parent_mutation'] is False
assert d['exclusive_action_owner'] is True
assert d['single_low_env_step_per_tick'] is True
assert d['reward_contract']=={'goal':1,'own_goal':-1,'other':0}
owners=[x['owner'] for x in d['records']]
assert 'r858_kick' in owners and 'r855_recovery' in owners and 'v4d_locomotion' in owners
print('TRAINING=FALSE')
print('PARENT_MUTATION=FALSE')
print('REWARD_CONTRACT=goal:+1,own_goal:-1,other:0')
print('OWNERS='+','.join(owners))
print('STATE=SMOKE_PASS')
print('VERDICT=R860_READY_FOR_ISAAC_ROUTER_INTEGRATION')
PY
