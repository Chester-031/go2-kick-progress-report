#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
RUN_ROOT="/scratch/lj2634/runs/go2_soccer_r8_51_kick64_true_contact"
JOB_NAME="r851-kick64"

JOB="$(squeue -h -u "$USER" -n "$JOB_NAME" -o '%A|%V' 2>/dev/null | sort -t'|' -k2,2r | awk -F'|' 'NR==1{print $1}')"
if [[ -z "$JOB" ]]; then
  START="$(date -d '14 days ago' +%F)"
  JOB="$(sacct -nX -u "$USER" -S "$START" --name="$JOB_NAME" --format=JobIDRaw,Submit --parsable2 2>/dev/null | awk -F'|' '$1~/^[0-9]+$/{print $1"|"$2}' | sort -t'|' -k2,2r | awk -F'|' 'NR==1{print $1}')"
fi
if [[ -z "$JOB" ]]; then echo "VERDICT=NO_R851_JOB"; exit 2; fi

LOG="$PROJECT/slurm-r851-${JOB}.out"
RUN_DIR="$RUN_ROOT/r851_kick64_${JOB}"
SUMMARY="$RUN_DIR/output/r851_summary.json"

echo "============================================================"
echo " R8.51 KICK64 TRUE-CONTACT CHECKER"
echo "============================================================"
echo "TIME=$(date -Is)"
echo "JOB=$JOB"
echo "LOG=$LOG"
echo "RUN_DIR=$RUN_DIR"
echo

echo "==================== STATUS ===================="
squeue -j "$JOB" -o '%.18i %.18j %.24a %.16P %.12T %.12M %.12l %.6D %R' 2>/dev/null || true
sacct -X -j "$JOB" --format=JobID,JobName%20,Account%24,Partition%16,State%18,ExitCode,Elapsed,Timelimit,AllocTRES%45,MaxRSS,NodeList 2>/dev/null || true
echo

echo "==================== CONTRACT ===================="
if [[ -s "$RUN_DIR/contract.txt" ]]; then cat "$RUN_DIR/contract.txt"; else echo CONTRACT_NOT_READY; fi
echo

echo "==================== PROGRESS ===================="
if [[ -s "$LOG" ]]; then
  grep -En 'R851_PATCH_OK|R851_UPDATE|R851_CONFIRM|R851_RESULT|R851_KICK64_OK|R851_WORKER_OK|R851_FATAL' "$LOG" | tail -n 160 || true
else echo LOG_NOT_READY; fi
echo

echo "==================== LATEST LOG ===================="
if [[ -s "$LOG" ]]; then tail -n 100 "$LOG"; else echo NO_LOG; fi
echo

echo "==================== FATAL ===================="
FATAL=""
if [[ -s "$LOG" ]]; then
  FATAL="$(grep -Ein 'traceback|runtimeerror|R851_FATAL|cuda.*error|physx.*error|disk quota exceeded|out of memory|segmentation fault|killed|assertionerror|(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)' "$LOG" 2>/dev/null || true)"
fi
if [[ -n "$FATAL" ]]; then printf '%s\n' "$FATAL"; echo FATAL_SCAN=FAILED; else echo FATAL_SCAN=CLEAN; fi
echo

echo "==================== ARTIFACTS ===================="
for file in contract.txt train_r851_kick64.py output/kick64_latest.pt output/kick64_final.pt output/r851_summary.json SHA256SUMS; do
  if [[ -s "$RUN_DIR/$file" ]]; then stat --printf='READY %n %s bytes %y\n' "$RUN_DIR/$file"; else echo "NOT_READY $RUN_DIR/$file"; fi
done
echo

if [[ -s "$SUMMARY" ]]; then
  echo "==================== METRICS ===================="
  python - "$SUMMARY" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding='utf-8'))
print('VERSION=',d.get('version'),sep='')
print('OBS_CONTRACT=',d.get('obs_contract'),sep='')
print('ACTUAL_CONTACT=',d.get('actual_contact'),sep='')
print('FROZEN_CHAIN=',d.get('frozen_load_lift_swing_recovery'),sep='')
print('TOP_GEOMETRY=',d.get('top_geometry'),sep='')
print('UPDATES=',d.get('updates'),sep='')
print('ROLLOUT_STEPS=',d.get('rollout_steps'),sep='')
print('STRENGTH_MONOTONIC=',d.get('strength_monotonic'),sep='')
for x in d.get('strength_confirm',[]):
 print('STRENGTH',x['strength'],'episodes',x['episodes'],'selected',f"{x['selected_rate']:.4f}",'clean',f"{x['clean_rate']:.4f}",'real',f"{x['real_strike_rate']:.4f}",'recovery|real',f"{x['recovery_given_real']:.4f}",'wrong',f"{x['wrong_rate']:.4f}",'body',f"{x['body_rate']:.4f}",'peakV',f"{x['mean_peak_direction_speed_mps']:.4f}",'disp',f"{x['mean_directional_disp_m']:.4f}")
print('CHECKPOINT=',d.get('checkpoint'),sep='')
print('ACCEPTED=',d.get('accepted'),sep='')
print('VERDICT=',d.get('verdict'),sep='')
PY
fi
echo

echo "==================== HASH ===================="
if [[ -s "$RUN_DIR/SHA256SUMS" ]]; then (cd "$RUN_DIR" && sha256sum -c SHA256SUMS); else echo HASH_NOT_READY; fi
echo

STATE="$(sacct -nX -j "$JOB" --format=State 2>/dev/null | awk 'NF{print $1;exit}')"; STATE="${STATE:-UNKNOWN}"
echo "STATE=$STATE"
if [[ -n "$FATAL" ]]; then echo VERDICT=FAILED_FATAL
elif [[ "$STATE" == PENDING* ]]; then echo VERDICT=PENDING
elif [[ "$STATE" == RUNNING* ]]; then echo VERDICT=RUNNING_CLEAN
elif [[ "$STATE" == COMPLETED* && ! -s "$SUMMARY" ]]; then echo VERDICT=FAILED_MISSING_SUMMARY
elif [[ "$STATE" == COMPLETED* ]]; then
  python - "$SUMMARY" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
print('VERDICT=COMPLETED_READY_FOR_ROUTER' if d.get('accepted') else 'VERDICT=COMPLETED_KEEP_TRAINING_NOT_ROUTER_READY')
PY
else echo VERDICT=CHECK_STATE_AND_LOG; fi

