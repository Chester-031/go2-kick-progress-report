#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"; ROOT="/scratch/lj2634/runs/go2_soccer_r8_55_touchdown_hold"; NAME="r855-hold"
JOB="$(squeue -h -u "$USER" -n "$NAME" -o '%A|%V' 2>/dev/null|sort -t'|' -k2,2r|awk -F'|' 'NR==1{print $1}')"
if [[ -z "$JOB" ]]; then START="$(date -d '14 days ago' +%F)"; JOB="$(sacct -nX -u "$USER" -S "$START" --name="$NAME" --format=JobIDRaw,Submit --parsable2 2>/dev/null|awk -F'|' '$1~/^[0-9]+$/{print $1"|"$2}'|sort -t'|' -k2,2r|awk -F'|' 'NR==1{print $1}')"; fi
[[ -n "$JOB" ]]||{ echo VERDICT=NO_R855_JOB; exit 2; }
LOG="$PROJECT/slurm-r855-${JOB}.out"; RUN="$ROOT/r855_hold_${JOB}"; SUMMARY="$RUN/output/r855_summary.json"
echo '============================================================'; echo ' R8.55 TOUCHDOWN-HOLD CHECKER'; echo '============================================================'; echo "TIME=$(date -Is)"; echo "JOB=$JOB"; echo "LOG=$LOG"; echo "RUN=$RUN"; echo
echo '==================== STATUS ===================='; squeue -j "$JOB" -o '%.18i %.18j %.24a %.16P %.12T %.12M %.12l %.6D %R' 2>/dev/null||true; sacct -X -j "$JOB" --format=JobID,JobName%20,Account%24,Partition%16,State%18,ExitCode,Elapsed,Timelimit,AllocTRES%45,MaxRSS,NodeList 2>/dev/null||true; echo
echo '==================== CONTRACT ===================='; [[ -s "$RUN/contract.txt" ]]&&cat "$RUN/contract.txt"||echo CONTRACT_NOT_READY; echo
echo '==================== PROGRESS ===================='; [[ -s "$LOG" ]]&&grep -En 'R855_PATCH_OK|R855_BASELINE|R855_UPDATE|R855_CONFIRM|R855_RESULT|R855_TOUCHDOWN_HOLD_OK|R855_WORKER_OK|R855_FATAL|R8_4_FATAL|R8_FATAL' "$LOG"|tail -n 280||echo LOG_NOT_READY; echo
echo '==================== LATEST LOG ===================='; [[ -s "$LOG" ]]&&tail -n 140 "$LOG"||echo NO_LOG; echo
echo '==================== FATAL ===================='; FATAL=''; [[ -s "$LOG" ]]&&FATAL="$(grep -Ein 'traceback|keyerror|runtimeerror|R855_FATAL|R8_4_FATAL|R8_FATAL|cuda.*error|disk quota exceeded|out of memory|segmentation fault|killed|bus error|illegal memory|assertionerror|(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)' "$LOG" 2>/dev/null||true)"; if [[ -n "$FATAL" ]]; then printf '%s\n' "$FATAL"; echo FATAL_SCAN=FAILED; else echo FATAL_SCAN=CLEAN; fi; echo
echo '==================== ARTIFACTS ===================='; for f in contract.txt train_r855_touchdown_hold.py output/recovery64_latest.pt output/recovery64_final.pt output/r855_summary.json SHA256SUMS; do [[ -s "$RUN/$f" ]]&&stat --printf='READY %n %s bytes %y\n' "$RUN/$f"||echo "NOT_READY $RUN/$f"; done; echo
if [[ -s "$SUMMARY" ]]; then
 echo '==================== METRICS ===================='
 python - "$SUMMARY" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
for k in ('version','training','actual_contact','kick_frozen','r854_warm_start','touchdown_latch','curriculum','dense_reward','updates','rollout_steps','strength_monotonic','checkpoint','accepted','verdict'):
 print(f'{k.upper()}={d.get(k)}')
def rows(title,items):
 print(title)
 for x in items:
  print(' strength',x['strength'],'episodes',x['episodes'],'real',f"{x['real_rate']:.4f}",'wrong',f"{x['wrong_rate']:.4f}",'body',f"{x['body_rate']:.4f}",'return|real',f"{x['return_given_real']:.4f}",'contact|real',f"{x['contact_given_real']:.4f}",'touchdown|real',f"{x['touchdown_given_real']:.4f}",'latch|real',f"{x['latch_given_real']:.4f}",'recovery3|real',f"{x['recovery_given_real']:.4f}",'peakV',f"{x['mean_peak_speed_mps']:.4f}",'disp',f"{x['mean_disp_m']:.4f}")
rows('LATCHED_BASELINE',d.get('baseline_latched',[]))
print('LATEST_UPDATES (real/latch/recovery-step)')
for x in d.get('history',[])[-10:]:
 print(' update',x['update'],'stage',x['stage'],'by_strength',x['by_strength'],'actorLoss',f"{x['actor_loss']:.5f}",'valueLoss',f"{x['value_loss']:.5f}",'anchor',f"{x['anchor']:.7f}")
rows('FINAL_CONFIRM',d.get('strength_confirm',[]))
PY
fi
echo
echo '==================== HASH ===================='; [[ -s "$RUN/SHA256SUMS" ]]&&(cd "$RUN"&&sha256sum -c SHA256SUMS)||echo HASH_NOT_READY; echo
STATE="$(sacct -nX -j "$JOB" --format=State 2>/dev/null|awk 'NF{print $1;exit}')"; STATE="${STATE:-UNKNOWN}"; echo "STATE=$STATE"
if [[ -n "$FATAL" ]]; then echo VERDICT=FAILED_FATAL
elif [[ "$STATE" == PENDING* ]]; then echo VERDICT=PENDING
elif [[ "$STATE" == RUNNING* ]]; then echo VERDICT=RUNNING_CLEAN
elif [[ "$STATE" == COMPLETED* && ! -s "$SUMMARY" ]]; then echo VERDICT=FAILED_MISSING_SUMMARY
elif [[ "$STATE" == COMPLETED* ]]; then
 python - "$SUMMARY" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
print('VERDICT=COMPLETED_READY_FOR_4_SKILL_ROUTER' if d.get('accepted') else 'VERDICT=COMPLETED_KEEP_R855_NOT_ROUTER_READY')
PY
else echo VERDICT=CHECK_STATE_AND_LOG
fi
