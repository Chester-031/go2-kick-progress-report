#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"; RUN_ROOT="/scratch/lj2634/runs/go2_soccer_r8_54_strength_conditioned_recovery"; JOB_NAME="r854-rec64"
JOB="$(squeue -h -u "$USER" -n "$JOB_NAME" -o '%A|%V' 2>/dev/null | sort -t'|' -k2,2r | awk -F'|' 'NR==1{print $1}')"; if [[ -z "$JOB" ]]; then START="$(date -d '14 days ago' +%F)"; JOB="$(sacct -nX -u "$USER" -S "$START" --name="$JOB_NAME" --format=JobIDRaw,Submit --parsable2 2>/dev/null | awk -F'|' '$1~/^[0-9]+$/{print $1"|"$2}' | sort -t'|' -k2,2r | awk -F'|' 'NR==1{print $1}')"; fi; [[ -n "$JOB" ]] || { echo VERDICT=NO_R854_JOB; exit 2; }
LOG="$PROJECT/slurm-r854-${JOB}.out"; RUN_DIR="$RUN_ROOT/r854_recovery64_${JOB}"; SUMMARY="$RUN_DIR/output/r854_summary.json"
echo '============================================================'; echo ' R8.54 STRENGTH-CONDITIONED RECOVERY64 CHECKER'; echo '============================================================'; echo "TIME=$(date -Is)"; echo "JOB=$JOB"; echo "LOG=$LOG"; echo "RUN_DIR=$RUN_DIR"; echo
echo '==================== STATUS ===================='; squeue -j "$JOB" -o '%.18i %.18j %.24a %.16P %.12T %.12M %.12l %.6D %R' 2>/dev/null || true; sacct -X -j "$JOB" --format=JobID,JobName%20,Account%24,Partition%16,State%18,ExitCode,Elapsed,Timelimit,AllocTRES%45,MaxRSS,NodeList 2>/dev/null || true; echo
echo '==================== CONTRACT ===================='; if [[ -s "$RUN_DIR/contract.txt" ]]; then cat "$RUN_DIR/contract.txt"; else echo CONTRACT_NOT_READY; fi; echo
echo '==================== PROGRESS ===================='; if [[ -s "$LOG" ]]; then grep -En 'R854_PATCH_OK|R854_UPDATE|R854_CONFIRM|R854_RESULT|R854_STRENGTH_RECOVERY_OK|R854_WORKER_OK|R854_FATAL|R8_4_FATAL|R8_FATAL' "$LOG" | tail -n 260 || true; else echo LOG_NOT_READY; fi; echo
echo '==================== LATEST LOG ===================='; if [[ -s "$LOG" ]]; then tail -n 130 "$LOG"; else echo NO_LOG; fi; echo
echo '==================== FATAL ===================='; FATAL=''; if [[ -s "$LOG" ]]; then FATAL="$(grep -Ein 'traceback|keyerror|runtimeerror|R854_FATAL|R8_4_FATAL|R8_FATAL|cuda.*error|disk quota exceeded|out of memory|segmentation fault|killed|bus error|illegal memory|assertionerror|(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)' "$LOG" 2>/dev/null || true)"; fi; if [[ -n "$FATAL" ]]; then printf '%s\n' "$FATAL"; echo FATAL_SCAN=FAILED; else echo FATAL_SCAN=CLEAN; fi; echo
echo '==================== ARTIFACTS ===================='; for file in contract.txt train_r854_strength_recovery.py output/recovery64_latest.pt output/recovery64_final.pt output/r854_summary.json SHA256SUMS; do if [[ -s "$RUN_DIR/$file" ]]; then stat --printf='READY %n %s bytes %y\n' "$RUN_DIR/$file"; else echo "NOT_READY $RUN_DIR/$file"; fi; done; echo
if [[ -s "$SUMMARY" ]]; then echo '==================== METRICS ===================='; python - "$SUMMARY" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
for k in ('version','training','actual_contact','kick_frozen','r853_warm_start','strength_observation','curriculum','dense_reward','geometry','strength_gain','updates','rollout_steps','strength_monotonic','checkpoint','accepted','verdict'): print(f'{k.upper()}={d.get(k)}')
print('LATEST_UPDATES (per strength real/all-step/recovery-step)')
for x in d.get('history',[])[-10:]: print(' update',x['update'],'stage',x['stage'],'by_strength',x['by_strength'],'actorLoss',f"{x['actor_loss']:.5f}",'valueLoss',f"{x['value_loss']:.5f}",'anchor',f"{x['anchor']:.7f}")
print('STRENGTH_CONFIRM')
for x in d.get('strength_confirm',[]): print(' strength',x['strength'],'episodes',x['episodes'],'clean',f"{x['clean_rate']:.4f}",'real',f"{x['real_rate']:.4f}",'wrong',f"{x['wrong_rate']:.4f}",'body',f"{x['body_rate']:.4f}",'return|real',f"{x['return_given_real']:.4f}",'contact|real',f"{x['contact_given_real']:.4f}",'support4|real',f"{x['support4_given_real']:.4f}",'touchdown|real',f"{x['touchdown_given_real']:.4f}",'all|real',f"{x['all_now_given_real']:.4f}",'recovery3|real',f"{x['recovery_given_real']:.4f}",'peakV',f"{x['mean_peak_speed_mps']:.4f}",'disp',f"{x['mean_disp_m']:.4f}")
PY
fi; echo
echo '==================== HASH ===================='; if [[ -s "$RUN_DIR/SHA256SUMS" ]]; then (cd "$RUN_DIR" && sha256sum -c SHA256SUMS); else echo HASH_NOT_READY; fi; echo
STATE="$(sacct -nX -j "$JOB" --format=State 2>/dev/null | awk 'NF{print $1;exit}')"; STATE="${STATE:-UNKNOWN}"; echo "STATE=$STATE"; if [[ -n "$FATAL" ]]; then echo VERDICT=FAILED_FATAL; elif [[ "$STATE" == PENDING* ]]; then echo VERDICT=PENDING; elif [[ "$STATE" == RUNNING* ]]; then echo VERDICT=RUNNING_CLEAN; elif [[ "$STATE" == COMPLETED* && ! -s "$SUMMARY" ]]; then echo VERDICT=FAILED_MISSING_SUMMARY; elif [[ "$STATE" == COMPLETED* ]]; then python - "$SUMMARY" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8')); print('VERDICT=COMPLETED_READY_FOR_4_SKILL_ROUTER' if d.get('accepted') else 'VERDICT=COMPLETED_KEEP_R854_NOT_ROUTER_READY')
PY
else echo VERDICT=CHECK_STATE_AND_LOG; fi
