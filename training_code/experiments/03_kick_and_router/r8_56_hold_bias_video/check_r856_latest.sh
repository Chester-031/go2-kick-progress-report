#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"; ROOT="/scratch/lj2634/runs/go2_soccer_r8_56_hold_bias"; NAME="r856-scan"; JOB="$(squeue -h -u "$USER" -n "$NAME" -o '%A|%V' 2>/dev/null|sort -t'|' -k2,2r|awk -F'|' 'NR==1{print $1}')"; if [[ -z "$JOB" ]]; then START="$(date -d '14 days ago' +%F)"; JOB="$(sacct -nX -u "$USER" -S "$START" --name="$NAME" --format=JobIDRaw,Submit --parsable2 2>/dev/null|awk -F'|' '$1~/^[0-9]+$/{print $1"|"$2}'|sort -t'|' -k2,2r|awk -F'|' 'NR==1{print $1}')"; fi; [[ -n "$JOB" ]]||{ echo VERDICT=NO_R856_JOB; exit 2; }
LOG="$PROJECT/slurm-r856-${JOB}.out"; RUN="$ROOT/r856_scan_${JOB}"; SUMMARY="$RUN/output/r856_summary.json"; echo '============================================================'; echo ' R8.56 FL HOLD-BIAS CHECKER'; echo '============================================================'; echo "TIME=$(date -Is)"; echo "JOB=$JOB"; echo "LOG=$LOG"; echo "RUN=$RUN"; echo
echo '==================== STATUS ===================='; squeue -j "$JOB" -o '%.18i %.18j %.24a %.16P %.12T %.12M %.12l %.6D %R' 2>/dev/null||true; sacct -X -j "$JOB" --format=JobID,JobName%20,Account%24,Partition%16,State%18,ExitCode,Elapsed,Timelimit,AllocTRES%45,MaxRSS,NodeList 2>/dev/null||true; echo
echo '==================== CONTRACT ===================='; [[ -s "$RUN/contract.txt" ]]&&cat "$RUN/contract.txt"||echo CONTRACT_NOT_READY; echo
echo '==================== PROGRESS ===================='; [[ -s "$LOG" ]]&&grep -En 'R856_PATCH_OK|R856_JOINTS|R856_SCAN|R856_CONFIRM|R856_RESULT|R856_HOLD_SCAN_OK|R856_WORKER_OK|R856_FATAL|R8_4_FATAL|R8_FATAL' "$LOG"|tail -n 180||echo LOG_NOT_READY; echo
echo '==================== LATEST LOG ===================='; [[ -s "$LOG" ]]&&tail -n 130 "$LOG"||echo NO_LOG; echo
echo '==================== FATAL ===================='; FATAL=''; [[ -s "$LOG" ]]&&FATAL="$(grep -Ein 'traceback|keyerror|runtimeerror|R856_FATAL|R8_4_FATAL|R8_FATAL|cuda.*error|disk quota exceeded|out of memory|segmentation fault|killed|bus error|illegal memory|assertionerror|(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)' "$LOG" 2>/dev/null||true)"; if [[ -n "$FATAL" ]]; then printf '%s\n' "$FATAL"; echo FATAL_SCAN=FAILED; else echo FATAL_SCAN=CLEAN; fi; echo
echo '==================== ARTIFACTS ===================='; for f in contract.txt train_r856_hold_bias.py output/r856_hold_winner.json output/r856_summary.json SHA256SUMS; do [[ -s "$RUN/$f" ]]&&stat --printf='READY %n %s bytes %y\n' "$RUN/$f"||echo "NOT_READY $RUN/$f"; done; echo
if [[ -s "$SUMMARY" ]]; then echo '==================== METRICS ===================='; python - "$SUMMARY" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
for k in ('version','training','actual_contact','network_frozen','grid_values','episodes_per_case','confirm_episodes','winner_file','accepted','verdict'): print(f'{k.upper()}={d.get(k)}')
print('WINNERS')
for s,w in d.get('winners',{}).items(): print(' strength',s,'case',w['case'],'bias',w['bias'],'real',f"{w['real_rate']:.4f}",'touchdown|real',f"{w['touchdown_given_real']:.4f}",'recovery3|real',f"{w['recovery_given_real']:.4f}",'wrong',f"{w['wrong_rate']:.4f}",'body',f"{w['body_rate']:.4f}")
print('TOP5_PER_STRENGTH')
for s,rows in d.get('scans',{}).items():
 print(' strength',s)
 for x in sorted(rows,key=lambda z:(z['recovery_given_real'],z['touchdown_given_real']),reverse=True)[:5]: print('  case',x['case'],'bias',x['bias'],'real',f"{x['real_rate']:.4f}",'td',f"{x['touchdown_given_real']:.4f}",'rec3',f"{x['recovery_given_real']:.4f}")
print('CONFIRM')
for x in d.get('confirm',[]): print(' strength',x['strength'],'bias',x['bias'],'episodes',x['episodes'],'real',f"{x['real_rate']:.4f}",'td',f"{x['touchdown_given_real']:.4f}",'rec3',f"{x['recovery_given_real']:.4f}",'wrong',f"{x['wrong_rate']:.4f}",'body',f"{x['body_rate']:.4f}")
PY
fi; echo
echo '==================== HASH ===================='; [[ -s "$RUN/SHA256SUMS" ]]&&(cd "$RUN"&&sha256sum -c SHA256SUMS)||echo HASH_NOT_READY; echo
STATE="$(sacct -nX -j "$JOB" --format=State 2>/dev/null|awk 'NF{print $1;exit}')"; STATE="${STATE:-UNKNOWN}"; echo "STATE=$STATE"; if [[ -n "$FATAL" ]]; then echo VERDICT=FAILED_FATAL; elif [[ "$STATE" == PENDING* ]]; then echo VERDICT=PENDING; elif [[ "$STATE" == RUNNING* ]]; then echo VERDICT=RUNNING_CLEAN; elif [[ "$STATE" == COMPLETED* && ! -s "$SUMMARY" ]]; then echo VERDICT=FAILED_MISSING_SUMMARY; elif [[ "$STATE" == COMPLETED* ]]; then python - "$SUMMARY" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8')); print('VERDICT=COMPLETED_READY_TO_BUILD_VIDEO' if d.get('accepted') else 'VERDICT=COMPLETED_KEEP_R856_NOT_READY')
PY
else echo VERDICT=CHECK_STATE_AND_LOG; fi
