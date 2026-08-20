#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"; ROOT="/scratch/lj2634/runs/go2_soccer_r8_58_targeted_hold"; NAME="r858-hold"
JOB="$(squeue -h -u "$USER" -n "$NAME" -o '%A|%V' 2>/dev/null | sort -t'|' -k2,2r | awk -F'|' 'NR==1{print $1}')"
if [[ -z "$JOB" ]]; then START="$(date -d '14 days ago' +%F)"; JOB="$(sacct -nX -u "$USER" -S "$START" --name="$NAME" --format=JobIDRaw,Submit --parsable2 2>/dev/null | awk -F'|' '$1~/^[0-9]+$/{print $1"|"$2}' | sort -t'|' -k2,2r | awk -F'|' 'NR==1{print $1}')"; fi
[[ -n "$JOB" ]] || { echo VERDICT=NO_R858_JOB; exit 2; }
LOG="$PROJECT/slurm-r858-${JOB}.out"; RUN="$ROOT/r858_hold_${JOB}"; SUMMARY="$RUN/output/r858_summary.json"
echo '============================================================'; echo ' R8.58 TARGETED HOLD CHECKER'; echo '============================================================'; echo "TIME=$(date -Is)"; echo "JOB=$JOB"; echo "LOG=$LOG"; echo "RUN=$RUN"; echo
echo '==================== STATUS ===================='; sacct -X -j "$JOB" --format=JobID,JobName%20,Account%24,Partition%16,State%18,ExitCode,Elapsed,Timelimit,AllocTRES%45,MaxRSS,NodeList 2>/dev/null || true
echo '==================== CONTRACT ===================='; [[ -s "$RUN/contract.txt" ]] && cat "$RUN/contract.txt" || echo CONTRACT_NOT_READY; echo
echo '==================== PROGRESS ===================='; [[ -s "$LOG" ]] && grep -En 'R858_PATCH_OK|R856_JOINTS|R858_SEMIFINAL|R858_CONFIRM|R858_RESULT|R858_TARGETED_HOLD_OK|R858_WORKER_OK|R858_FATAL|R8_4_FATAL|R8_FATAL' "$LOG" | tail -n 160 || echo LOG_NOT_READY; echo
echo '==================== LATEST LOG ===================='; [[ -s "$LOG" ]] && tail -n 120 "$LOG" || echo NO_LOG; echo
echo '==================== FATAL ===================='; FATAL=''; [[ -s "$LOG" ]] && FATAL="$(grep -Ein 'traceback|keyerror|runtimeerror|R858_FATAL|R8_4_FATAL|R8_FATAL|cuda.*error|disk quota exceeded|out of memory|segmentation fault|killed|assertionerror|(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)' "$LOG" 2>/dev/null || true)"; [[ -n "$FATAL" ]] && { printf '%s\n' "$FATAL"; echo FATAL_SCAN=FAILED; } || echo FATAL_SCAN=CLEAN; echo
echo '==================== ARTIFACTS ===================='; for f in contract.txt train_r858_targeted.py output/r858_hold_winner.json output/r858_summary.json SHA256SUMS; do [[ -s "$RUN/$f" ]] && stat --printf='READY %n %s bytes %y\n' "$RUN/$f" || echo "NOT_READY $RUN/$f"; done; echo
if [[ -s "$SUMMARY" ]]; then echo '==================== METRICS ===================='; python - "$SUMMARY" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
for k in ('version','training','network_frozen','reused_proven_chain','search_strengths','high_strength_frozen','candidate_count','screen_episodes_per_case','semifinal_topk','semifinal_episodes','final_episodes_per_condition','final_conditions','winner_file','accepted','verdict'): print(f'{k.upper()}={d.get(k)}')
print('WINNERS')
for s,w in d['winners'].items(): print(' strength',s,'controller',w['controller'])
print('FINAL_CONFIRM')
for s,rows in d['final_confirm'].items():
 for x in rows: print(' strength',s,'condition',('exact' if x['geometry_jitter_m']==0 else 'jitter4mm'),'episodes',x['episodes'],'real',f"{x['real_rate']:.4f}",'td',f"{x['touchdown_given_real']:.4f}",'rec3',f"{x['recovery_given_real']:.4f}",'retain',f"{x['retention_given_touchdown']:.4f}",'wrong',f"{x['wrong_rate']:.4f}",'body',f"{x['body_rate']:.4f}")
print('SEMIFINAL_TOP3')
for s,rows in d['semifinals'].items():
 print(' strength',s)
 for x in sorted(rows,key=lambda z:z['recovery_given_real'],reverse=True)[:3]: print('  ctrl',x['controller'],'real',f"{x['real_rate']:.4f}",'td',f"{x['touchdown_given_real']:.4f}",'rec3',f"{x['recovery_given_real']:.4f}")
PY
fi; echo
echo '==================== HASH ===================='; [[ -s "$RUN/SHA256SUMS" ]] && (cd "$RUN" && sha256sum -c SHA256SUMS) || echo HASH_NOT_READY
STATE="$(sacct -nX -j "$JOB" --format=State 2>/dev/null | awk 'NF{print $1;exit}')"; STATE="${STATE:-UNKNOWN}"; echo "STATE=$STATE"
if [[ -n "$FATAL" ]]; then echo VERDICT=FAILED_FATAL
elif [[ "$STATE" == PENDING* ]]; then echo VERDICT=PENDING
elif [[ "$STATE" == RUNNING* ]]; then echo VERDICT=RUNNING_CLEAN
elif [[ "$STATE" == COMPLETED* && ! -s "$SUMMARY" ]]; then echo VERDICT=FAILED_MISSING_SUMMARY
elif [[ "$STATE" == COMPLETED* ]]; then python -c 'import json,sys; d=json.load(open(sys.argv[1])); print("VERDICT=COMPLETED_READY_TO_BUILD_VIDEO" if d.get("accepted") else "VERDICT=COMPLETED_STOP_CONTROLLER_CLASS")' "$SUMMARY"
else echo VERDICT=CHECK_STATE_AND_LOG; fi
