#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"; ROOT="/scratch/lj2634/runs/go2_soccer_r8_52_gain_recovery_scan"; NAME="r852-scan"
JOB="$(squeue -h -u "$USER" -n "$NAME" -o '%A|%V' 2>/dev/null|sort -t'|' -k2,2r|awk -F'|' 'NR==1{print $1}')"
if [[ -z "$JOB" ]]; then START="$(date -d '14 days ago' +%F)"; JOB="$(sacct -nX -u "$USER" -S "$START" --name="$NAME" --format=JobIDRaw,Submit --parsable2 2>/dev/null|awk -F'|' '$1~/^[0-9]+$/{print $1"|"$2}'|sort -t'|' -k2,2r|awk -F'|' 'NR==1{print $1}')"; fi
[[ -n "$JOB" ]] || { echo VERDICT=NO_R852_JOB; exit 2; }
LOG="$PROJECT/slurm-r852-${JOB}.out"; RUN="$ROOT/r852_scan_${JOB}"; SUMMARY="$RUN/output/r852_summary.json"
echo "JOB=$JOB"; echo "LOG=$LOG"; echo "RUN=$RUN"
squeue -j "$JOB" -o '%.18i %.18j %.24a %.16P %.12T %.12M %.12l %R' 2>/dev/null||true
sacct -X -j "$JOB" --format=JobID,Account%24,Partition,State,ExitCode,Elapsed,AllocTRES%45,MaxRSS,NodeList 2>/dev/null||true
echo '===== CONTRACT ====='; [[ -s "$RUN/contract.txt" ]]&&cat "$RUN/contract.txt"||echo NOT_READY
echo '===== PROGRESS ====='; [[ -s "$LOG" ]]&&grep -En 'R852_PATCH_OK|R852_RESULT|R852_GAIN_RECOVERY_SCAN_OK|R852_WORKER_OK|R852_FATAL' "$LOG"|tail -n 100||true
echo '===== LOG TAIL ====='; [[ -s "$LOG" ]]&&tail -n 100 "$LOG"||echo LOG_NOT_READY
echo '===== FATAL ====='; FATAL=""; [[ -s "$LOG" ]]&&FATAL="$(grep -Ein 'traceback|runtimeerror|keyerror|R8_4_FATAL|R8_FATAL|R852_FATAL|cuda.*error|physx.*error|disk quota exceeded|out of memory|segmentation fault|killed|assertionerror|(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)' "$LOG"||true)"; [[ -n "$FATAL" ]]&&{ printf '%s\n' "$FATAL"; echo FATAL_SCAN=FAILED; }||echo FATAL_SCAN=CLEAN
echo '===== ARTIFACTS ====='; for f in contract.txt probe_r852_scan.py output/r852_summary.json output/r852_episode_rows.jsonl SHA256SUMS; do [[ -s "$RUN/$f" ]]&&stat --printf='READY %n %s bytes\n' "$RUN/$f"||echo "NOT_READY $RUN/$f"; done
if [[ -s "$SUMMARY" ]]; then
python - "$SUMMARY" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
print('VERSION=',d['version'],sep=''); print('GEOMETRY=',d['geometry'],sep=''); print('TRAINING=',d['training'],sep='')
print('GAIN_MONOTONIC_BY_DELAY')
for delay,x in d['strength_by_delay'].items(): print(' delay',delay,'speeds',','.join(f'{v:.4f}' for v in x['speeds']),'monotonic',x['strict_monotonic'])
print('TOP_RECOVERY_CASES')
ranked=sorted(d['cases'],key=lambda x:(x['recovery_given_real'],x['all_now_given_real'],x['touchdown_given_real'],x['mean_peak_speed_mps']),reverse=True)
for x in ranked[:8]: print(' gain',x['gain'],'delay',x['delay_steps'],'real',f"{x['real_rate']:.4f}",'speed',f"{x['mean_peak_speed_mps']:.4f}",'disp',f"{x['mean_directional_disp_m']:.4f}",'candidate',f"{x['rec_candidate_given_real']:.4f}",'return',f"{x['return_given_real']:.4f}",'contact',f"{x['contact_given_real']:.4f}",'support4',f"{x['support4_given_real']:.4f}",'touchdown',f"{x['touchdown_given_real']:.4f}",'all',f"{x['all_now_given_real']:.4f}",'recovery',f"{x['recovery_given_real']:.4f}")
print('VERDICT=',d['verdict'],sep='')
PY
fi
echo '===== HASH ====='; [[ -s "$RUN/SHA256SUMS" ]]&&(cd "$RUN"&&sha256sum -c SHA256SUMS)||echo HASH_NOT_READY
STATE="$(sacct -nX -j "$JOB" --format=State 2>/dev/null|awk 'NF{print $1;exit}')"; echo "STATE=${STATE:-UNKNOWN}"
if [[ -n "$FATAL" ]]; then echo VERDICT=FAILED_FATAL; elif [[ "$STATE" == RUNNING* ]]; then echo VERDICT=RUNNING_CLEAN; elif [[ "$STATE" == COMPLETED* && -s "$SUMMARY" ]]; then echo VERDICT=COMPLETED_READY_FOR_R853_DECISION; else echo VERDICT=CHECK_STATE_AND_LOG; fi

