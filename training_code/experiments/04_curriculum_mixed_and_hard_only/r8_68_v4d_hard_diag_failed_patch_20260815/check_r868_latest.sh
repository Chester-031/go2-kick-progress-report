#!/usr/bin/env bash
set -u
ROOT="/scratch/lj2634/runs/go2_soccer_r8_68_v4d_hard_diag"
PTR="$ROOT/latest_r868.txt"
echo "============================================================"
echo " R8.68 V4D HARD DIAGNOSTIC CHECK"
echo "============================================================"
echo "TIME=$(date -Is)"
echo
ACTIVE="$(squeue -h -u "$USER" -o '%A|%j|%T|%M|%R' | awk -F'|' '$2 ~ /^r868-diag/ {print}')"
if [[ -n "$ACTIVE" ]]; then
  echo "ACTIVE=$ACTIVE"
else
  echo "ACTIVE=NONE"
fi
if [[ ! -s "$PTR" ]]; then
  echo "POINTER=ABSENT"
  echo "VERDICT=R868_WAITING_FOR_SUBMISSION_OR_COMPLETION"
else
  RUN="$(tr -d '\r\n' < "$PTR")"
  echo "RUN=$RUN"
  [[ -f "$RUN/contract.txt" ]] && cat "$RUN/contract.txt"
  echo "================ MARKERS ================"
  grep -E '\[R868_MARKER\]|PROCESS_RC|Traceback|RuntimeError|ModuleNotFoundError|Timeout' "$RUN/worker.log" || true
  echo "================ STACK TAIL ============="
  grep -nE 'Stack (most recent call last)|File .*train|File .*debug|File .*metric' "$RUN/worker.log" | tail -n 80 || true
  echo "================ OUTPUTS ================="
  find "$RUN/summaries" "$RUN/trajectories" -type f -size +0c -printf 'READY %p %s bytes\n' 2>/dev/null || true
  if [[ -s "$RUN/SHA256SUMS" ]]; then (cd "$RUN" && sha256sum -c SHA256SUMS); fi
  echo "TRAINING=FALSE"
  echo "VIDEO_GENERATION=FALSE"
  echo "STATE=HARD_DIAG_COLLECTED"
  echo "VERDICT=R868_READY_FOR_ROOT_CAUSE"
fi
