#!/usr/bin/env bash
set -u
ROOT="/scratch/lj2634/runs/go2_soccer_r8_74b_v4d_fresh_no_curriculum_fix"
PTR="$ROOT/latest_r874.txt"
echo "============================================================"
echo " R8.74 V4D FRESH NO-CURRICULUM TRAINING CHECK"
echo "============================================================"
echo "TIME=$(date -Is)"
ACTIVE="$(squeue -h -u "$USER" -o '%A|%j|%T|%M|%R' | awk -F'|' '$2 ~ /^r874b-fresh/ {print}')"
echo "ACTIVE=${ACTIVE:-NONE}"
if [[ ! -s "$PTR" ]]; then
    echo "POINTER=ABSENT"
    echo "VERDICT=R874B_WAITING_FOR_SUBMISSION"
    exit 0
fi
RUN="$(tr -d '\r\n' < "$PTR")"
echo "RUN=$RUN"
[[ -f "$RUN/contract.txt" ]] && cat "$RUN/contract.txt"
echo "==================== MARKERS =============================="
grep -E '\[R874B_MARKER\]|\[R873_MARKER\]|stage=|iteration=|PROCESS_RC|Traceback|RuntimeError|CUDA out of memory|Error' "$RUN/worker.log" | tail -n 160 || true
echo "==================== CHECKPOINTS =========================="
find "$RUN/checkpoints" "$RUN/logs" -maxdepth 1 -type f \
    \( -name '*model*.pt' -o -name '*.pth' \) -size +0c \
    -printf 'READY %p %s bytes\n' 2>/dev/null || true
[[ -s "$RUN/SHA256SUMS" ]] && (cd "$RUN" && sha256sum -c SHA256SUMS) || true
RC="$(grep -oE 'PROCESS_RC=[0-9]+' "$RUN/worker.log" | tail -n 1 | cut -d= -f2)"
HIGH_COUNT=$(find "$RUN/checkpoints" "$RUN/logs" -maxdepth 1 -type f -name 'high_model_final.pt' -size +0c 2>/dev/null | wc -l)
if [[ "$RC" == 0 && "$HIGH_COUNT" -gt 0 ]]; then
    echo "TRAINING=TRUE"
    echo "FRESH_HIGH_POLICY=TRUE"
    echo "LOW_POLICY=FROZEN_WARM_START"
    echo "PROGRESSIVE_CURRICULUM=FALSE"
    echo "STATE=FRESH_NO_CURRICULUM_CHECKPOINT_READY"
    echo "VERDICT=R874B_READY_FOR_13_SCENARIO_EVALUATION"
else
    echo "TRAINING=TRUE"
    echo "STATE=TRAINING_INCOMPLETE"
    echo "VERDICT=R874B_NOT_READY"
fi
