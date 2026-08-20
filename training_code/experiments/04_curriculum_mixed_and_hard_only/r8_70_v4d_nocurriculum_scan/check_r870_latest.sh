#!/usr/bin/env bash
set -u

ROOT="/scratch/lj2634/runs/go2_soccer_r8_70_v4d_nocurriculum_scan"
PTR="$ROOT/latest_r870.txt"

echo "============================================================"
echo " R8.70 V4D NO-CURRICULUM METRIC SCAN CHECK"
echo "============================================================"
echo "TIME=$(date -Is)"
echo

ACTIVE="$(squeue -h -u "$USER" -o '%A|%j|%T|%M|%R' | awk -F'|' '$2 ~ /^r870-scan/ {print}')"
if [[ -n "$ACTIVE" ]]; then
    echo "ACTIVE=$ACTIVE"
else
    echo "ACTIVE=NONE"
fi

if [[ ! -s "$PTR" ]]; then
    echo "POINTER=ABSENT"
    echo "VERDICT=R870_WAITING_FOR_SUBMISSION_OR_COMPLETION"
else
    RUN="$(tr -d '\r\n' < "$PTR")"
    echo "RUN=$RUN"
    [[ -f "$RUN/contract.txt" ]] && cat "$RUN/contract.txt"

    echo "==================== SEED STATUS ===================="
    for log in "$RUN"/logs/seed_*.log; do
        [[ -f "$log" ]] || continue
        seed="${log##*/seed_}"
        seed="${seed%.log}"
        rc="$(grep -o 'PROCESS_RC=[0-9]*' "$log" | tail -n 1 || true)"
        echo "seed=$seed ${rc:-PROCESS_RC=UNKNOWN}"
        grep -E '\[R869_MARKER\]|\[R868_MARKER\]|Traceback|RuntimeError|ModuleNotFoundError|Timeout' "$log" | tail -n 8 || true
    done

    echo "==================== SUMMARIES ====================="
    find "$RUN/summaries" -type f -size +0c -printf 'READY %p %s bytes\n' 2>/dev/null || true
    find "$RUN/trajectories" -type f -size +0c -printf 'READY %p %s bytes\n' 2>/dev/null || true

    if [[ -s "$RUN/aggregate_summary.json" ]]; then
        python - "$RUN/aggregate_summary.json" <<'PY'
import json
import sys

p = sys.argv[1]
with open(p, encoding="utf-8") as handle:
    aggregate = json.load(handle)
for key in (
    "summary_count", "goal_seed_count", "goal_seed_rate",
    "own_goal_seed_count", "own_goal_seed_rate", "total_goal_count",
    "total_own_goal_count", "total_done_count", "total_push_stall_count",
):
    print(f"{key}={aggregate.get(key)}")
print("skill_counts=" + json.dumps(aggregate.get("skill_counts", {}), sort_keys=True))
PY
    else
        echo "AGGREGATE_SUMMARY=MISSING"
    fi

    echo "==================== HASH ==========================="
    if [[ -s "$RUN/SHA256SUMS" ]]; then
        (cd "$RUN" && sha256sum -c SHA256SUMS)
    else
        echo "SHA256SUMS=MISSING"
    fi

    echo "TRAINING=FALSE"
    echo "VIDEO_GENERATION=FALSE"
    summary_count="$(find "$RUN/summaries" -type f -size +0c 2>/dev/null | wc -l)"
    if [[ -s "$RUN/aggregate_summary.json" && "$summary_count" -eq 8 ]]; then
        echo "STATE=NO_CURRICULUM_METRICS_COMPLETE"
        echo "VERDICT=R870_READY_FOR_BEHAVIOR_DECISION"
    else
        echo "STATE=NO_CURRICULUM_METRICS_INCOMPLETE"
        echo "VERDICT=R870_SCAN_INCOMPLETE"
    fi
fi
