#!/usr/bin/env bash
set -u

ROOT="/scratch/lj2634/runs/go2_soccer_r8_71_v4d_targeted_scan"
PTR="$ROOT/latest_r871.txt"

echo "============================================================"
echo " R8.71 V4D TARGETED EXTREME-STATE CHECK"
echo "============================================================"
echo "TIME=$(date -Is)"
echo

ACTIVE="$(squeue -h -u "$USER" -o '%A|%j|%T|%M|%R' | awk -F'|' '$2 ~ /^r871-targ/ {print}')"
echo "ACTIVE=${ACTIVE:-NONE}"

if [[ ! -s "$PTR" ]]; then
    echo "POINTER=ABSENT"
    echo "VERDICT=R871_WAITING_FOR_SUBMISSION_OR_COMPLETION"
else
    RUN="$(tr -d '\r\n' < "$PTR")"
    echo "RUN=$RUN"
    [[ -f "$RUN/contract.txt" ]] && cat "$RUN/contract.txt"

    echo "==================== CASE STATUS ==================="
    for log in "$RUN"/logs/*.log; do
        [[ -f "$log" ]] || continue
        echo "----- $(basename "$log") -----"
        grep -E '\[R871_SCENARIO\]|\[R869_MARKER\]|\[R868_MARKER\]|PROCESS_RC|Traceback|RuntimeError|Timeout' "$log" | tail -n 12 || true
    done

    echo "==================== TARGETED RESULT ==============="
    if [[ -s "$RUN/targeted_aggregate.json" ]]; then
        python - "$RUN/targeted_aggregate.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    a = json.load(handle)
for key in ("scenario_count", "goal_count", "own_goal_count", "wall_stall_count", "avoidable_wall_stall_count"):
    print(f"{key}={a.get(key)}")
for row in a.get("scenarios", []):
    print(
        f"CASE scenario={row['scenario']} goal={int(row['goal'])} "
        f"own_goal={int(row['own_goal'])} wall_stall={int(row['wall_stall'])} "
        f"wall_escape={int(row['wall_escape_detected'])} "
        f"avoidable_wall_stall={int(row['avoidable_wall_stall'])}"
    )
PY
    else
        echo "TARGETED_AGGREGATE=MISSING"
    fi

    echo "==================== ARTIFACTS ====================="
    summary_count="$(find "$RUN/summaries" -maxdepth 1 -name '*.json' ! -name '*.scenario.json' -size +0c 2>/dev/null | wc -l)"
    trajectory_count="$(find "$RUN/trajectories" -maxdepth 1 -name '*.csv' -size +0c 2>/dev/null | wc -l)"
    echo "SUMMARY_COUNT=$summary_count"
    echo "TRAJECTORY_COUNT=$trajectory_count"

    if [[ -s "$RUN/SHA256SUMS" ]]; then
        (cd "$RUN" && sha256sum -c SHA256SUMS)
    fi
    echo "TRAINING=FALSE"
    echo "VIDEO_GENERATION=FALSE"
    if [[ -s "$RUN/targeted_aggregate.json" && "$summary_count" -eq 13 && "$trajectory_count" -eq 13 ]]; then
        echo "STATE=TARGETED_EXTREME_METRICS_COMPLETE"
        echo "VERDICT=R871_READY_FOR_BEHAVIOR_DECISION"
    else
        echo "STATE=TARGETED_EXTREME_METRICS_INCOMPLETE"
        echo "VERDICT=R871_SCAN_INCOMPLETE"
    fi
fi
