#!/usr/bin/env bash
# R8_3_REAL_LOAD_TRANSFER_PPO
set -uo pipefail

RUN_ROOT="/scratch/lj2634/runs/go2_soccer_r8_3_real_load_transfer_skill"
POINTER="${RUN_ROOT}/latest_r8_3_real_load_transfer_skill.txt"
PY="/scratch/lj2634/envs/isaaclab232_v2/bin/python"

test -s "${POINTER}" || {
  echo "ERROR: missing ${POINTER}" >&2
  exit 20
}

ROOT="$(tr -d '\r\n' < "${POINTER}")"
PILOT_JOB="$(tr -d '\r\n' < "${ROOT}/pilot/job_id.txt")"
VIDEO_JOB="$(tr -d '\r\n' < "${ROOT}/video/job_id.txt")"
LOG="${ROOT}/pilot/worker.log"
SUMMARY="${ROOT}/pilot/logs/r8_support_summary.json"

echo "============================================================"
echo "R8.3 SHORT REAL-LOAD REPORT"
echo "ROOT=${ROOT}"
echo "PILOT_JOB=${PILOT_JOB}"
echo "VIDEO_JOB=${VIDEO_JOB}"
echo "============================================================"

echo
echo "===== SLURM ====="
sacct \
  -j "${PILOT_JOB},${VIDEO_JOB}" \
  --format=JobID,JobName%22,State,ExitCode,Elapsed \
  2>/dev/null || true

echo
echo "===== PHYSICS / GATES ====="
grep -E \
'R8_3_CONTACT_SOURCE|R8_3_LOAD_BASELINE|R8_3_STAGE_START|R8_3_STAGE_GATE|R8_3_STAGE_CLOSE|R8_3_STAGE_TIMEOUT|R8_3_CURRICULUM_COMPLETE|R8_3_FINAL_GUARD|R8_3_FATAL|RUN_RC=' \
  "${LOG}" 2>/dev/null | tail -n 70 || true

echo
echo "===== COMPACT QUALITY ====="
if [ -s "${SUMMARY}" ]; then
"${PY}" - "${SUMMARY}" <<'PY'
import json, sys
from collections import defaultdict

data = json.load(
    open(sys.argv[1], "r", encoding="utf-8")
)
records = data.get("records", [])
final = data.get("final_metrics", {})

print(f"records={len(records)}")
print(
    f"curriculum_complete="
    f"{data.get('curriculum_complete')}"
)
print(
    f"highest_passed_stage="
    f"{data.get('highest_passed_stage')}"
)
print(
    f"evaluation_stage="
    f"{data.get('evaluation_stage')}"
)
print(f"stop_reason={data.get('stop_reason')}")
print(
    "final: "
    f"stage_success={final.get('stage_success_rate')} "
    f"load_ratio={final.get('selected_load_ratio')} "
    f"FL={final.get('selected_fz_N')}N "
    f"SUP3={final.get('support3_fz_N')}N "
    f"load_ready={final.get('load_ready_fraction')} "
    f"base={final.get('base_displacement')} "
    f"upright={final.get('upright_error')} "
    f"violation={final.get('violation_fraction')}"
)

print("stage_gates:")
for row in data.get("stage_gate_history", []):
    print(
        f"  {str(row.get('stage')):12s} "
        f"status={row.get('status')} "
        f"iters={row.get('stage_iterations')} "
        f"success={row.get('recent_success')} "
        f"episodes={row.get('recent_completed_episodes')}"
    )

groups = defaultdict(list)
for row in records:
    groups[str(row.get("stage"))].append(row)

print("tail10:")
for stage, rows in groups.items():
    tail = rows[-min(10, len(rows)):]
    def mean(key):
        values = [
            float(row[key])
            for row in tail
            if isinstance(row.get(key), (int, float))
        ]
        return (
            sum(values) / len(values)
            if values else None
        )
    def minimum(key):
        values = [
            float(row[key])
            for row in tail
            if isinstance(row.get(key), (int, float))
        ]
        return min(values) if values else None

    print(
        f"  {stage:12s} n={len(rows):3d} "
        f"success={mean('stage_success_rate')} "
        f"load_ratio={mean('selected_load_ratio')} "
        f"min_ratio={minimum('min_selected_load_ratio')} "
        f"FL={mean('selected_fz')}N "
        f"SUP3={mean('support3_fz')}N "
        f"ready={mean('load_ready_seen')} "
        f"base={mean('base_displacement')} "
        f"upright={mean('upright_error')}"
    )
PY
else
  echo "SUMMARY_MISSING=${SUMMARY}"
fi

echo
echo "===== VIDEO ====="
ls -lh \
  "${ROOT}/video/r8_support_pilot.mp4" \
  "${ROOT}/video/r8_support_video_summary.json" \
  2>/dev/null || true
