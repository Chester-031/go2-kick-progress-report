#!/usr/bin/env bash
# R8_1_GATED_UNLOAD_CURRICULUM_PPO
set -uo pipefail

RUN_ROOT="/scratch/lj2634/runs/go2_soccer_r8_1_gated_support_skill"
POINTER="${RUN_ROOT}/latest_r8_1_gated_support_skill.txt"

if [ ! -s "${POINTER}" ]; then
  echo "ERROR: missing R8.1 pointer: ${POINTER}" >&2
  exit 20
fi

ROOT="$(tr -d '\r\n' < "${POINTER}")"
PILOT_JOB="$(tr -d '\r\n' < "${ROOT}/pilot/job_id.txt")"
VIDEO_JOB="$(tr -d '\r\n' < "${ROOT}/video/job_id.txt")"
LOG="${ROOT}/pilot/worker.log"
SUMMARY="${ROOT}/pilot/logs/r8_support_summary.json"
PY="/scratch/lj2634/envs/isaaclab232_v2/bin/python"

echo "============================================================"
echo "R8.1 SHORT QUALITY REPORT"
echo "ROOT=${ROOT}"
echo "PILOT_JOB=${PILOT_JOB}"
echo "VIDEO_JOB=${VIDEO_JOB}"
echo "============================================================"

echo
echo "===== SLURM ====="
sacct \
  -j "${PILOT_JOB},${VIDEO_JOB}" \
  --format=JobID,JobName%22,State,ExitCode,Elapsed,NodeList \
  2>/dev/null || true

echo
echo "===== STAGE GATES / FINAL ====="
grep -E \
'R8_1_STAGE_START|R8_1_STAGE_GATE|R8_1_STAGE_CLOSE|R8_1_STAGE_TIMEOUT|R8_1_CURRICULUM_COMPLETE|R8_1_FINAL_GUARD|R8_1_PILOT_OUTCOME|R8_1_SUPPORT_PILOT_COMPLETE|R8_1_FATAL|RUN_RC=' \
  "${LOG}" 2>/dev/null | tail -n 80 || true

echo
echo "===== COMPACT QUALITY ====="
if [ -s "${SUMMARY}" ]; then
  "${PY}" - "${SUMMARY}" <<'PY'
from __future__ import annotations

import json
import sys
from collections import defaultdict

path = sys.argv[1]
data = json.load(open(path, "r", encoding="utf-8"))
records = data.get("records", [])
final = data.get("final_metrics", {})

print(f"records={len(records)}")
print(f"curriculum_complete={data.get('curriculum_complete')}")
print(f"highest_passed_stage={data.get('highest_passed_stage')}")
print(f"evaluation_stage={data.get('evaluation_stage')}")
print(f"stop_reason={data.get('stop_reason')}")
print(
    "final: "
    f"stage_success={final.get('stage_success_rate')} "
    f"strict={final.get('strict_success_rate')} "
    f"lift={final.get('foot_lift_fraction')} "
    f"swing={final.get('foot_swing_fraction')} "
    f"base={final.get('base_displacement')} "
    f"upright={final.get('upright_error')} "
    f"violation={final.get('violation_fraction')}"
)

print("stage_gate_history:")
for row in data.get("stage_gate_history", []):
    print(
        f"  {row.get('stage'):12s} "
        f"status={row.get('status')} "
        f"iters={row.get('stage_iterations')} "
        f"recent_success={row.get('recent_success')} "
        f"episodes={row.get('recent_completed_episodes')}"
    )

groups = defaultdict(list)
for row in records:
    groups[str(row.get("stage"))].append(row)

print("representative_tail:")
for stage, rows in groups.items():
    tail = rows[-min(10, len(rows)):]
    def vals(key):
        return [
            float(r[key]) for r in tail
            if isinstance(r.get(key), (int, float))
        ]
    def mean(key):
        x = vals(key)
        return sum(x) / len(x) if x else None
    def mx(key):
        x = vals(key)
        return max(x) if x else None
    print(
        f"  {stage:12s} n={len(rows):3d} "
        f"tail_stage_success={mean('stage_success_rate')} "
        f"tail_reward={mean('reward')} "
        f"tail_foot_z={mean('selected_foot_height')} "
        f"tail_base={mean('base_displacement')} "
        f"tail_upright={mean('upright_error')} "
        f"best_strict_lift_fraction_tail={mx('foot_lift_fraction')} "
        f"best_air_swing_fraction_tail={mx('foot_swing_fraction')}"
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

if [ -s "${ROOT}/video/r8_support_video_summary.json" ]; then
  "${PY}" - "${ROOT}/video/r8_support_video_summary.json" <<'PY'
import json, sys
p=json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(
    "video_quality: "
    f"stage={p.get('evaluation_stage')} "
    f"stage_success={p.get('stage_success_rate')} "
    f"strict_success={p.get('success_rate')} "
    f"episodes={p.get('episodes')}"
)
PY
fi
