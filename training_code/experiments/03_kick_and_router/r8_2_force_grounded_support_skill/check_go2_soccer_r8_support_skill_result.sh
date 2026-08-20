#!/usr/bin/env bash
# R8_2_FORCE_GROUNDED_UNLOAD_CURRICULUM_PPO
set -uo pipefail

RUN_ROOT="/scratch/lj2634/runs/go2_soccer_r8_2_force_grounded_support_skill"
POINTER="${RUN_ROOT}/latest_r8_2_force_grounded_support_skill.txt"

if [ ! -s "${POINTER}" ]; then
  echo "ERROR: missing R8.2 pointer: ${POINTER}" >&2
  exit 20
fi

ROOT="$(tr -d '\r\n' < "${POINTER}")"
PILOT_JOB="$(tr -d '\r\n' < "${ROOT}/pilot/job_id.txt")"
VIDEO_JOB="$(tr -d '\r\n' < "${ROOT}/video/job_id.txt")"
LOG="${ROOT}/pilot/worker.log"
SUMMARY="${ROOT}/pilot/logs/r8_support_summary.json"
PY="/scratch/lj2634/envs/isaaclab232_v2/bin/python"

echo "============================================================"
echo "R8.2 FORCE-GROUNDED SHORT QUALITY REPORT"
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
echo "===== CONTACT SENSOR / GATES / FINAL ====="
grep -E \
'R8_2_CONTACT_SENSOR|R8_2_CONTACT_MAPPING|R8_2_STAGE_START|R8_2_STAGE_GATE|R8_2_STAGE_CLOSE|R8_2_STAGE_TIMEOUT|R8_2_CURRICULUM_COMPLETE|R8_2_FINAL_GUARD|R8_2_PILOT_OUTCOME|R8_2_SUPPORT_PILOT_COMPLETE|R8_2_FATAL|R8_FATAL|RUN_RC=' \
  "${LOG}" 2>/dev/null | tail -n 100 || true

echo
echo "===== COMPACT QUALITY ====="
if [ -s "${SUMMARY}" ]; then
  "${PY}" - "${SUMMARY}" <<'PY'
from __future__ import annotations
import json, sys
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
    f"violation={final.get('violation_fraction')} "
    f"load_rel={final.get('selected_load_relative')} "
    f"load_frac={final.get('selected_load_fraction')} "
    f"unload_seen={final.get('unload_seen_fraction')} "
    f"baseline_valid={final.get('contact_baseline_valid_fraction')}"
)

print("stage_gate_history:")
for row in data.get("stage_gate_history", []):
    print(
        f"  {str(row.get('stage')):13s} "
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
        return [float(r[key]) for r in tail if isinstance(r.get(key), (int, float))]
    def mean(key):
        x = vals(key)
        return sum(x) / len(x) if x else None
    def mn(key):
        x = vals(key)
        return min(x) if x else None
    print(
        f"  {stage:13s} n={len(rows):3d} "
        f"tail_success={mean('stage_success_rate')} "
        f"reward={mean('reward')} "
        f"load_rel={mean('selected_load_relative')} "
        f"min_load_rel={mn('min_selected_load_relative')} "
        f"unload_seen={mean('unload_seen_fraction')} "
        f"foot_z={mean('selected_foot_height')} "
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
