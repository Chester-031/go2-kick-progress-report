#!/usr/bin/env bash
# R8_4_LOAD_PRESERVING_PROGRESSIVE_LIFT_PPO
set -uo pipefail

RUN_ROOT="/scratch/lj2634/runs/go2_soccer_r8_4_load_preserving_progressive_lift"
POINTER="${RUN_ROOT}/latest_r8_4_load_preserving_progressive_lift.txt"
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
echo "R8.4 SHORT LOAD-PRESERVING LIFT REPORT"
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
echo "===== GATES ====="
grep -E \
'R8_4_CONTACT_SOURCE|R8_4_LOAD_BASELINE|R8_4_STAGE_START|R8_4_STAGE_GATE|R8_4_STAGE_CLOSE|R8_4_STAGE_TIMEOUT|R8_4_CURRICULUM_COMPLETE|R8_4_FINAL_GUARD|R8_4_FATAL|RUN_RC=' \
  "${LOG}" 2>/dev/null | tail -n 90 || true

echo
echo "===== QUALITY ====="
if [ -s "${SUMMARY}" ]; then
"${PY}" - "${SUMMARY}" <<'PY'
import json, sys
from collections import defaultdict

data=json.load(open(sys.argv[1], "r", encoding="utf-8"))
records=data.get("records", [])
final=data.get("final_metrics", {})

print(f"records={len(records)}")
print(f"curriculum_complete={data.get('curriculum_complete')}")
print(f"highest_passed_stage={data.get('highest_passed_stage')}")
print(f"evaluation_stage={data.get('evaluation_stage')}")
print(f"stop_reason={data.get('stop_reason')}")
print(
    "final: "
    f"stage_success={final.get('stage_success_rate')} "
    f"strict={final.get('strict_success_rate')} "
    f"load_ratio={final.get('selected_load_ratio')} "
    f"FL={final.get('selected_fz_N')}N "
    f"SUP3={final.get('support3_fz_N')}N "
    f"load_ready={final.get('load_ready_fraction')} "
    f"lift={final.get('foot_lift_fraction')} "
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

groups=defaultdict(list)
for row in records:
    groups[str(row.get("stage"))].append(row)

print("tail10:")
for stage, rows in groups.items():
    tail=rows[-min(10,len(rows)):]
    def mean(key):
        vals=[
            float(r[key]) for r in tail
            if isinstance(r.get(key),(int,float))
        ]
        return sum(vals)/len(vals) if vals else None
    def maximum(key):
        vals=[
            float(r[key]) for r in rows
            if isinstance(r.get(key),(int,float))
        ]
        return max(vals) if vals else None
    print(
        f"  {stage:12s} n={len(rows):3d} "
        f"success={mean('stage_success_rate')} "
        f"load_ratio={mean('selected_load_ratio')} "
        f"FL={mean('selected_fz')}N "
        f"SUP3={mean('support3_fz')}N "
        f"ready={mean('load_ready_seen')} "
        f"prelift={mean('prelift_unload_seen')} "
        f"best_lift_frac={maximum('foot_lift_fraction')} "
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

grep -E \
'R8_4_EVAL_STAGE|R8_EVAL_VIDEO_READY|R8_4_FATAL|RUN_RC=' \
  "${ROOT}/video/worker.log" 2>/dev/null | tail -n 12 || true
