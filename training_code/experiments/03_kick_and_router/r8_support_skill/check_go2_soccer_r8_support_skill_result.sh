#!/usr/bin/env bash
set -uo pipefail

RUN_ROOT="/scratch/lj2634/runs/go2_soccer_r8_support_skill"
POINTER="${RUN_ROOT}/latest_r8_support_skill.txt"

if [ ! -s "${POINTER}" ]; then
  echo "ERROR: missing R8 pointer: ${POINTER}" >&2
  exit 20
fi

ROOT="$(tr -d '\r\n' < "${POINTER}")"
PILOT_JOB="$(tr -d '\r\n' < "${ROOT}/pilot/job_id.txt")"
VIDEO_JOB="$(tr -d '\r\n' < "${ROOT}/video/job_id.txt")"

echo "ROOT=${ROOT}"
echo "PILOT_JOB=${PILOT_JOB}"
echo "VIDEO_JOB=${VIDEO_JOB}"

echo
echo "===== Slurm ====="
sacct \
  -j "${PILOT_JOB},${VIDEO_JOB}" \
  --format=JobID,JobName%24,State,ExitCode,Elapsed,NodeList

echo
echo "===== Pilot diagnostics ====="
grep -nE \
'R8_STAGE_START|R8_STAGE_GATE|R8_ITER|R8_FINAL_GUARD|R8_SUPPORT_PILOT_COMPLETE|R8_FATAL|RUN_RC=' \
  "${ROOT}/pilot/worker.log" 2>/dev/null | tail -n 220 || true

echo
echo "===== Summary ====="
cat "${ROOT}/pilot/logs/r8_support_summary.json" 2>/dev/null || true

echo
echo "===== Video ====="
ls -lh \
  "${ROOT}/video/r8_support_pilot.mp4" \
  "${ROOT}/video/r8_support_video_summary.json" \
  2>/dev/null || true

if [ -f "${ROOT}/pilot/R8_SUPPORT_PILOT_COMPLETE" ]; then
  echo "R8_PILOT_TRAINING_COMPLETE"
fi
if [ -f "${ROOT}/video/R8_SUPPORT_VIDEO_READY" ]; then
  echo "R8_PILOT_VIDEO_READY"
fi
