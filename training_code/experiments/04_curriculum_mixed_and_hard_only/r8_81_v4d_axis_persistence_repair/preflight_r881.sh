#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
EXP="$PROJECT/experiments/r8_81_v4d_axis_persistence_repair"
PARENT="/scratch/lj2634/runs/go2_soccer_r8_79_v4d_directional_low_repair/directional_15874488_20260816_201731/source/train_r879.py"
PY="${ISAAC_PYTHON:-/scratch/lj2634/envs/isaaclab232_v2/bin/python}"
for FILE in "$EXP/patch_r881_axis_persistence.py" "$EXP/run_r881_axis_persistence.sbatch" \
  "$EXP/check_r881_latest.sh" "$PARENT"; do
  [[ -s "$FILE" ]] || { echo "R881_PREFLIGHT_FATAL=missing:$FILE"; exit 21; }
done
bash -n "$EXP/run_r881_axis_persistence.sbatch" "$EXP/check_r881_latest.sh"
"$PY" -m py_compile "$EXP/patch_r881_axis_persistence.py"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
"$PY" "$EXP/patch_r881_axis_persistence.py" --source "$PARENT" --output "$TMP/train_r881.py"
"$PY" -m py_compile "$TMP/train_r881.py"
for M in axis_persistence_failure angular_requested R881_SUSTAINED_LOW_REPAIR 'Refresh after installing'; do
  grep -Fq "$M" "$TMP/train_r881.py" || { echo "R881_PREFLIGHT_FATAL=missing_marker:$M"; exit 22; }
done
echo "R881_PREFLIGHT_OK"
