#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
EXP="$PROJECT/experiments/r8_82_v4d_turn_target_repair"
PARENT="/scratch/lj2634/runs/go2_soccer_r8_81_v4d_axis_persistence_repair/axis_15876323_20260816_211259/source/train_r881.py"
PY="${ISAAC_PYTHON:-/scratch/lj2634/envs/isaaclab232_v2/bin/python}"

for FILE in \
  "$EXP/patch_r882_turn_target.py" \
  "$EXP/run_r882_turn_target.sbatch" \
  "$EXP/check_r882_latest.sh" \
  "$PARENT"
do
  [[ -s "$FILE" ]] || {
    echo "R882_PREFLIGHT_FATAL=missing:$FILE"
    exit 21
  }
done

bash -n "$EXP/run_r882_turn_target.sbatch" \
        "$EXP/check_r882_latest.sh"

TMP="$(mktemp -d "${TMPDIR:-/tmp}/r882_preflight.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

"$PY" "$EXP/patch_r882_turn_target.py" \
  --source "$PARENT" \
  --output "$TMP/train_r882.py"

"$PY" -m py_compile "$TMP/train_r882.py"

grep -Fq \
  'terminal_velocity[:, 2] = 0.25 * action[:, 2]' \
  "$TMP/train_r882.py"

echo "R882_PREFLIGHT_OK"
