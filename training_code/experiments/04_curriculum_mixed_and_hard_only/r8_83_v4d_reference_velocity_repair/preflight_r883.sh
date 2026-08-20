#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
EXP="$PROJECT/experiments/r8_83_v4d_reference_velocity_repair"
PARENT="/scratch/lj2634/runs/go2_soccer_r8_81_v4d_axis_persistence_repair/axis_15876323_20260816_211259/source/train_r881.py"
PY="${ISAAC_PYTHON:-/scratch/lj2634/envs/isaaclab232_v2/bin/python}"

for FILE in \
  "$EXP/patch_r883_reference_velocity.py" \
  "$EXP/run_r883_reference_velocity.sbatch" \
  "$EXP/check_r883_latest.sh" \
  "$PARENT"
do
  [[ -s "$FILE" ]] || {
    echo "R883_PREFLIGHT_FATAL=missing:$FILE"
    exit 21
  }
done

bash -n "$EXP/run_r883_reference_velocity.sbatch" \
        "$EXP/check_r883_latest.sh"

TMP="$(mktemp -d "${TMPDIR:-/tmp}/r883_preflight.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

"$PY" "$EXP/patch_r883_reference_velocity.py" \
  --source "$PARENT" \
  --output "$TMP/train_r883.py"

"$PY" -m py_compile "$TMP/train_r883.py"

grep -Fq \
  'terminal_velocity[:, 2] = 0.25 * action[:, 2]' \
  "$TMP/train_r883.py"

grep -Fq   'terminal_velocity[:, 2] = 0.25 * action[:, 2]'   "$TMP/train_r883.py"

grep -Fq   'requested_velocity = self.target_terminal_velocity'   "$TMP/train_r883.py"

echo "R883_PREFLIGHT_OK"
