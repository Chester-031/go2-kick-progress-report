#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
EXP="$PROJECT/experiments/r8_80_v4d_mean_noise_attribution"
PARENT="/scratch/lj2634/runs/go2_soccer_r8_79_v4d_directional_low_repair/directional_15874488_20260816_201731/source/train_r879.py"
PY="${ISAAC_PYTHON:-/scratch/lj2634/envs/isaaclab232_v2/bin/python}"
for FILE in "$EXP/patch_r880_mean_noise_eval.py" "$EXP/run_r880_mean_noise_attribution.sbatch" \
  "$EXP/check_r880_latest.sh" "$PARENT"; do
  [[ -s "$FILE" ]] || { echo "R880_PREFLIGHT_FATAL=missing:$FILE"; exit 21; }
done
bash -n "$EXP/run_r880_mean_noise_attribution.sbatch" "$EXP/check_r880_latest.sh"
"$PY" -m py_compile "$EXP/patch_r880_mean_noise_eval.py"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
"$PY" "$EXP/patch_r880_mean_noise_eval.py" --source "$PARENT" --output "$TMP/train_r880.py"
"$PY" -m py_compile "$TMP/train_r880.py"
grep -Fq 'R880_MEAN_NOISE_ATTRIBUTION' "$TMP/train_r880.py"
grep -Fq 'stochastic=True' "$TMP/train_r880.py"
echo "R880_PREFLIGHT_OK"
