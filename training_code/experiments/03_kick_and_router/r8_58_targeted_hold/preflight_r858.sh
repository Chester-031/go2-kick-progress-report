#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
EXP="$PROJECT/experiments/r8_58_targeted_hold"
bash -n "$EXP/run_r858_targeted.sbatch"
bash -n "$EXP/check_r858_latest.sh"
bash -n "$EXP/build_r858_video_bundle.sh"
bash -n "$EXP/check_r858_video_bundle.sh"
python -m py_compile "$EXP/patch_r858_source.py"
for f in "$EXP/run_r858_targeted.sbatch" "$EXP/check_r858_latest.sh" "$EXP/patch_r858_source.py"; do [[ -s "$f" ]] || { echo "MISSING $f"; exit 2; }; done
grep -q '^#SBATCH --account=torch_pr_1066_general' "$EXP/run_r858_targeted.sbatch"
grep -q '^#SBATCH --partition=l40s_public' "$EXP/run_r858_targeted.sbatch"
grep -q 'XDG_CACHE_HOME=' "$EXP/run_r858_targeted.sbatch"
echo R858_PREFLIGHT_OK
