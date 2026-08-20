#!/usr/bin/env bash
#SBATCH --account=torch_pr_1066_general
#SBATCH --partition=l40s_public
#SBATCH --job-name=r896-turn
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-r896-%j.out

set -Eeuo pipefail

PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
PY="/scratch/lj2634/envs/isaaclab232_v2/bin/python"
ROOT="/scratch/lj2634/runs/go2_soccer_r8_96_v4d_turn_repair_oversample"
SOURCE="$PROJECT/experiments/r8_96_v4d_turn_repair_oversample/train_r896.py"

LOW="/scratch/lj2634/runs/go2_soccer_r8_83_v4d_reference_velocity_repair/axis_15877796_20260816_222323/logs/low_model_final.pt"
HIGH="/scratch/lj2634/runs/go2_soccer_v4d_joint/v4d_staged_joint_20260803_223518/full/logs/high_model_final.pt"

[[ -s "$SOURCE" ]] || exit 21
[[ -s "$LOW" ]] || exit 22
[[ -s "$HIGH" ]] || exit 23

PERIOD=16
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="$ROOT/period16_fixed_${SLURM_JOB_ID}_${STAMP}"

mkdir -p "$RUN"/{source,logs,runtime/home,runtime/cache,runtime/config,runtime/data,runtime/tmp,runtime/mpl}
cp -a "$SOURCE" "$RUN/source/train_r896.py"
printf '%s\n' "$RUN" > "$ROOT/latest_period16.txt"

export PYTHONPATH="$PROJECT/source/isaac_go2_low_level_rl:$PROJECT/source:${PYTHONPATH:-}"
export HOME="$RUN/runtime/home"
export XDG_CACHE_HOME="$RUN/runtime/cache"
export XDG_CONFIG_HOME="$RUN/runtime/config"
export XDG_DATA_HOME="$RUN/runtime/data"
export TMPDIR="$RUN/runtime/tmp"
export MPLCONFIGDIR="$RUN/runtime/mpl"
export PYTHONUNBUFFERED=1
export QT_QPA_PLATFORM=offscreen
export R873_HEADLESS_STRIP=1
export R876_CURRICULUM_HIGH_REPAIR=1
export R890_SUSTAINED_LOW_REPAIR=1
export R890_FORCE_LOCOMOTION_PRIMITIVES=1
export R890_LOW_LR=0.00005
export R890_GAIT_PERIOD=16.0
export R896_REPAIR_TURN_COVERAGE=0.50
export R896_REPAIR_ACTION_STD=0.15
export V4D_STAGE1_ITERATIONS=75
export V4D_STAGE2_ITERATIONS=125
export V4D_STAGE3_ITERATIONS=50
export V4D_HIGH_CHECKPOINT="$HIGH"

set +e
setsid timeout --foreground --kill-after=30s 75m \
  "$PY" "$RUN/source/train_r896.py" \
  --headless \
  --num_envs 512 \
  --max_iterations 250 \
  --seed 89016 \
  --low_level_checkpoint "$LOW" \
  --log_dir "$RUN/logs" \
  --run_name r896_period16_turn_repair \
  > "$RUN/worker.log" 2>&1
RC=$?
set -e

printf 'PROCESS_RC=%s\n' "$RC" >> "$RUN/worker.log"
echo "R896_RUN=$RUN"
exit "$RC"
