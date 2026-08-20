#!/usr/bin/env bash
#SBATCH --account=torch_pr_1066_general
#SBATCH --partition=l40s_public
#SBATCH --job-name=r895-fix1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --array=0-2
#SBATCH --output=slurm-r895-fix1-%A_%a.out

set -Eeuo pipefail

PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
PY="/scratch/lj2634/envs/isaaclab232_v2/bin/python"
ROOT="/scratch/lj2634/runs/go2_soccer_r8_95_v4d_receding_turn_fix1"
LOW="/scratch/lj2634/runs/go2_soccer_r8_83_v4d_reference_velocity_repair/axis_15877796_20260816_222323/logs/low_model_final.pt"
HIGH="/scratch/lj2634/runs/go2_soccer_v4d_joint/v4d_staged_joint_20260803_223518/full/logs/high_model_final.pt"

case "${SLURM_ARRAY_TASK_ID}" in
  0) PERIOD=16 ;;
  1) PERIOD=24 ;;
  2) PERIOD=32 ;;
  *) exit 20 ;;
esac

SOURCE="$PROJECT/experiments/r8_95_v4d_receding_turn/train_r895.py"

[[ -s "$SOURCE" ]] || exit 21
[[ -s "$LOW" ]] || exit 22
[[ -s "$HIGH" ]] || exit 23

"$PY" -m py_compile "$SOURCE"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="$ROOT/period${PERIOD}_fixed_${SLURM_JOB_ID}_${STAMP}"
mkdir -p "$RUN"/{source,logs,runtime/home,runtime/cache,runtime/config,runtime/data,runtime/tmp,runtime/mpl}
cp -a "$SOURCE" "$RUN/source/train_r890.py"
printf '%s\n' "$RUN" > "$ROOT/latest_period${PERIOD}.txt"

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
export R890_TURN_OVERSAMPLE=1
export R890_TURN_COVERAGE=0.50
export R890_GAIT_PERIOD="${PERIOD}.0"
export R895_TURN_LOOKAHEAD_S=1.0
export R890_LOW_LR=0.00005
export V4D_STAGE1_ITERATIONS=75
export V4D_STAGE2_ITERATIONS=125
export V4D_STAGE3_ITERATIONS=50
export V4D_HIGH_CHECKPOINT="$HIGH"

set +e
setsid timeout --foreground --kill-after=30s 75m \
  "$PY" "$RUN/source/train_r890.py" \
  --headless \
  --num_envs 512 \
  --max_iterations 250 \
  --seed "$((89000 + PERIOD))" \
  --low_level_checkpoint "$LOW" \
  --log_dir "$RUN/logs" \
  --run_name "r895_period${PERIOD}_receding" \
  > "$RUN/worker.log" 2>&1
RC=$?
set -e

printf 'PROCESS_RC=%s\n' "$RC" >> "$RUN/worker.log"
echo "R890_RUN=$RUN"
exit "$RC"
