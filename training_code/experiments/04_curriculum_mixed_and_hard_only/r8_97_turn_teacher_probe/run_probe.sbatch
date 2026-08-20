#!/usr/bin/env bash
#SBATCH --account=torch_pr_1066_general
#SBATCH --partition=l40s_public
#SBATCH --job-name=r897-probe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-r897-probe-%j.out

set -Eeuo pipefail

PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
PY="/scratch/lj2634/envs/isaaclab232_v2/bin/python"
ROOT="/scratch/lj2634/runs/go2_soccer_r8_97_turn_teacher_probe"
SOURCE="$PROJECT/experiments/r8_97_turn_teacher_probe/train_r897.py"

LOW="/scratch/lj2634/runs/go2_soccer_r8_83_v4d_reference_velocity_repair/axis_15877796_20260816_222323/logs/low_model_final.pt"
HIGH="/scratch/lj2634/runs/go2_soccer_v4d_joint/v4d_staged_joint_20260803_223518/full/logs/high_model_final.pt"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="$ROOT/probe_${SLURM_JOB_ID}_${STAMP}"

mkdir -p "$RUN"/{source,logs,runtime/home,runtime/cache,runtime/config,runtime/data,runtime/tmp,runtime/mpl}
cp -a "$SOURCE" "$RUN/source/train_r897.py"
printf '%s\n' "$RUN" > "$ROOT/latest_probe.txt"

export PYTHONPATH="$PROJECT/source/isaac_go2_low_level_rl:$PROJECT/source:${PYTHONPATH:-}"
export R873_HEADLESS_STRIP=1
export HOME="$RUN/runtime/home"
export XDG_CACHE_HOME="$RUN/runtime/cache"
export XDG_CONFIG_HOME="$RUN/runtime/config"
export XDG_DATA_HOME="$RUN/runtime/data"
export TMPDIR="$RUN/runtime/tmp"
export MPLCONFIGDIR="$RUN/runtime/mpl"
export PYTHONUNBUFFERED=1
export QT_QPA_PLATFORM=offscreen

export R890_SUSTAINED_LOW_REPAIR=1
export R890_FORCE_LOCOMOTION_PRIMITIVES=1
export R890_LOW_LR=0.0
export R897_TURN_TEACHER_ONLY=1
export R897_TURN_TEACHER_SCALE=1.0
export V4D_HIGH_CHECKPOINT="$HIGH"

set +e
setsid timeout --foreground --kill-after=15s 12m \
  "$PY" "$RUN/source/train_r897.py" \
  --headless \
  --num_envs 512 \
  --max_iterations 1 \
  --seed 89716 \
  --low_level_checkpoint "$LOW" \
  --log_dir "$RUN/logs" \
  --run_name r897_turn_teacher_probe \
  > "$RUN/worker.log" 2>&1
RC=$?
set -e

printf 'PROCESS_RC=%s\n' "$RC" >> "$RUN/worker.log"
echo "R897_RUN=$RUN"
exit "$RC"
