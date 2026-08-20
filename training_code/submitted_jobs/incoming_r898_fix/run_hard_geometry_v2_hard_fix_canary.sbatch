#!/usr/bin/env bash
#SBATCH --account=torch_pr_1066_general
#SBATCH --partition=l40s_public
#SBATCH --job-name=r898-hardfix
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/lj2634/runs/go2_soccer_r8_98_restore_native_command_probe/hard_fix_canary_slurm/slurm-%j.out
#SBATCH --error=/scratch/lj2634/runs/go2_soccer_r8_98_restore_native_command_probe/hard_fix_canary_slurm/slurm-%j.err
#SBATCH --chdir=/scratch/lj2634/Isaac-Go2-Low-Level-RL

set -Eeuo pipefail

PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
PY="/scratch/lj2634/envs/isaaclab232_v2/bin/python"
ROOT="/scratch/lj2634/runs/go2_soccer_r8_98_restore_native_command_probe"
SOURCE="$PROJECT/experiments/r8_98_restore_native_command_probe/train_r898_hard_fix.py"

LOW="/scratch/lj2634/runs/go2_soccer_r8_98_restore_native_command_probe/probe_15987578_20260818_204233/logs/low_model_final.pt"
HIGH="/scratch/lj2634/runs/go2_soccer_r8_98_restore_native_command_probe/probe_15987578_20260818_204233/logs/high_model_final.pt"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="$ROOT/hard_fix_canary_${SLURM_JOB_ID}_${STAMP}"

mkdir -p "$RUN"/{source,logs,runtime/home,runtime/cache,runtime/config,runtime/data,runtime/tmp,runtime/mpl}
cp -a "$SOURCE" "$RUN/source/train_r898.py"
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

unset R898_RESTORE_NATIVE_COMMAND
unset R899_KEEP_NATIVE_COMMAND_WEIGHTS
export R899_TURN_STAND_REPLAY=1
unset R899_LEFT_BIASED_TURN_REPAIR
unset R899_LEFT_ONLY_TURN_REPAIR
export R899_REPAIR_ACTION_STD=0.15
export R899_ACTION_DIAGNOSTIC=1
export R899_EXEC_DIAGNOSTIC=1
export V4D_HIGH_CHECKPOINT="$HIGH"

LOW="/scratch/lj2634/runs/go2_soccer_r8_98_restore_native_command_probe/probe_15987578_20260818_204233/logs/low_model_final.pt"
export R876_EASY_END=20
export R876_INTERMEDIATE_END=40
export R876_HARD_END=80
export R876_CURRICULUM_HIGH_REPAIR=1
export V4D_CHECKPOINT_INTERVAL=20
export V4D_STAGE1_ITERATIONS=20
export V4D_STAGE2_ITERATIONS=50
export V4D_STAGE3_ITERATIONS=10

# R898 strategy repair: bounded high actions, staging-potential high reward,
# and the accepted low-level controller frozen during the canary.
export R898_STRATEGY_FIX=1
# Log saturation during the canary; do not abort before the repaired actor
# has had a chance to move away from the old checkpoint's Gaussian tails.
export R898_REQUIRE_ACTION_BOUND=0
export R898_MAX_HIGH_ACTION_SATURATION=0.35
export R874B_FREEZE_LOW=1
export V4D_HIGH_ROLLOUT_STEPS=48
export V4D_HIGH_GAMMA=0.995
export V4D_HIGH_ENTROPY_COEF=0.002
export V4D_HIGH_CLIP_PARAM=0.12
export V4D_HIGH_EPOCHS=4
export V4D_HIGH_MINI_BATCHES=4
export V4D_HIGH_DESIRED_KL=0.004

export R876_HARD_GEOMETRY_V2=1
export R876_HARD_REWARD_V2=1
set +e
setsid timeout --foreground --kill-after=15s 50m \
  "$PY" "$RUN/source/train_r898.py" \
  --headless \
  --num_envs 512 \
  --max_iterations 80 \
  --seed 89716 \
  --low_level_checkpoint "$LOW" \
  --log_dir "$RUN/logs" \
  --run_name r898_strategy_fix_canary \
  > "$RUN/worker.log" 2>&1
RC=$?
set -e

printf 'PROCESS_RC=%s\n' "$RC" >> "$RUN/worker.log"
echo "R897_RUN=$RUN"
exit "$RC"
