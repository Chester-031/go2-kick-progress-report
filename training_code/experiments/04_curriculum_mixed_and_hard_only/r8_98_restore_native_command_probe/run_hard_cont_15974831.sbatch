#!/usr/bin/env bash
#SBATCH --account=torch_pr_1066_general
#SBATCH --partition=l40s_public
#SBATCH --job-name=r898-probe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/lj2634/runs/go2_soccer_r8_98_restore_native_command_probe/hard_cont_slurm_logs/slurm-%j.out
#SBATCH --error=/scratch/lj2634/runs/go2_soccer_r8_98_restore_native_command_probe/hard_cont_slurm_logs/slurm-%j.err

set -Eeuo pipefail

PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
PY="/scratch/lj2634/envs/isaaclab232_v2/bin/python"
ROOT="/scratch/lj2634/runs/go2_soccer_r8_98_restore_native_command_probe"
SOURCE="/scratch/lj2634/runs/go2_soccer_r8_98_restore_native_command_probe/probe_15974831_20260818_170115/source/train_r898.py"

HIGH="/scratch/lj2634/runs/go2_soccer_r8_98_restore_native_command_probe/probe_15974831_20260818_170115/logs/high_model_final.pt"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="$ROOT/probe_${SLURM_JOB_ID}_${STAMP}"

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

# R890 repair disabled
LOW="/scratch/lj2634/runs/go2_soccer_r8_98_restore_native_command_probe/probe_15974831_20260818_170115/logs/low_model_final.pt"
export R890_LOW_LR=1e-8
# repair iterations disabled
export R899_REPAIR_ACTION_STD=0.15
export R890_FORCE_LOCOMOTION_PRIMITIVES=1
unset R898_RESTORE_NATIVE_COMMAND
unset R899_KEEP_NATIVE_COMMAND_WEIGHTS
export R899_PURE_TURN_REPAIR=1
export R899_TURN_STAND_REPLAY=1
unset R899_LEFT_BIASED_TURN_REPAIR
unset R899_LEFT_ONLY_TURN_REPAIR
export R899_REPAIR_ACTION_STD=0.15
export R899_ACTION_DIAGNOSTIC=1
export R899_EXEC_DIAGNOSTIC=1
export V4D_HIGH_CHECKPOINT="$HIGH"

set +e
setsid timeout --foreground --kill-after=15s 25m \
  "$PY" "$RUN/source/train_r898.py" \
  --headless \
  --num_envs 512 \
  --max_iterations 300 \
  --seed 89716 \
  --low_level_checkpoint "$LOW" \
  --log_dir "$RUN/logs" \
  --run_name r899_turn_repair_canary \
  > "$RUN/worker.log" 2>&1
RC=$?
set -e

printf 'PROCESS_RC=%s\n' "$RC" >> "$RUN/worker.log"
echo "R897_RUN=$RUN"
exit "$RC"

export LOW=/scratch/lj2634/runs/go2_soccer_r8_98_restore_native_command_probe/probe_15969289_20260818_150104/logs/low_model_final.pt

unset R874B_FRESH_NO_CURRICULUM
export R874B_FREEZE_LOW=0

# Ball-spawn curriculum
export R876_EASY_END=50
export R876_INTERMEDIATE_END=125
export R876_HARD_END=200

# V4D training stages: 90 + 150 + 60 = 300
export V4D_STAGE1_ITERATIONS=90
export V4D_STAGE2_ITERATIONS=150
export V4D_STAGE3_ITERATIONS=60
export V4D_CHECKPOINT_INTERVAL=50
