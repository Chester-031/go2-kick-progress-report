#!/usr/bin/env bash
#SBATCH --account=torch_pr_1066_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --job-name=r898-none-v11
#SBATCH --output=/scratch/lj2634/runs/go2_soccer_r8_98_restore_native_command_probe/none_curriculum_fix_v1_1_%A_%a.out
#SBATCH --error=/scratch/lj2634/runs/go2_soccer_r8_98_restore_native_command_probe/none_curriculum_fix_v1_1_%A_%a.err

set +e
set +u
set +o pipefail

PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
ROOT="/scratch/lj2634/runs/go2_soccer_r8_98_restore_native_command_probe"
PY="/scratch/lj2634/envs/isaaclab232_v2/bin/python"

SOURCE="$ROOT/incoming_r898_fix/train_r898_fix_v1_1_baseline.py"
BASE_RUN="/scratch/lj2634/runs/go2_soccer_r8_98_restore_native_command_probe/hard_fix_long600_v2_16027021_20260819_125302"

for REQUIRED in \
  "$PY" \
  "$SOURCE" \
  "$BASE_RUN/logs/high_model_final.pt" \
  "$BASE_RUN/logs/low_model_final.pt"
do
    if [ ! -s "$REQUIRED" ]; then
        echo "[STOP] REQUIRED_FILE_MISSING=$REQUIRED"
        exit 2
    fi
done

TAG="none_curriculum_fix_v1_1_long600_seed$((87701 + SLURM_ARRAY_TASK_ID))"
RUN="$ROOT/${TAG}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p "$RUN/source" "$RUN/logs"
cp -p "$SOURCE" "$RUN/source/train_r898.py"

NODE_CACHE="/state/partition1/job-${SLURM_JOB_ID}/r898_cache"
RUNTIME="$NODE_CACHE/runtime"

mkdir -p \
  "$RUNTIME/home" \
  "$RUNTIME/cache" \
  "$RUNTIME/config" \
  "$RUNTIME/data" \
  "$RUNTIME/mpl" \
  "$RUNTIME/omni" \
  "$RUNTIME/xdg" \
  "$RUNTIME/torch"

export HOME="$RUNTIME/home"
export XDG_CACHE_HOME="$RUNTIME/cache"
export XDG_CONFIG_HOME="$RUNTIME/config"
export XDG_DATA_HOME="$RUNTIME/data"
export MPLCONFIGDIR="$RUNTIME/mpl"
export OMNI_USER_HOME="$RUNTIME/omni"
export TORCH_HOME="$RUNTIME/torch"
export LD_LIBRARY_PATH="/scratch/lj2634/vendor/libglu-9.0.0/lib:${LD_LIBRARY_PATH:-}"

export PYTHONPATH="$PROJECT/source/isaac_go2_low_level_rl:$PROJECT/source:$PROJECT:$RUN/source:${PYTHONPATH:-}"
export V4D_HIGH_CHECKPOINT="$BASE_RUN/logs/high_model_final.pt"

export R876_CURRICULUM_HIGH_REPAIR=1
export R876_HARD_GEOMETRY_V2=1
export R876_HARD_REWARD_V2=1
export R898_STRATEGY_FIX=1
export R898_ACTION_SCALE=0.75

export R876_EASY_END=0
export R876_INTERMEDIATE_END=0
export R876_HARD_END=600

# 600-iteration hard-only schedule.
# Matches the successful 80-step proportion:
# high_adapter -> joint_finetune -> stabilization.
export V4D_STAGE1_ITERATIONS=150
export V4D_STAGE2_ITERATIONS=336
export V4D_STAGE3_ITERATIONS=114

# Preserve the validated V1.1 baseline: low policy remains frozen.
export V4D_HIGH_LR_STAGE1=0.00001
export V4D_HIGH_LR_STAGE2=0.000005
export V4D_HIGH_LR_STAGE3=0.000002
export V4D_LOW_LR_STAGE2=0.000001

export R898_STAGING_PUSH_GATE=1
export R898_STRICT_PUSH_GATE=1
export R898_STAGING_MAX=0.45
export R898_CONTACT_MIN=0.50
export R898_CONTACT_MAX=0.80
export R898_BODY_X_MIN=0.15
export R898_GOAL_BODY_X_MIN=0.35
export R898_CONTACT_LATERAL_MAX=0.30
export R898_PUSH_HOLD_STEPS=8
export R898_PUSH_HOLD_LATERAL_MAX=0.42

export R898_PREPUSH_ALIGN=1
export R898_PREPUSH_OFFSET=0.65
export R898_PREPUSH_LIMIT=0.90

# V1.1 baseline: disable the failed V1.2/V1.3 rotate-first override.
export R898_PREPUSH_ROTATE_FIRST=0

export R873_HEADLESS_STRIP=1
export PYTHONUNBUFFERED=1

echo "RUN=$RUN"
echo "TAG=$TAG"
echo "BASE_RUN=$BASE_RUN"
echo "ROTATE_FIRST=$R898_PREPUSH_ROTATE_FIRST"

setsid timeout --foreground --kill-after=60s 110m "$PY" \
  "$RUN/source/train_r898.py" \
  --headless \
  --task Isaac-Go2-Soccer-Milestone2-v0 \
  --num_envs 512 \
  --seed $((87701 + SLURM_ARRAY_TASK_ID)) \
  --max_iterations 600 \
  --log_dir "$RUN/logs" \
  --run_name "$TAG" \
  --low_level_checkpoint "$BASE_RUN/logs/low_model_final.pt" \
  2>&1 | tee "$RUN/worker.log"

rc=${PIPESTATUS[0]}

if [ "$rc" -eq 0 ]   && grep -q "V4D_JOINT_TRAINING_COMPLETE" "$RUN/worker.log" 2>/dev/null   && test -s "$RUN/logs/high_model_final.pt"   && test -s "$RUN/logs/low_model_final.pt"
then
    touch "$RUN/logs/V1_1_LONG600_OK"
else
    echo "[STOP] TRAINING_COMPLETION_CONTRACT_FAILED"
    rc=1
fi

echo "$rc" > "$RUN/PROCESS_RC"
echo "PROCESS_RC=$rc"
exit "$rc"
