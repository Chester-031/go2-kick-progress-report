#!/bin/bash
# Strict push canary sweep v2: node-local caches, durable worker.log, and
# artifact-backed success.  No job may report success without 80 iterations.
#SBATCH --account=torch_pr_1066_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-2
#SBATCH --job-name=r898-push-canary
#SBATCH --output=/scratch/lj2634/runs/go2_soccer_r8_98_restore_native_command_probe/strict_push_canary_v2_%A_%a.out
#SBATCH --error=/scratch/lj2634/runs/go2_soccer_r8_98_restore_native_command_probe/strict_push_canary_v2_%A_%a.err

set -u
set -o pipefail

PROJECT=/scratch/lj2634/Isaac-Go2-Low-Level-RL
ROOT=/scratch/lj2634/runs/go2_soccer_r8_98_restore_native_command_probe
PY=/scratch/lj2634/envs/isaaclab232_v2/bin/python
SOURCE=/scratch/lj2634/incoming_r898_fix/train_r898_hard_fix_v4.py

BASE_RUN=$(find "$ROOT" -maxdepth 1 -type d \
  -name 'hard_fix_long600_v2_[0-9]*' -print 2>/dev/null | sort | tail -n 1)
if [ -z "$BASE_RUN" ]; then
  echo "[STOP] hard_fix_long600_v2 checkpoint not found"
  exit 2
fi

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) TAG=gate025; STAGING_MAX=0.25; CONTACT_MIN=0.50; CONTACT_MAX=0.80; BODY_X_MIN=0.15; GOAL_BODY_X_MIN=0.35 ;;
  1) TAG=gate035; STAGING_MAX=0.35; CONTACT_MIN=0.45; CONTACT_MAX=0.85; BODY_X_MIN=0.10; GOAL_BODY_X_MIN=0.25 ;;
  2) TAG=gate045; STAGING_MAX=0.45; CONTACT_MIN=0.45; CONTACT_MAX=0.95; BODY_X_MIN=0.05; GOAL_BODY_X_MIN=0.20 ;;
  *) echo "[STOP] unknown array index=${SLURM_ARRAY_TASK_ID}"; exit 2 ;;
esac

RUN="$ROOT/strict_push_canary_v2_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${TAG}"
mkdir -p "$RUN/source" "$RUN/logs"
cp "$SOURCE" "$RUN/source/train_r898.py"

# Never write caches into the quota-limited login home.  Prefer the job-local
# filesystem and retain a scratch fallback for clusters without /state.
NODE_CACHE=/state/partition1/job-${SLURM_JOB_ID}/r898_cache
if ! mkdir -p "$NODE_CACHE" 2>/dev/null; then
  NODE_CACHE="$ROOT/node_cache_${SLURM_JOB_ID}"
  mkdir -p "$NODE_CACHE"
fi
mkdir -p \
  "$NODE_CACHE/xdg" \
  "$NODE_CACHE/warp" \
  "$NODE_CACHE/nvidia" \
  "$NODE_CACHE/torch" \
  "$NODE_CACHE/pycache" \
  "$NODE_CACHE/tmp" \
  "$NODE_CACHE/omniverse"

export XDG_CACHE_HOME="$NODE_CACHE/xdg"
export WARP_CACHE_PATH="$NODE_CACHE/warp"
export CUDA_CACHE_PATH="$NODE_CACHE/nvidia"
export TORCH_HOME="$NODE_CACHE/torch"
export PYTHONPYCACHEPREFIX="$NODE_CACHE/pycache"
export TMPDIR="$NODE_CACHE/tmp"
export OMNI_USER_DIR="$NODE_CACHE/omniverse"

export PYTHONPATH="$PROJECT/source/isaac_go2_low_level_rl:$PROJECT/source:$RUN/source:${PYTHONPATH:-}"
export V4D_HIGH_CHECKPOINT="$BASE_RUN/logs/high_model_final.pt"
export R876_HARD_GEOMETRY_V2=1
export R876_HARD_REWARD_V2=1
export R898_STRATEGY_FIX=1
export R898_STAGING_PUSH_GATE=1
export R898_STRICT_PUSH_GATE=1
export R898_ACTION_SCALE=0.75
export R898_STAGING_MAX="$STAGING_MAX"
export R898_CONTACT_MIN="$CONTACT_MIN"
export R898_CONTACT_MAX="$CONTACT_MAX"
export R898_BODY_X_MIN="$BODY_X_MIN"
export R898_GOAL_BODY_X_MIN="$GOAL_BODY_X_MIN"
export R874B_FREEZE_LOW=1
export R876_EASY_END=0
export R876_INTERMEDIATE_END=0
export R876_HARD_END=80
export V4D_STAGE1_ITERATIONS=20
export V4D_STAGE2_ITERATIONS=45
export V4D_STAGE3_ITERATIONS=15
export V4D_HIGH_LR_STAGE1=1e-5
export V4D_HIGH_LR_STAGE2=5e-6
export V4D_HIGH_LR_STAGE3=2e-6
export R873_HEADLESS_STRIP=1
export QT_QPA_PLATFORM=offscreen
export PYTHONUNBUFFERED=1
unset R890_FORCE_LOCOMOTION_PRIMITIVES

echo "CANARY_START tag=$TAG staging_max=$STAGING_MAX contact=[$CONTACT_MIN,$CONTACT_MAX] body_x_min=$BODY_X_MIN goal_body_x_min=$GOAL_BODY_X_MIN"
echo "CANARY_RUN=$RUN"
echo "NODE_CACHE=$NODE_CACHE"
# R898_RUNTIME_HOME_FIX
RUNTIME="$NODE_CACHE/runtime"
export HOME="$RUNTIME/home"
export XDG_CACHE_HOME="$RUNTIME/cache"
export XDG_CONFIG_HOME="$RUNTIME/config"
export XDG_DATA_HOME="$RUNTIME/data"
export TMPDIR="$RUNTIME/tmp"
export MPLCONFIGDIR="$RUNTIME/mpl"
export OMNI_USER_HOME="$RUNTIME/omni"
export LD_LIBRARY_PATH="/scratch/lj2634/vendor/libglu-9.0.0/lib:${LD_LIBRARY_PATH:-}"

mkdir -p \
  "$HOME" \
  "$XDG_CACHE_HOME" \
  "$XDG_CONFIG_HOME" \
  "$XDG_DATA_HOME" \
  "$TMPDIR" \
  "$MPLCONFIGDIR" \
  "$OMNI_USER_HOME"


setsid timeout --foreground --kill-after=30s 25m "$PY" \
  "$RUN/source/train_r898.py" \
  --headless \
  --task Isaac-Go2-Soccer-Milestone2-v0 \
  --num_envs "${R898_NUM_ENVS:-512}" \
  --seed $((87701 + SLURM_ARRAY_TASK_ID)) \
  --max_iterations 80 \
  --log_dir "$RUN/logs" \
  --run_name "R898_STRICT_PUSH_CANARY_${TAG}" \
  --low_level_checkpoint "$BASE_RUN/logs/low_model_final.pt" \
  2>&1 | tee "$RUN/worker.log"
rc=${PIPESTATUS[0]}

echo "$rc" > "$RUN/PROCESS_RC"
echo "CANARY_PROCESS_RC=$rc"

iterations=$(grep -c '^\[V4D_JOINT_ITER\]' "$RUN/worker.log" 2>/dev/null || true)
echo "CANARY_ITERATIONS=$iterations"

if [ "$rc" -ne 0 ]; then
  echo "[STOP] player failed rc=$rc"
  exit "$rc"
fi
if [ "$iterations" -ne 80 ]; then
  echo "[STOP] expected 80 iterations, observed $iterations"
  exit 91
fi
if [ ! -f "$RUN/logs/V4D_JOINT_OK" ]; then
  echo "[STOP] V4D_JOINT_OK missing"
  exit 92
fi
if [ ! -s "$RUN/logs/high_model_final.pt" ]; then
  echo "[STOP] high_model_final.pt missing"
  exit 93
fi

echo "CANARY_SUCCESS tag=$TAG iterations=$iterations"
