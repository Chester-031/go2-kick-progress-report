#!/usr/bin/env bash
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=%x-%j.out

set -Eeuo pipefail

: "${RUN_DIR:?RUN_DIR is required}"
: "${SOURCE_CHECKPOINT:?SOURCE_CHECKPOINT is required}"
: "${R8_MODE:?R8_MODE is required}"
: "${R8_PROFILE:?R8_PROFILE is required}"
: "${R8_NUM_ENVS:?R8_NUM_ENVS is required}"
: "${R8_SOURCE:?R8_SOURCE is required}"

PROJECT_DIR="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
ISAACLAB_ROOT="${R8_ISAACLAB_ROOT:-/scratch/lj2634/IsaacLab_232_src_37ddf62}"
ENV_ROOT="${R8_ENV_ROOT:-/scratch/lj2634/envs/isaaclab232_v2}"
PY="${R8_PYTHON_BIN:-${ENV_ROOT}/bin/python}"
GLU_ROOT="${R8_GLU_ROOT:-/scratch/lj2634/vendor/libglu-9.0.0}"

test -x "${PY}"
test -d "${PROJECT_DIR}"
test -d "${ISAACLAB_ROOT}"
test -s "${R8_SOURCE}"
test -s "${SOURCE_CHECKPOINT}"
test -e "${ENV_ROOT}/lib/libstdc++.so.6"
test -e "${GLU_ROOT}/lib/libGLU.so.1"

mkdir -p "${RUN_DIR}/logs"
cp -a "${R8_SOURCE}" "${RUN_DIR}/train_go2_soccer_r8_support_skill.snapshot.py"

export LD_LIBRARY_PATH="${GLU_ROOT}/lib:${ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${ENV_ROOT}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"
export PYTHONPATH="${PROJECT_DIR}/source/isaac_go2_low_level_rl:${ISAACLAB_ROOT}/source/isaaclab:${ISAACLAB_ROOT}/source/isaaclab_tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export GIT_PYTHON_REFRESH=quiet

RUNTIME_ROOT="/scratch/lj2634/job_runtime/${SLURM_JOB_ID}_r8_support_${R8_MODE}_${R8_PROFILE}"
export HOME="${RUNTIME_ROOT}/home"
export XDG_CACHE_HOME="${RUNTIME_ROOT}/cache"
export XDG_CONFIG_HOME="${RUNTIME_ROOT}/config"
export XDG_DATA_HOME="${RUNTIME_ROOT}/data"
export TMPDIR="${RUNTIME_ROOT}/tmp"
export MPLCONFIGDIR="${RUNTIME_ROOT}/matplotlib"
mkdir -p \
  "${HOME}" \
  "${XDG_CACHE_HOME}" \
  "${XDG_CONFIG_HOME}" \
  "${XDG_DATA_HOME}" \
  "${TMPDIR}" \
  "${MPLCONFIGDIR}"

export V4K1_SETTLE_MAX_STEPS=150
export V4K1_SETTLE_MIN_STEPS=25
export V4K1_SETTLE_REQUIRED_STEPS=10
export V4K1_MIN_STABLE_ACTIVATION_RATE=1.0
export V4K1_BALL_ARM_STEPS=4

if [ "${R8_PROFILE}" = "pilot" ]; then
  export R8_STAGE1_ITERATIONS=100
  export R8_STAGE2_ITERATIONS=150
  export R8_STAGE3_ITERATIONS=175
  export R8_STAGE4_ITERATIONS=175
  export R8_ROLLOUT_STEPS=48
  export R8_CHECKPOINT_INTERVAL=50
else
  export R8_STAGE1_ITERATIONS=250
  export R8_STAGE2_ITERATIONS=300
  export R8_STAGE3_ITERATIONS=325
  export R8_STAGE4_ITERATIONS=325
  export R8_ROLLOUT_STEPS=60
  export R8_CHECKPOINT_INTERVAL=50
  export R8_FULL_MIN_STRICT_SUCCESS=0.80
fi

export R8_INIT_NOISE_STD=0.25
export R8_ENTROPY_COEF=0.006
export R8_EPOCHS=3
export R8_MINI_BATCHES=4
export R8_GAMMA=0.995
export R8_DESIRED_KL=0.012

echo "============================================================"
echo "R8 VERIFIED-RUNTIME SUPPORT-TRANSFER SKILL PPO"
echo "============================================================"
echo "JOB_ID=${SLURM_JOB_ID}"
echo "HOST=$(hostname)"
echo "MODE=${R8_MODE}"
echo "PROFILE=${R8_PROFILE}"
echo "RUN_DIR=${RUN_DIR}"
echo "PY=${PY}"
echo "ISAACLAB_ROOT=${ISAACLAB_ROOT}"
echo "RUNTIME_ROOT=${RUNTIME_ROOT}"
echo "HOME=${HOME}"
echo "XDG_CACHE_HOME=${XDG_CACHE_HOME}"

ARGS=(
  "${R8_SOURCE}"
  --task Isaac-Go2-Soccer-Milestone2-v0
  --low_level_checkpoint "${SOURCE_CHECKPOINT}"
  --log_dir "${RUN_DIR}/logs"
  --run_name "$(basename "${RUN_DIR}")"
  --num_envs "${R8_NUM_ENVS}"
  --max_iterations "${R8_MAX_ITERATIONS:-600}"
  --kick_episode_steps 80
  --kick_fixed_leg left
  --r8_profile "${R8_PROFILE}"
  --seed "${R8_SEED:-43}"
  --device cuda:0
)

if [ "${R8_MODE}" = "eval" ]; then
  : "${R8_TRAINED_CHECKPOINT:?R8_TRAINED_CHECKPOINT is required for eval}"
  : "${R8_VIDEO_PATH:?R8_VIDEO_PATH is required for eval}"
  ARGS+=(
    --enable_cameras
    --trained_low_checkpoint "${R8_TRAINED_CHECKPOINT}"
    --eval_video "${R8_VIDEO_PATH}"
    --trajectory_csv "${RUN_DIR}/r8_support_trajectory.csv"
    --eval_summary "${RUN_DIR}/r8_support_video_summary.json"
    --eval_episodes 8
    --eval_fps 25
  )
fi

cd "${RUN_DIR}"
set +e
srun \
  --export=ALL \
  --unbuffered \
  "${PY}" \
  "${ARGS[@]}" \
  2>&1 | tee "${RUN_DIR}/worker.log"
RC=${PIPESTATUS[0]}
set -e

echo "RUN_RC=${RC}" | tee -a "${RUN_DIR}/worker.log"
if [ "${RC}" -ne 0 ]; then
  touch "${RUN_DIR}/R8_SUPPORT_JOB_FAILED"
  exit "${RC}"
fi

if [ "${R8_MODE}" = "train" ]; then
  test -s "${RUN_DIR}/logs/support_model_final.pt"
  test -s "${RUN_DIR}/logs/r8_support_summary.json"
  test -s "${RUN_DIR}/logs/r8_support_contract.json"
  if [ "${R8_PROFILE}" = "full" ]; then
    test -f "${RUN_DIR}/logs/R8_SUPPORT_SKILL_ACCEPTED"
    touch "${RUN_DIR}/R8_SUPPORT_FULL_ACCEPTED"
  else
    test -f "${RUN_DIR}/logs/R8_SUPPORT_PILOT_COMPLETE"
    touch "${RUN_DIR}/R8_SUPPORT_PILOT_COMPLETE"
  fi
else
  test -s "${R8_VIDEO_PATH}"
  test -f "${R8_VIDEO_PATH}.ready"
  test -s "${RUN_DIR}/r8_support_video_summary.json"
  grep -qF 'R8_EVAL_VIDEO_READY' "${RUN_DIR}/worker.log"
  touch "${RUN_DIR}/R8_SUPPORT_VIDEO_READY"
fi
