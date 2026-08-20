#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G

set -Eeuo pipefail

PROJECT_DIR="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
ISAACLAB_ROOT="/scratch/lj2634/IsaacLab_232_src_37ddf62"
ENV_ROOT="/scratch/lj2634/envs/isaaclab232_v2"
GLU_ROOT="/scratch/lj2634/vendor/libglu-9.0.0"

PY="${ENV_ROOT}/bin/python"
TRAIN_SCRIPT="${PROJECT_DIR}/experiments/v3_zero_sum/train_go2_soccer_v3_zero_sum.py"

: "${TASK:?TASK is required}"
: "${LOW_CHECKPOINT:?LOW_CHECKPOINT is required}"
: "${PHASEA_WARM_START_CHECKPOINT:?warm start is required}"
: "${RUN_DIR:?RUN_DIR is required}"
: "${LOG_DIR:?LOG_DIR is required}"
: "${RUN_NAME:?RUN_NAME is required}"
: "${NUM_ENVS:?NUM_ENVS is required}"
: "${MAX_ITERATIONS:?MAX_ITERATIONS is required}"
: "${SEED:?SEED is required}"

mkdir -p "${RUN_DIR}" "${LOG_DIR}"
exec > >(tee -a "${RUN_DIR}/training.log") 2>&1

echo "============================================================"
echo "V3 ZERO-SUM HIGH-LEVEL TRAINING"
echo "============================================================"
echo "JOB_ID=${SLURM_JOB_ID}"
echo "HOST=$(hostname)"
echo "RUN_DIR=${RUN_DIR}"
echo "NUM_ENVS=${NUM_ENVS}"
echo "MAX_ITERATIONS=${MAX_ITERATIONS}"
echo "WARM_START=${PHASEA_WARM_START_CHECKPOINT}"

test -x "${PY}"
test -s "${TRAIN_SCRIPT}"
test -s "${LOW_CHECKPOINT}"
test -s "${PHASEA_WARM_START_CHECKPOINT}"

export LD_LIBRARY_PATH="${GLU_ROOT}/lib:${ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${ENV_ROOT}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"
export PYTHONPATH="${PROJECT_DIR}/source/isaac_go2_low_level_rl:${ISAACLAB_ROOT}/source/isaaclab:${ISAACLAB_ROOT}/source/isaaclab_tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export GIT_PYTHON_REFRESH=quiet

RUNTIME_ROOT="/scratch/lj2634/job_runtime/${SLURM_JOB_ID}_v3_zero_sum"
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

git -C "${PROJECT_DIR}" rev-parse HEAD > "${RUN_DIR}/git_commit.txt"
git -C "${PROJECT_DIR}" branch --show-current > "${RUN_DIR}/git_branch.txt"
git -C "${PROJECT_DIR}" status --short > "${RUN_DIR}/git_status.txt"

cp -a "${TRAIN_SCRIPT}" "${RUN_DIR}/train_v3_zero_sum.snapshot.py"

sha256sum \
    "${TRAIN_SCRIPT}" \
    "${LOW_CHECKPOINT}" \
    "${PHASEA_WARM_START_CHECKPOINT}" \
    > "${RUN_DIR}/input_sha256.txt"

cat > "${RUN_DIR}/resolved_launch_config.txt" <<CONFIG
version=v3
reward_goal=1
reward_own_goal=-1
reward_all_other=0
reward_base_done=0
reward_timeout=0
dense_shaping=false
state_dependent_command_limits=false
command_slew_limit=false
low_level_frozen=true
warm_start_checkpoint=${PHASEA_WARM_START_CHECKPOINT}
warm_start_mode=actor_critic_weights_only
optimizer_loaded=false
num_envs=${NUM_ENVS}
max_iterations=${MAX_ITERATIONS}
seed=${SEED}
learning_rate=${PHASEA_LEARNING_RATE}
entropy_coef=${PHASEA_ENTROPY_COEF}
init_noise_std=${PHASEA_INIT_NOISE_STD}
clip_param=${PHASEA_CLIP_PARAM}
desired_kl=${PHASEA_DESIRED_KL}
gamma=${PHASEA_GAMMA}
obs_normalization=${PHASEA_OBS_NORMALIZATION}
CONFIG

cd "${RUN_DIR}"

set +e

srun \
    --export=ALL \
    --unbuffered \
    "${PY}" \
    "${TRAIN_SCRIPT}" \
    --task "${TASK}" \
    --low_level_checkpoint "${LOW_CHECKPOINT}" \
    --log_dir "${LOG_DIR}" \
    --run_name "${RUN_NAME}" \
    --num_envs "${NUM_ENVS}" \
    --max_iterations "${MAX_ITERATIONS}" \
    --low_level_steps_per_action 5 \
    --high_level_episode_steps 300 \
    --seed "${SEED}" \
    --device cuda:0 \
    --headless

RUN_RC=$?
set -e

echo "RUN_RC=${RUN_RC}"

MODEL_FINAL="${LOG_DIR}/model_final.pt"

if \
    [ "${RUN_RC}" -eq 0 ] && \
    [ -s "${MODEL_FINAL}" ] && \
    grep -qF '[V3_ZERO_SUM_CONTRACT]' "${RUN_DIR}/training.log" && \
    grep -qF '[V3_ZERO_SUM_PROGRESS]' "${RUN_DIR}/training.log" && \
    ! grep -qF '[PHASEA_FATAL]' "${RUN_DIR}/training.log"
then
    touch "${RUN_DIR}/V3_ZERO_SUM_TRAINING_OK"
    echo "V3_ZERO_SUM_TRAINING_ACCEPTED"
    echo "MODEL_FINAL=${MODEL_FINAL}"
    exit 0
fi

echo "V3_ZERO_SUM_TRAINING_FAILED"

grep -nEi \
    'fatal|error|exception|traceback|cuda|failed' \
    "${RUN_DIR}/training.log" \
    "${RUN_DIR}"/slurm-*.out \
    "${RUN_DIR}"/slurm-*.err \
    2>/dev/null | tail -n 160 || true

exit 2
