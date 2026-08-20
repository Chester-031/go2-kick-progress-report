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
TRAIN_SCRIPT="${PROJECT_DIR}/experiments/v3a_high_level_finetune/train_go2_soccer_v3a.py"

: "${TASK:?TASK is required}"
: "${LOW_LEVEL_CHECKPOINT:?LOW_LEVEL_CHECKPOINT is required}"
: "${RUN_DIR:?RUN_DIR is required}"
: "${LOG_DIR:?LOG_DIR is required}"
: "${RUN_NAME:?RUN_NAME is required}"
: "${NUM_ENVS:?NUM_ENVS is required}"
: "${MAX_ITERATIONS:?MAX_ITERATIONS is required}"
: "${SEED:?SEED is required}"

mkdir -p "${RUN_DIR}" "${LOG_DIR}"

exec > >(tee -a "${RUN_DIR}/training.log") 2>&1

echo "============================================================"
echo "V3A HIGH-LEVEL COMMAND-STABILITY TRAINING"
echo "============================================================"
echo "JOB_ID=${SLURM_JOB_ID}"
echo "HOST=$(hostname)"
echo "TASK=${TASK}"
echo "TRAIN_SCRIPT=${TRAIN_SCRIPT}"
echo "LOW_LEVEL_CHECKPOINT=${LOW_LEVEL_CHECKPOINT}"
echo "WARM_START=${PHASEA_WARM_START_CHECKPOINT:-unset}"
echo "RUN_DIR=${RUN_DIR}"
echo "LOG_DIR=${LOG_DIR}"
echo "RUN_NAME=${RUN_NAME}"
echo "NUM_ENVS=${NUM_ENVS}"
echo "MAX_ITERATIONS=${MAX_ITERATIONS}"
echo "SEED=${SEED}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

test -x "${PY}"
test -s "${TRAIN_SCRIPT}"
test -s "${LOW_LEVEL_CHECKPOINT}"
test -s "${PHASEA_WARM_START_CHECKPOINT}"
test -e "${ENV_ROOT}/lib/libstdc++.so.6"

if [ -e "${GLU_ROOT}/lib/libGLU.so.1" ]; then
    export LD_LIBRARY_PATH="${GLU_ROOT}/lib:${ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"
else
    export LD_LIBRARY_PATH="${ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"
fi

export LD_PRELOAD="${ENV_ROOT}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"
export PYTHONPATH="${PROJECT_DIR}/source/isaac_go2_low_level_rl:${ISAACLAB_ROOT}/source/isaaclab:${ISAACLAB_ROOT}/source/isaaclab_tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export GIT_PYTHON_REFRESH=quiet

RUNTIME_ROOT="/scratch/lj2634/job_runtime/${SLURM_JOB_ID}_v3a"

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

git -C "${PROJECT_DIR}" rev-parse HEAD \
    > "${RUN_DIR}/git_commit.txt"

git -C "${PROJECT_DIR}" branch --show-current \
    > "${RUN_DIR}/git_branch.txt"

git -C "${PROJECT_DIR}" status --short \
    > "${RUN_DIR}/git_status.txt"

sha256sum \
    "${TRAIN_SCRIPT}" \
    "${LOW_LEVEL_CHECKPOINT}" \
    "${PHASEA_WARM_START_CHECKPOINT}" \
    > "${RUN_DIR}/input_sha256.txt"

cp -a "${TRAIN_SCRIPT}" \
    "${RUN_DIR}/train_go2_soccer_v3a.snapshot.py"

cat > "${RUN_DIR}/resolved_launch_config.txt" <<CONFIG
phase=v3a_high_level_only
low_level_frozen=true
task=${TASK}
low_level_checkpoint=${LOW_LEVEL_CHECKPOINT}
warm_start_checkpoint=${PHASEA_WARM_START_CHECKPOINT}
warm_start_mode=actor_critic_weights_only
optimizer_loaded=false
num_envs=${NUM_ENVS}
max_iterations=${MAX_ITERATIONS}
seed=${SEED}
low_level_steps_per_action=5
high_level_episode_steps=300
learning_rate=${PHASEA_LEARNING_RATE}
entropy_coef=${PHASEA_ENTROPY_COEF}
init_noise_std=${PHASEA_INIT_NOISE_STD}
clip_param=${PHASEA_CLIP_PARAM}
desired_kl=${PHASEA_DESIRED_KL}
gamma=${PHASEA_GAMMA}
obs_normalization=${PHASEA_OBS_NORMALIZATION}
command_global_vy_limit=0.25
command_global_wz_limit=0.55
push_distance_threshold=0.85
push_alignment_threshold=0.965
push_vy_limit=0.10
push_wz_limit=0.22
command_delta_limits=0.12,0.05,0.08
CONFIG

cd "${RUN_DIR}"

set +e

srun \
    --export=ALL \
    --unbuffered \
    "${PY}" \
    "${TRAIN_SCRIPT}" \
    --task "${TASK}" \
    --low_level_checkpoint "${LOW_LEVEL_CHECKPOINT}" \
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

echo
echo "===== Training process result ====="
echo "RUN_RC=${RUN_RC}"

find "${RUN_DIR}" \
    -maxdepth 5 \
    -type f \
    -printf '%TY-%Tm-%Td %TH:%TM:%TS | %12s | %p\n' \
    | sort \
    | tail -n 120

MODEL_FINAL="${LOG_DIR}/model_final.pt"

if \
    [ "${RUN_RC}" -eq 0 ] &&
    [ -s "${MODEL_FINAL}" ] &&
    grep -qF 'PHASEA_V3A_COMMAND_STABILITY_V1' \
        "${RUN_DIR}/train_go2_soccer_v3a.snapshot.py" &&
    grep -qF '[V3A_PROGRESS]' \
        "${RUN_DIR}/training.log" &&
    ! grep -qF '[PHASEA_FATAL]' \
        "${RUN_DIR}/training.log"
then
    touch "${RUN_DIR}/V3A_TRAINING_OK"

    echo
    echo "V3A_TRAINING_ACCEPTED"
    echo "MODEL_FINAL=${MODEL_FINAL}"
    exit 0
fi

echo
echo "===== Important errors ====="

grep -nEi \
    'fatal|error|exception|traceback|cuda|failed' \
    "${RUN_DIR}/training.log" \
    "${RUN_DIR}"/slurm-*.out \
    "${RUN_DIR}"/slurm-*.err \
    2>/dev/null \
    | tail -n 160 \
    || true

echo
echo "V3A_TRAINING_FAILED"
exit 2
