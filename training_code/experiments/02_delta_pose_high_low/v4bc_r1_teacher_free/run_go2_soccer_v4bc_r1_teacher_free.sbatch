#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G

set -Eeuo pipefail

PROJECT_DIR="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
ISAACLAB_ROOT="/scratch/lj2634/IsaacLab_232_src_37ddf62"
ENV_ROOT="/scratch/lj2634/envs/isaaclab232_v2"
GLU_ROOT="/scratch/lj2634/vendor/libglu-9.0.0"

PY="${ENV_ROOT}/bin/python"
TRAIN_SCRIPT="${PROJECT_DIR}/experiments/v4bc_r1_teacher_free/train_go2_soccer_v4bc_r1_teacher_free.py"

: "${TASK:?TASK is required}"
: "${HIGH_CHECKPOINT:?HIGH_CHECKPOINT is required}"
: "${SOURCE_LOW_CHECKPOINT:?SOURCE_LOW_CHECKPOINT is required}"
: "${RUN_DIR:?RUN_DIR is required}"
: "${LOG_DIR:?LOG_DIR is required}"
: "${RUN_NAME:?RUN_NAME is required}"
: "${NUM_ENVS:?NUM_ENVS is required}"
: "${MAX_ITERATIONS:?MAX_ITERATIONS is required}"
: "${SEED:?SEED is required}"

mkdir -p "${RUN_DIR}" "${LOG_DIR}"
exec > >(tee -a "${RUN_DIR}/worker.log") 2>&1

echo "============================================================"
echo "V4BC-R1 TEACHER-FREE DIRECT POSE + PUSH"
echo "============================================================"
echo "JOB_ID=${SLURM_JOB_ID}"
echo "HOST=$(hostname)"
echo "RUN_DIR=${RUN_DIR}"
echo "NUM_ENVS=${NUM_ENVS}"
echo "MAX_ITERATIONS=${MAX_ITERATIONS}"
echo "HIGH_CHECKPOINT=${HIGH_CHECKPOINT}"
echo "SOURCE_LOW_CHECKPOINT=${SOURCE_LOW_CHECKPOINT}"

test -x "${PY}"
test -s "${TRAIN_SCRIPT}"
test -s "${HIGH_CHECKPOINT}"
test -s "${SOURCE_LOW_CHECKPOINT}"

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
export V4BC_HIGH_CHECKPOINT="${HIGH_CHECKPOINT}"

RUNTIME_ROOT="/scratch/lj2634/job_runtime/${SLURM_JOB_ID}_v4bc_r1_teacher_free"

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

cp -a "${TRAIN_SCRIPT}" \
    "${RUN_DIR}/train_go2_soccer_v4bc_r1_teacher_free.snapshot.py"

git -C "${PROJECT_DIR}" rev-parse HEAD \
    > "${RUN_DIR}/git_commit.txt"

git -C "${PROJECT_DIR}" status --short \
    > "${RUN_DIR}/git_status.txt"

sha256sum \
    "${TRAIN_SCRIPT}" \
    "${HIGH_CHECKPOINT}" \
    "${SOURCE_LOW_CHECKPOINT}" \
    > "${RUN_DIR}/input_sha256.txt"

cat > "${RUN_DIR}/resolved_launch_config.txt" <<CONFIG
version=v4bc_r1_teacher_free
purpose=merge_pose_conditioned_locomotion_and_continuous_push_before_kick

high_policy=v3_zero_sum_frozen
high_checkpoint=${HIGH_CHECKPOINT}

low_policy=direct_pose_conditioned_trainable
source_low_checkpoint=${SOURCE_LOW_CHECKPOINT}
low_observation=native48_plus_pose_velocity_time_skill_ball_context
low_action=12_joint_position_residuals

skills=stand,locomotion,push
kick=false

teacher_reference=actor_observation_never
teacher_free_fraction=1.00

soccer_goal_reward=diagnostic_only
low_reward=pose_tracking_terminal_velocity_posture_push_quality

num_envs=${NUM_ENVS}
iterations=${MAX_ITERATIONS}
high_rollout_steps=${DELTA_HIGH_ROLLOUT_STEPS}
low_steps_per_high_action=5
low_learning_rate=${DELTA_LOW_LEARNING_RATE}
seed=${SEED}
CONFIG

cd "${RUN_DIR}"

set +e

srun \
    --export=ALL \
    --unbuffered \
    "${PY}" \
    "${TRAIN_SCRIPT}" \
    --task "${TASK}" \
    --low_level_checkpoint "${SOURCE_LOW_CHECKPOINT}" \
    --log_dir "${LOG_DIR}" \
    --run_name "${RUN_NAME}" \
    --num_envs "${NUM_ENVS}" \
    --max_iterations "${MAX_ITERATIONS}" \
    --low_level_steps_per_action 5 \
    --high_level_episode_steps 300 \
    --seed "${SEED}" \
    --device cuda:0 \
    --headless \
    2>&1 |
tee "${RUN_DIR}/training.log"

RUN_RC="${PIPESTATUS[0]}"
set -e

echo
echo "RUN_RC=${RUN_RC}"

LOW_MODEL="${LOG_DIR}/low_model_final.pt"
HIGH_REFERENCE="${LOG_DIR}/high_v3_frozen_reference.pt"
SUMMARY="${LOG_DIR}/v4bc_r1_summary.json"
CONTRACT="${LOG_DIR}/v4bc_r1_contract.json"

if \
    [ "${RUN_RC}" -eq 0 ] &&
    [ -s "${LOW_MODEL}" ] &&
    [ -s "${HIGH_REFERENCE}" ] &&
    [ -s "${SUMMARY}" ] &&
    [ -s "${CONTRACT}" ] &&
    [ -f "${LOG_DIR}/V4BC_R1_TEACHER_FREE_OK" ] &&
    grep -qF '[V4BC_R1_CONTRACT]' "${RUN_DIR}/training.log" &&
    grep -qF '[V4BC_R1_ITER]' "${RUN_DIR}/training.log" &&
    grep -qF 'V4BC_R1_TEACHER_FREE_TRAINING_COMPLETE' "${RUN_DIR}/training.log" &&
    ! grep -qF '[V4BC_R1_FATAL]' "${RUN_DIR}/training.log"
then
    touch "${RUN_DIR}/V4BC_R1_TEACHER_FREE_ACCEPTED"

    sha256sum \
        "${LOW_MODEL}" \
        "${HIGH_REFERENCE}" \
        "${SUMMARY}" \
        "${CONTRACT}" \
        > "${RUN_DIR}/output_sha256.txt"

    echo
    echo "V4BC_R1_TEACHER_FREE_ACCEPTED"
    echo "LOW_MODEL=${LOW_MODEL}"
    echo "HIGH_REFERENCE=${HIGH_REFERENCE}"
    echo "SUMMARY=${SUMMARY}"
    exit 0
fi

echo
echo "===== Important errors ====="

grep -nEi \
    'fatal|error|exception|traceback|mismatch|overflow|underflow|cuda|nan|failed' \
    "${RUN_DIR}/training.log" \
    "${RUN_DIR}/worker.log" \
    "${RUN_DIR}"/slurm-*.out \
    "${RUN_DIR}"/slurm-*.err \
    2>/dev/null |
tail -n 240 || true

echo
echo "V4BC_R1_TEACHER_FREE_FAILED"
exit 2
