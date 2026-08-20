#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G

set -Eeuo pipefail

PROJECT_DIR="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
ISAACLAB_ROOT="/scratch/lj2634/IsaacLab_232_src_37ddf62"
ENV_ROOT="/scratch/lj2634/envs/isaaclab232_v2"
GLU_ROOT="/scratch/lj2634/vendor/libglu-9.0.0"

PY="${ENV_ROOT}/bin/python"
TRAIN_SCRIPT="${PROJECT_DIR}/experiments/v4a_skill_router/train_go2_soccer_v4a_skill_router.py"

: "${TASK:?TASK is required}"
: "${HIGH_CHECKPOINT:?HIGH_CHECKPOINT is required}"
: "${LOW_CHECKPOINT:?LOW_CHECKPOINT is required}"
: "${RUN_DIR:?RUN_DIR is required}"
: "${LOG_DIR:?LOG_DIR is required}"
: "${RUN_NAME:?RUN_NAME is required}"
: "${NUM_ENVS:?NUM_ENVS is required}"
: "${MAX_ITERATIONS:?MAX_ITERATIONS is required}"
: "${SEED:?SEED is required}"

mkdir -p "${RUN_DIR}" "${LOG_DIR}"

exec > >(tee -a "${RUN_DIR}/worker.log") 2>&1

echo "============================================================"
echo "V4A SKILL-ROUTER PROTOTYPE"
echo "============================================================"
echo "JOB_ID=${SLURM_JOB_ID}"
echo "HOST=$(hostname)"
echo "RUN_DIR=${RUN_DIR}"
echo "NUM_ENVS=${NUM_ENVS}"
echo "MAX_ITERATIONS=${MAX_ITERATIONS}"
echo "HIGH_CHECKPOINT=${HIGH_CHECKPOINT}"
echo "LOW_CHECKPOINT=${LOW_CHECKPOINT}"

test -x "${PY}"
test -s "${TRAIN_SCRIPT}"
test -s "${HIGH_CHECKPOINT}"
test -s "${LOW_CHECKPOINT}"
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

RUNTIME_ROOT="/scratch/lj2634/job_runtime/${SLURM_JOB_ID}_v4a_skill_router"

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

cp -a \
    "${TRAIN_SCRIPT}" \
    "${RUN_DIR}/train_go2_soccer_v4a_skill_router.snapshot.py"

git -C "${PROJECT_DIR}" rev-parse HEAD \
    > "${RUN_DIR}/git_commit.txt"

git -C "${PROJECT_DIR}" status --short \
    > "${RUN_DIR}/git_status.txt"

sha256sum \
    "${TRAIN_SCRIPT}" \
    "${HIGH_CHECKPOINT}" \
    "${LOW_CHECKPOINT}" \
    > "${RUN_DIR}/input_sha256.txt"

cat > "${RUN_DIR}/resolved_launch_config.txt" <<CONFIG
version=v4a_skill_router_smoke
purpose=test_skill_library_before_training_a_physical_kick_expert

high_network_output=forward_intent,lateral_intent,yaw_intent
high_network_output_dimension=3
v3_actor_head_compatible=true

skill_0=stand
skill_1=locomotion
skill_2=push
skill_3=kick_request

stand_execution=zero_velocity_command
locomotion_execution=frozen_round2_velocity_expert
push_execution=persistent_forward_velocity_expert
kick_request_execution=push_fallback
learned_kick_expert_available=false

high_policy_trainable=true
low_locomotion_expert_trainable=false

high_reward=goal:+1,own_goal:-1,all_other:0
low_reward=not_used_for_training
low_locomotion_reward=diagnostic_only

high_checkpoint=${HIGH_CHECKPOINT}
low_checkpoint=${LOW_CHECKPOINT}
num_envs=${NUM_ENVS}
iterations=${MAX_ITERATIONS}
high_rollout_steps=${DELTA_HIGH_ROLLOUT_STEPS}
low_steps_per_high_action=5
high_learning_rate=${DELTA_HIGH_LEARNING_RATE}
high_init_noise_std=${DELTA_HIGH_INIT_NOISE_STD}
seed=${SEED}
CONFIG

cd "${RUN_DIR}"

export DELTA_HIGH_CHECKPOINT="${HIGH_CHECKPOINT}"

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
    --headless \
    2>&1 |
tee "${RUN_DIR}/training.log"

RUN_RC="${PIPESTATUS[0]}"

set -e

echo
echo "RUN_RC=${RUN_RC}"

HIGH_MODEL="${LOG_DIR}/high_model_final.pt"
LOW_REFERENCE="${LOG_DIR}/low_model_frozen_reference.pt"
SUMMARY="${LOG_DIR}/v4a_skill_summary.json"
CONTRACT="${LOG_DIR}/v4a_skill_contract.json"

if \
    [ "${RUN_RC}" -eq 0 ] &&
    [ -s "${HIGH_MODEL}" ] &&
    [ -s "${LOW_REFERENCE}" ] &&
    [ -s "${SUMMARY}" ] &&
    [ -s "${CONTRACT}" ] &&
    [ -f "${LOG_DIR}/V4A_SKILL_ROUTER_OK" ] &&
    grep -qF '[V4A_CONTRACT]' "${RUN_DIR}/training.log" &&
    grep -qF '[V4A_HIGH_ITER]' "${RUN_DIR}/training.log" &&
    grep -qF 'V4A_SKILL_ROUTER_TRAINING_COMPLETE' "${RUN_DIR}/training.log" &&
    ! grep -qF '[V4A_FATAL]' "${RUN_DIR}/training.log"
then
    touch "${RUN_DIR}/V4A_SKILL_ROUTER_ACCEPTED"

    sha256sum \
        "${HIGH_MODEL}" \
        "${LOW_REFERENCE}" \
        "${SUMMARY}" \
        "${CONTRACT}" \
        > "${RUN_DIR}/output_sha256.txt"

    echo
    echo "V4A_SKILL_ROUTER_ACCEPTED"
    echo "HIGH_MODEL=${HIGH_MODEL}"
    echo "LOW_REFERENCE=${LOW_REFERENCE}"
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
echo "V4A_SKILL_ROUTER_FAILED"
exit 2
