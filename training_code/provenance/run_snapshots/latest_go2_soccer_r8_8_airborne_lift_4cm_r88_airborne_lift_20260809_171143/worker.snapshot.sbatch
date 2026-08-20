#!/usr/bin/env bash
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -Eeuo pipefail

: "${RUN_DIR:?RUN_DIR required}"
: "${R88_SOURCE:?R88_SOURCE required}"
: "${R88_PARENT:?R88_PARENT required}"

PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"

ISAACLAB_ROOT="/scratch/lj2634/IsaacLab_232_src_37ddf62"
ENV_ROOT="/scratch/lj2634/envs/isaaclab232_v2"
PY="${ENV_ROOT}/bin/python"
GLU_ROOT="/scratch/lj2634/vendor/libglu-9.0.0"

for path in \
    "${PY}" \
    "${R88_SOURCE}" \
    "${R88_PARENT}"
do
    test -e "${path}"
done

export LD_LIBRARY_PATH="${GLU_ROOT}/lib:${ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"

export LD_PRELOAD="${ENV_ROOT}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"

export PYTHONPATH="${PROJECT}/source/isaac_go2_low_level_rl:${ISAACLAB_ROOT}/source/isaaclab:${ISAACLAB_ROOT}/source/isaaclab_tasks:${PYTHONPATH:-}"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export GIT_PYTHON_REFRESH=quiet


RUNTIME_ROOT="/scratch/lj2634/job_runtime/${SLURM_JOB_ID}_r88"

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
    "${MPLCONFIGDIR}" \
    "${RUN_DIR}/logs"


# ============================================================
# FIXED physical/runtime contract.
# ============================================================

export V4K1_SETTLE_MAX_STEPS=150
export V4K1_SETTLE_MIN_STEPS=25
export V4K1_SETTLE_REQUIRED_STEPS=10
export V4K1_MIN_STABLE_ACTIVATION_RATE=1.0
export V4K1_BALL_ARM_STEPS=4

export R852_BRIDGE_HOLD_STEPS=3


# ============================================================
# R8.8 curriculum variables.
# ============================================================

export R88_LR_2CM=0.000004
export R88_LR_3CM=0.000004
export R88_LR_4CM=0.000003

export R88_GATE_2CM=0.012
export R88_GATE_3CM=0.008
export R88_GATE_4CM=0.004

# Informational only.
# Existing learned 70-D action std is preserved.
export R88_NOMINAL_STD_2CM=0.26
export R88_NOMINAL_STD_3CM=0.26
export R88_NOMINAL_STD_4CM=0.26


# Keep PPO structure unchanged.
export R8_ROLLOUT_STEPS=48
export R8_ENTROPY_COEF=0.004
export R8_EPOCHS=3
export R8_MINI_BATCHES=4
export R8_GAMMA=0.995
export R8_DESIRED_KL=0.012

export R8_CHECKPOINT_INTERVAL=5

export R84_GATE_WINDOW=20
export R84_GATE_MIN_EPISODES=256


echo "============================================================"
echo "R8.8 2CM -> 3CM -> 4CM AIRBORNE LIFT"
echo "============================================================"
echo "JOB=${SLURM_JOB_ID}"
echo "PARENT=${R88_PARENT}"
echo "BALL=false"
echo "HIGH_LEVEL=false"
echo "VIDEO=false"
echo "FORWARD_SWING=false"


set +e

srun \
    --export=ALL \
    --unbuffered \
    "${PY}" \
    "${R88_SOURCE}" \
    --task Isaac-Go2-Soccer-Milestone2-v0 \
    --low_level_checkpoint "${R88_PARENT}" \
    --log_dir "${RUN_DIR}/logs" \
    --run_name r88_airborne_lift \
    --num_envs 512 \
    --max_iterations 70 \
    --kick_episode_steps 80 \
    --kick_fixed_leg left \
    --r8_profile pilot \
    --seed 43 \
    --device cuda:0 \
    --headless \
    2>&1 |
tee "${RUN_DIR}/worker.log"

RC=${PIPESTATUS[0]}

set -e

echo "RUN_RC=${RC}" |
tee -a "${RUN_DIR}/worker.log"

if [ "${RC}" -ne 0 ]; then
    touch "${RUN_DIR}/R88_JOB_FAILED"
    exit "${RC}"
fi

test -s \
    "${RUN_DIR}/logs/support_model_final.pt"

test -s \
    "${RUN_DIR}/logs/r8_support_summary.json"

touch \
    "${RUN_DIR}/R88_TRAINING_COMPLETE"

echo "R88_TRAINING_COMPLETE"
