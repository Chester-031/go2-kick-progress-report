#!/usr/bin/env bash
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -Eeuo pipefail

: "${RUN_DIR:?RUN_DIR required}"
: "${R87_SOURCE:?R87_SOURCE required}"
: "${R87_PARENT:?R87_PARENT required}"

PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"

ISAACLAB_ROOT="/scratch/lj2634/IsaacLab_232_src_37ddf62"
ENV_ROOT="/scratch/lj2634/envs/isaaclab232_v2"
PY="${ENV_ROOT}/bin/python"
GLU_ROOT="/scratch/lj2634/vendor/libglu-9.0.0"

for path in \
    "${PY}" \
    "${R87_SOURCE}" \
    "${R87_PARENT}"
do
    test -e "${path}"
done

export LD_LIBRARY_PATH="${GLU_ROOT}/lib:${ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"

export LD_PRELOAD="${ENV_ROOT}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"

export PYTHONPATH="${PROJECT}/source/isaac_go2_low_level_rl:${ISAACLAB_ROOT}/source/isaaclab:${ISAACLAB_ROOT}/source/isaaclab_tasks:${PYTHONPATH:-}"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export GIT_PYTHON_REFRESH=quiet


RUNTIME_ROOT="/scratch/lj2634/job_runtime/${SLURM_JOB_ID}_r87"

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
# FIXED validated physical/runtime contract
# ============================================================

export V4K1_SETTLE_MAX_STEPS=150
export V4K1_SETTLE_MIN_STEPS=25
export V4K1_SETTLE_REQUIRED_STEPS=10
export V4K1_MIN_STABLE_ACTIVATION_RATE=1.0
export V4K1_BALL_ARM_STEPS=4

# Corrected simultaneous physical bridge remains unchanged.
export R852_BRIDGE_HOLD_STEPS=3


# ============================================================
# R8.7 variables
# ============================================================

export R87_LR_1CM=0.000004
export R87_LR_2CM=0.000004

export R87_GATE_1CM=0.030
export R87_GATE_2CM=0.015

# These remain informational; learned 70-D policy std is preserved.
export R87_NOMINAL_STD_1CM=0.26
export R87_NOMINAL_STD_2CM=0.26

# Keep PPO settings unchanged to isolate the curriculum change.
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
echo "R8.7 PROGRESSIVE LIFT 1CM -> 2CM"
echo "============================================================"
echo "JOB=${SLURM_JOB_ID}"
echo "PARENT=${R87_PARENT}"
echo "BALL=false"
echo "HIGH_LEVEL=false"
echo "VIDEO=false"
echo "TARGET=2cm"


set +e

srun \
    --export=ALL \
    --unbuffered \
    "${PY}" \
    "${R87_SOURCE}" \
    --task Isaac-Go2-Soccer-Milestone2-v0 \
    --low_level_checkpoint "${R87_PARENT}" \
    --log_dir "${RUN_DIR}/logs" \
    --run_name r87_lift_2cm \
    --num_envs 512 \
    --max_iterations 60 \
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
    touch "${RUN_DIR}/R87_JOB_FAILED"
    exit "${RC}"
fi

test -s \
    "${RUN_DIR}/logs/support_model_final.pt"

test -s \
    "${RUN_DIR}/logs/r8_support_summary.json"

touch \
    "${RUN_DIR}/R87_TRAINING_COMPLETE"

echo "R87_TRAINING_COMPLETE"
