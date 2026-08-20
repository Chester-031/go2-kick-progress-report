#!/usr/bin/env bash
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=52G

set -Eeuo pipefail

: "${RUN_DIR:?}"
: "${R821_SOURCE:?}"
: "${AUDIT_SOURCE:?}"
: "${PARENT:?}"

PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
ISAACLAB_ROOT="/scratch/lj2634/IsaacLab_232_src_37ddf62"
ENV_ROOT="/scratch/lj2634/envs/isaaclab232_v2"
GLU_ROOT="/scratch/lj2634/vendor/libglu-9.0.0"

PY="${ENV_ROOT}/bin/python"

export LD_LIBRARY_PATH="${GLU_ROOT}/lib:${ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${ENV_ROOT}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"

export PYTHONPATH="${PROJECT}/source/isaac_go2_low_level_rl:${ISAACLAB_ROOT}/source/isaaclab:${ISAACLAB_ROOT}/source/isaaclab_tasks:${PYTHONPATH:-}"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export GIT_PYTHON_REFRESH=quiet

RUNTIME="/scratch/lj2634/job_runtime/${SLURM_JOB_ID}_r821"

export HOME="${RUNTIME}/home"
export XDG_CACHE_HOME="${RUNTIME}/cache"
export XDG_CONFIG_HOME="${RUNTIME}/config"
export XDG_DATA_HOME="${RUNTIME}/data"
export TMPDIR="${RUNTIME}/tmp"
export MPLCONFIGDIR="${RUNTIME}/mpl"

mkdir -p \
    "${HOME}" \
    "${XDG_CACHE_HOME}" \
    "${XDG_CONFIG_HOME}" \
    "${XDG_DATA_HOME}" \
    "${TMPDIR}" \
    "${MPLCONFIGDIR}" \
    "${RUN_DIR}/train/logs" \
    "${RUN_DIR}/audit"


# Fixed validated physics.
export V4K1_SETTLE_MAX_STEPS=150
export V4K1_SETTLE_MIN_STEPS=25
export V4K1_SETTLE_REQUIRED_STEPS=10
export V4K1_MIN_STABLE_ACTIVATION_RATE=1.0
export V4K1_BALL_ARM_STEPS=4

export R852_BRIDGE_HOLD_STEPS=3
export R89_SWING_HOLD_STEPS=2


# Actual std control was verified by R8.15.
export R815_FORCE_STD=0.20

export R821_ITERATIONS=10
export R821_LR=0.000004

export R8_ENTROPY_COEF=0.0
export R8_ROLLOUT_STEPS=48
export R8_EPOCHS=3
export R8_MINI_BATCHES=4
export R8_GAMMA=0.995
export R8_DESIRED_KL=0.012

export R8_CHECKPOINT_INTERVAL=5
export R84_GATE_WINDOW=20
export R84_GATE_MIN_EPISODES=256


echo "R821_LIFT_TRAIN_START"

srun \
    --export=ALL \
    --unbuffered \
    "${PY}" \
    "${R821_SOURCE}" \
    --task Isaac-Go2-Soccer-Milestone2-v0 \
    --low_level_checkpoint "${PARENT}" \
    --log_dir "${RUN_DIR}/train/logs" \
    --run_name r821_lift_only \
    --num_envs 256 \
    --max_iterations 10 \
    --kick_episode_steps 80 \
    --kick_fixed_leg left \
    --r8_profile pilot \
    --seed 43 \
    --device cuda:0 \
    --headless \
    > "${RUN_DIR}/train/worker.log" \
    2>&1

test -s "${RUN_DIR}/train/logs/support_model_final.pt"

grep -qF \
    '[R815_STD_APPLIED]' \
    "${RUN_DIR}/train/worker.log"

echo "R821_LIFT_TRAIN_COMPLETE"


MODEL="${RUN_DIR}/train/logs/support_model_final.pt"


# ============================================================
# Deterministic/stochastic audit.
# ============================================================

export R814_AUDIT_ONLY=1
export R814_EPISODES=256
export R814_CHECKPOINT_NAME=r821_lift_only
export R814_SUMMARY="${RUN_DIR}/audit/audit.json"

srun \
    --export=ALL \
    --unbuffered \
    "${PY}" \
    "${AUDIT_SOURCE}" \
    --task Isaac-Go2-Soccer-Milestone2-v0 \
    --low_level_checkpoint "${MODEL}" \
    --log_dir "${RUN_DIR}/audit/logs" \
    --run_name r821_lift_audit \
    --num_envs 128 \
    --max_iterations 1 \
    --kick_episode_steps 80 \
    --kick_fixed_leg left \
    --r8_profile pilot \
    --seed 43 \
    --device cuda:0 \
    --headless \
    > "${RUN_DIR}/audit/worker.log" \
    2>&1

test -s "${RUN_DIR}/audit/audit.json"

grep -qF \
    'R814_AUDIT_OK' \
    "${RUN_DIR}/audit/worker.log"

touch "${RUN_DIR}/R821_COMPLETE"

echo "R821_INDEPENDENT_LIFT_COMPLETE"
