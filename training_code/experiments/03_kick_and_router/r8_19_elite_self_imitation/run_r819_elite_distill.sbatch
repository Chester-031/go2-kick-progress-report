#!/usr/bin/env bash
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G

set -Eeuo pipefail

: "${RUN_DIR:?}"
: "${R819_SOURCE:?}"
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

RUNTIME="/scratch/lj2634/job_runtime/${SLURM_JOB_ID}_r819"

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
    "${RUN_DIR}/distill" \
    "${RUN_DIR}/audit"


export V4K1_SETTLE_MAX_STEPS=150
export V4K1_SETTLE_MIN_STEPS=25
export V4K1_SETTLE_REQUIRED_STEPS=10
export V4K1_MIN_STABLE_ACTIVATION_RATE=1.0
export V4K1_BALL_ARM_STEPS=4

export R852_BRIDGE_HOLD_STEPS=3
export R89_SWING_HOLD_STEPS=2


# ============================================================
# Self-imitation collection/distillation.
# ============================================================

export R819_DISTILL_ONLY=1

export R819_EPISODES=4096
export R819_HISTORY_STEPS=8

export R819_MAX_ELITE_SAMPLES=10000
export R819_MAX_ANCHOR_SAMPLES=6000

export R819_OPT_STEPS=250
export R819_BATCH_SIZE=512
export R819_DISTILL_LR=0.0001
export R819_ANCHOR_WEIGHT=0.20

export R819_OUTPUT_CHECKPOINT="${RUN_DIR}/distill/model_distilled.pt"
export R819_SUMMARY="${RUN_DIR}/distill/summary.json"


srun \
    --export=ALL \
    --unbuffered \
    "${PY}" \
    "${R819_SOURCE}" \
    --task Isaac-Go2-Soccer-Milestone2-v0 \
    --low_level_checkpoint "${PARENT}" \
    --log_dir "${RUN_DIR}/distill/logs" \
    --run_name r819_distill \
    --num_envs 256 \
    --max_iterations 1 \
    --kick_episode_steps 80 \
    --kick_fixed_leg left \
    --r8_profile pilot \
    --seed 43 \
    --device cuda:0 \
    --headless \
    > "${RUN_DIR}/distill/worker.log" \
    2>&1


test -s "${R819_OUTPUT_CHECKPOINT}"
test -s "${R819_SUMMARY}"

grep -qF \
    'R819_DISTILL_OK' \
    "${RUN_DIR}/distill/worker.log"

echo "R819_DISTILL_COMPLETE"


# ============================================================
# Deterministic/stochastic audit using same validated source.
# ============================================================

unset R819_DISTILL_ONLY

export R814_AUDIT_ONLY=1
export R814_EPISODES=256

export R814_CHECKPOINT_NAME=r819_distilled
export R814_SUMMARY="${RUN_DIR}/audit/audit.json"


srun \
    --export=ALL \
    --unbuffered \
    "${PY}" \
    "${R819_SOURCE}" \
    --task Isaac-Go2-Soccer-Milestone2-v0 \
    --low_level_checkpoint "${R819_OUTPUT_CHECKPOINT}" \
    --log_dir "${RUN_DIR}/audit/logs" \
    --run_name r819_audit \
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


touch "${RUN_DIR}/R819_COMPLETE"

echo "R819_SELF_IMITATION_COMPLETE"
