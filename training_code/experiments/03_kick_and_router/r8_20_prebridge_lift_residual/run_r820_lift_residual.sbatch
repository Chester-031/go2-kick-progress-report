#!/usr/bin/env bash
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G

set -Eeuo pipefail

: "${RUN_DIR:?}"
: "${R820_SOURCE:?}"
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


RUNTIME="/scratch/lj2634/job_runtime/${SLURM_JOB_ID}_r820"

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
    "${RUN_DIR}"


# Existing validated physics.
export V4K1_SETTLE_MAX_STEPS=150
export V4K1_SETTLE_MIN_STEPS=25
export V4K1_SETTLE_REQUIRED_STEPS=10
export V4K1_MIN_STABLE_ACTIVATION_RATE=1.0
export V4K1_BALL_ARM_STEPS=4

export R852_BRIDGE_HOLD_STEPS=3
export R89_SWING_HOLD_STEPS=2


# Structural pilot.
export R820_RESIDUAL_ONLY=1

export R820_EPISODES=4096
export R820_HISTORY_STEPS=8

export R820_MAX_ELITE=12000
export R820_MAX_ANCHOR=5000

export R820_OPT_STEPS=300
export R820_BATCH=512

export R820_LR=0.0003
export R820_ANCHOR_WEIGHT=0.05

# Bound residual magnitude.
export R820_RESIDUAL_SCALE=0.30

export R820_EVAL_EPISODES=1024

export R820_BUNDLE="${RUN_DIR}/lift_residual_bundle.pt"
export R820_SUMMARY="${RUN_DIR}/summary.json"


srun \
    --export=ALL \
    --unbuffered \
    "${PY}" \
    "${R820_SOURCE}" \
    --task Isaac-Go2-Soccer-Milestone2-v0 \
    --low_level_checkpoint "${PARENT}" \
    --log_dir "${RUN_DIR}/logs" \
    --run_name r820_lift_residual \
    --num_envs 256 \
    --max_iterations 1 \
    --kick_episode_steps 80 \
    --kick_fixed_leg left \
    --r8_profile pilot \
    --seed 43 \
    --device cuda:0 \
    --headless \
    > "${RUN_DIR}/worker.log" \
    2>&1


test -s "${RUN_DIR}/summary.json"
test -s "${RUN_DIR}/lift_residual_bundle.pt"

grep -qF \
    'R820_RESIDUAL_OK' \
    "${RUN_DIR}/worker.log"

touch "${RUN_DIR}/R820_COMPLETE"

echo "R820_STRUCTURAL_PILOT_COMPLETE"
