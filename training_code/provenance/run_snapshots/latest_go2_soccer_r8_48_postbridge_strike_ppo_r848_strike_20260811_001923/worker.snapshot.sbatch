#!/usr/bin/env bash
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -Eeuo pipefail

: "${RUN_DIR:?}"
: "${R848_SOURCE:?}"
: "${FREEZE_DIR:?}"

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

RUNTIME="/scratch/lj2634/job_runtime/${SLURM_JOB_ID}_r848"

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
    "${RUN_DIR}/output"

export V4K1_SETTLE_MAX_STEPS=150
export V4K1_SETTLE_MIN_STEPS=25
export V4K1_SETTLE_REQUIRED_STEPS=10
export V4K1_MIN_STABLE_ACTIVATION_RATE=1.0
export V4K1_BALL_ARM_STEPS=4

export R852_BRIDGE_HOLD_STEPS=3
export R89_SWING_HOLD_STEPS=2

export R848_STRIKE_PPO_ONLY=1
export R848_UPDATES=32
export R848_ROLLOUT_STEPS=256

export R848_FREEZE_DIR="${FREEZE_DIR}"
export R848_OUTPUT="${RUN_DIR}/output"

(
    cd "${FREEZE_DIR}"
    sha256sum -c SHA256SUMS
) > "${RUN_DIR}/freeze_integrity.log"

srun \
    --export=ALL \
    --unbuffered \
    "${PY}" \
    "${R848_SOURCE}" \
    --task Isaac-Go2-Soccer-Milestone2-v0 \
    --low_level_checkpoint "${FREEZE_DIR}/artifacts/r821_lift_base.pt" \
    --log_dir "${RUN_DIR}/logs" \
    --run_name r848_strike_ppo \
    --num_envs 128 \
    --max_iterations 1 \
    --kick_episode_steps 80 \
    --kick_fixed_leg left \
    --r8_profile pilot \
    --seed 43 \
    --device cuda:0 \
    --headless \
    > "${RUN_DIR}/worker.log" \
    2>&1

test -s \
    "${RUN_DIR}/output/summary.json"

test -s \
    "${RUN_DIR}/output/winner_strike_u32.pt"

grep -qF \
    'R848_STRIKE_PPO_OK' \
    "${RUN_DIR}/worker.log"

touch \
    "${RUN_DIR}/R848_COMPLETE"

echo "R848_COMPLETE"
