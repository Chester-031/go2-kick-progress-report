#!/usr/bin/env bash
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G

set -Eeuo pipefail

: "${RUN_DIR:?}"
: "${R814_SOURCE:?}"
: "${CKPT_R88:?}"
: "${CKPT_R89:?}"
: "${CKPT_R811_PRE:?}"
: "${CKPT_R811_POST:?}"

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


RUNTIME="/scratch/lj2634/job_runtime/${SLURM_JOB_ID}_r814"

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


# Existing validated physics/runtime contract.
export V4K1_SETTLE_MAX_STEPS=150
export V4K1_SETTLE_MIN_STEPS=25
export V4K1_SETTLE_REQUIRED_STEPS=10
export V4K1_MIN_STABLE_ACTIVATION_RATE=1.0
export V4K1_BALL_ARM_STEPS=4

export R852_BRIDGE_HOLD_STEPS=3
export R89_SWING_HOLD_STEPS=2


# Audit only.
export R814_AUDIT_ONLY=1
export R814_EPISODES=256


run_checkpoint() {

    local name="$1"
    local checkpoint="$2"

    local dst="${RUN_DIR}/${name}"

    mkdir -p "${dst}"

    export R814_CHECKPOINT_NAME="${name}"
    export R814_SUMMARY="${dst}/audit.json"

    echo
    echo "============================================================"
    echo "R814_CHECKPOINT_START=${name}"
    echo "CHECKPOINT=${checkpoint}"
    echo "============================================================"

    srun \
        --export=ALL \
        --unbuffered \
        "${PY}" \
        "${R814_SOURCE}" \
        --task Isaac-Go2-Soccer-Milestone2-v0 \
        --low_level_checkpoint "${checkpoint}" \
        --log_dir "${dst}/logs" \
        --run_name "${name}" \
        --num_envs 128 \
        --max_iterations 1 \
        --kick_episode_steps 80 \
        --kick_fixed_leg left \
        --r8_profile pilot \
        --seed 43 \
        --device cuda:0 \
        --headless \
        > "${dst}/worker.log" \
        2>&1

    test -s "${dst}/audit.json"

    grep -qF \
        'R814_AUDIT_OK' \
        "${dst}/worker.log"

    echo "R814_CHECKPOINT_DONE=${name}"
}


run_checkpoint \
    r88_lift4_pass \
    "${CKPT_R88}"

run_checkpoint \
    r89_swing6_pass \
    "${CKPT_R89}"

run_checkpoint \
    r811_pre_recovery \
    "${CKPT_R811_PRE}"

run_checkpoint \
    r811_post_recovery \
    "${CKPT_R811_POST}"


touch "${RUN_DIR}/R814_ROOT_AUDIT_COMPLETE"

echo
echo "R814_ROOT_AUDIT_COMPLETE"
