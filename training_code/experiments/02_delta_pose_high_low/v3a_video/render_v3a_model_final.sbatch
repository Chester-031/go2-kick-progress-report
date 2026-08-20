#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=00:35:00

set -Eeuo pipefail

PROJECT_DIR="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
ISAACLAB_ROOT="/scratch/lj2634/IsaacLab_232_src_37ddf62"
ENV_ROOT="/scratch/lj2634/envs/isaaclab232_v2"
GLU_ROOT="/scratch/lj2634/vendor/libglu-9.0.0"

PY="${ENV_ROOT}/bin/python"
PLAYER="${PROJECT_DIR}/experiments/v3a_video/play_v3a_model_final.py"

: "${LOW_CHECKPOINT:?LOW_CHECKPOINT is required}"
: "${V3A_EVAL_CHECKPOINT:?V3A_EVAL_CHECKPOINT is required}"
: "${V3A_EVAL_VIDEO:?V3A_EVAL_VIDEO is required}"
: "${V3A_EVAL_CSV:?V3A_EVAL_CSV is required}"
: "${V3A_VIDEO_DIR:?V3A_VIDEO_DIR is required}"

mkdir -p "${V3A_VIDEO_DIR}"

RUNTIME_ROOT="/scratch/lj2634/job_runtime/${SLURM_JOB_ID}_v3a_video"

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

export LD_LIBRARY_PATH="${GLU_ROOT}/lib:${ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${ENV_ROOT}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"

export PYTHONPATH="${PROJECT_DIR}/source/isaac_go2_low_level_rl:${ISAACLAB_ROOT}/source/isaaclab:${ISAACLAB_ROOT}/source/isaaclab_tasks:${PYTHONPATH:-}"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export GIT_PYTHON_REFRESH=quiet

# Do not reload the original V2 checkpoint through the training warm-start hook.
unset PHASEA_WARM_START_CHECKPOINT || true

export PHASEA_LEARNING_RATE="0.00015"
export PHASEA_ENTROPY_COEF="0.0015"
export PHASEA_INIT_NOISE_STD="0.30"
export PHASEA_CLIP_PARAM="0.12"
export PHASEA_DESIRED_KL="0.003"
export PHASEA_GAMMA="0.99"
export PHASEA_OBS_NORMALIZATION="0"

RUN_LOG="${V3A_VIDEO_DIR}/render.log"

rm -f \
    "${V3A_EVAL_VIDEO}" \
    "${V3A_EVAL_CSV}"

echo "============================================================"
echo "V3A MODEL-FINAL DETERMINISTIC VIDEO"
echo "============================================================"
echo "JOB_ID=${SLURM_JOB_ID}"
echo "HOST=$(hostname)"
echo "CHECKPOINT=${V3A_EVAL_CHECKPOINT}"
echo "VIDEO=${V3A_EVAL_VIDEO}"
echo "CSV=${V3A_EVAL_CSV}"

set +e

srun \
    --export=ALL \
    --unbuffered \
    "${PY}" \
    "${PLAYER}" \
    --task Isaac-Go2-Soccer-Milestone2-v0 \
    --low_level_checkpoint "${LOW_CHECKPOINT}" \
    --log_dir "${V3A_VIDEO_DIR}/eval_logs" \
    --run_name v3a_model_final_video \
    --num_envs 1 \
    --max_iterations 1 \
    --low_level_steps_per_action 5 \
    --high_level_episode_steps 300 \
    --seed 42 \
    --device cuda:0 \
    --headless \
    --enable_cameras \
    2>&1 | tee "${RUN_LOG}"

RUN_RC="${PIPESTATUS[0]}"

set -e

echo
echo "RUN_RC=${RUN_RC}"

if \
    [ "${RUN_RC}" -eq 0 ] &&
    [ -s "${V3A_EVAL_VIDEO}" ] &&
    [ -s "${V3A_EVAL_CSV}" ] &&
    grep -qF 'V3A_EVAL_VIDEO_OK' "${RUN_LOG}"
then
    sha256sum "${V3A_EVAL_VIDEO}" \
        > "${V3A_EVAL_VIDEO}.sha256"

    touch "${V3A_VIDEO_DIR}/V3A_VIDEO_OK"

    echo
    echo "V3A_VIDEO_ACCEPTED"
    echo "VIDEO=${V3A_EVAL_VIDEO}"
    echo "CSV=${V3A_EVAL_CSV}"

    ls -lh \
        "${V3A_EVAL_VIDEO}" \
        "${V3A_EVAL_CSV}"

    exit 0
fi

echo
echo "===== Important video errors ====="

grep -nEi \
    'fatal|error|exception|traceback|blank|uniform|failed|vulkan|neuray|libglu' \
    "${RUN_LOG}" |
tail -n 160 || true

echo
echo "V3A_VIDEO_FAILED"

exit 2
