#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -Eeuo pipefail

PROJECT_DIR="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
ISAACLAB_ROOT="/scratch/lj2634/IsaacLab_232_src_37ddf62"
ENV_ROOT="/scratch/lj2634/envs/isaaclab232_v2"
GLU_ROOT="/scratch/lj2634/vendor/libglu-9.0.0"

PYTHON="${ENV_ROOT}/bin/python"
PLAYER="${PROJECT_DIR}/experiments/sway_diagnostics/cd_direct/play_phaseA_cd_direct.py"

: "${TEST_LABEL:?}"
: "${TEST_MODE:?}"
: "${RESULT_DIR:?}"
: "${LOW_CHECKPOINT:?}"
: "${V2_CHECKPOINT:?}"
: "${FIXED_VX:?}"
: "${FIXED_VY:?}"
: "${FIXED_WZ:?}"
: "${EVAL_STEPS:?}"

mkdir -p "${RESULT_DIR}"

RUNTIME_ROOT="/scratch/lj2634/job_runtime/${SLURM_JOB_ID}_${TEST_LABEL}"

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

export PYTHONPATH="${PROJECT_DIR}/source/isaac_go2_low_level_rl:${ISAACLAB_ROOT}/source/isaaclab:${ISAACLAB_ROOT}/source/isaaclab_tasks:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${GLU_ROOT}/lib:${ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${ENV_ROOT}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export GIT_PYTHON_REFRESH=quiet

export PHASEA_CD_DIAG_DIR="${RESULT_DIR}"
export PHASEA_CD_DIAG_LABEL="${TEST_LABEL}"
export PHASEA_CD_DIAG_MODE="${TEST_MODE}"
export PHASEA_CD_FIXED_VX="${FIXED_VX}"
export PHASEA_CD_FIXED_VY="${FIXED_VY}"
export PHASEA_CD_FIXED_WZ="${FIXED_WZ}"

PLACEHOLDER="${RESULT_DIR}/placeholder.mp4"
LOG="${RESULT_DIR}/diagnostic.log"

set +e

srun \
    --export=ALL \
    --unbuffered \
    "${PYTHON}" \
    "${PLAYER}" \
    --task Isaac-Go2-Soccer-Milestone2-v0 \
    --low_level_checkpoint "${LOW_CHECKPOINT}" \
    --log_dir "${RESULT_DIR}/eval" \
    --run_name "${TEST_LABEL}" \
    --num_envs 1 \
    --max_iterations 1 \
    --low_level_steps_per_action 5 \
    --high_level_episode_steps 300 \
    --seed 42 \
    --device cuda:0 \
    --headless \
    --enable_cameras \
    --eval_checkpoint "${V2_CHECKPOINT}" \
    --eval_video "${PLACEHOLDER}" \
    --eval_steps "${EVAL_STEPS}" \
    --eval_fps 30 \
    2>&1 | tee "${LOG}"

RC="${PIPESTATUS[0]}"

set -e

if \
    [ "${RC}" -eq 0 ] &&
    [ -s "${RESULT_DIR}/low_level.csv" ] &&
    [ -s "${RESULT_DIR}/high_level.csv" ] &&
    ! grep -q '\[PHASEA_FATAL\]' "${LOG}"
then
    echo "GO2_CD_TEST_ACCEPTED=${TEST_LABEL}"
    exit 0
fi

echo "GO2_CD_TEST_FAILED=${TEST_LABEL}"
echo "PLAYER_RC=${RC}"

grep -nEi \
    'fatal|error|exception|traceback|cuda|failed' \
    "${LOG}" |
tail -n 100 || true

exit 2
