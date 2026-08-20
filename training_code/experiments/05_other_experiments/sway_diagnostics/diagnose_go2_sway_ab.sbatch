#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00

set -Eeuo pipefail

PROJECT_DIR="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
ISAACLAB_ROOT="/scratch/lj2634/IsaacLab_232_src_37ddf62"
ENV_ROOT="/scratch/lj2634/envs/isaaclab232_v2"
GLU_ROOT="/scratch/lj2634/vendor/libglu-9.0.0"

PYTHON="${ENV_ROOT}/bin/python"
PLAYER="${PROJECT_DIR}/experiments/sway_diagnostics/diagnose_go2_sway_ab.py"

: "${TEST_LABEL:?TEST_LABEL is required}"
: "${RESULT_DIR:?RESULT_DIR is required}"
: "${CHECKPOINT:?CHECKPOINT is required}"
: "${CMD_VX:?CMD_VX is required}"
: "${CMD_VY:?CMD_VY is required}"
: "${CMD_WZ:?CMD_WZ is required}"
: "${TEST_STEPS:?TEST_STEPS is required}"

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

echo "============================================================"
echo "GO2 SWAY DIAGNOSTIC"
echo "============================================================"
echo "JOB_ID=${SLURM_JOB_ID}"
echo "TEST_LABEL=${TEST_LABEL}"
echo "RESULT_DIR=${RESULT_DIR}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "COMMAND=${CMD_VX},${CMD_VY},${CMD_WZ}"
echo "TEST_STEPS=${TEST_STEPS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

RUN_LOG="${RESULT_DIR}/diagnostic.log"

set +e

srun \
    --export=ALL \
    --unbuffered \
    "${PYTHON}" \
    "${PLAYER}" \
    --task Isaac-Go2-Low-Level-Round2-v0 \
    --checkpoint "${CHECKPOINT}" \
    --result_dir "${RESULT_DIR}" \
    --test_label "${TEST_LABEL}" \
    --command_vx "${CMD_VX}" \
    --command_vy "${CMD_VY}" \
    --command_wz "${CMD_WZ}" \
    --steps "${TEST_STEPS}" \
    --warmup_steps 50 \
    --seed 43 \
    --stop_on_done \
    --device cuda:0 \
    --headless \
    2>&1 | tee "${RUN_LOG}"

PLAYER_RC="${PIPESTATUS[0]}"

set -e

echo
echo "===== Result inventory ====="

find "${RESULT_DIR}" \
    -maxdepth 1 \
    -type f \
    -printf '%TY-%Tm-%Td %TH:%TM:%TS | %12s | %f\n' |
sort

if \
    [ "${PLAYER_RC}" -eq 0 ] &&
    [ -s "${RESULT_DIR}/low_level_diagnostic.csv" ] &&
    [ -s "${RESULT_DIR}/summary.json" ] &&
    ! grep -q '\[SWAY_FATAL\]' "${RUN_LOG}"
then
    echo
    echo "GO2_SWAY_DIAGNOSTIC_ACCEPTED"
    exit 0
fi

echo
echo "GO2_SWAY_DIAGNOSTIC_FAILED"
echo "PLAYER_RC=${PLAYER_RC}"

exit 2
