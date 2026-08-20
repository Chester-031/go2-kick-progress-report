#!/usr/bin/env bash
# R8_2_CONTACT_LOAD_SIGNAL_PROBE
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=%x-%j.out

set -Eeuo pipefail

: "${RUN_DIR:?RUN_DIR is required}"
: "${SOURCE_CHECKPOINT:?SOURCE_CHECKPOINT is required}"
: "${R82_TRAINED_CHECKPOINT:?R82_TRAINED_CHECKPOINT is required}"
: "${R82_SOURCE:?R82_SOURCE is required}"

PROJECT_DIR="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
ISAACLAB_ROOT="${R8_ISAACLAB_ROOT:-/scratch/lj2634/IsaacLab_232_src_37ddf62}"
ENV_ROOT="${R8_ENV_ROOT:-/scratch/lj2634/envs/isaaclab232_v2}"
PY="${R8_PYTHON_BIN:-${ENV_ROOT}/bin/python}"
GLU_ROOT="${R8_GLU_ROOT:-/scratch/lj2634/vendor/libglu-9.0.0}"

for path in \
  "${PY}" \
  "${R82_SOURCE}" \
  "${SOURCE_CHECKPOINT}" \
  "${R82_TRAINED_CHECKPOINT}"
do
  test -e "${path}" || {
    echo "ERROR_MISSING=${path}" >&2
    exit 20
  }
done

mkdir -p "${RUN_DIR}"
cp -a "${R82_SOURCE}" "${RUN_DIR}/train_r8_2_contact_probe.snapshot.py"

export LD_LIBRARY_PATH="${GLU_ROOT}/lib:${ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${ENV_ROOT}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"
export PYTHONPATH="${PROJECT_DIR}/source/isaac_go2_low_level_rl:${ISAACLAB_ROOT}/source/isaaclab:${ISAACLAB_ROOT}/source/isaaclab_tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export GIT_PYTHON_REFRESH=quiet

RUNTIME_ROOT="/scratch/lj2634/job_runtime/${SLURM_JOB_ID}_r82_contact_probe"
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

export V4K1_SETTLE_MAX_STEPS=150
export V4K1_SETTLE_MIN_STEPS=25
export V4K1_SETTLE_REQUIRED_STEPS=10
export V4K1_MIN_STABLE_ACTIVATION_RATE=1.0
export V4K1_BALL_ARM_STEPS=4

OUT_JSON="${RUN_DIR}/r8_2_contact_load_probe.json"

echo "============================================================"
echo "R8.2 CONTACT/LOAD SIGNAL PROBE"
echo "============================================================"
echo "JOB_ID=${SLURM_JOB_ID}"
echo "HOST=$(hostname)"
echo "RUN_DIR=${RUN_DIR}"
echo "R81_CHECKPOINT=${R82_TRAINED_CHECKPOINT}"
echo "NUM_ENVS=${R82_NUM_ENVS:-128}"
echo "PROBE_STEPS=${R82_PROBE_STEPS:-160}"

ARGS=(
  "${R82_SOURCE}"
  --task Isaac-Go2-Soccer-Milestone2-v0
  --low_level_checkpoint "${SOURCE_CHECKPOINT}"
  --trained_low_checkpoint "${R82_TRAINED_CHECKPOINT}"
  --contact_probe_json "${OUT_JSON}"
  --contact_probe_steps "${R82_PROBE_STEPS:-160}"
  --log_dir "${RUN_DIR}/logs"
  --run_name "$(basename "${RUN_DIR}")"
  --num_envs "${R82_NUM_ENVS:-128}"
  --max_iterations 1
  --kick_episode_steps 80
  --kick_fixed_leg left
  --r8_profile pilot
  --seed 43
  --device cuda:0
)

cd "${RUN_DIR}"
set +e
srun \
  --export=ALL \
  --unbuffered \
  "${PY}" \
  "${ARGS[@]}" \
  2>&1 | tee "${RUN_DIR}/worker.log"
RC=${PIPESTATUS[0]}
set -e

echo "RUN_RC=${RC}" | tee -a "${RUN_DIR}/worker.log"
if [ "${RC}" -ne 0 ]; then
  touch "${RUN_DIR}/R8_2_CONTACT_PROBE_FAILED"
  exit "${RC}"
fi

test -s "${OUT_JSON}"
grep -qF 'R8_2_CONTACT_PROBE_COMPLETE' "${RUN_DIR}/worker.log"
touch "${RUN_DIR}/R8_2_CONTACT_PROBE_COMPLETE"
