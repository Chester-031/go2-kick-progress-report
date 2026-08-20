#!/usr/bin/env bash
# R8_4_LOAD_PRESERVING_PROGRESSIVE_LIFT_PPO
set -Eeuo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
ISAACLAB_ROOT="${R8_ISAACLAB_ROOT:-/scratch/lj2634/IsaacLab_232_src_37ddf62}"
ENV_ROOT="${R8_ENV_ROOT:-/scratch/lj2634/envs/isaaclab232_v2}"
PYTHON_BIN="${R8_PYTHON_BIN:-${ENV_ROOT}/bin/python}"
GLU_ROOT="${R8_GLU_ROOT:-/scratch/lj2634/vendor/libglu-9.0.0}"
SLURM_ACCOUNT="${R8_SLURM_ACCOUNT:-torch_pr_1066_general}"
INSTALL_DIR="${PROJECT_DIR}/experiments/r8_4_load_preserving_progressive_lift"
RUN_ROOT="/scratch/lj2634/runs/go2_soccer_r8_4_load_preserving_progressive_lift"
R83_RUN_ROOT="/scratch/lj2634/runs/go2_soccer_r8_3_real_load_transfer_skill"
R83_POINTER="${R83_RUN_ROOT}/latest_r8_3_real_load_transfer_skill.txt"
test -s "${R83_POINTER}" || {
  echo "ERROR: R8.3 pointer missing: ${R83_POINTER}" >&2
  exit 25
}
R83_ROOT="$(tr -d '\r\n' < "${R83_POINTER}")"
R83_FINAL="${R83_ROOT}/pilot/logs/support_model_final.pt"
BASE_CHECKPOINT="${R84_SOURCE_CHECKPOINT:-${R83_FINAL}}"
SOURCE_NAME="train_go2_soccer_r8_support_skill.py"
WORKER_NAME="run_go2_soccer_r8_support_skill.sbatch"
VALIDATOR_NAME="validate_go2_soccer_r8_support_skill.py"
CHECKER_NAME="check_go2_soccer_r8_support_skill_result.sh"
README_NAME="README.md"
VALIDATION_RUNTIME_ROOT="${RUN_ROOT}/runtime_validation_${USER}_$$"

run_with_validation_home() {
  env \
    HOME="${VALIDATION_RUNTIME_ROOT}/home" \
    XDG_CACHE_HOME="${VALIDATION_RUNTIME_ROOT}/cache" \
    XDG_CONFIG_HOME="${VALIDATION_RUNTIME_ROOT}/config" \
    XDG_DATA_HOME="${VALIDATION_RUNTIME_ROOT}/data" \
    TMPDIR="${VALIDATION_RUNTIME_ROOT}/tmp" \
    MPLCONFIGDIR="${VALIDATION_RUNTIME_ROOT}/matplotlib" \
    PYTHONPATH="${PROJECT_DIR}/source/isaac_go2_low_level_rl:${ISAACLAB_ROOT}/source/isaaclab:${ISAACLAB_ROOT}/source/isaaclab_tasks:${PYTHONPATH:-}" \
    "$@"
}

submit_job() {
  local output rc job
  set +e
  output="$(sbatch --parsable --account="${SLURM_ACCOUNT}" "$@" 2>&1)"
  rc=$?
  set -e
  echo "${output}" >&2
  if [ "${rc}" -ne 0 ]; then
    echo "ERROR: sbatch submission failed with rc=${rc}" >&2
    return 40
  fi
  job="${output%%;*}"
  if ! [[ "${job}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: invalid Slurm job id: ${output}" >&2
    return 41
  fi
  printf '%s\n' "${job}"
}

for file in \
  "${SOURCE_NAME}" \
  "${WORKER_NAME}" \
  "${VALIDATOR_NAME}" \
  "${CHECKER_NAME}" \
  "${README_NAME}"
do
  test -s "${PACKAGE_DIR}/${file}" || {
    echo "ERROR: missing ${file}" >&2
    exit 20
  }
done

test -x "${PYTHON_BIN}" || {
  echo "ERROR: verified Isaac Lab Python missing: ${PYTHON_BIN}" >&2
  exit 21
}
test -d "${ISAACLAB_ROOT}" || exit 22
test -e "${ENV_ROOT}/lib/libstdc++.so.6" || exit 23
test -e "${GLU_ROOT}/lib/libGLU.so.1" || exit 24
test -s "${BASE_CHECKPOINT}" || {
  echo "ERROR: R8.4 source checkpoint missing: ${BASE_CHECKPOINT}" >&2
  exit 25
}

mkdir -p \
  "${VALIDATION_RUNTIME_ROOT}/home" \
  "${VALIDATION_RUNTIME_ROOT}/cache" \
  "${VALIDATION_RUNTIME_ROOT}/config" \
  "${VALIDATION_RUNTIME_ROOT}/data" \
  "${VALIDATION_RUNTIME_ROOT}/tmp" \
  "${VALIDATION_RUNTIME_ROOT}/matplotlib"

run_with_validation_home "${PYTHON_BIN}" - <<'PY'
import importlib.util
import sys
import torch
required = ("torch", "toml", "rsl_rl", "isaaclab", "isaaclab.app")
missing = [name for name in required if importlib.util.find_spec(name) is None]
print(f"R8_VALIDATION_PYTHON={sys.executable}")
print(f"R8_VALIDATION_TORCH={torch.__version__}")
print(f"R8_VALIDATION_MISSING={missing}")
if missing:
    raise SystemExit(1)
print("R8_VERIFIED_RUNTIME_PYTHON_READY")
PY

run_with_validation_home \
  "${PYTHON_BIN}" \
  "${PACKAGE_DIR}/${VALIDATOR_NAME}" \
  "${PACKAGE_DIR}" | tee "${PACKAGE_DIR}/package_audit_runtime.log"
grep -qF 'R8_4_SUPPORT_PACKAGE_AUDIT_CLEAN' \
  "${PACKAGE_DIR}/package_audit_runtime.log"

mkdir -p "${INSTALL_DIR}" "${RUN_ROOT}"
install -m 0644 "${PACKAGE_DIR}/${SOURCE_NAME}" "${INSTALL_DIR}/${SOURCE_NAME}"
install -m 0755 "${PACKAGE_DIR}/${WORKER_NAME}" "${INSTALL_DIR}/${WORKER_NAME}"
install -m 0755 "${PACKAGE_DIR}/${VALIDATOR_NAME}" "${INSTALL_DIR}/${VALIDATOR_NAME}"
install -m 0755 "${PACKAGE_DIR}/${CHECKER_NAME}" "${INSTALL_DIR}/${CHECKER_NAME}"
install -m 0755 "${PACKAGE_DIR}/install_submit_go2_soccer_r8_support_skill.sh" "${INSTALL_DIR}/install_submit_go2_soccer_r8_support_skill.sh"
install -m 0644 "${PACKAGE_DIR}/${README_NAME}" "${INSTALL_DIR}/${README_NAME}"

INSTALLED_AUDIT="${RUN_ROOT}/installed_audit_$(date +%Y%m%d_%H%M%S).log"
run_with_validation_home \
  "${PYTHON_BIN}" \
  "${INSTALL_DIR}/${VALIDATOR_NAME}" \
  "${INSTALL_DIR}" | tee "${INSTALLED_AUDIT}"
grep -qF 'R8_4_SUPPORT_PACKAGE_AUDIT_CLEAN' "${INSTALLED_AUDIT}"
echo "R8_4_SUPPORT_INSTALLED_AUDIT_CLEAN"
echo "R8_SUPPORT_INSTALLED_AUDIT_CLEAN"

STAMP="$(date +%Y%m%d_%H%M%S)"
ROOT="${RUN_ROOT}/r8_4_load_preserving_progressive_lift_${STAMP}"
PILOT_DIR="${ROOT}/pilot"
VIDEO_DIR="${ROOT}/video"
mkdir -p "${PILOT_DIR}" "${VIDEO_DIR}"

cat > "${ROOT}/experiment_contract.txt" <<EOF2
version=r8_4_load_preserving_progressive_lift_ppo_pilot
created_at=$(date --iso-8601=seconds)
training=true
ppo=true
behavior_cloning=false
scope=isolated_low_level_no_ball_load_preserving_progressive_lift
source_checkpoint=${BASE_CHECKPOINT}
actor_observation=70D_teacher_free
low_action=12D_joint_position_residuals
high_level_policy_used=false
high_level_router_interface=unchanged
high_level_sparse_reward=unchanged_not_used_in_this_job
dense_reward=low_level_skill_only
ball_contact_training=deferred_until_pilot_video_review
pilot_num_envs=256
pilot_iterations=1000
pilot_curriculum=success_gated_load10_then_lift1_lift2_lift3_lift4
runtime_python=${PYTHON_BIN}
runtime_env=${ENV_ROOT}
isaaclab_root=${ISAACLAB_ROOT}
glu_root=${GLU_ROOT}
slurm_account=${SLURM_ACCOUNT}
runtime_home=scratch_per_job
EOF2

SOURCE="${INSTALL_DIR}/${SOURCE_NAME}"
WORKER="${INSTALL_DIR}/${WORKER_NAME}"
VIDEO="${VIDEO_DIR}/r8_support_pilot.mp4"
PILOT_CHECKPOINT="${PILOT_DIR}/logs/support_model_final.pt"

unset SBATCH_ACCOUNT

PILOT_JOB="$(
  submit_job \
    --job-name=r84lift-pilot \
    --time=03:00:00 \
    --output="${PILOT_DIR}/slurm-%j.out" \
    --export=ALL,RUN_DIR="${PILOT_DIR}",SOURCE_CHECKPOINT="${BASE_CHECKPOINT}",R8_MODE=train,R8_PROFILE=pilot,R8_NUM_ENVS=256,R8_MAX_ITERATIONS=1000,R8_SOURCE="${SOURCE}",R8_SEED=43 \
    "${WORKER}"
)"
echo "${PILOT_JOB}" > "${PILOT_DIR}/job_id.txt"

VIDEO_JOB="$(
  submit_job \
    --dependency=afterok:"${PILOT_JOB}" \
    --job-name=r84lift-video \
    --time=01:00:00 \
    --output="${VIDEO_DIR}/slurm-%j.out" \
    --export=ALL,RUN_DIR="${VIDEO_DIR}",SOURCE_CHECKPOINT="${BASE_CHECKPOINT}",R8_MODE=eval,R8_PROFILE=pilot,R8_NUM_ENVS=1,R8_MAX_ITERATIONS=1,R8_SOURCE="${SOURCE}",R8_SEED=43,R8_TRAINED_CHECKPOINT="${PILOT_CHECKPOINT}",R8_VIDEO_PATH="${VIDEO}" \
    "${WORKER}"
)"
echo "${VIDEO_JOB}" > "${VIDEO_DIR}/job_id.txt"

printf '%s\n' "${ROOT}" > "${RUN_ROOT}/latest_r8_4_load_preserving_progressive_lift.txt"
printf '%s\n' "${PILOT_JOB}" > "${ROOT}/pilot_job_id.txt"
printf '%s\n' "${VIDEO_JOB}" > "${ROOT}/video_job_id.txt"

echo "R8.4 LOAD-PRESERVING PROGRESSIVE LIFT PPO PILOT SUBMITTED"
echo "ROOT=${ROOT}"
echo "PILOT_JOB=${PILOT_JOB}"
echo "VIDEO_JOB=${VIDEO_JOB}"
echo "VIDEO_DEPENDENCY=afterok:${PILOT_JOB}"
echo "R8_RUNTIME_PYTHON=${PYTHON_BIN}"
echo "R8_4_SUBMIT_STATUS=0"
