#!/usr/bin/env bash
set -Eeuo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
INSTALL_DIR="${PROJECT_DIR}/experiments/v4k1_r5_stable_kick"
RUN_ROOT="/scratch/lj2634/runs/go2_soccer_v4k1_r5_stable_kick"
V4D_RUN_ROOT="/scratch/lj2634/runs/go2_soccer_v4d_joint"
ACCOUNT="torch_pr_1066_general"
PARTITION="l40s_public"
TASK="Isaac-Go2-Soccer-Milestone2-v0"
SEED="42"
PY="/scratch/lj2634/envs/isaaclab232_v2/bin/python"

TRAIN_NAME="train_go2_soccer_v4k1_r5_stable_kick.py"
PLAY_NAME="play_go2_soccer_v4k1_r5_stable_kick.py"
WORKER_NAME="run_go2_soccer_v4k1_r5_stable_kick.sbatch"
VALIDATOR_NAME="validate_v4k1_r5_package.py"

resolve_v4d_low() {
  local pointer="${V4D_RUN_ROOT}/latest_v4d_joint.txt"
  local root="" candidate="" marker=""
  if [ -s "${pointer}" ]; then
    root="$(tr -d '\r\n' < "${pointer}")"
    candidate="${root}/full/logs/low_model_final.pt"
    if [ -s "${candidate}" ]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  fi
  marker="$(
    find "${V4D_RUN_ROOT}" \
      -type f \
      -name V4D_JOINT_ACCEPTED \
      -printf '%T@ %p\n' \
      2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-
  )"
  if [ -n "${marker}" ]; then
    candidate="$(dirname "${marker}")/logs/low_model_final.pt"
  fi
  [ -s "${candidate}" ] || return 1
  printf '%s\n' "${candidate}"
}

BASE_LOW_CHECKPOINT="${V4K1_BASE_LOW_CHECKPOINT:-}"
if [ -z "${BASE_LOW_CHECKPOINT}" ]; then
  BASE_LOW_CHECKPOINT="$(resolve_v4d_low || true)"
fi

mkdir -p "${RUN_ROOT}"
AUDIT_STAMP="$(date +%Y%m%d_%H%M%S)"
PACKAGE_AUDIT_REPORT="${RUN_ROOT}/package_audit_${AUDIT_STAMP}.log"
INSTALL_AUDIT_REPORT="${RUN_ROOT}/installed_audit_${AUDIT_STAMP}.log"

printf '%s\n' "============================================================"
printf '%s\n' "V4K1-R5 STABLE-STANCE PACKAGE AUDIT AND SUBMISSION"
printf '%s\n' "============================================================"

# No file is installed and no GPU job is submitted until this complete,
# collect-all audit reaches V4K1_R5_PACKAGE_AUDIT_CLEAN.
set +e
"${PY}" "${PACKAGE_DIR}/${VALIDATOR_NAME}" "${PACKAGE_DIR}" \
  2>&1 | tee "${PACKAGE_AUDIT_REPORT}"
PACKAGE_AUDIT_RC="${PIPESTATUS[0]}"
set -e
if [ "${PACKAGE_AUDIT_RC}" -ne 0 ] \
  || ! grep -qF 'V4K1_R5_PACKAGE_AUDIT_CLEAN' "${PACKAGE_AUDIT_REPORT}"
then
  echo "ERROR: package audit failed; no files installed and no jobs submitted." >&2
  echo "PACKAGE_AUDIT_REPORT=${PACKAGE_AUDIT_REPORT}" >&2
  exit 30
fi

echo "V4K1_R5_PACKAGE_AUDIT_CLEAN"

ACTIVE="$(
  squeue -h -u "${USER}" -o '%A|%j|%T' 2>/dev/null \
    | awk -F'|' '$2 ~ /^v4k/ && $3 ~ /^(PENDING|RUNNING|CONFIGURING|COMPLETING)$/ {print}'
)"
if [ -n "${ACTIVE}" ]; then
  echo "ERROR: active V4K job exists:" >&2
  printf '%s\n' "${ACTIVE}" >&2
  exit 31
fi

for path in \
  "${BASE_LOW_CHECKPOINT}" \
  "${PACKAGE_DIR}/${TRAIN_NAME}" \
  "${PACKAGE_DIR}/${PLAY_NAME}" \
  "${PACKAGE_DIR}/${WORKER_NAME}" \
  "${PACKAGE_DIR}/${VALIDATOR_NAME}"
do
  [ -s "${path}" ] || {
    echo "ERROR: required input missing: ${path:-EMPTY}" >&2
    exit 32
  }
done

mkdir -p "${INSTALL_DIR}"
install -m 0644 "${PACKAGE_DIR}/${TRAIN_NAME}" "${INSTALL_DIR}/${TRAIN_NAME}"
install -m 0644 "${PACKAGE_DIR}/${PLAY_NAME}" "${INSTALL_DIR}/${PLAY_NAME}"
install -m 0755 "${PACKAGE_DIR}/${WORKER_NAME}" "${INSTALL_DIR}/${WORKER_NAME}"
install -m 0755 "${PACKAGE_DIR}/${VALIDATOR_NAME}" "${INSTALL_DIR}/${VALIDATOR_NAME}"
install -m 0755 "${PACKAGE_DIR}/install_submit_v4k1_r5_stable_kick.sh" "${INSTALL_DIR}/install_submit_v4k1_r5_stable_kick.sh"
install -m 0644 "${PACKAGE_DIR}/README.md" "${INSTALL_DIR}/README.md"
install -m 0644 "${PACKAGE_DIR}/V4K1_DESIGN.md" "${INSTALL_DIR}/V4K1_DESIGN.md"

set +e
"${PY}" "${INSTALL_DIR}/${VALIDATOR_NAME}" "${INSTALL_DIR}" \
  2>&1 | tee "${INSTALL_AUDIT_REPORT}"
INSTALL_AUDIT_RC="${PIPESTATUS[0]}"
set -e
if [ "${INSTALL_AUDIT_RC}" -ne 0 ] \
  || ! grep -qF 'V4K1_R5_PACKAGE_AUDIT_CLEAN' "${INSTALL_AUDIT_REPORT}"
then
  echo "ERROR: installed-source audit failed; no jobs submitted." >&2
  echo "INSTALL_AUDIT_REPORT=${INSTALL_AUDIT_REPORT}" >&2
  exit 33
fi

echo "V4K1_R5_INSTALLED_SOURCE_AUDIT_CLEAN"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="v4k1_r5_stable_kick_${STAMP}"
ROOT="${RUN_ROOT}/${RUN_ID}"
PRE_DIR="${ROOT}/preflight"
PILOT_DIR="${ROOT}/pilot"
FULL_DIR="${ROOT}/full"
mkdir -p "${PRE_DIR}/logs" "${PILOT_DIR}/logs" "${FULL_DIR}/logs"

cat > "${ROOT}/experiment_contract.txt" <<EOF
version=v4k1_r5_stable_stance_delta_pose_kick
created_at=$(date --iso-8601=seconds)
base_low_checkpoint=${BASE_LOW_CHECKPOINT}
training_scope=isolated_low_level_kick_only
high_policy=not_used
soccer_goal_reward=not_used
delta_pose=planar_base_delta_pose_direct_context
kick_context=phase,leg,direction,strength
low_action=12_joint_position_residuals
teacher_in_actor_observation=false
stable_reset_required=true
ball_parked_during_settle=true
ball_placed_from_actual_selected_foot=true
learned_action_disabled_before_arm=true
selected_foot_lift_and_swing_required=true
precontact_motion_invalid=true
body_or_wrong_foot_contact_invalid=true
short_contact_required=true
post_kick_recovery_required=true
left_leg_only=true
right_leg_deferred=true
preflight=32_envs_3_iters_stages_1_1_1
pilot=128_envs_150_iters_stages_60_60_30
full=512_envs_600_iters_stages_150_300_150
package_audit=${PACKAGE_AUDIT_REPORT}
installed_audit=${INSTALL_AUDIT_REPORT}
EOF

COMMON="ALL"
COMMON+=",TASK=${TASK}"
COMMON+=",SEED=${SEED}"
COMMON+=",V4K1_ROLLOUT_STEPS=60"
COMMON+=",V4K1_KICK_EPISODE_STEPS=60"
COMMON+=",V4K1_FIXED_LEG=left"
COMMON+=",V4K1_SETTLE_MIN_STEPS=25"
COMMON+=",V4K1_SETTLE_MAX_STEPS=75"
COMMON+=",V4K1_SETTLE_REQUIRED_STEPS=10"
COMMON+=",V4K1_BALL_ARM_STEPS=6"
COMMON+=",V4K1_INIT_NOISE_STD=0.30"
COMMON+=",V4K1_ENTROPY_COEF=0.008"
COMMON+=",V4K1_CLIP_PARAM=0.20"
COMMON+=",V4K1_EPOCHS=4"
COMMON+=",V4K1_MINI_BATCHES=4"
COMMON+=",V4K1_DESIRED_KL=0.010"
COMMON+=",V4K1_LR_STAGE1=0.000020"
COMMON+=",V4K1_LR_STAGE2=0.000012"
COMMON+=",V4K1_LR_STAGE3=0.000004"
COMMON+=",V4K1_STD_STAGE1=0.30"
COMMON+=",V4K1_STD_STAGE2=0.20"
COMMON+=",V4K1_STD_STAGE3=0.12"
COMMON+=",V4K1_CONTACT_DISTANCE=0.155"
COMMON+=",V4K1_IMPACT_SPEED=0.060"
COMMON+=",V4K1_MAX_UPRIGHT_ERROR=0.04"
COMMON+=",V4K1_MAX_ACTION_RATE=0.90"
COMMON+=",V4K1_CHECKPOINT_INTERVAL=50"

unset SBATCH_ACCOUNT || true

submit_job() {
  local dependency="$1"
  local job_name="$2"
  local run_dir="$3"
  local checkpoint="$4"
  local envs="$5"
  local iterations="$6"
  local s1="$7"
  local s2="$8"
  local s3="$9"
  local time_limit="${10}"
  local min_success="${11}"
  local min_contact="${12}"
  local min_impact="${13}"
  local stage1_contact="${14}"
  local stage2_impact="${15}"
  local stage2_kick="${16}"
  local max_violation="${17}"
  local min_stable_activation="${18}"
  local output="" job=""

  local args=(
    --parsable
    --account="${ACCOUNT}"
    --partition="${PARTITION}"
    --gres=gpu:l40s:1
    --time="${time_limit}"
    --job-name="${job_name}"
    --chdir="${PROJECT_DIR}"
    --output="${run_dir}/slurm-%j.out"
    --error="${run_dir}/slurm-%j.err"
  )
  [ -z "${dependency}" ] || args+=(--dependency="afterok:${dependency}")

  output="$(
    env -u SBATCH_ACCOUNT \
    sbatch "${args[@]}" \
      --export="${COMMON},LOW_CHECKPOINT=${checkpoint},RUN_DIR=${run_dir},LOG_DIR=${run_dir}/logs,RUN_NAME=${RUN_ID}_${job_name},NUM_ENVS=${envs},MAX_ITERATIONS=${iterations},V4K1_STAGE1_ITERATIONS=${s1},V4K1_STAGE2_ITERATIONS=${s2},V4K1_STAGE3_ITERATIONS=${s3},V4K1_MIN_SUCCESS_RATE=${min_success},V4K1_MIN_CONTACT_RATE=${min_contact},V4K1_MIN_IMPACT_RATE=${min_impact},V4K1_STAGE1_MIN_CONTACT_RATE=${stage1_contact},V4K1_STAGE2_MIN_IMPACT_RATE=${stage2_impact},V4K1_STAGE2_MIN_KICK_RATE=${stage2_kick},V4K1_MAX_VIOLATION_RATE=${max_violation},V4K1_MIN_STABLE_ACTIVATION_RATE=${min_stable_activation}" \
      "${INSTALL_DIR}/${WORKER_NAME}"
  )"
  job="${output%%;*}"
  [[ "${job}" =~ ^[0-9]+$ ]] || {
    echo "ERROR: invalid job ID: ${output}" >&2
    return 1
  }
  printf '%s\n' "${job}"
}

PRE_JOB="$(
  submit_job '' v4k5-pre "${PRE_DIR}" "${BASE_LOW_CHECKPOINT}" \
    32 3 1 1 1 00:30:00 \
    0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.95
)"
printf '%s\n' "${PRE_JOB}" > "${PRE_DIR}/job_id.txt"

PILOT_JOB="$(
  submit_job "${PRE_JOB}" v4k5-pilot "${PILOT_DIR}" "${BASE_LOW_CHECKPOINT}" \
    128 150 60 60 30 03:00:00 \
    0.02 0.25 0.08 0.20 0.06 0.0 0.05 0.95
)"
printf '%s\n' "${PILOT_JOB}" > "${PILOT_DIR}/job_id.txt"

FULL_SOURCE="${PILOT_DIR}/logs/kick_model_final.pt"
FULL_JOB="$(
  submit_job "${PILOT_JOB}" v4k5-full "${FULL_DIR}" "${FULL_SOURCE}" \
    512 600 150 300 150 08:00:00 \
    0.20 0.45 0.30 0.35 0.20 0.05 0.02 0.97
)"
printf '%s\n' "${FULL_JOB}" > "${FULL_DIR}/job_id.txt"

cat > "${ROOT}/jobs.tsv" <<EOF
stage	job_id	run_dir	dependency	low_checkpoint
preflight	${PRE_JOB}	${PRE_DIR}	none	${BASE_LOW_CHECKPOINT}
pilot	${PILOT_JOB}	${PILOT_DIR}	afterok:${PRE_JOB}	${BASE_LOW_CHECKPOINT}
full	${FULL_JOB}	${FULL_DIR}	afterok:${PILOT_JOB}	${FULL_SOURCE}
EOF

printf '%s\n' "${ROOT}" > "${RUN_ROOT}/latest_v4k1_r5_stable_kick.txt"

echo "============================================================"
echo "V4K1-R5 STABLE-STANCE KICK TRAINING SUBMITTED"
echo "============================================================"
echo "ROOT=${ROOT}"
echo "PREFLIGHT_JOB=${PRE_JOB}"
echo "PILOT_JOB=${PILOT_JOB}"
echo "FULL_JOB=${FULL_JOB}"
echo "PILOT_DEPENDENCY=afterok:${PRE_JOB}"
echo "FULL_DEPENDENCY=afterok:${PILOT_JOB}"
echo "PACKAGE_AUDIT_REPORT=${PACKAGE_AUDIT_REPORT}"
echo "INSTALL_AUDIT_REPORT=${INSTALL_AUDIT_REPORT}"
squeue -j "${PRE_JOB},${PILOT_JOB},${FULL_JOB}" \
  -o "%.18i %.20P %.24j %.10u %.3t %.12M %.4D %R" || true
echo "V4K1_R5_SUBMIT_STATUS=0"
