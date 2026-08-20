#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G

set -Eeuo pipefail

PROJECT_DIR="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
ISAACLAB_ROOT="/scratch/lj2634/IsaacLab_232_src_37ddf62"
ENV_ROOT="/scratch/lj2634/envs/isaaclab232_v2"
GLU_ROOT="/scratch/lj2634/vendor/libglu-9.0.0"
PY="${ENV_ROOT}/bin/python"
TRAIN_SCRIPT="${PROJECT_DIR}/experiments/v4k1_r5_2_stable_kick/train_go2_soccer_v4k1_r5_stable_kick.py"

for name in \
  TASK LOW_CHECKPOINT RUN_DIR LOG_DIR RUN_NAME NUM_ENVS MAX_ITERATIONS SEED \
  V4K1_STAGE1_ITERATIONS V4K1_STAGE2_ITERATIONS V4K1_STAGE3_ITERATIONS
do
  [ -n "${!name:-}" ] || {
    echo "ERROR: ${name} is required." >&2
    exit 20
  }
done

mkdir -p "${RUN_DIR}" "${LOG_DIR}"
exec > >(tee -a "${RUN_DIR}/worker.log") 2>&1

echo "============================================================"
echo "V4K1-R5.2 STABLE-STANCE ISOLATED DELTA-POSE KICK"
echo "============================================================"
echo "JOB_ID=${SLURM_JOB_ID}"
echo "HOST=$(hostname)"
echo "RUN_DIR=${RUN_DIR}"
echo "NUM_ENVS=${NUM_ENVS}"
echo "MAX_ITERATIONS=${MAX_ITERATIONS}"
echo "LOW_CHECKPOINT=${LOW_CHECKPOINT}"

test -x "${PY}"
test -s "${TRAIN_SCRIPT}"
test -s "${LOW_CHECKPOINT}"

if [ -e "${GLU_ROOT}/lib/libGLU.so.1" ]; then
  export LD_LIBRARY_PATH="${GLU_ROOT}/lib:${ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="${ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"
fi
export LD_PRELOAD="${ENV_ROOT}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"
export PYTHONPATH="${PROJECT_DIR}/source/isaac_go2_low_level_rl:${ISAACLAB_ROOT}/source/isaaclab:${ISAACLAB_ROOT}/source/isaaclab_tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export GIT_PYTHON_REFRESH=quiet

RUNTIME_ROOT="/scratch/lj2634/job_runtime/${SLURM_JOB_ID}_v4k1_r5"
export HOME="${RUNTIME_ROOT}/home"
export XDG_CACHE_HOME="${RUNTIME_ROOT}/cache"
export XDG_CONFIG_HOME="${RUNTIME_ROOT}/config"
export XDG_DATA_HOME="${RUNTIME_ROOT}/data"
export TMPDIR="${RUNTIME_ROOT}/tmp"
export MPLCONFIGDIR="${RUNTIME_ROOT}/matplotlib"
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_DATA_HOME}" "${TMPDIR}" "${MPLCONFIGDIR}"

cp -a "${TRAIN_SCRIPT}" "${RUN_DIR}/train_go2_soccer_v4k1_r5_stable_kick.snapshot.py"
git -C "${PROJECT_DIR}" rev-parse HEAD > "${RUN_DIR}/git_commit.txt"
git -C "${PROJECT_DIR}" status --short > "${RUN_DIR}/git_status.txt"
sha256sum "${TRAIN_SCRIPT}" "${LOW_CHECKPOINT}" > "${RUN_DIR}/input_sha256.txt"

cat > "${RUN_DIR}/resolved_launch_config.txt" <<CONFIG
version=v4k1_r5_2_stable_stance_delta_pose_kick
training_scope=isolated_low_level_kick_only
high_policy=not_used
soccer_goal_reward=not_used
source_low_checkpoint=${LOW_CHECKPOINT}
delta_pose=planar_base_delta_pose_direct_context
kick_context=phase,leg,direction,strength
stable_reset_required=true
ball_parked_during_settle=true
ball_placed_from_actual_selected_foot=true
selected_foot_requires_lift_and_swing=true
precontact_ball_motion_invalid=true
body_or_wrong_foot_contact_invalid=true
short_contact_required=true
post_kick_recovery_required=true
stage1_fixed_leg=left
stage2_fixed_leg=left
stage3_fixed_leg=left
right_leg_deferred=true
settle_min_steps=${V4K1_SETTLE_MIN_STEPS:-25}
settle_max_steps=${V4K1_SETTLE_MAX_STEPS:-150}
settle_required_steps=${V4K1_SETTLE_REQUIRED_STEPS:-10}
ball_arm_steps=${V4K1_BALL_ARM_STEPS:-6}
max_violation_rate=${V4K1_MAX_VIOLATION_RATE:-0.02}
min_stable_activation_rate=${V4K1_MIN_STABLE_ACTIVATION_RATE:-0.95}
min_completed_env_fraction=${V4K1_MIN_COMPLETED_ENV_FRACTION:-0.0}
rollout_steps=${V4K1_ROLLOUT_STEPS:-60}
stage1_min_contact_rate=${V4K1_STAGE1_MIN_CONTACT_RATE:-0}
stage2_min_impact_rate=${V4K1_STAGE2_MIN_IMPACT_RATE:-0}
stage2_min_kick_rate=${V4K1_STAGE2_MIN_KICK_RATE:-0}
final_min_success_rate=${V4K1_MIN_SUCCESS_RATE:-0}
final_min_contact_rate=${V4K1_MIN_CONTACT_RATE:-0}
final_min_impact_rate=${V4K1_MIN_IMPACT_RATE:-0}
low_action=12_joint_position_residuals
teacher_in_actor_observation=false
num_envs=${NUM_ENVS}
iterations=${MAX_ITERATIONS}
active_kick_steps=${V4K1_KICK_EPISODE_STEPS:-60}
stage1_iterations=${V4K1_STAGE1_ITERATIONS}
stage2_iterations=${V4K1_STAGE2_ITERATIONS}
stage3_iterations=${V4K1_STAGE3_ITERATIONS}
seed=${SEED}
CONFIG

cd "${RUN_DIR}"
set +e
srun --export=ALL --unbuffered "${PY}" "${TRAIN_SCRIPT}" \
  --task "${TASK}" \
  --low_level_checkpoint "${LOW_CHECKPOINT}" \
  --log_dir "${LOG_DIR}" \
  --run_name "${RUN_NAME}" \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${MAX_ITERATIONS}" \
  --kick_episode_steps "${V4K1_KICK_EPISODE_STEPS:-60}" \
  --kick_fixed_leg left \
  --seed "${SEED}" \
  --device cuda:0 \
  --headless \
  2>&1 | tee "${RUN_DIR}/training.log"
RUN_RC="${PIPESTATUS[0]}"
set -e

echo "RUN_RC=${RUN_RC}"
MODEL="${LOG_DIR}/kick_model_final.pt"
SUMMARY="${LOG_DIR}/v4k1_kick_summary.json"
CONTRACT="${LOG_DIR}/v4k1_kick_contract.json"

if [ "${RUN_RC}" -eq 0 ] \
  && [ -s "${MODEL}" ] \
  && [ -s "${SUMMARY}" ] \
  && [ -s "${CONTRACT}" ] \
  && [ -f "${LOG_DIR}/V4K1_ISOLATED_KICK_OK" ] \
  && grep -qF '[V4K1_CONTRACT]' "${RUN_DIR}/training.log" \
  && grep -qF '[V4K1_INPUT_GUARD]' "${RUN_DIR}/training.log" \
  && grep -qF '[V4K1_R5_STABLE_RESET_GUARD]' "${RUN_DIR}/training.log" \
  && grep -qF '[V4K1_ITER]' "${RUN_DIR}/training.log" \
  && grep -qF '[V4K1_STAGE_GATE]' "${RUN_DIR}/training.log" \
  && grep -qF '[V4K1_FINAL_GUARD]' "${RUN_DIR}/training.log" \
  && grep -qF 'V4K1_R5_STABLE_STANCE_CONTACT_CAUSALITY=true' "${RUN_DIR}/training.log" \
  && grep -qF 'V4K1_ISOLATED_KICK_TRAINING_COMPLETE' "${RUN_DIR}/training.log" \
  && ! grep -qF '[V4K1_FATAL]' "${RUN_DIR}/training.log"
then
  touch "${RUN_DIR}/V4K1_ISOLATED_KICK_ACCEPTED"
  sha256sum "${MODEL}" "${SUMMARY}" "${CONTRACT}" > "${RUN_DIR}/output_sha256.txt"
  echo "V4K1_ISOLATED_KICK_ACCEPTED"
  echo "MODEL=${MODEL}"
  echo "SUMMARY=${SUMMARY}"
  exit 0
fi

echo "===== Important errors ====="
grep -nEi \
  'fatal|error|exception|traceback|mismatch|overflow|underflow|cuda|nan|failed|guard' \
  "${RUN_DIR}/training.log" \
  "${RUN_DIR}/worker.log" \
  "${RUN_DIR}"/slurm-*.out \
  "${RUN_DIR}"/slurm-*.err \
  2>/dev/null | tail -n 300 || true

touch "${RUN_DIR}/V4K1_ISOLATED_KICK_FAILED"
echo "V4K1_ISOLATED_KICK_FAILED"
exit 2
