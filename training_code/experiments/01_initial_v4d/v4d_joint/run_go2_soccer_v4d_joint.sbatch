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
TRAIN_SCRIPT="${PROJECT_DIR}/experiments/v4d_joint/train_go2_soccer_v4d_joint.py"
for name in TASK HIGH_CHECKPOINT LOW_CHECKPOINT RUN_DIR LOG_DIR RUN_NAME NUM_ENVS MAX_ITERATIONS SEED V4D_STAGE1_ITERATIONS V4D_STAGE2_ITERATIONS V4D_STAGE3_ITERATIONS; do
  [ -n "${!name:-}" ] || { echo "ERROR: ${name} is required." >&2; exit 20; }
done
mkdir -p "${RUN_DIR}" "${LOG_DIR}"
exec > >(tee -a "${RUN_DIR}/worker.log") 2>&1
echo "============================================================"
echo "V4D STAGED ZERO-SUM HIGH/LOW JOINT TRAINING"
echo "============================================================"
echo "JOB_ID=${SLURM_JOB_ID}"
echo "HOST=$(hostname)"
echo "RUN_DIR=${RUN_DIR}"
echo "NUM_ENVS=${NUM_ENVS}"
echo "MAX_ITERATIONS=${MAX_ITERATIONS}"
echo "STAGES=${V4D_STAGE1_ITERATIONS},${V4D_STAGE2_ITERATIONS},${V4D_STAGE3_ITERATIONS}"
echo "HIGH_CHECKPOINT=${HIGH_CHECKPOINT}"
echo "LOW_CHECKPOINT=${LOW_CHECKPOINT}"
test -x "${PY}"; test -s "${TRAIN_SCRIPT}"; test -s "${HIGH_CHECKPOINT}"; test -s "${LOW_CHECKPOINT}"
if [ -e "${GLU_ROOT}/lib/libGLU.so.1" ]; then export LD_LIBRARY_PATH="${GLU_ROOT}/lib:${ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"; else export LD_LIBRARY_PATH="${ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"; fi
export LD_PRELOAD="${ENV_ROOT}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"
export PYTHONPATH="${PROJECT_DIR}/source/isaac_go2_low_level_rl:${ISAACLAB_ROOT}/source/isaaclab:${ISAACLAB_ROOT}/source/isaaclab_tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 GIT_PYTHON_REFRESH=quiet V4D_HIGH_CHECKPOINT="${HIGH_CHECKPOINT}"
RUNTIME_ROOT="/scratch/lj2634/job_runtime/${SLURM_JOB_ID}_v4d_joint"
export HOME="${RUNTIME_ROOT}/home" XDG_CACHE_HOME="${RUNTIME_ROOT}/cache" XDG_CONFIG_HOME="${RUNTIME_ROOT}/config" XDG_DATA_HOME="${RUNTIME_ROOT}/data" TMPDIR="${RUNTIME_ROOT}/tmp" MPLCONFIGDIR="${RUNTIME_ROOT}/matplotlib"
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_DATA_HOME}" "${TMPDIR}" "${MPLCONFIGDIR}"
cp -a "${TRAIN_SCRIPT}" "${RUN_DIR}/train_go2_soccer_v4d_joint.snapshot.py"
git -C "${PROJECT_DIR}" rev-parse HEAD > "${RUN_DIR}/git_commit.txt"
git -C "${PROJECT_DIR}" status --short > "${RUN_DIR}/git_status.txt"
sha256sum "${TRAIN_SCRIPT}" "${HIGH_CHECKPOINT}" "${LOW_CHECKPOINT}" > "${RUN_DIR}/input_sha256.txt"
cat > "${RUN_DIR}/resolved_launch_config.txt" <<CONFIG
version=v4d_staged_zero_sum_joint
high_reward=goal_plus1_own_goal_minus1_other0
high_dense_shaping=false
low_reward=execution_quality_only
reward_mixing=false
router=stand_locomotion_push_unchanged
kick=false
delta_pose=direct_low_level_context
teacher_in_actor_observation=false
high_checkpoint=${HIGH_CHECKPOINT}
low_checkpoint=${LOW_CHECKPOINT}
num_envs=${NUM_ENVS}
iterations=${MAX_ITERATIONS}
stage1_iterations=${V4D_STAGE1_ITERATIONS}
stage2_iterations=${V4D_STAGE2_ITERATIONS}
stage3_iterations=${V4D_STAGE3_ITERATIONS}
high_lr_stage1=${V4D_HIGH_LR_STAGE1}
high_lr_stage2=${V4D_HIGH_LR_STAGE2}
high_lr_stage3=${V4D_HIGH_LR_STAGE3}
low_lr_stage2=${V4D_LOW_LR_STAGE2}
seed=${SEED}
CONFIG
cd "${RUN_DIR}"
set +e
srun --export=ALL --unbuffered "${PY}" "${TRAIN_SCRIPT}" \
  --task "${TASK}" --low_level_checkpoint "${LOW_CHECKPOINT}" --log_dir "${LOG_DIR}" --run_name "${RUN_NAME}" \
  --num_envs "${NUM_ENVS}" --max_iterations "${MAX_ITERATIONS}" --low_level_steps_per_action 5 --high_level_episode_steps 300 \
  --seed "${SEED}" --device cuda:0 --headless 2>&1 | tee "${RUN_DIR}/training.log"
RUN_RC="${PIPESTATUS[0]}"
set -e
echo "RUN_RC=${RUN_RC}"
HIGH_MODEL="${LOG_DIR}/high_model_final.pt"; LOW_MODEL="${LOG_DIR}/low_model_final.pt"; SUMMARY="${LOG_DIR}/v4d_joint_summary.json"; CONTRACT="${LOG_DIR}/v4d_joint_contract.json"
if [ "${RUN_RC}" -eq 0 ] && [ -s "${HIGH_MODEL}" ] && [ -s "${LOW_MODEL}" ] && [ -s "${SUMMARY}" ] && [ -s "${CONTRACT}" ] \
  && [ -f "${LOG_DIR}/V4D_JOINT_OK" ] && grep -qF '[V4D_CONTRACT]' "${RUN_DIR}/training.log" \
  && grep -qF '[V4D_ZERO_SUM_GUARD]' "${RUN_DIR}/training.log" && grep -qF '[V4D_JOINT_ITER]' "${RUN_DIR}/training.log" \
  && grep -qF '[V4D_FINAL_GUARD]' "${RUN_DIR}/training.log" && grep -qF 'V4D_JOINT_TRAINING_COMPLETE' "${RUN_DIR}/training.log" \
  && ! grep -qF '[V4D_FATAL]' "${RUN_DIR}/training.log"; then
  touch "${RUN_DIR}/V4D_JOINT_ACCEPTED"
  sha256sum "${HIGH_MODEL}" "${LOW_MODEL}" "${SUMMARY}" "${CONTRACT}" > "${RUN_DIR}/output_sha256.txt"
  echo "V4D_JOINT_ACCEPTED"; echo "HIGH_MODEL=${HIGH_MODEL}"; echo "LOW_MODEL=${LOW_MODEL}"; echo "SUMMARY=${SUMMARY}"; exit 0
fi
echo "===== Important errors ====="
grep -nEi 'fatal|error|exception|traceback|mismatch|overflow|underflow|cuda|nan|failed|guard' "${RUN_DIR}/training.log" "${RUN_DIR}/worker.log" "${RUN_DIR}"/slurm-*.out "${RUN_DIR}"/slurm-*.err 2>/dev/null | tail -n 260 || true
echo "V4D_JOINT_FAILED"; exit 2
