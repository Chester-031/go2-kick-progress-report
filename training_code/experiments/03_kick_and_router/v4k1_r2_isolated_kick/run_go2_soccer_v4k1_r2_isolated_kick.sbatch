#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
set -Eeuo pipefail
echo "V4K1_R2_WORKER_ARCH_INFERENCE=true"
PROJECT_DIR="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
ISAACLAB_ROOT="/scratch/lj2634/IsaacLab_232_src_37ddf62"
ENV_ROOT="/scratch/lj2634/envs/isaaclab232_v2"
GLU_ROOT="/scratch/lj2634/vendor/libglu-9.0.0"
PY="${ENV_ROOT}/bin/python"
TRAIN_SCRIPT="${PROJECT_DIR}/experiments/v4k1_r2_isolated_kick/train_go2_soccer_v4k1_r2_isolated_kick.py"
for name in TASK LOW_CHECKPOINT RUN_DIR LOG_DIR RUN_NAME NUM_ENVS MAX_ITERATIONS SEED V4K1_STAGE1_ITERATIONS V4K1_STAGE2_ITERATIONS V4K1_STAGE3_ITERATIONS; do
  [ -n "${!name:-}" ] || { echo "ERROR: ${name} is required." >&2; exit 20; }
done
mkdir -p "${RUN_DIR}" "${LOG_DIR}"
exec > >(tee -a "${RUN_DIR}/worker.log") 2>&1
echo "============================================================"
echo "V4K1 ISOLATED DELTA-POSE KICK TRAINING"
echo "============================================================"
echo "JOB_ID=${SLURM_JOB_ID}"
echo "HOST=$(hostname)"
echo "RUN_DIR=${RUN_DIR}"
echo "NUM_ENVS=${NUM_ENVS}"
echo "MAX_ITERATIONS=${MAX_ITERATIONS}"
echo "STAGES=${V4K1_STAGE1_ITERATIONS},${V4K1_STAGE2_ITERATIONS},${V4K1_STAGE3_ITERATIONS}"
echo "LOW_CHECKPOINT=${LOW_CHECKPOINT}"
test -x "${PY}"; test -s "${TRAIN_SCRIPT}"; test -s "${LOW_CHECKPOINT}"
if [ -e "${GLU_ROOT}/lib/libGLU.so.1" ]; then export LD_LIBRARY_PATH="${GLU_ROOT}/lib:${ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"; else export LD_LIBRARY_PATH="${ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"; fi
export LD_PRELOAD="${ENV_ROOT}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"
export PYTHONPATH="${PROJECT_DIR}/source/isaac_go2_low_level_rl:${ISAACLAB_ROOT}/source/isaaclab:${ISAACLAB_ROOT}/source/isaaclab_tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 GIT_PYTHON_REFRESH=quiet
RUNTIME_ROOT="/scratch/lj2634/job_runtime/${SLURM_JOB_ID}_v4k1_kick"
export HOME="${RUNTIME_ROOT}/home" XDG_CACHE_HOME="${RUNTIME_ROOT}/cache" XDG_CONFIG_HOME="${RUNTIME_ROOT}/config" XDG_DATA_HOME="${RUNTIME_ROOT}/data" TMPDIR="${RUNTIME_ROOT}/tmp" MPLCONFIGDIR="${RUNTIME_ROOT}/matplotlib"
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_DATA_HOME}" "${TMPDIR}" "${MPLCONFIGDIR}"
cp -a "${TRAIN_SCRIPT}" "${RUN_DIR}/train_go2_soccer_v4k1_r2_isolated_kick.snapshot.py"
git -C "${PROJECT_DIR}" rev-parse HEAD > "${RUN_DIR}/git_commit.txt"
git -C "${PROJECT_DIR}" status --short > "${RUN_DIR}/git_status.txt"
sha256sum "${TRAIN_SCRIPT}" "${LOW_CHECKPOINT}" > "${RUN_DIR}/input_sha256.txt"
cat > "${RUN_DIR}/resolved_launch_config.txt" <<CONFIG
version=v4k1_isolated_delta_pose_kick
training_scope=isolated_low_level_kick_only
high_policy=not_used
soccer_goal_reward=not_used
source_low_checkpoint=${LOW_CHECKPOINT}
delta_pose=planar_base_delta_pose_direct_context
kick_context=phase,leg,direction,strength
low_action=12_joint_position_residuals
teacher_in_actor_observation=false
num_envs=${NUM_ENVS}
iterations=${MAX_ITERATIONS}
episode_steps=${V4K1_KICK_EPISODE_STEPS:-60}
stage1_iterations=${V4K1_STAGE1_ITERATIONS}
stage2_iterations=${V4K1_STAGE2_ITERATIONS}
stage3_iterations=${V4K1_STAGE3_ITERATIONS}
seed=${SEED}
CONFIG
cd "${RUN_DIR}"
set +e
srun --export=ALL --unbuffered "${PY}" "${TRAIN_SCRIPT}" \
  --task "${TASK}" --low_level_checkpoint "${LOW_CHECKPOINT}" --log_dir "${LOG_DIR}" --run_name "${RUN_NAME}" \
  --num_envs "${NUM_ENVS}" --max_iterations "${MAX_ITERATIONS}" --kick_episode_steps "${V4K1_KICK_EPISODE_STEPS:-60}" \
  --kick_fixed_leg "${V4K1_FIXED_LEG:-alternate}" --seed "${SEED}" --device cuda:0 --headless 2>&1 | tee "${RUN_DIR}/training.log"
RUN_RC="${PIPESTATUS[0]}"
set -e
echo "RUN_RC=${RUN_RC}"
MODEL="${LOG_DIR}/kick_model_final.pt"; SUMMARY="${LOG_DIR}/v4k1_kick_summary.json"; CONTRACT="${LOG_DIR}/v4k1_kick_contract.json"
if [ "${RUN_RC}" -eq 0 ] && [ -s "${MODEL}" ] && [ -s "${SUMMARY}" ] && [ -s "${CONTRACT}" ] \
  && [ -f "${LOG_DIR}/V4K1_ISOLATED_KICK_OK" ] && grep -qF '[V4K1_CONTRACT]' "${RUN_DIR}/training.log" \
  && grep -qF '[V4K1_INPUT_GUARD]' "${RUN_DIR}/training.log" && grep -qF '[V4K1_ITER]' "${RUN_DIR}/training.log" \
  && grep -qF '[V4K1_FINAL_GUARD]' "${RUN_DIR}/training.log" && grep -qF 'V4K1_ISOLATED_KICK_TRAINING_COMPLETE' "${RUN_DIR}/training.log" \
  && ! grep -qF '[V4K1_FATAL]' "${RUN_DIR}/training.log"; then
  touch "${RUN_DIR}/V4K1_ISOLATED_KICK_ACCEPTED"
  sha256sum "${MODEL}" "${SUMMARY}" "${CONTRACT}" > "${RUN_DIR}/output_sha256.txt"
  echo "V4K1_ISOLATED_KICK_ACCEPTED"; echo "MODEL=${MODEL}"; echo "SUMMARY=${SUMMARY}"; exit 0
fi
echo "===== Important errors ====="
grep -nEi 'fatal|error|exception|traceback|mismatch|overflow|underflow|cuda|nan|failed|guard' "${RUN_DIR}/training.log" "${RUN_DIR}/worker.log" "${RUN_DIR}"/slurm-*.out "${RUN_DIR}"/slurm-*.err 2>/dev/null | tail -n 260 || true
echo "V4K1_ISOLATED_KICK_FAILED"; exit 2
