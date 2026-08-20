#!/usr/bin/env bash
#SBATCH --job-name=r854-rec64
#SBATCH --account=torch_pr_1066_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-r854-%j.out

set -Eeuo pipefail
PROJECT="${SLURM_SUBMIT_DIR:?submit from project root}"; EXP="$PROJECT/experiments/r8_54_strength_conditioned_recovery"
RUN_ROOT="/scratch/lj2634/runs/go2_soccer_r8_54_strength_conditioned_recovery"; RUN_DIR="$RUN_ROOT/r854_recovery64_${SLURM_JOB_ID}"; RUNTIME="/scratch/lj2634/job_runtime/${SLURM_JOB_ID}_r854"
FREEZE="/scratch/lj2634/runs/go2_soccer_kick_freeze/ballfree_r841_grace6_20260810_185244"; ISAACLAB_ROOT="/scratch/lj2634/IsaacLab_232_src_37ddf62"; ENV_ROOT="/scratch/lj2634/envs/isaaclab232_v2"; GLU_ROOT="/scratch/lj2634/vendor/libglu-9.0.0"; PY="$ENV_ROOT/bin/python"
export HOME="$RUNTIME/home" XDG_CACHE_HOME="$RUNTIME/cache" XDG_CONFIG_HOME="$RUNTIME/config" XDG_DATA_HOME="$RUNTIME/data" TMPDIR="$RUNTIME/tmp" MPLCONFIGDIR="$RUNTIME/mpl"
export LD_LIBRARY_PATH="$GLU_ROOT/lib:$ENV_ROOT/lib:${LD_LIBRARY_PATH:-}"; export LD_PRELOAD="$ENV_ROOT/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"; export PYTHONPATH="$PROJECT/source/isaac_go2_low_level_rl:$ISAACLAB_ROOT/source/isaaclab:$ISAACLAB_ROOT/source/isaaclab_tasks:${PYTHONPATH:-}"; export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 GIT_PYTHON_REFRESH=quiet
mkdir -p "$RUN_DIR/output" "$HOME" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$TMPDIR" "$MPLCONFIGDIR" "$RUN_ROOT"
printf '%s\n' "$RUN_DIR" > "$RUN_ROOT/latest_r854.txt"; printf '%s\n' "$RUN_DIR" > "$EXP/latest_r854.txt"

R853_RUN="$(tr -d '\r\n' < /scratch/lj2634/runs/go2_soccer_r8_53_phase_recovery64/latest_r853.txt)"; R851_RUN="$(tr -d '\r\n' < /scratch/lj2634/runs/go2_soccer_r8_51_kick64_true_contact/latest_r851.txt)"; R850_RUN="$(tr -d '\r\n' < "$PROJECT/experiments/r8_50_real_contact_truth/latest_r850.txt")"
SOURCE="$R853_RUN/train_r853_phase_recovery64.py"; R853_CHECKPOINT="$R853_RUN/output/recovery64_final.pt"; KICK="$R851_RUN/output/kick64_final.pt"; R851_SUMMARY="$R851_RUN/output/r851_summary.json"; BODY_JSON="$R850_RUN/body_names.json"
for required in "$SOURCE" "$R853_CHECKPOINT" "$KICK" "$R851_SUMMARY" "$BODY_JSON" "$FREEZE/artifacts/r821_lift_base.pt"; do [[ -s "$required" ]] || { echo "R854_FATAL missing parent: $required" >&2; exit 32; }; done
SOURCE_HASH="$(sha256sum "$SOURCE" | awk '{print $1}')"; "$PY" "$EXP/patch_r854_source.py" --source "$SOURCE" --output "$RUN_DIR/train_r854_strength_recovery.py"; "$PY" -m py_compile "$RUN_DIR/train_r854_strength_recovery.py"

export R854_STRENGTH_RECOVERY_ONLY=1 R854_FREEZE_DIR="$FREEZE" R854_KICK="$KICK" R854_R851_SUMMARY="$R851_SUMMARY" R854_BODY_JSON="$BODY_JSON" R851_BODY_JSON="$BODY_JSON" R854_R853_CHECKPOINT="$R853_CHECKPOINT" R854_OUTPUT="$RUN_DIR/output"
export R854_UPDATES="${R854_UPDATES:-28}" R854_ROLLOUT_STEPS="${R854_ROLLOUT_STEPS:-256}" R854_EVAL_EPISODES="${R854_EVAL_EPISODES:-1024}" V4K1_SETTLE_MAX_STEPS=180 V4K1_BALL_ARM_STEPS=4
{
  printf 'job_id=%s\nsource=%s\nsource_sha256=%s\n' "$SLURM_JOB_ID" "$SOURCE" "$SOURCE_HASH"; printf 'r853_warm_start=%s\nr851_kick=%s\nr850_contact_parent=%s\n' "$R853_CHECKPOINT" "$KICK" "$R850_RUN"; printf 'kick_frozen=true\nload_lift_swing_frozen=true\nr849_used=false\n'; printf 'strength_observation=%s\n' 'Recovery64_slot63=desired_strength;env_strength=1.0'; printf 'curriculum=%s\ndense_reward=%s\n' 'u1-8:1.25,u9-16:1.0+1.25,u17+:all' 'pose+contact+support4+touchdown+all_now+hold3'; printf 'num_envs=%s\nupdates=%s\nrollout_steps=%s\neval_episodes=%s\n' "${R854_NUM_ENVS:-128}" "$R854_UPDATES" "$R854_ROLLOUT_STEPS" "$R854_EVAL_EPISODES"; printf 'actual_filtered_contact=true\naccept_each_strength=real>=0.95,touchdown>=0.80,recovery3>=0.60,wrong<=0.02,body<=0.02\n'
} > "$RUN_DIR/contract.txt"
cd "$PROJECT"; "$PY" "$RUN_DIR/train_r854_strength_recovery.py" --task Isaac-Go2-Soccer-Milestone2-v0 --num_envs "${R854_NUM_ENVS:-128}" --device cuda:0 --headless --max_iterations 1 --run_name r854_recovery64 --low_level_checkpoint "$FREEZE/artifacts/r821_lift_base.pt" --log_dir "$RUN_DIR"
for required in recovery64_latest.pt recovery64_final.pt r854_summary.json; do [[ -s "$RUN_DIR/output/$required" ]] || { echo "R854_FATAL missing output: $required" >&2; exit 41; }; done
sha256sum "$RUN_DIR/train_r854_strength_recovery.py" "$RUN_DIR/contract.txt" "$RUN_DIR/output/recovery64_latest.pt" "$RUN_DIR/output/recovery64_final.pt" "$RUN_DIR/output/r854_summary.json" > "$RUN_DIR/SHA256SUMS"
echo "R854_WORKER_OK run=$RUN_DIR"
