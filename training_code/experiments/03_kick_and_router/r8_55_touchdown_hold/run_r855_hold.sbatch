#!/usr/bin/env bash
#SBATCH --job-name=r855-hold
#SBATCH --account=torch_pr_1066_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=14:00:00
#SBATCH --output=slurm-r855-%j.out
set -Eeuo pipefail
PROJECT="${SLURM_SUBMIT_DIR:?submit from project root}"; EXP="$PROJECT/experiments/r8_55_touchdown_hold"; RUN_ROOT="/scratch/lj2634/runs/go2_soccer_r8_55_touchdown_hold"; RUN_DIR="$RUN_ROOT/r855_hold_${SLURM_JOB_ID}"; RUNTIME="/scratch/lj2634/job_runtime/${SLURM_JOB_ID}_r855"; FREEZE="/scratch/lj2634/runs/go2_soccer_kick_freeze/ballfree_r841_grace6_20260810_185244"; ISAACLAB_ROOT="/scratch/lj2634/IsaacLab_232_src_37ddf62"; ENV_ROOT="/scratch/lj2634/envs/isaaclab232_v2"; GLU_ROOT="/scratch/lj2634/vendor/libglu-9.0.0"; PY="$ENV_ROOT/bin/python"
export HOME="$RUNTIME/home" XDG_CACHE_HOME="$RUNTIME/cache" XDG_CONFIG_HOME="$RUNTIME/config" XDG_DATA_HOME="$RUNTIME/data" TMPDIR="$RUNTIME/tmp" MPLCONFIGDIR="$RUNTIME/mpl"; export LD_LIBRARY_PATH="$GLU_ROOT/lib:$ENV_ROOT/lib:${LD_LIBRARY_PATH:-}" LD_PRELOAD="$ENV_ROOT/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}" PYTHONPATH="$PROJECT/source/isaac_go2_low_level_rl:$ISAACLAB_ROOT/source/isaaclab:$ISAACLAB_ROOT/source/isaaclab_tasks:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 GIT_PYTHON_REFRESH=quiet
mkdir -p "$RUN_DIR/output" "$HOME" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$TMPDIR" "$MPLCONFIGDIR" "$RUN_ROOT"; printf '%s\n' "$RUN_DIR" > "$RUN_ROOT/latest_r855.txt"; printf '%s\n' "$RUN_DIR" > "$EXP/latest_r855.txt"
R854_RUN="$(tr -d '\r\n' < /scratch/lj2634/runs/go2_soccer_r8_54_strength_conditioned_recovery/latest_r854.txt)"; R851_RUN="$(tr -d '\r\n' < /scratch/lj2634/runs/go2_soccer_r8_51_kick64_true_contact/latest_r851.txt)"; R850_RUN="$(tr -d '\r\n' < "$PROJECT/experiments/r8_50_real_contact_truth/latest_r850.txt")"; SOURCE="$R854_RUN/train_r854_strength_recovery.py"; PARENT="$R854_RUN/output/recovery64_final.pt"; KICK="$R851_RUN/output/kick64_final.pt"; R851_SUMMARY="$R851_RUN/output/r851_summary.json"; BODY_JSON="$R850_RUN/body_names.json"
for f in "$SOURCE" "$PARENT" "$KICK" "$R851_SUMMARY" "$BODY_JSON" "$FREEZE/artifacts/r821_lift_base.pt"; do [[ -s "$f" ]] || { echo "R855_FATAL missing parent: $f" >&2; exit 32; }; done
SOURCE_HASH="$(sha256sum "$SOURCE"|awk '{print $1}')"; "$PY" "$EXP/patch_r855_source.py" --source "$SOURCE" --output "$RUN_DIR/train_r855_touchdown_hold.py"; "$PY" -m py_compile "$RUN_DIR/train_r855_touchdown_hold.py"
export R855_TOUCHDOWN_HOLD_ONLY=1 R855_FREEZE_DIR="$FREEZE" R855_KICK="$KICK" R855_R851_SUMMARY="$R851_SUMMARY" R855_BODY_JSON="$BODY_JSON" R851_BODY_JSON="$BODY_JSON" R855_R854_CHECKPOINT="$PARENT" R855_OUTPUT="$RUN_DIR/output"; export R855_UPDATES="${R855_UPDATES:-32}" R855_ROLLOUT_STEPS="${R855_ROLLOUT_STEPS:-256}" R855_BASELINE_EPISODES="${R855_BASELINE_EPISODES:-256}" R855_EVAL_EPISODES="${R855_EVAL_EPISODES:-1024}" V4K1_SETTLE_MAX_STEPS=180 V4K1_BALL_ARM_STEPS=4
{
 printf 'job_id=%s\nsource=%s\nsource_sha256=%s\nr854_warm_start=%s\nkick_parent=%s\n' "$SLURM_JOB_ID" "$SOURCE" "$SOURCE_HASH" "$PARENT" "$KICK"; printf 'kick_frozen=true\nload_lift_swing_frozen=true\ntouchdown_latch=%s\n' 'first_all_now_joint_target_replayed_3_steps'; printf 'curriculum=%s\ndense_reward=%s\n' 'u1-8:low,u9-16:mid,u17-24:low+mid,u25+:weighted_all' 'pose+selected_fz+support4+touchdown+all_now+hold1/2/3'; printf 'num_envs=%s\nupdates=%s\nrollout_steps=%s\nbaseline_episodes=%s\neval_episodes=%s\n' "${R855_NUM_ENVS:-128}" "$R855_UPDATES" "$R855_ROLLOUT_STEPS" "$R855_BASELINE_EPISODES" "$R855_EVAL_EPISODES"; printf 'actual_filtered_contact=true\naccept_each_strength=real>=0.95,touchdown>=0.80,recovery3>=0.60,wrong<=0.02,body<=0.02\n'
} > "$RUN_DIR/contract.txt"
cd "$PROJECT"; "$PY" "$RUN_DIR/train_r855_touchdown_hold.py" --task Isaac-Go2-Soccer-Milestone2-v0 --num_envs "${R855_NUM_ENVS:-128}" --device cuda:0 --headless --max_iterations 1 --run_name r855_hold --low_level_checkpoint "$FREEZE/artifacts/r821_lift_base.pt" --log_dir "$RUN_DIR"
for f in recovery64_latest.pt recovery64_final.pt r855_summary.json; do [[ -s "$RUN_DIR/output/$f" ]] || { echo "R855_FATAL missing output: $f" >&2; exit 41; }; done
sha256sum "$RUN_DIR/train_r855_touchdown_hold.py" "$RUN_DIR/contract.txt" "$RUN_DIR/output/recovery64_latest.pt" "$RUN_DIR/output/recovery64_final.pt" "$RUN_DIR/output/r855_summary.json" > "$RUN_DIR/SHA256SUMS"; echo "R855_WORKER_OK run=$RUN_DIR"
