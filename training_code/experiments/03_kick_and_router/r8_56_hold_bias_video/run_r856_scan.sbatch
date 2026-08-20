#!/usr/bin/env bash
#SBATCH --job-name=r856-scan
#SBATCH --account=torch_pr_1066_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-r856-%j.out
set -Eeuo pipefail
PROJECT="${SLURM_SUBMIT_DIR:?submit from project root}"; EXP="$PROJECT/experiments/r8_56_hold_bias_video"; ROOT="/scratch/lj2634/runs/go2_soccer_r8_56_hold_bias"; RUN="$ROOT/r856_scan_${SLURM_JOB_ID}"; RUNTIME="/scratch/lj2634/job_runtime/${SLURM_JOB_ID}_r856"; FREEZE="/scratch/lj2634/runs/go2_soccer_kick_freeze/ballfree_r841_grace6_20260810_185244"; ISAACLAB_ROOT="/scratch/lj2634/IsaacLab_232_src_37ddf62"; ENV_ROOT="/scratch/lj2634/envs/isaaclab232_v2"; GLU_ROOT="/scratch/lj2634/vendor/libglu-9.0.0"; PY="$ENV_ROOT/bin/python"
export HOME="$RUNTIME/home" XDG_CACHE_HOME="$RUNTIME/cache" XDG_CONFIG_HOME="$RUNTIME/config" XDG_DATA_HOME="$RUNTIME/data" TMPDIR="$RUNTIME/tmp" MPLCONFIGDIR="$RUNTIME/mpl"; export LD_LIBRARY_PATH="$GLU_ROOT/lib:$ENV_ROOT/lib:${LD_LIBRARY_PATH:-}" LD_PRELOAD="$ENV_ROOT/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}" PYTHONPATH="$PROJECT/source/isaac_go2_low_level_rl:$ISAACLAB_ROOT/source/isaaclab:$ISAACLAB_ROOT/source/isaaclab_tasks:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 GIT_PYTHON_REFRESH=quiet
mkdir -p "$RUN/output" "$HOME" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$TMPDIR" "$MPLCONFIGDIR" "$ROOT"; printf '%s\n' "$RUN" > "$ROOT/latest_r856.txt"; printf '%s\n' "$RUN" > "$EXP/latest_r856.txt"
R855_RUN="$(tr -d '\r\n' < /scratch/lj2634/runs/go2_soccer_r8_55_touchdown_hold/latest_r855.txt)"; R851_RUN="$(tr -d '\r\n' < /scratch/lj2634/runs/go2_soccer_r8_51_kick64_true_contact/latest_r851.txt)"; R850_RUN="$(tr -d '\r\n' < "$PROJECT/experiments/r8_50_real_contact_truth/latest_r850.txt")"; SOURCE="$R855_RUN/train_r855_touchdown_hold.py"; RECOVERY="$R855_RUN/output/recovery64_final.pt"; KICK="$R851_RUN/output/kick64_final.pt"; S851="$R851_RUN/output/r851_summary.json"; BODY="$R850_RUN/body_names.json"
for f in "$SOURCE" "$RECOVERY" "$KICK" "$S851" "$BODY" "$FREEZE/artifacts/r821_lift_base.pt"; do [[ -s "$f" ]]||{ echo "R856_FATAL missing: $f" >&2; exit 32; }; done
HASH="$(sha256sum "$SOURCE"|awk '{print $1}')"; "$PY" "$EXP/patch_r856_source.py" --source "$SOURCE" --output "$RUN/train_r856_hold_bias.py"; "$PY" -m py_compile "$RUN/train_r856_hold_bias.py"
export R856_HOLD_SCAN_ONLY=1 R856_FREEZE_DIR="$FREEZE" R856_KICK="$KICK" R856_R851_SUMMARY="$S851" R856_BODY_JSON="$BODY" R851_BODY_JSON="$BODY" R856_RECOVERY="$RECOVERY" R856_OUTPUT="$RUN/output" R856_EPISODES_PER_CASE="${R856_EPISODES_PER_CASE:-64}" R856_CONFIRM_EPISODES="${R856_CONFIRM_EPISODES:-1024}" V4K1_SETTLE_MAX_STEPS=180 V4K1_BALL_ARM_STEPS=4
{
 printf 'job_id=%s\nsource=%s\nsource_sha256=%s\nr855_parent=%s\nrecovery_parent=%s\nkick_parent=%s\n' "$SLURM_JOB_ID" "$SOURCE" "$HASH" "$R855_RUN" "$RECOVERY" "$KICK"; printf 'training=false\nnetwork_frozen=true\ngrid=FL_thigh_x_FL_calf_-0.12_to_0.12_step_0.04\nhold_steps=5\n'; printf 'num_envs=%s\nepisodes_per_case=%s\nconfirm_episodes=%s\nactual_filtered_contact=true\n' "${R856_NUM_ENVS:-128}" "$R856_EPISODES_PER_CASE" "$R856_CONFIRM_EPISODES"
} > "$RUN/contract.txt"
cd "$PROJECT"; "$PY" "$RUN/train_r856_hold_bias.py" --task Isaac-Go2-Soccer-Milestone2-v0 --num_envs "${R856_NUM_ENVS:-128}" --device cuda:0 --headless --max_iterations 1 --run_name r856_scan --low_level_checkpoint "$FREEZE/artifacts/r821_lift_base.pt" --log_dir "$RUN"
for f in r856_hold_winner.json r856_summary.json; do [[ -s "$RUN/output/$f" ]]||{ echo "R856_FATAL missing output: $f" >&2; exit 41; }; done
sha256sum "$RUN/train_r856_hold_bias.py" "$RUN/contract.txt" "$RUN/output/r856_hold_winner.json" "$RUN/output/r856_summary.json" > "$RUN/SHA256SUMS"; echo "R856_WORKER_OK run=$RUN"
