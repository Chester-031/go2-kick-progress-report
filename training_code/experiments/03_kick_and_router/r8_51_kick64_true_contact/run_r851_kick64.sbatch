#!/usr/bin/env bash
#SBATCH --job-name=r851-kick64
#SBATCH --account=torch_pr_1066_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-r851-%j.out

set -Eeuo pipefail

PROJECT="${SLURM_SUBMIT_DIR:?submit from project root}"
EXP="$PROJECT/experiments/r8_51_kick64_true_contact"
SOURCE="$PROJECT/experiments/r8_49_recovery_finetune/train_r849_recovery.py"
FREEZE="/scratch/lj2634/runs/go2_soccer_kick_freeze/ballfree_r841_grace6_20260810_185244"
RUN_ROOT="/scratch/lj2634/runs/go2_soccer_r8_51_kick64_true_contact"
RUN_DIR="$RUN_ROOT/r851_kick64_${SLURM_JOB_ID}"
RUNTIME="/scratch/lj2634/job_runtime/${SLURM_JOB_ID}_r851"
ISAACLAB_ROOT="/scratch/lj2634/IsaacLab_232_src_37ddf62"
ENV_ROOT="/scratch/lj2634/envs/isaaclab232_v2"
GLU_ROOT="/scratch/lj2634/vendor/libglu-9.0.0"
PY="$ENV_ROOT/bin/python"

export HOME="$RUNTIME/home"
export XDG_CACHE_HOME="$RUNTIME/cache"
export XDG_CONFIG_HOME="$RUNTIME/config"
export XDG_DATA_HOME="$RUNTIME/data"
export TMPDIR="$RUNTIME/tmp"
export MPLCONFIGDIR="$RUNTIME/mpl"
export LD_LIBRARY_PATH="$GLU_ROOT/lib:$ENV_ROOT/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$ENV_ROOT/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
export PYTHONPATH="$PROJECT/source/isaac_go2_low_level_rl:$ISAACLAB_ROOT/source/isaaclab:$ISAACLAB_ROOT/source/isaaclab_tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export GIT_PYTHON_REFRESH=quiet

mkdir -p "$RUN_DIR/output" "$HOME" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" \
  "$XDG_DATA_HOME" "$TMPDIR" "$MPLCONFIGDIR"
mkdir -p "$RUN_ROOT"
printf '%s\n' "$RUN_DIR" > "$RUN_ROOT/latest_r851.txt"
printf '%s\n' "$RUN_DIR" > "$EXP/latest_r851.txt"

test -s "$SOURCE"
SOURCE_HASH="$(sha256sum "$SOURCE" | awk '{print $1}')"
if [[ "$SOURCE_HASH" != "8abfe17a5b6f348aa32e75227280e71af4681f99f7f7b9556bd3098e8a7d4844" ]]; then
  echo "R851_FATAL unexpected R849 source hash: $SOURCE_HASH" >&2
  exit 31
fi

R850_RUN="$(tr -d '\r\n' < "$PROJECT/experiments/r8_50_real_contact_truth/latest_r850.txt")"
R848_RUN="$(tr -d '\r\n' < /scratch/lj2634/runs/go2_soccer_r8_48_postbridge_strike_ppo/latest_r848.txt)"
R849_RUN="$(tr -d '\r\n' < /scratch/lj2634/runs/go2_soccer_r8_49_recovery_finetune/latest_r849.txt)"

R850_SUMMARY="$R850_RUN/r850_real_contact_summary.json"
BODY_JSON="$R850_RUN/body_names.json"
STRIKE="$R848_RUN/output/winner_strike_u32.pt"
RECOVERY="$R849_RUN/output/recovery_vector_finetuned.pt"

for required in "$R850_SUMMARY" "$BODY_JSON" "$STRIKE" "$RECOVERY" \
  "$FREEZE/source_r841_grace6.py"; do
  if [[ ! -s "$required" ]]; then
    echo "R851_FATAL missing parent: $required" >&2
    exit 32
  fi
done

"$PY" "$EXP/patch_r851_source.py" --source "$SOURCE" --output "$RUN_DIR/train_r851_kick64.py"

export R851_KICK64_ONLY=1
export R851_FREEZE_DIR="$FREEZE"
export R851_STRIKE="$STRIKE"
export R851_RECOVERY="$RECOVERY"
export R851_R850_SUMMARY="$R850_SUMMARY"
export R851_BODY_JSON="$BODY_JSON"
export R851_OUTPUT="$RUN_DIR/output"
export R851_UPDATES="${R851_UPDATES:-12}"
export R851_ROLLOUT_STEPS="${R851_ROLLOUT_STEPS:-192}"
export R851_EVAL_EPISODES="${R851_EVAL_EPISODES:-512}"
export V4K1_SETTLE_MAX_STEPS=180
export V4K1_BALL_ARM_STEPS=4

{
  printf 'job_id=%s\n' "$SLURM_JOB_ID"
  printf 'source=%s\n' "$SOURCE"
  printf 'source_sha256=%s\n' "$SOURCE_HASH"
  printf 'r850_parent=%s\n' "$R850_RUN"
  printf 'strike_parent=%s\n' "$STRIKE"
  printf 'recovery_parent=%s\n' "$RECOVERY"
  printf 'freeze_parent=%s\n' "$FREEZE"
  printf 'num_envs=%s\n' "${R851_NUM_ENVS:-128}"
  printf 'updates=%s\n' "$R851_UPDATES"
  printf 'rollout_steps=%s\n' "$R851_ROLLOUT_STEPS"
  printf 'eval_episodes=%s\n' "$R851_EVAL_EPISODES"
  printf 'obs_contract=%s\n' '64D_from_70D_columns_0_to_62_and_69'
  printf 'actual_filtered_contact=true\n'
  printf 'high_level_reward=unchanged_not_used\n'
} > "$RUN_DIR/contract.txt"

cd "$PROJECT"
"$PY" "$RUN_DIR/train_r851_kick64.py" \
  --task Isaac-Go2-Soccer-Milestone2-v0 \
  --num_envs "${R851_NUM_ENVS:-128}" \
  --device cuda:0 --headless --max_iterations 1 \
  --run_name r851_kick64 \
  --low_level_checkpoint "$FREEZE/artifacts/r821_lift_base.pt" \
  --log_dir "$RUN_DIR"

for required in kick64_latest.pt kick64_final.pt r851_summary.json; do
  if [[ ! -s "$RUN_DIR/output/$required" ]]; then
    echo "R851_FATAL missing output: $RUN_DIR/output/$required" >&2
    exit 41
  fi
done

sha256sum "$RUN_DIR/train_r851_kick64.py" "$RUN_DIR/contract.txt" \
  "$RUN_DIR/output/kick64_latest.pt" "$RUN_DIR/output/kick64_final.pt" \
  "$RUN_DIR/output/r851_summary.json" > "$RUN_DIR/SHA256SUMS"

echo "R851_WORKER_OK run=$RUN_DIR"
