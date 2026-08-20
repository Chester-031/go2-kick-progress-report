#!/usr/bin/env bash
#SBATCH --job-name=r853-rec64
#SBATCH --account=torch_pr_1066_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=slurm-r853-%j.out

set -Eeuo pipefail

PROJECT="${SLURM_SUBMIT_DIR:?submit from project root}"
EXP="$PROJECT/experiments/r8_53_phase_recovery64"
RUN_ROOT="/scratch/lj2634/runs/go2_soccer_r8_53_phase_recovery64"
RUN_DIR="$RUN_ROOT/r853_recovery64_${SLURM_JOB_ID}"
RUNTIME="/scratch/lj2634/job_runtime/${SLURM_JOB_ID}_r853"
FREEZE="/scratch/lj2634/runs/go2_soccer_kick_freeze/ballfree_r841_grace6_20260810_185244"
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
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 GIT_PYTHON_REFRESH=quiet

mkdir -p "$RUN_DIR/output" "$HOME" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" \
  "$XDG_DATA_HOME" "$TMPDIR" "$MPLCONFIGDIR" "$RUN_ROOT"
printf '%s\n' "$RUN_DIR" > "$RUN_ROOT/latest_r853.txt"
printf '%s\n' "$RUN_DIR" > "$EXP/latest_r853.txt"

R851_RUN="$(tr -d '\r\n' < /scratch/lj2634/runs/go2_soccer_r8_51_kick64_true_contact/latest_r851.txt)"
R850_RUN="$(tr -d '\r\n' < "$PROJECT/experiments/r8_50_real_contact_truth/latest_r850.txt")"
SOURCE="$R851_RUN/train_r851_kick64.py"
KICK="$R851_RUN/output/kick64_final.pt"
R851_SUMMARY="$R851_RUN/output/r851_summary.json"
BODY_JSON="$R850_RUN/body_names.json"
R839_VECTOR="$FREEZE/artifacts/r839_recovery_vector_u32.pt"
R835_SCALE="$FREEZE/artifacts/r835_swing_stability_u32.pt"

for required in "$SOURCE" "$KICK" "$R851_SUMMARY" "$BODY_JSON" \
  "$R839_VECTOR" "$R835_SCALE" "$FREEZE/artifacts/r821_lift_base.pt"; do
  if [[ ! -s "$required" ]]; then
    echo "R853_FATAL missing parent: $required" >&2
    exit 32
  fi
done

SOURCE_HASH="$(sha256sum "$SOURCE" | awk '{print $1}')"
"$PY" "$EXP/patch_r853_source.py" --source "$SOURCE" --output "$RUN_DIR/train_r853_phase_recovery64.py"
"$PY" -m py_compile "$RUN_DIR/train_r853_phase_recovery64.py"

export R853_PHASE_RECOVERY64_ONLY=1
export R853_FREEZE_DIR="$FREEZE"
export R853_KICK="$KICK"
export R853_R851_SUMMARY="$R851_SUMMARY"
export R853_BODY_JSON="$BODY_JSON"
# The inherited R8.51 contact-sensor bootstrap reads this alias.
export R851_BODY_JSON="$BODY_JSON"
export R853_OUTPUT="$RUN_DIR/output"
export R853_UPDATES="${R853_UPDATES:-20}"
export R853_ROLLOUT_STEPS="${R853_ROLLOUT_STEPS:-256}"
export R853_EVAL_EPISODES="${R853_EVAL_EPISODES:-1024}"
export V4K1_SETTLE_MAX_STEPS=180 V4K1_BALL_ARM_STEPS=4

{
  printf 'job_id=%s\n' "$SLURM_JOB_ID"
  printf 'source=%s\nsource_sha256=%s\n' "$SOURCE" "$SOURCE_HASH"
  printf 'r851_parent=%s\nkick_parent=%s\n' "$R851_RUN" "$KICK"
  printf 'r850_contact_parent=%s\nbody_json=%s\n' "$R850_RUN" "$BODY_JSON"
  printf 'recovery_parent=%s\nrecovery_scale_parent=%s\n' "$R839_VECTOR" "$R835_SCALE"
  printf 'r849_recovery_used=false\nkick_frozen=true\nload_lift_swing_frozen=true\n'
  printf 'obs_contract=%s\n' 'Recovery64=normalized_70D[0:63,69]'
  printf 'recovery_gate=%s\n' 'real_strike&swing_seen&phase>=0.78'
  printf 'strict_gate_phase=%s\n' 'phase>=0.88'
  printf 'strength_gain=%s\n' '0.75:0.5,1.0:1.0,1.25:2.0'
  printf 'num_envs=%s\nupdates=%s\nrollout_steps=%s\neval_episodes=%s\n' \
    "${R853_NUM_ENVS:-128}" "$R853_UPDATES" "$R853_ROLLOUT_STEPS" "$R853_EVAL_EPISODES"
  printf 'actual_filtered_contact=true\nhigh_level_reward=unchanged_not_used\n'
} > "$RUN_DIR/contract.txt"

cd "$PROJECT"
"$PY" "$RUN_DIR/train_r853_phase_recovery64.py" \
  --task Isaac-Go2-Soccer-Milestone2-v0 \
  --num_envs "${R853_NUM_ENVS:-128}" \
  --device cuda:0 --headless --max_iterations 1 \
  --run_name r853_recovery64 \
  --low_level_checkpoint "$FREEZE/artifacts/r821_lift_base.pt" \
  --log_dir "$RUN_DIR"

for required in recovery64_latest.pt recovery64_final.pt r853_summary.json; do
  if [[ ! -s "$RUN_DIR/output/$required" ]]; then
    echo "R853_FATAL missing output: $RUN_DIR/output/$required" >&2
    exit 41
  fi
done

sha256sum "$RUN_DIR/train_r853_phase_recovery64.py" "$RUN_DIR/contract.txt" \
  "$RUN_DIR/output/recovery64_latest.pt" "$RUN_DIR/output/recovery64_final.pt" \
  "$RUN_DIR/output/r853_summary.json" > "$RUN_DIR/SHA256SUMS"

echo "R853_WORKER_OK run=$RUN_DIR"
