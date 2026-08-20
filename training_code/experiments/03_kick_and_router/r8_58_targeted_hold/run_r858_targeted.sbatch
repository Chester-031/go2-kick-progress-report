#!/usr/bin/env bash
#SBATCH --job-name=r858-hold
#SBATCH --account=torch_pr_1066_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=slurm-r858-%j.out
set -Eeuo pipefail
PROJECT="${SLURM_SUBMIT_DIR:?submit from project root}"
EXP="$PROJECT/experiments/r8_58_targeted_hold"
ROOT="/scratch/lj2634/runs/go2_soccer_r8_58_targeted_hold"
RUN="$ROOT/r858_hold_${SLURM_JOB_ID}"
RUNTIME="/scratch/lj2634/job_runtime/${SLURM_JOB_ID}_r858"
FREEZE="/scratch/lj2634/runs/go2_soccer_kick_freeze/ballfree_r841_grace6_20260810_185244"
ISAACLAB_ROOT="/scratch/lj2634/IsaacLab_232_src_37ddf62"
ENV_ROOT="/scratch/lj2634/envs/isaaclab232_v2"
GLU_ROOT="/scratch/lj2634/vendor/libglu-9.0.0"
PY="$ENV_ROOT/bin/python"
export HOME="$RUNTIME/home" XDG_CACHE_HOME="$RUNTIME/cache" XDG_CONFIG_HOME="$RUNTIME/config" XDG_DATA_HOME="$RUNTIME/data" TMPDIR="$RUNTIME/tmp" MPLCONFIGDIR="$RUNTIME/mpl"
export LD_LIBRARY_PATH="$GLU_ROOT/lib:$ENV_ROOT/lib:${LD_LIBRARY_PATH:-}" LD_PRELOAD="$ENV_ROOT/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}" PYTHONPATH="$PROJECT/source/isaac_go2_low_level_rl:$ISAACLAB_ROOT/source/isaaclab:$ISAACLAB_ROOT/source/isaaclab_tasks:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 GIT_PYTHON_REFRESH=quiet
mkdir -p "$RUN/output" "$HOME" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$TMPDIR" "$MPLCONFIGDIR" "$ROOT"
printf '%s\n' "$RUN" > "$ROOT/latest_r858.txt"
printf '%s\n' "$RUN" > "$EXP/latest_r858.txt"
R857="$(tr -d '\r\n' < /scratch/lj2634/runs/go2_soccer_r8_57_balanced_hold/latest_r857.txt)"
R855="$(tr -d '\r\n' < /scratch/lj2634/runs/go2_soccer_r8_55_touchdown_hold/latest_r855.txt)"
R851="$(tr -d '\r\n' < /scratch/lj2634/runs/go2_soccer_r8_51_kick64_true_contact/latest_r851.txt)"
R850="$(tr -d '\r\n' < "$PROJECT/experiments/r8_50_real_contact_truth/latest_r850.txt")"
SOURCE="$R857/train_r857_balanced.py"
PRIOR="$R857/output/r857_hold_winner.json"
RECOVERY="$R855/output/recovery64_final.pt"
KICK="$R851/output/kick64_final.pt"
S851="$R851/output/r851_summary.json"
BODY="$R850/body_names.json"
for f in "$SOURCE" "$PRIOR" "$RECOVERY" "$KICK" "$S851" "$BODY" "$FREEZE/artifacts/r821_lift_base.pt"; do [[ -s "$f" ]] || { echo "R858_FATAL missing: $f" >&2; exit 32; }; done
HASH="$(sha256sum "$SOURCE" | awk '{print $1}')"
"$PY" "$EXP/patch_r858_source.py" --source "$SOURCE" --output "$RUN/train_r858_targeted.py"
"$PY" -m py_compile "$RUN/train_r858_targeted.py"
export R858_TARGETED_HOLD_ONLY=1 R858_FREEZE_DIR="$FREEZE" R858_KICK="$KICK" R858_R851_SUMMARY="$S851" R858_BODY_JSON="$BODY" R851_BODY_JSON="$BODY" R858_RECOVERY="$RECOVERY" R858_R857_WINNER="$PRIOR" R858_OUTPUT="$RUN/output"
export R858_SCREEN_EPISODES="${R858_SCREEN_EPISODES:-24}" R858_SEMIFINAL_EPISODES="${R858_SEMIFINAL_EPISODES:-256}" R858_FINAL_EPISODES="${R858_FINAL_EPISODES:-512}" V4K1_SETTLE_MAX_STEPS=180 V4K1_BALL_ARM_STEPS=4
{
  printf 'job_id=%s\nsource=%s\nsource_sha256=%s\nr857_parent=%s\nr855_recovery=%s\n' "$SLURM_JOB_ID" "$SOURCE" "$HASH" "$R857" "$RECOVERY"
  printf 'training=false\nnetwork_frozen=true\nsearch_strengths=0.75,1.0\nhigh_strength=frozen_r857\ncontroller=static_latch_plus_age_ramp\ncandidate_count=144\n'
  printf 'screen_episodes_per_case=%s\nsemifinal_topk=8\nsemifinal_episodes=%s\nfinal_episodes_per_condition=%s\nfinal_conditions=exact,uniform_4mm_geometry_jitter\n' "$R858_SCREEN_EPISODES" "$R858_SEMIFINAL_EPISODES" "$R858_FINAL_EPISODES"
} > "$RUN/contract.txt"
cd "$PROJECT"
"$PY" "$RUN/train_r858_targeted.py" --task Isaac-Go2-Soccer-Milestone2-v0 --num_envs "${R858_NUM_ENVS:-128}" --device cuda:0 --headless --max_iterations 1 --run_name r858_hold --low_level_checkpoint "$FREEZE/artifacts/r821_lift_base.pt" --log_dir "$RUN"
for f in r858_hold_winner.json r858_summary.json; do [[ -s "$RUN/output/$f" ]] || { echo "R858_FATAL missing output: $f" >&2; exit 41; }; done
sha256sum "$RUN/train_r858_targeted.py" "$RUN/contract.txt" "$RUN/output/r858_hold_winner.json" "$RUN/output/r858_summary.json" > "$RUN/SHA256SUMS"
echo "R858_WORKER_OK run=$RUN"
