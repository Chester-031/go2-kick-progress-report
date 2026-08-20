#!/usr/bin/env bash
#SBATCH --job-name=r850-contact
#SBATCH --account=torch_pr_1066_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-r850-%j.out

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?submit from the project root}"
EXP_DIR="$PROJECT_ROOT/experiments/r8_50_real_contact_truth"
FREEZE_DIR="${R850_FREEZE_DIR:-$PROJECT_ROOT/experiments/r8_42_ballfree_formal_acceptance/accepted_ballfree}"
RUN_DIR="$PROJECT_ROOT/runs/r850_real_contact_${SLURM_JOB_ID}"
ISAACLAB_ROOT="/scratch/lj2634/IsaacLab_232_src_37ddf62"
ENV_ROOT="/scratch/lj2634/envs/isaaclab232_v2"
GLU_ROOT="/scratch/lj2634/vendor/libglu-9.0.0"
PYTHON_BIN="${ISAAC_PYTHON:-$ENV_ROOT/bin/python}"
RUNTIME="/scratch/lj2634/job_runtime/${SLURM_JOB_ID}_r850"

export HOME="$RUNTIME/home"
export XDG_CACHE_HOME="$RUNTIME/cache"
export XDG_CONFIG_HOME="$RUNTIME/config"
export XDG_DATA_HOME="$RUNTIME/data"
export TMPDIR="$RUNTIME/tmp"
export MPLCONFIGDIR="$RUNTIME/mpl"
export LD_LIBRARY_PATH="$GLU_ROOT/lib:$ENV_ROOT/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$ENV_ROOT/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
export PYTHONPATH="$PROJECT_ROOT/source/isaac_go2_low_level_rl:$ISAACLAB_ROOT/source/isaaclab:$ISAACLAB_ROOT/source/isaaclab_tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export GIT_PYTHON_REFRESH=quiet

mkdir -p "$RUN_DIR" "$HOME" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" \
  "$XDG_DATA_HOME" "$TMPDIR" "$MPLCONFIGDIR"
cd "$FREEZE_DIR"
sha256sum -c SHA256SUMS
cd "$PROJECT_ROOT"

"$PYTHON_BIN" "$EXP_DIR/discover_r850_bodies.py" \
  --task Isaac-Go2-Soccer-Milestone2-v0 \
  --device cuda:0 \
  --headless \
  --output "$RUN_DIR/body_names.json"

if [[ ! -s "$RUN_DIR/body_names.json" ]]; then
  echo "R850_FATAL body discovery did not produce body_names.json" >&2
  exit 41
fi

"$PYTHON_BIN" "$EXP_DIR/patch_r850_source.py" \
  --source "$FREEZE_DIR/source_r841_grace6.py" \
  --output "$RUN_DIR/probe_r850_real_contact.py"

export R850_REAL_CONTACT_ONLY=1
export R850_FREEZE_DIR="$FREEZE_DIR"
export R850_OUTPUT="$RUN_DIR"
export R850_BODY_NAMES_JSON="$RUN_DIR/body_names.json"
export R850_EPISODES_PER_CASE="${R850_EPISODES_PER_CASE:-128}"
export V4K1_SETTLE_MAX_STEPS=180
export V4K1_BALL_ARM_STEPS=4

"$PYTHON_BIN" "$RUN_DIR/probe_r850_real_contact.py" \
  --task Isaac-Go2-Soccer-Milestone2-v0 \
  --num_envs "${R850_NUM_ENVS:-128}" \
  --device cuda:0 \
  --headless \
  --max_iterations 1 \
  --run_name r850_real_contact \
  --low_level_checkpoint "$FREEZE_DIR/artifacts/r821_lift_base.pt" \
  --log_dir "$RUN_DIR"

for required in body_names.json probe_r850_real_contact.py \
  r850_real_contact_summary.json r850_episode_rows.jsonl; do
  if [[ ! -s "$RUN_DIR/$required" ]]; then
    echo "R850_FATAL missing required artifact: $RUN_DIR/$required" >&2
    exit 42
  fi
done

printf '%s\n' "$RUN_DIR" > "$EXP_DIR/latest_r850.txt"
sha256sum "$RUN_DIR"/body_names.json "$RUN_DIR"/probe_r850_real_contact.py \
  "$RUN_DIR"/r850_real_contact_summary.json "$RUN_DIR"/r850_episode_rows.jsonl \
  > "$RUN_DIR/SHA256SUMS"
