#!/usr/bin/env bash
#SBATCH --job-name=r852-scan
#SBATCH --account=torch_pr_1066_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-r852-%j.out

set -Eeuo pipefail
PROJECT="${SLURM_SUBMIT_DIR:?submit from project root}"
EXP="$PROJECT/experiments/r8_52_gain_recovery_scan"
RUN_ROOT="/scratch/lj2634/runs/go2_soccer_r8_52_gain_recovery_scan"
RUN_DIR="$RUN_ROOT/r852_scan_${SLURM_JOB_ID}"
RUNTIME="/scratch/lj2634/job_runtime/${SLURM_JOB_ID}_r852"
ISAACLAB_ROOT="/scratch/lj2634/IsaacLab_232_src_37ddf62"
ENV_ROOT="/scratch/lj2634/envs/isaaclab232_v2"
GLU_ROOT="/scratch/lj2634/vendor/libglu-9.0.0"
PY="$ENV_ROOT/bin/python"
export HOME="$RUNTIME/home" XDG_CACHE_HOME="$RUNTIME/cache" XDG_CONFIG_HOME="$RUNTIME/config"
export XDG_DATA_HOME="$RUNTIME/data" TMPDIR="$RUNTIME/tmp" MPLCONFIGDIR="$RUNTIME/mpl"
export LD_LIBRARY_PATH="$GLU_ROOT/lib:$ENV_ROOT/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$ENV_ROOT/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
export PYTHONPATH="$PROJECT/source/isaac_go2_low_level_rl:$ISAACLAB_ROOT/source/isaaclab:$ISAACLAB_ROOT/source/isaaclab_tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 GIT_PYTHON_REFRESH=quiet
mkdir -p "$RUN_DIR/output" "$HOME" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$TMPDIR" "$MPLCONFIGDIR" "$RUN_ROOT"
printf '%s\n' "$RUN_DIR" > "$RUN_ROOT/latest_r852.txt"; printf '%s\n' "$RUN_DIR" > "$EXP/latest_r852.txt"

R851_RUN="$(tr -d '\r\n' < /scratch/lj2634/runs/go2_soccer_r8_51_kick64_true_contact/latest_r851.txt)"
R850_RUN="$(tr -d '\r\n' < "$PROJECT/experiments/r8_50_real_contact_truth/latest_r850.txt")"
R849_RUN="$(tr -d '\r\n' < /scratch/lj2634/runs/go2_soccer_r8_49_recovery_finetune/latest_r849.txt)"
SOURCE="$R851_RUN/train_r851_kick64.py"
KICK="$R851_RUN/output/kick64_final.pt"
SUMMARY851="$R851_RUN/output/r851_summary.json"
BODY_JSON="$R850_RUN/body_names.json"
RECOVERY="$R849_RUN/output/recovery_vector_finetuned.pt"
FREEZE="/scratch/lj2634/runs/go2_soccer_kick_freeze/ballfree_r841_grace6_20260810_185244"
for f in "$SOURCE" "$KICK" "$SUMMARY851" "$BODY_JSON" "$RECOVERY"; do test -s "$f" || { echo "R852_FATAL missing parent $f" >&2; exit 31; }; done

"$PY" "$EXP/patch_r852_source.py" --source "$SOURCE" --output "$RUN_DIR/probe_r852_scan.py"
export R852_GAIN_RECOVERY_ONLY=1 R852_FREEZE_DIR="$FREEZE" R852_KICK="$KICK" R852_RECOVERY="$RECOVERY"
export R852_R851_SUMMARY="$SUMMARY851" R852_BODY_JSON="$BODY_JSON" R852_OUTPUT="$RUN_DIR/output"
export R851_BODY_JSON="$BODY_JSON"
export R852_EPISODES_PER_CASE="${R852_EPISODES_PER_CASE:-128}" V4K1_SETTLE_MAX_STEPS=180 V4K1_BALL_ARM_STEPS=4
{
 printf 'job_id=%s\nsource=%s\nsource_sha256=%s\nkick_parent=%s\nr851_summary=%s\nrecovery_parent=%s\n' "$SLURM_JOB_ID" "$SOURCE" "$(sha256sum "$SOURCE"|awk '{print $1}')" "$KICK" "$SUMMARY851" "$RECOVERY"
 printf 'gains=0.0,0.5,1.0,1.5,2.0\ndelays=0,2,4,6\nepisodes_per_case=%s\ntraining=false\n' "$R852_EPISODES_PER_CASE"
} > "$RUN_DIR/contract.txt"
cd "$PROJECT"
"$PY" "$RUN_DIR/probe_r852_scan.py" --task Isaac-Go2-Soccer-Milestone2-v0 --num_envs "${R852_NUM_ENVS:-128}" --device cuda:0 --headless --max_iterations 1 --run_name r852_scan --low_level_checkpoint "$FREEZE/artifacts/r821_lift_base.pt" --log_dir "$RUN_DIR"
for f in r852_summary.json r852_episode_rows.jsonl; do test -s "$RUN_DIR/output/$f" || { echo "R852_FATAL missing output $f" >&2; exit 41; }; done
sha256sum "$RUN_DIR/probe_r852_scan.py" "$RUN_DIR/contract.txt" "$RUN_DIR/output/r852_summary.json" "$RUN_DIR/output/r852_episode_rows.jsonl" > "$RUN_DIR/SHA256SUMS"
echo "R852_WORKER_OK run=$RUN_DIR"

