#!/usr/bin/env bash
#SBATCH --account=torch_pr_1066_general
#SBATCH --partition=l40s_public
#SBATCH --job-name=r865-behavior
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-r865-%j.out
set -Eeuo pipefail
PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
R863_ROOT="/scratch/lj2634/runs/go2_soccer_r8_63_full_support_v4d"
RUN_ROOT="/scratch/lj2634/runs/go2_soccer_r8_65_v4d_behavior"
PYTHON_BIN="${ISAAC_PYTHON:-/scratch/lj2634/envs/isaaclab232_v2/bin/python}"
R863_RUN="$(tr -d '\r\n' < "$R863_ROOT/latest_r863_candidate.txt")"
SOURCE="$R863_RUN/source/train_v4d_full_support.py"
HIGH="/scratch/lj2634/runs/go2_soccer_high_level/v3_zero_sum_from_v3a_20260730_233921/full/logs/model_final.pt"
LOW="/scratch/lj2634/runs/go2_soccer_v4d_joint/v4d_staged_joint_20260803_223518/full/logs/low_model_final.pt"
STAMP="$(date +%Y%m%d_%H%M%S)"; RUN="$RUN_ROOT/r865_behavior_${SLURM_JOB_ID:-manual}_${STAMP}"
RUNTIME="$RUN/runtime"; mkdir -p "$RUN/videos" "$RUN/summaries" "$RUNTIME"/{home,cache,config,data,tmp,mpl}
export HOME="$RUNTIME/home" XDG_CACHE_HOME="$RUNTIME/cache" XDG_CONFIG_HOME="$RUNTIME/config" XDG_DATA_HOME="$RUNTIME/data" TMPDIR="$RUNTIME/tmp" MPLCONFIGDIR="$RUNTIME/mpl" V4D_HIGH_CHECKPOINT="$HIGH"
[[ -s "$SOURCE" && -s "$HIGH" && -s "$LOW" ]] || { echo R865_FATAL=source_or_checkpoint_missing; exit 21; }
for seed in 86501 86502 86503 86504 86505 86506 86507 86508; do
  echo "[R865_CASE] seed=$seed"
  "$PYTHON_BIN" "$SOURCE" --num_envs 1 --seed "$seed" --eval_video "$RUN/videos/seed_${seed}.mp4" --eval_steps 900 --eval_fps 30 --eval_summary "$RUN/summaries/seed_${seed}.json" --low_level_checkpoint "$LOW" --trained_low_checkpoint "$LOW" >> "$RUN/worker.log" 2>&1
done
mkdir -p "$RUN_ROOT" "$PROJECT/experiments/r8_65_v4d_behavior_scan"
printf '%s\n' "$RUN" > "$RUN_ROOT/latest_r865.txt"
printf 'version=r8_65_v4d_behavior_scan\ntraining=false\ncheckpoint_frozen=true\nseeds=86501,86502,86503,86504,86505,86506,86507,86508\n' > "$RUN/contract.txt"
(cd "$RUN" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
echo R865_BEHAVIOR_SCAN_OK; echo "RUN=$RUN"; echo VERDICT=R865_DATA_READY_FOR_CHECK
