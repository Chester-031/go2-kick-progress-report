#!/usr/bin/env bash
#SBATCH --account=torch_pr_1066_general
#SBATCH --partition=l40s_public
#SBATCH --job-name=r866-metric
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-r866-%j.out
set -Eeuo pipefail
PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"; R863_ROOT="/scratch/lj2634/runs/go2_soccer_r8_63_full_support_v4d"; ROOT="/scratch/lj2634/runs/go2_soccer_r8_66_v4d_metric_smoke"; PY="${ISAAC_PYTHON:-/scratch/lj2634/envs/isaaclab232_v2/bin/python}"
cd "$PROJECT"; export PYTHONPATH="$PROJECT/source/isaac_go2_low_level_rl:$PROJECT/source:${PYTHONPATH:-}"
BASE="$(tr -d '\r\n' < "$R863_ROOT/latest_r863_candidate.txt")"; SRC="$BASE/source/train_v4d_full_support.py"; HIGH="/scratch/lj2634/runs/go2_soccer_high_level/v3_zero_sum_from_v3a_20260730_233921/full/logs/model_final.pt"; LOW="/scratch/lj2634/runs/go2_soccer_v4d_joint/v4d_staged_joint_20260803_223518/full/logs/low_model_final.pt"
STAMP="$(date +%Y%m%d_%H%M%S)"; RUN="$ROOT/r866_metric_${SLURM_JOB_ID:-manual}_${STAMP}"; mkdir -p "$RUN/source" "$RUN/summaries" "$RUN/trajectories" "$RUN/runtime"/{home,cache,config,data,tmp,mpl}; export HOME="$RUN/runtime/home" XDG_CACHE_HOME="$RUN/runtime/cache" XDG_CONFIG_HOME="$RUN/runtime/config" XDG_DATA_HOME="$RUN/runtime/data" TMPDIR="$RUN/runtime/tmp" MPLCONFIGDIR="$RUN/runtime/mpl" V4D_HIGH_CHECKPOINT="$HIGH" R865_METRIC_ONLY=1
cp -a "$SRC" "$RUN/source/source_v4d_full_support.py"; cp -a "$PROJECT/experiments/r8_66_v4d_metric_smoke/patch_metric_only.py" "$RUN/source/"; "$PY" "$RUN/source/patch_metric_only.py" --source "$RUN/source/source_v4d_full_support.py" --out "$RUN/source/train_v4d_metric_only.py"
timeout 600 "$PY" "$RUN/source/train_v4d_metric_only.py" --num_envs 1 --seed 86601 --log_dir "$RUN/logs" --run_name r866_metric_smoke --eval_steps 60 --eval_fps 30 --eval_summary "$RUN/summaries/metric.json" --trajectory_csv "$RUN/trajectories/metric.csv" --low_level_checkpoint "$LOW" --trained_low_checkpoint "$LOW" > "$RUN/worker.log" 2>&1
printf '%s\n' "$RUN" > "$ROOT/latest_r866.txt"; printf 'version=r8_66_v4d_metric_only\ntraining=false\nvideo=false\neval_steps=60\n' > "$RUN/contract.txt"; (cd "$RUN" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS); echo R866_METRIC_SMOKE_OK; echo "RUN=$RUN"; echo VERDICT=R866_READY_FOR_CHECK
