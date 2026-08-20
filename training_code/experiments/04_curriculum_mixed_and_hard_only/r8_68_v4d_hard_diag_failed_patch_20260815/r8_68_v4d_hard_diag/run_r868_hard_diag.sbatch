#!/usr/bin/env bash
#SBATCH --account=torch_pr_1066_general
#SBATCH --partition=l40s_public
#SBATCH --job-name=r868-diag
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-r868-%j.out
set -Eeuo pipefail
PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
R863_ROOT="/scratch/lj2634/runs/go2_soccer_r8_63_full_support_v4d"
ROOT="/scratch/lj2634/runs/go2_soccer_r8_68_v4d_hard_diag"
PY="${ISAAC_PYTHON:-/scratch/lj2634/envs/isaaclab232_v2/bin/python}"
cd "$PROJECT"
export PYTHONPATH="$PROJECT/source/isaac_go2_low_level_rl:$PROJECT/source:${PYTHONPATH:-}"
BASE="$(tr -d '\r\n' < "$R863_ROOT/latest_r863_candidate.txt")"
SRC="$BASE/source/train_v4d_full_support.py"
LOW="/scratch/lj2634/runs/go2_soccer_v4d_joint/v4d_staged_joint_20260803_223518/full/logs/low_model_final.pt"
HIGH="/scratch/lj2634/runs/go2_soccer_high_level/v3_zero_sum_from_v3a_20260730_233921/full/logs/model_final.pt"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="$ROOT/diag_${SLURM_JOB_ID:-manual}_${STAMP}"
mkdir -p "$RUN/source" "$RUN/summaries" "$RUN/trajectories" "$RUN/logs" "$RUN/runtime"/{home,cache,config,data,tmp,mpl}
export HOME="$RUN/runtime/home" XDG_CACHE_HOME="$RUN/runtime/cache" XDG_CONFIG_HOME="$RUN/runtime/config" XDG_DATA_HOME="$RUN/runtime/data" TMPDIR="$RUN/runtime/tmp" MPLCONFIGDIR="$RUN/runtime/mpl" V4D_HIGH_CHECKPOINT="$HIGH" R865_METRIC_ONLY=1 PYTHONUNBUFFERED=1
cp -a "$SRC" "$RUN/source/source.py"
cp -a "$PROJECT/experiments/r8_68_v4d_hard_diag/patch_metric_only.py" "$RUN/source/"
cp -a "$PROJECT/experiments/r8_68_v4d_hard_diag/patch_hard_diag.py" "$RUN/source/"
"$PY" "$RUN/source/patch_metric_only.py" --source "$RUN/source/source.py" --out "$RUN/source/metric.py"
"$PY" "$RUN/source/patch_hard_diag.py" --source "$RUN/source/metric.py" --out "$RUN/source/debug.py"
printf 'version=r8_68_v4d_hard_diag\ntraining=false\nvideo=false\ntimeout_seconds=120\ncheckpoint_frozen=true\n' > "$RUN/contract.txt"
set +e
setsid timeout --foreground --kill-after=15s 120s "$PY" "$RUN/source/debug.py" \
  --num_envs 1 --seed 86801 --log_dir "$RUN/logs" --run_name r868_diag \
  --eval_steps 1 --eval_fps 30 \
  --eval_summary "$RUN/summaries/debug.json" \
  --trajectory_csv "$RUN/trajectories/debug.csv" \
  --low_level_checkpoint "$LOW" --trained_low_checkpoint "$LOW" \
  > "$RUN/worker.log" 2>&1
RC=$?
set -e
printf 'PROCESS_RC=%s\n' "$RC" >> "$RUN/worker.log"
printf '%s\n' "$RUN" > "$ROOT/latest_r868.txt"
(cd "$RUN" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
echo "R868_HARD_DIAG_DONE rc=$RC"
echo "RUN=$RUN"
if [[ "$RC" == 124 || "$RC" == 137 ]]; then echo "VERDICT=R868_TIMEOUT_WITH_STACK"; else echo "VERDICT=R868_DIAG_COMPLETE"; fi
