#!/usr/bin/env bash
#SBATCH --account=torch_pr_1066_general
#SBATCH --partition=l40s_public
#SBATCH --job-name=r870-scan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-r870-%j.out
set -Eeuo pipefail

PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
R863_ROOT="/scratch/lj2634/runs/go2_soccer_r8_63_full_support_v4d"
ROOT="/scratch/lj2634/runs/go2_soccer_r8_70_v4d_nocurriculum_scan"
PY="${ISAAC_PYTHON:-/scratch/lj2634/envs/isaaclab232_v2/bin/python}"
SEEDS=(87001 87002 87003 87004 87005 87006 87007 87008)
EVAL_STEPS="${R870_EVAL_STEPS:-600}"

cd "$PROJECT"
export PYTHONPATH="$PROJECT/source/isaac_go2_low_level_rl:$PROJECT/source:${PYTHONPATH:-}"
BASE="$(tr -d '\r\n' < "$R863_ROOT/latest_r863_candidate.txt")"
SRC="$BASE/source/train_v4d_full_support.py"
LOW="/scratch/lj2634/runs/go2_soccer_v4d_joint/v4d_staged_joint_20260803_223518/full/logs/low_model_final.pt"
HIGH="/scratch/lj2634/runs/go2_soccer_high_level/v3_zero_sum_from_v3a_20260730_233921/full/logs/model_final.pt"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="$ROOT/scan_${SLURM_JOB_ID:-manual}_${STAMP}"
mkdir -p "$RUN/source" "$RUN/summaries" "$RUN/trajectories" "$RUN/logs" "$RUN/runtime"/{home,cache,config,data,tmp,mpl}
export HOME="$RUN/runtime/home" XDG_CACHE_HOME="$RUN/runtime/cache" XDG_CONFIG_HOME="$RUN/runtime/config" XDG_DATA_HOME="$RUN/runtime/data" TMPDIR="$RUN/runtime/tmp" MPLCONFIGDIR="$RUN/runtime/mpl" V4D_HIGH_CHECKPOINT="$HIGH" R865_METRIC_ONLY=1 PYTHONUNBUFFERED=1

cp -a "$SRC" "$RUN/source/source.py"
cp -a "$PROJECT/experiments/r8_70_v4d_nocurriculum_scan/patch_metric_only.py" "$RUN/source/"
cp -a "$PROJECT/experiments/r8_70_v4d_nocurriculum_scan/patch_headless.py" "$RUN/source/"
cp -a "$PROJECT/experiments/r8_70_v4d_nocurriculum_scan/patch_hard_diag.py" "$RUN/source/"
"$PY" "$RUN/source/patch_metric_only.py" --source "$RUN/source/source.py" --out "$RUN/source/metric.py"
"$PY" "$RUN/source/patch_headless.py" --source "$RUN/source/metric.py" --out "$RUN/source/headless.py"
"$PY" "$RUN/source/patch_hard_diag.py" --source "$RUN/source/headless.py" --out "$RUN/source/scan.py"

printf 'version=r8_70_v4d_nocurriculum_scan\ntraining=false\nvideo=false\ncheckpoint_frozen=true\nprogressive_curriculum=false\neval_steps=%s\nseeds=%s\n' "$EVAL_STEPS" "${SEEDS[*]}" > "$RUN/contract.txt"

overall_rc=0
for seed in "${SEEDS[@]}"; do
    seed_log="$RUN/logs/seed_${seed}.log"
    set +e
    setsid timeout --foreground --kill-after=15s 180s "$PY" "$RUN/source/scan.py" \
        --headless --num_envs 1 --seed "$seed" \
        --log_dir "$RUN/logs/seed_${seed}" --run_name "r870_seed_${seed}" \
        --eval_steps "$EVAL_STEPS" --eval_fps 30 \
        --eval_summary "$RUN/summaries/seed_${seed}.json" \
        --trajectory_csv "$RUN/trajectories/seed_${seed}.csv" \
        --low_level_checkpoint "$LOW" \
        --trained_low_checkpoint "$LOW" \
        > "$seed_log" 2>&1
    rc=$?
    set -e
    printf 'SEED=%s PROCESS_RC=%s\n' "$seed" "$rc" >> "$seed_log"
    if [[ "$rc" != 0 ]]; then
        overall_rc=1
    fi
done

"$PY" - "$RUN" "$EVAL_STEPS" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
steps = int(sys.argv[2])
summaries = []
for path in sorted((run / "summaries").glob("seed_*.json")):
    item = json.loads(path.read_text(encoding="utf-8"))
    item["summary_file"] = str(path)
    summaries.append(item)

goal_seeds = sum(int(x.get("goal_count", 0) > 0) for x in summaries)
own_goal_seeds = sum(int(x.get("own_goal_count", 0) > 0) for x in summaries)
aggregate = {
    "version": "r8_70_v4d_nocurriculum_scan",
    "training": False,
    "video": False,
    "progressive_curriculum": False,
    "checkpoint_frozen": True,
    "eval_steps": steps,
    "summary_count": len(summaries),
    "goal_seed_count": goal_seeds,
    "own_goal_seed_count": own_goal_seeds,
    "goal_seed_rate": goal_seeds / len(summaries) if summaries else 0.0,
    "own_goal_seed_rate": own_goal_seeds / len(summaries) if summaries else 0.0,
    "total_goal_count": sum(int(x.get("goal_count", 0)) for x in summaries),
    "total_own_goal_count": sum(int(x.get("own_goal_count", 0)) for x in summaries),
    "total_done_count": sum(int(x.get("done_count", 0)) for x in summaries),
    "total_push_stall_count": sum(int(x.get("push_stall_count", 0)) for x in summaries),
    "skill_counts": {
        name: sum(int(x.get("skill_counts", {}).get(name, 0)) for x in summaries)
        for name in sorted({name for x in summaries for name in x.get("skill_counts", {})})
    },
    "seeds": summaries,
}
(run / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({k: aggregate[k] for k in ("summary_count", "goal_seed_count", "own_goal_seed_count", "total_goal_count", "total_own_goal_count", "total_done_count", "total_push_stall_count")}, sort_keys=True))
PY

printf '%s\n' "$RUN" > "$ROOT/latest_r870.txt"
(cd "$RUN" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
echo "R870_NOCURRICULUM_SCAN_DONE overall_rc=$overall_rc"
echo "RUN=$RUN"
echo "VERDICT=R870_METRICS_READY"
