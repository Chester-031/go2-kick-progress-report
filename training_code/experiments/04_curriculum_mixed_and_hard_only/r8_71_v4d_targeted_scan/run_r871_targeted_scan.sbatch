#!/usr/bin/env bash
#SBATCH --account=torch_pr_1066_general
#SBATCH --partition=l40s_public
#SBATCH --job-name=r871-target
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-r871-%j.out
set -Eeuo pipefail

PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
R863_ROOT="/scratch/lj2634/runs/go2_soccer_r8_63_full_support_v4d"
ROOT="/scratch/lj2634/runs/go2_soccer_r8_71_v4d_targeted_scan"
PY="${ISAAC_PYTHON:-/scratch/lj2634/envs/isaaclab232_v2/bin/python}"
EVAL_STEPS="${R871_EVAL_STEPS:-600}"
SCENARIOS=(
  center_front center_back
  own_penalty_front own_penalty_back
  enemy_goal_diagonal_left enemy_goal_diagonal_right
  own_penalty_back_diagonal
  near_left_wall_avoidable near_right_wall_avoidable
  near_left_wall_back near_right_wall_back
  corner_candidate_left corner_candidate_right
)

cd "$PROJECT"
export PYTHONPATH="$PROJECT/source/isaac_go2_low_level_rl:$PROJECT/source:${PYTHONPATH:-}"
BASE="$(tr -d '\r\n' < "$R863_ROOT/latest_r863_candidate.txt")"
SRC="$BASE/source/train_v4d_full_support.py"
LOW="/scratch/lj2634/runs/go2_soccer_v4d_joint/v4d_staged_joint_20260803_223518/full/logs/low_model_final.pt"
HIGH="/scratch/lj2634/runs/go2_soccer_high_level/v3_zero_sum_from_v3a_20260730_233921/full/logs/model_final.pt"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="$ROOT/targeted_${SLURM_JOB_ID:-manual}_${STAMP}"
mkdir -p "$RUN/source" "$RUN/summaries" "$RUN/trajectories" "$RUN/logs" "$RUN/runtime"/{home,cache,config,data,tmp,mpl}
export HOME="$RUN/runtime/home" XDG_CACHE_HOME="$RUN/runtime/cache" XDG_CONFIG_HOME="$RUN/runtime/config" XDG_DATA_HOME="$RUN/runtime/data" TMPDIR="$RUN/runtime/tmp" MPLCONFIGDIR="$RUN/runtime/mpl" V4D_HIGH_CHECKPOINT="$HIGH" R865_METRIC_ONLY=1 PYTHONUNBUFFERED=1

cp -a "$SRC" "$RUN/source/source.py"
cp -a "$PROJECT/experiments/r8_71_v4d_targeted_scan/patch_metric_only.py" "$RUN/source/"
cp -a "$PROJECT/experiments/r8_71_v4d_targeted_scan/patch_headless.py" "$RUN/source/"
cp -a "$PROJECT/experiments/r8_71_v4d_targeted_scan/patch_scenarios.py" "$RUN/source/"
cp -a "$PROJECT/experiments/r8_71_v4d_targeted_scan/patch_hard_diag.py" "$RUN/source/"
"$PY" "$RUN/source/patch_metric_only.py" --source "$RUN/source/source.py" --out "$RUN/source/metric.py"
"$PY" "$RUN/source/patch_headless.py" --source "$RUN/source/metric.py" --out "$RUN/source/headless.py"
"$PY" "$RUN/source/patch_scenarios.py" --source "$RUN/source/headless.py" --out "$RUN/source/scenario.py"
"$PY" "$RUN/source/patch_hard_diag.py" --source "$RUN/source/scenario.py" --out "$RUN/source/targeted.py"

printf 'version=r8_71_v4d_targeted_scan\ntraining=false\nvideo=false\ncheckpoint_frozen=true\nprogressive_curriculum=false\neval_steps=%s\nscenario_count=%s\n' "$EVAL_STEPS" "${#SCENARIOS[@]}" > "$RUN/contract.txt"

overall_rc=0
index=0
for scenario in "${SCENARIOS[@]}"; do
    index=$((index + 1))
    slug="${index}_$scenario"
    seed=$((87100 + index))
    seed_log="$RUN/logs/${slug}.log"
    spec="$RUN/summaries/${slug}.scenario.json"
    printf '{"scenario":"%s","seed":%d}\n' "$scenario" "$seed" > "$spec"
    set +e
    R871_SCENARIO="$scenario" setsid timeout --foreground --kill-after=15s 180s "$PY" "$RUN/source/targeted.py" \
        --headless --num_envs 1 --seed "$seed" \
        --log_dir "$RUN/logs/${slug}" --run_name "r871_${slug}" \
        --eval_steps "$EVAL_STEPS" --eval_fps 30 \
        --eval_summary "$RUN/summaries/${slug}.json" \
        --trajectory_csv "$RUN/trajectories/${slug}.csv" \
        --low_level_checkpoint "$LOW" --trained_low_checkpoint "$LOW" \
        > "$seed_log" 2>&1
    rc=$?
    set -e
    printf 'SCENARIO=%s PROCESS_RC=%s\n' "$scenario" "$rc" >> "$seed_log"
    if [[ "$rc" != 0 ]]; then overall_rc=1; fi
done

"$PY" - "$RUN" "$EVAL_STEPS" <<'PY'
import csv, json, sys
from pathlib import Path

run = Path(sys.argv[1]); steps = int(sys.argv[2])
rows = []
for summary_path in sorted(run.glob("summaries/*.json")):
    if summary_path.name.endswith(".scenario.json"):
        continue
    item = json.loads(summary_path.read_text(encoding="utf-8"))
    scenario = summary_path.stem.split("_", 1)[1]
    spec_path = summary_path.with_name(summary_path.stem + ".scenario.json")
    spec = json.loads(spec_path.read_text(encoding="utf-8")) if spec_path.exists() else {}
    traj_path = run / "trajectories" / (summary_path.stem + ".csv")
    traj = list(csv.DictReader(traj_path.open(encoding="utf-8"))) if traj_path.exists() else []
    initial = traj[0] if traj else {}
    final = traj[-1] if traj else {}
    wall_case = scenario.startswith("near_") or scenario.startswith("corner_")
    wall_sign = 1 if "left" in scenario else (-1 if "right" in scenario else 0)
    initial_ball_y = float(initial.get("ball_y", "nan")) if initial else float("nan")
    final_ball_y = float(final.get("ball_y", "nan")) if final else float("nan")
    inward = wall_sign * (initial_ball_y - final_ball_y) > 0.12 if wall_sign else False
    goal = int(item.get("goal_count", 0)) > 0
    own_goal = int(item.get("own_goal_count", 0)) > 0
    stall = int(item.get("push_stall_count", 0)) > 0 and not goal
    rows.append({
        "scenario": scenario, "seed": spec.get("seed"), "goal": goal,
        "own_goal": own_goal, "done_count": item.get("done_count", 0),
        "push_stall_count": item.get("push_stall_count", 0),
        "wall_case": wall_case, "wall_escape_detected": bool(inward),
        "wall_stall": bool(wall_case and stall),
        "avoidable_wall_stall": bool(wall_case and stall and "avoidable" in scenario and not inward),
        "summary": str(summary_path), "trajectory": str(traj_path),
    })
aggregate = {
    "version": "r8_71_v4d_targeted_scan", "training": False, "video": False,
    "progressive_curriculum": False, "checkpoint_frozen": True,
    "eval_steps": steps, "scenario_count": len(rows), "scenarios": rows,
    "goal_count": sum(int(x["goal"]) for x in rows),
    "own_goal_count": sum(int(x["own_goal"]) for x in rows),
    "wall_stall_count": sum(int(x["wall_stall"]) for x in rows),
    "avoidable_wall_stall_count": sum(int(x["avoidable_wall_stall"]) for x in rows),
}
(run / "targeted_aggregate.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({k: aggregate[k] for k in ("scenario_count", "goal_count", "own_goal_count", "wall_stall_count", "avoidable_wall_stall_count")}, sort_keys=True))
PY

printf '%s\n' "$RUN" > "$ROOT/latest_r871.txt"
(cd "$RUN" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
echo "R871_TARGETED_SCAN_DONE overall_rc=$overall_rc"
echo "RUN=$RUN"
echo "VERDICT=R871_TARGETED_METRICS_READY"
