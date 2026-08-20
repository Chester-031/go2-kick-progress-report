#!/usr/bin/env bash
#SBATCH --account=torch_pr_1066_general
#SBATCH --partition=l40s_public
#SBATCH --job-name=r876-high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-r876-%j.out

set -Eeuo pipefail

PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
EXP="$PROJECT/experiments/r8_76_v4d_curriculum_high_repair"
ROOT="/scratch/lj2634/runs/go2_soccer_r8_76_v4d_curriculum_high_repair"
PY="${ISAAC_PYTHON:-/scratch/lj2634/envs/isaaclab232_v2/bin/python}"
PARENT="/scratch/lj2634/runs/go2_soccer_v4d_joint/v4d_staged_joint_20260803_223518/full"
HIGH="$PARENT/logs/high_model_final.pt"
LOW="$PARENT/logs/low_model_final.pt"
MAX_ITERS="${R876_MAX_ITERATIONS:-600}"
NUM_ENVS="${R876_NUM_ENVS:-512}"

EXPECTED_HIGH="1a05964f6330b532fe944d24028c67b880664eee3218a984cf17db6bf8010397"
EXPECTED_LOW="52acb55160f4d4db0ac5c3da41aadbc5383a90555c64ab41851ba7383989996a"

cd "$PROJECT"
for FILE in "$EXP/train_r876.py" "$HIGH" "$LOW"; do
  [[ -s "$FILE" ]] || {
    echo "R876_FATAL=missing:$FILE" >&2
    exit 21
  }
done

verify_sha() {
  local file="$1" expected="$2" actual
  actual="$(sha256sum "$file" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || {
    echo "R876_FATAL=sha_mismatch:$file expected=$expected actual=$actual" >&2
    exit 22
  }
}
verify_sha "$HIGH" "$EXPECTED_HIGH"
verify_sha "$LOW" "$EXPECTED_LOW"

[[ "$MAX_ITERS" == "600" ]] || {
  echo "R876_FATAL=max_iterations_must_remain_600" >&2
  exit 23
}

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="$ROOT/curriculum_${SLURM_JOB_ID:-manual}_${STAMP}"
mkdir -p "$RUN/source" "$RUN/logs" "$RUN/checkpoints" \
  "$RUN/runtime"/{home,cache,config,data,tmp,mpl}
printf '%s\n' "$RUN" > "$ROOT/latest_r876.txt"
install -m 0644 "$EXP/train_r876.py" "$RUN/source/train_r876.py"
"$PY" -m py_compile "$RUN/source/train_r876.py"

cat > "$RUN/contract.txt" <<EOF
version=r8_76_v4d_curriculum_high_repair
training=true
training_history=curriculum_assisted
final_evaluation_distribution=stationary_full_support_no_curriculum
parent_high=$HIGH
parent_high_sha256=$EXPECTED_HIGH
parent_low=$LOW
parent_low_sha256=$EXPECTED_LOW
high_trainable=true
low_trainable=false
low_exact_checkpoint_load=true
reward_contract=goal=1,own_goal=-1,other=0
dense_high_reward=false
iterations=$MAX_ITERS
num_envs=$NUM_ENVS
phase_easy=0:75
phase_intermediate=75:200
phase_hard_mix=200:450
phase_full_support=450:600
video=false
EOF

export PYTHONPATH="$PROJECT/source/isaac_go2_low_level_rl:$PROJECT/source:${PYTHONPATH:-}"
export HOME="$RUN/runtime/home"
export XDG_CACHE_HOME="$RUN/runtime/cache"
export XDG_CONFIG_HOME="$RUN/runtime/config"
export XDG_DATA_HOME="$RUN/runtime/data"
export TMPDIR="$RUN/runtime/tmp"
export MPLCONFIGDIR="$RUN/runtime/mpl"
export PYTHONUNBUFFERED=1
export QT_QPA_PLATFORM=offscreen

export R873_HEADLESS_STRIP=1
export R876_CURRICULUM_HIGH_REPAIR=1
export R876_FREEZE_LOW=1
export R874B_FREEZE_LOW=1
export V4D_HIGH_CHECKPOINT="$HIGH"
unset R874B_FRESH_NO_CURRICULUM R874_FRESH_NO_CURRICULUM

export R876_EASY_END=75
export R876_INTERMEDIATE_END=200
export R876_HARD_END=450
export V4D_STAGE1_ITERATIONS=75
export V4D_STAGE2_ITERATIONS=375
export V4D_STAGE3_ITERATIONS=150
export V4D_HIGH_LR_STAGE1=0.000010
export V4D_HIGH_LR_STAGE2=0.0000075
export V4D_HIGH_LR_STAGE3=0.000003
export V4D_LOW_LR_STAGE2=0.0
export V4D_CHECKPOINT_INTERVAL=50

{
  echo "RUN=$RUN"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader || true
} > "$RUN/resource_start.txt"

set +e
setsid timeout --foreground --kill-after=30s 165m \
  "$PY" "$RUN/source/train_r876.py" \
    --headless \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERS" \
    --seed 87601 \
    --low_level_checkpoint "$LOW" \
    --log_dir "$RUN/logs" \
    --run_name r876_curriculum_high_repair \
  > "$RUN/worker.log" 2>&1
RC=$?
set -e
printf 'PROCESS_RC=%s\n' "$RC" >> "$RUN/worker.log"

for FILE in "$RUN"/logs/high_model_final.pt "$RUN"/logs/low_model_final.pt \
  "$RUN"/logs/v4d_joint_summary.json "$RUN"/logs/v4d_joint_contract.json; do
  [[ -s "$FILE" ]] && cp -a "$FILE" "$RUN/checkpoints/"
done

VALIDATION_RC=0
if [[ "$RC" -eq 0 && -s "$RUN/logs/v4d_joint_summary.json" ]]; then
  "$PY" - "$RUN/logs/v4d_joint_summary.json" <<'PY' \
    > "$RUN/post_training_validation.txt" 2>&1 || VALIDATION_RC=$?
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
records = data.get("records", [])
low_delta = data.get("low_policy", {}).get("parameter_delta", {})
high_delta = data.get("high_policy", {}).get("parameter_delta", {})
curriculum = data.get("curriculum", {})
support = set(data.get("high_reward_support", []))

assert len(records) == 600, len(records)
assert records[0]["curriculum_phase"] == "easy"
assert records[75]["curriculum_phase"] == "intermediate"
assert records[200]["curriculum_phase"] == "hard_mix"
assert records[450]["curriculum_phase"] == "full_support"
assert records[-1]["curriculum_phase"] == "full_support"
assert float(low_delta.get("max_abs", -1.0)) == 0.0, low_delta
assert float(high_delta.get("max_abs", 0.0)) > 0.0, high_delta
assert support.issubset({-1, 0, 1}), support
assert curriculum.get("training_history") == "curriculum_assisted"

phase_ranges = {
    "easy": records[0:75],
    "intermediate": records[75:200],
    "hard_mix": records[200:450],
    "full_support": records[450:600],
}
phase_nonzero = {
    phase: sum(
        float(row.get("high_nonzero_reward_fraction") or 0.0)
        for row in rows
    )
    for phase, rows in phase_ranges.items()
}
phase_goal = {
    phase: sum(float(row.get("goal_fraction") or 0.0) for row in rows)
    for phase, rows in phase_ranges.items()
}
for phase, total in phase_nonzero.items():
    assert total > 0.0, (
        f"R876 phase produced no nonzero high-level reward: {phase}"
    )
for phase, total in phase_goal.items():
    assert total > 0.0, f"R876 phase produced no goal event: {phase}"

print("R876_POST_TRAINING_VALIDATION_OK")
print("HIGH_DELTA=" + json.dumps(high_delta, sort_keys=True))
print("LOW_DELTA=" + json.dumps(low_delta, sort_keys=True))
print("PHASE_NONZERO_REWARD_SUM=" + json.dumps(phase_nonzero, sort_keys=True))
print("PHASE_GOAL_FRACTION_SUM=" + json.dumps(phase_goal, sort_keys=True))
print("FINAL_20_GOAL_RATES=" + json.dumps([r.get("goal_rate") for r in records[-20:]]))
print("FINAL_20_WALL_PROXY=not_measured_training_requires_r875_eval")
PY
else
  VALIDATION_RC=1
fi

if [[ "$RC" -eq 0 && "$VALIDATION_RC" -eq 0 \
      && -s "$RUN/logs/high_model_final.pt" \
      && -s "$RUN/logs/low_model_final.pt" ]]; then
  touch "$RUN/R876_READY_FOR_FULL_SUPPORT_EVALUATION"
fi

(cd "$RUN" && find . -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum > SHA256SUMS)

echo "R876_PROCESS_RC=$RC"
echo "R876_VALIDATION_RC=$VALIDATION_RC"
echo "R876_RUN=$RUN"
if [[ -f "$RUN/R876_READY_FOR_FULL_SUPPORT_EVALUATION" ]]; then
  echo "VERDICT=R876_READY_FOR_METRIC_EVALUATION"
else
  echo "VERDICT=R876_TRAINING_OR_VALIDATION_FAILED"
fi
exit "$(( RC != 0 ? RC : VALIDATION_RC ))"
