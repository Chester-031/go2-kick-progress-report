#!/usr/bin/env bash
#SBATCH --account=torch_pr_1066_general
#SBATCH --partition=l40s_public
#SBATCH --job-name=r878-low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-r878-%j.out

set -Eeuo pipefail

PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
EXP="$PROJECT/experiments/r8_78_v4d_sustained_low_repair"
R876_EXP="$PROJECT/experiments/r8_76_v4d_curriculum_high_repair"
R877_ROOT="/scratch/lj2634/runs/go2_soccer_r8_77_v4d_checkpoint_gate"
R877_RUN="$(tr -d '\r\n' < "$R877_ROOT/latest_r877.txt")"
ROOT="/scratch/lj2634/runs/go2_soccer_r8_78_v4d_sustained_low_repair"
PY="${ISAAC_PYTHON:-/scratch/lj2634/envs/isaaclab232_v2/bin/python}"
PARENT="/scratch/lj2634/runs/go2_soccer_v4d_joint/v4d_staged_joint_20260803_223518/full"
HIGH="$PARENT/logs/high_model_final.pt"
LOW="$PARENT/logs/low_model_final.pt"
TRACE="$R877_RUN/results/hard_mix_449/r875_trace.csv.gz"
MAX_ITERS="${R878_MAX_ITERATIONS:-300}"
NUM_ENVS="${R878_NUM_ENVS:-512}"

EXPECTED_HIGH="1a05964f6330b532fe944d24028c67b880664eee3218a984cf17db6bf8010397"
EXPECTED_LOW="52acb55160f4d4db0ac5c3da41aadbc5383a90555c64ab41851ba7383989996a"

cd "$PROJECT"
for FILE in "$R876_EXP/train_r876.py" "$EXP/patch_r878_low_repair.py" \
  "$EXP/analyze_r877_sustained_trace.py" "$HIGH" "$LOW" "$TRACE"; do
  [[ -s "$FILE" ]] || { echo "R878_FATAL=missing:$FILE" >&2; exit 21; }
done

[[ "$(sha256sum "$HIGH" | awk '{print $1}')" == "$EXPECTED_HIGH" ]] || \
  { echo "R878_FATAL=accepted_high_hash_mismatch" >&2; exit 22; }
[[ "$(sha256sum "$LOW" | awk '{print $1}')" == "$EXPECTED_LOW" ]] || \
  { echo "R878_FATAL=accepted_low_hash_mismatch" >&2; exit 22; }
[[ "$MAX_ITERS" == "300" ]] || \
  { echo "R878_FATAL=max_iterations_must_remain_300" >&2; exit 23; }

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="$ROOT/low_repair_${SLURM_JOB_ID:-manual}_${STAMP}"
mkdir -p "$RUN"/{source,logs,results,checkpoints,runtime/home,runtime/cache,runtime/config,runtime/data,runtime/tmp,runtime/mpl}
printf '%s\n' "$RUN" > "$ROOT/latest_r878.txt"

install -m 0644 "$EXP/patch_r878_low_repair.py" "$RUN/source/patch_r878_low_repair.py"
install -m 0644 "$EXP/analyze_r877_sustained_trace.py" "$RUN/source/analyze_r877_sustained_trace.py"
"$PY" "$RUN/source/patch_r878_low_repair.py" \
  --source "$R876_EXP/train_r876.py" \
  --output "$RUN/source/train_r878.py"
"$PY" -m py_compile "$RUN/source"/*.py

"$PY" "$RUN/source/analyze_r877_sustained_trace.py" \
  --trace "$TRACE" \
  --output "$RUN/results/r877_sustained_reassessment.json" \
  > "$RUN/results/r877_sustained_reassessment.log" 2>&1
cat "$RUN/results/r877_sustained_reassessment.log"
grep -Fq 'R878_TRACE_VERDICT=ACCEPTED_LOW_SUSTAINED_FAILURE' \
  "$RUN/results/r877_sustained_reassessment.log" || {
    echo "R878_FATAL=legacy_trace_did_not_confirm_low_sustained_failure" >&2
    exit 24
  }

cat > "$RUN/contract.txt" <<EOF
version=r8_78_v4d_sustained_low_repair_v1
training=true
training_target=low_only
high_trainable=false
low_trainable=true
parent_high=$HIGH
parent_high_sha256=$EXPECTED_HIGH
parent_low=$LOW
parent_low_sha256=$EXPECTED_LOW
legacy_trace=$TRACE
pre_gate=existing_r877_trace_late_window
post_gate=fresh_early_late_signed_velocity_8_primitives
iterations=$MAX_ITERS
num_envs=$NUM_ENVS
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
export R878_SUSTAINED_LOW_REPAIR=1
export R878_LOW_LR=0.00002
export V4D_HIGH_CHECKPOINT="$HIGH"
unset R874B_FREEZE_LOW R876_FREEZE_LOW R874B_FRESH_NO_CURRICULUM R874_FRESH_NO_CURRICULUM

{
  echo "RUN=$RUN"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader || true
} > "$RUN/resource_start.txt"

set +e
setsid timeout --foreground --kill-after=30s 105m \
  "$PY" "$RUN/source/train_r878.py" \
    --headless \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERS" \
    --seed 87801 \
    --low_level_checkpoint "$LOW" \
    --log_dir "$RUN/logs" \
    --run_name r878_sustained_low_repair \
  > "$RUN/worker.log" 2>&1
RC=$?
set -e
printf 'PROCESS_RC=%s\n' "$RC" >> "$RUN/worker.log"

VALIDATION_RC=0
SUMMARY="$RUN/logs/r878_low_repair_summary.json"
if [[ "$RC" -eq 0 && -s "$SUMMARY" ]]; then
  "$PY" - "$SUMMARY" <<'PY' > "$RUN/post_training_validation.txt" 2>&1 || VALIDATION_RC=$?
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert data["training_target"] == "low_only"
assert data["high_trainable"] is False
assert data["low_trainable"] is True
assert len(data["records"]) == 300
assert data["pre_sustained"]["accepted"] is False
assert data["post_sustained"]["accepted"] is True
assert data["post_sustained"]["pass_count"] == 8
assert data["post_sustained"]["pass_count"] > data["pre_sustained"]["pass_count"]
assert float(data["high_parameter_delta"]["max_abs"]) == 0.0
assert float(data["low_parameter_delta"]["max_abs"]) > 0.0
assert data["accepted"] is True
print("R878_POST_TRAINING_VALIDATION_OK")
print("PRE_PASS_COUNT=", data["pre_sustained"]["pass_count"])
print("POST_PASS_COUNT=", data["post_sustained"]["pass_count"])
print("LOW_DELTA=", json.dumps(data["low_parameter_delta"], sort_keys=True))
PY
else
  VALIDATION_RC=1
fi

for FILE in "$RUN/logs/low_model_final.pt" "$RUN/logs/high_model_unchanged.pt" \
  "$SUMMARY"; do
  [[ -s "$FILE" ]] && cp -a "$FILE" "$RUN/checkpoints/"
done

if [[ "$RC" -eq 0 && "$VALIDATION_RC" -eq 0 \
      && -f "$RUN/logs/R878_LOW_REPAIR_ACCEPTED" ]]; then
  touch "$RUN/R878_READY_FOR_PAIRED_FULL_SUPPORT_GATE"
fi

(cd "$RUN" && find . -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum > SHA256SUMS)

echo "R878_PROCESS_RC=$RC"
echo "R878_VALIDATION_RC=$VALIDATION_RC"
echo "R878_RUN=$RUN"
if [[ -f "$RUN/R878_READY_FOR_PAIRED_FULL_SUPPORT_GATE" ]]; then
  echo "VERDICT=R878_LOW_REPAIR_READY_FOR_PAIRED_GATE"
else
  echo "VERDICT=R878_LOW_REPAIR_REJECTED"
fi
exit "$(( RC != 0 ? RC : VALIDATION_RC ))"
