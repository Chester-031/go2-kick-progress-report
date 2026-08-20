#!/usr/bin/env bash
#SBATCH --account=torch_pr_1066_general
#SBATCH --partition=l40s_public
#SBATCH --job-name=r879-low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-r879-%j.out

set -Eeuo pipefail

PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
EXP="$PROJECT/experiments/r8_79_v4d_directional_low_repair"
R878_RUN="/scratch/lj2634/runs/go2_soccer_r8_78_v4d_sustained_low_repair/low_repair_15869994_20260816_194303"
SOURCE="$R878_RUN/source/train_r878.py"
LOW="$R878_RUN/logs/low_model_final.pt"
R878_SUMMARY="$R878_RUN/logs/r878_low_repair_summary.json"
ROOT="/scratch/lj2634/runs/go2_soccer_r8_79_v4d_directional_low_repair"
PY="${ISAAC_PYTHON:-/scratch/lj2634/envs/isaaclab232_v2/bin/python}"
ACCEPTED="/scratch/lj2634/runs/go2_soccer_v4d_joint/v4d_staged_joint_20260803_223518/full"
HIGH="$ACCEPTED/logs/high_model_final.pt"
EXPECTED_HIGH="1a05964f6330b532fe944d24028c67b880664eee3218a984cf17db6bf8010397"
MAX_ITERS="${R879_MAX_ITERATIONS:-200}"
NUM_ENVS="${R879_NUM_ENVS:-512}"

cd "$PROJECT"
for FILE in "$EXP/patch_r879_directional_balance.py" "$SOURCE" "$LOW" \
  "$R878_SUMMARY" "$HIGH"; do
  [[ -s "$FILE" ]] || { echo "R879_FATAL=missing:$FILE" >&2; exit 21; }
done
[[ "$(sha256sum "$HIGH" | awk '{print $1}')" == "$EXPECTED_HIGH" ]] || \
  { echo "R879_FATAL=accepted_high_hash_mismatch" >&2; exit 22; }
[[ "$MAX_ITERS" == "200" ]] || \
  { echo "R879_FATAL=max_iterations_must_remain_200" >&2; exit 23; }

"$PY" - "$R878_SUMMARY" <<'PY'
import json, sys
from pathlib import Path
d = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert d["accepted"] is False
assert d["post_sustained"]["pass_count"] == 4
assert float(d["high_parameter_delta"]["max_abs"]) == 0.0
print("R879_PARENT_R878_REJECTION_CONFIRMED")
PY

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="$ROOT/directional_${SLURM_JOB_ID:-manual}_${STAMP}"
mkdir -p "$RUN"/{source,logs,checkpoints,runtime/home,runtime/cache,runtime/config,runtime/data,runtime/tmp,runtime/mpl}
printf '%s\n' "$RUN" > "$ROOT/latest_r879.txt"

install -m 0644 "$EXP/patch_r879_directional_balance.py" "$RUN/source/patch_r879_directional_balance.py"
"$PY" "$RUN/source/patch_r879_directional_balance.py" \
  --source "$SOURCE" --output "$RUN/source/train_r879.py"
"$PY" -m py_compile "$RUN/source"/*.py

LOW_SHA="$(sha256sum "$LOW" | awk '{print $1}')"
cat > "$RUN/contract.txt" <<EOF
version=r8_79_v4d_directional_low_repair_v1
training=true
continuation=true
iterations=$MAX_ITERS
num_envs=$NUM_ENVS
high_trainable=false
low_trainable=true
automatic_push_during_primitives=false
directional_reference_reward=true
parent_r878_run=$R878_RUN
parent_low=$LOW
parent_low_sha256=$LOW_SHA
accepted_high=$HIGH
accepted_high_sha256=$EXPECTED_HIGH
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
export R879_SUSTAINED_LOW_REPAIR=1
export R879_FORCE_LOCOMOTION_PRIMITIVES=1
export R879_LOW_LR=0.00001
export V4D_HIGH_CHECKPOINT="$HIGH"
unset R878_SUSTAINED_LOW_REPAIR R874B_FREEZE_LOW R876_FREEZE_LOW \
  R874B_FRESH_NO_CURRICULUM R874_FRESH_NO_CURRICULUM

set +e
setsid timeout --foreground --kill-after=30s 75m \
  "$PY" "$RUN/source/train_r879.py" \
    --headless --num_envs "$NUM_ENVS" --max_iterations "$MAX_ITERS" \
    --seed 87901 --low_level_checkpoint "$LOW" \
    --log_dir "$RUN/logs" --run_name r879_directional_low_repair \
  > "$RUN/worker.log" 2>&1
RC=$?
set -e
printf 'PROCESS_RC=%s\n' "$RC" >> "$RUN/worker.log"

VALIDATION_RC=0
SUMMARY="$RUN/logs/r879_low_repair_summary.json"
if [[ "$RC" -eq 0 && -s "$SUMMARY" ]]; then
  "$PY" - "$SUMMARY" <<'PY' > "$RUN/post_training_validation.txt" 2>&1 || VALIDATION_RC=$?
import json, sys
from pathlib import Path
d = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert len(d["records"]) == 200
assert d["pre_sustained"]["pass_count"] < 8
assert d["post_sustained"]["accepted"] is True
assert d["post_sustained"]["pass_count"] == 8
assert float(d["high_parameter_delta"]["max_abs"]) == 0.0
assert float(d["low_parameter_delta"]["max_abs"]) > 0.0
assert d["accepted"] is True
print("R879_POST_TRAINING_VALIDATION_OK")
print("PRE_PASS_COUNT=", d["pre_sustained"]["pass_count"])
print("POST_PASS_COUNT=", d["post_sustained"]["pass_count"])
print("LOW_DELTA=", json.dumps(d["low_parameter_delta"], sort_keys=True))
PY
else
  VALIDATION_RC=1
fi

for FILE in "$RUN/logs/low_model_final.pt" "$RUN/logs/high_model_unchanged.pt" "$SUMMARY"; do
  [[ -s "$FILE" ]] && cp -a "$FILE" "$RUN/checkpoints/"
done
if [[ "$RC" -eq 0 && "$VALIDATION_RC" -eq 0 \
      && -f "$RUN/logs/R879_LOW_REPAIR_ACCEPTED" ]]; then
  touch "$RUN/R879_READY_FOR_PAIRED_FULL_SUPPORT_GATE"
fi
(cd "$RUN" && find . -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum > SHA256SUMS)

echo "R879_PROCESS_RC=$RC"
echo "R879_VALIDATION_RC=$VALIDATION_RC"
echo "R879_RUN=$RUN"
if [[ -f "$RUN/R879_READY_FOR_PAIRED_FULL_SUPPORT_GATE" ]]; then
  echo "VERDICT=R879_DIRECTIONAL_LOW_READY_FOR_PAIRED_GATE"
else
  echo "VERDICT=R879_DIRECTIONAL_LOW_REJECTED"
fi
exit "$(( RC != 0 ? RC : VALIDATION_RC ))"
