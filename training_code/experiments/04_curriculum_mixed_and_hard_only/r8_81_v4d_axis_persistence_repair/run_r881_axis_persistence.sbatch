#!/usr/bin/env bash
#SBATCH --account=torch_pr_1066_general
#SBATCH --partition=l40s_public
#SBATCH --job-name=r881-low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-r881-%j.out

set -Eeuo pipefail
PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
EXP="$PROJECT/experiments/r8_81_v4d_axis_persistence_repair"
R879_RUN="/scratch/lj2634/runs/go2_soccer_r8_79_v4d_directional_low_repair/directional_15874488_20260816_201731"
SOURCE="$R879_RUN/source/train_r879.py"
LOW="$R879_RUN/logs/low_model_final.pt"
PARENT_SUMMARY="$R879_RUN/logs/r879_low_repair_summary.json"
ROOT="/scratch/lj2634/runs/go2_soccer_r8_81_v4d_axis_persistence_repair"
PY="${ISAAC_PYTHON:-/scratch/lj2634/envs/isaaclab232_v2/bin/python}"
HIGH="/scratch/lj2634/runs/go2_soccer_v4d_joint/v4d_staged_joint_20260803_223518/full/logs/high_model_final.pt"
EXPECTED_HIGH="1a05964f6330b532fe944d24028c67b880664eee3218a984cf17db6bf8010397"
MAX_ITERS="${R881_MAX_ITERATIONS:-250}"
NUM_ENVS="${R881_NUM_ENVS:-512}"

cd "$PROJECT"
for FILE in "$EXP/patch_r881_axis_persistence.py" "$SOURCE" "$LOW" "$PARENT_SUMMARY" "$HIGH"; do
  [[ -s "$FILE" ]] || { echo "R881_FATAL=missing:$FILE" >&2; exit 21; }
done
[[ "$(sha256sum "$HIGH" | awk '{print $1}')" == "$EXPECTED_HIGH" ]] || exit 22
[[ "$MAX_ITERS" == "250" ]] || { echo "R881_FATAL=iterations_must_be_250"; exit 23; }
"$PY" - "$PARENT_SUMMARY" <<'PY'
import json, sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
assert d["accepted"] is False
assert d["post_sustained"]["pass_count"] == 5
print("R881_PARENT_5_OF_8_CONFIRMED")
PY

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="$ROOT/axis_${SLURM_JOB_ID:-manual}_${STAMP}"
mkdir -p "$RUN"/{source,logs,checkpoints,runtime/home,runtime/cache,runtime/config,runtime/data,runtime/tmp,runtime/mpl}
printf '%s\n' "$RUN" > "$ROOT/latest_r881.txt"
"$PY" "$EXP/patch_r881_axis_persistence.py" --source "$SOURCE" --output "$RUN/source/train_r881.py"
"$PY" -m py_compile "$RUN/source/train_r881.py"

export PYTHONPATH="$PROJECT/source/isaac_go2_low_level_rl:$PROJECT/source:${PYTHONPATH:-}"
export HOME="$RUN/runtime/home" XDG_CACHE_HOME="$RUN/runtime/cache" XDG_CONFIG_HOME="$RUN/runtime/config"
export XDG_DATA_HOME="$RUN/runtime/data" TMPDIR="$RUN/runtime/tmp" MPLCONFIGDIR="$RUN/runtime/mpl"
export PYTHONUNBUFFERED=1 QT_QPA_PLATFORM=offscreen R873_HEADLESS_STRIP=1
export R876_CURRICULUM_HIGH_REPAIR=1
export R881_SUSTAINED_LOW_REPAIR=1
export R881_FORCE_LOCOMOTION_PRIMITIVES=1
export R881_LOW_LR=0.00001
export V4D_HIGH_CHECKPOINT="$HIGH"
unset R879_SUSTAINED_LOW_REPAIR R878_SUSTAINED_LOW_REPAIR R874B_FREEZE_LOW R876_FREEZE_LOW

set +e
setsid timeout --foreground --kill-after=30s 75m \
  "$PY" "$RUN/source/train_r881.py" --headless --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERS" --seed 88101 --low_level_checkpoint "$LOW" \
    --log_dir "$RUN/logs" --run_name r881_axis_persistence \
  > "$RUN/worker.log" 2>&1
RC=$?
set -e
printf 'PROCESS_RC=%s\n' "$RC" >> "$RUN/worker.log"

VALIDATION_RC=0
SUMMARY="$RUN/logs/r881_low_repair_summary.json"
if [[ "$RC" -eq 0 && -s "$SUMMARY" ]]; then
  "$PY" - "$SUMMARY" <<'PY' > "$RUN/validation.txt" 2>&1 || VALIDATION_RC=$?
import json, sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
assert len(d["records"]) == 250
assert d["pre_sustained"]["pass_count"] < 8
assert d["post_sustained"]["pass_count"] == 8
assert d["post_sustained"]["accepted"] is True
assert float(d["high_parameter_delta"]["max_abs"]) == 0.0
assert float(d["low_parameter_delta"]["max_abs"]) > 0.0
assert d["accepted"] is True
print("R881_VALIDATION_OK")
print("PRE_PASS_COUNT=",d["pre_sustained"]["pass_count"])
print("POST_PASS_COUNT=",d["post_sustained"]["pass_count"])
PY
else VALIDATION_RC=1
fi
for FILE in "$RUN/logs/low_model_final.pt" "$RUN/logs/high_model_unchanged.pt" "$SUMMARY"; do
  [[ -s "$FILE" ]] && cp -a "$FILE" "$RUN/checkpoints/"
done
if [[ "$RC" -eq 0 && "$VALIDATION_RC" -eq 0 && -f "$RUN/logs/R881_LOW_REPAIR_ACCEPTED" ]]; then
  touch "$RUN/R881_READY_FOR_PAIRED_FULL_SUPPORT_GATE"
fi
(cd "$RUN" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
cat "$RUN/validation.txt" 2>/dev/null || true
echo "R881_RUN=$RUN"
[[ -f "$RUN/R881_READY_FOR_PAIRED_FULL_SUPPORT_GATE" ]] && \
  echo "VERDICT=R881_AXIS_LOW_READY_FOR_PAIRED_GATE" || echo "VERDICT=R881_AXIS_LOW_REJECTED"
exit "$(( RC != 0 ? RC : VALIDATION_RC ))"
