#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
EXP="$PROJECT/experiments/r8_78_v4d_sustained_low_repair"
R876="$PROJECT/experiments/r8_76_v4d_curriculum_high_repair/train_r876.py"
PY="${ISAAC_PYTHON:-/scratch/lj2634/envs/isaaclab232_v2/bin/python}"

for FILE in "$EXP/patch_r878_low_repair.py" "$EXP/analyze_r877_sustained_trace.py" \
  "$EXP/run_r878_sustained_low_repair.sbatch" "$EXP/check_r878_latest.sh" "$R876"; do
  [[ -s "$FILE" ]] || { echo "R878_PREFLIGHT_FATAL=missing:$FILE"; exit 21; }
done

bash -n "$EXP/run_r878_sustained_low_repair.sbatch"
bash -n "$EXP/check_r878_latest.sh"
"$PY" -m py_compile "$EXP/patch_r878_low_repair.py" "$EXP/analyze_r877_sustained_trace.py"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
"$PY" "$EXP/patch_r878_low_repair.py" --source "$R876" --output "$TMP/train_r878.py"
"$PY" -m py_compile "$TMP/train_r878.py"

"$PY" - "$TMP/train_r878.py" <<'PY'
import ast, sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(encoding="utf-8")
ast.parse(text)
required = (
    "def run_r878_low_repair(",
    "sustained_stall = motion_requested & (robot_speed < 0.04)",
    "reward = reward - 2.0 * sustained_stall.float()",
    '"training_target": "low_only"',
    'post_eval["pass_count"] > pre_eval["pass_count"]',
    'high_delta["max_abs"] == 0.0',
    "R878_LOW_REPAIR_ACCEPTED",
)
missing = [item for item in required if item not in text]
assert not missing, missing
assert 'diag["robot_speed"][' not in text
assert 'diag["upright_error"][' not in text
print("R878_GENERATED_TRAINER_STATIC_CONTRACT_OK")
PY

echo "R878_PREFLIGHT_OK"
