#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
EXP="$PROJECT/experiments/r8_79_v4d_directional_low_repair"
PARENT="/scratch/lj2634/runs/go2_soccer_r8_78_v4d_sustained_low_repair/low_repair_15869994_20260816_194303/source/train_r878.py"
PY="${ISAAC_PYTHON:-/scratch/lj2634/envs/isaaclab232_v2/bin/python}"
for FILE in "$EXP/patch_r879_directional_balance.py" \
  "$EXP/run_r879_directional_low_repair.sbatch" "$EXP/check_r879_latest.sh" "$PARENT"; do
  [[ -s "$FILE" ]] || { echo "R879_PREFLIGHT_FATAL=missing:$FILE"; exit 21; }
done
bash -n "$EXP/run_r879_directional_low_repair.sbatch" "$EXP/check_r879_latest.sh"
"$PY" -m py_compile "$EXP/patch_r879_directional_balance.py"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
"$PY" "$EXP/patch_r879_directional_balance.py" --source "$PARENT" --output "$TMP/train_r879.py"
"$PY" -m py_compile "$TMP/train_r879.py"
"$PY" - "$TMP/train_r879.py" <<'PY'
import ast, sys
from pathlib import Path
t = Path(sys.argv[1]).read_text(encoding="utf-8")
ast.parse(t)
required = (
    'R879_FORCE_LOCOMOTION_PRIMITIVES',
    'push_mask = torch.zeros_like(push_mask)',
    'directional_velocity_track',
    'wrong_direction',
    'def run_r879_low_repair(',
    'R879_LOW_REPAIR_ACCEPTED',
)
missing = [x for x in required if x not in t]
assert not missing, missing
assert 'R878_SUSTAINED_LOW_REPAIR' not in t
print("R879_GENERATED_TRAINER_STATIC_CONTRACT_OK")
PY
echo "R879_PREFLIGHT_OK"
