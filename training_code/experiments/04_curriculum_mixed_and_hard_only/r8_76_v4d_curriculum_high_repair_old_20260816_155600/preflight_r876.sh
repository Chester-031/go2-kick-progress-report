#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
EXP="$PROJECT/experiments/r8_76_v4d_curriculum_high_repair"
PY="${ISAAC_PYTHON:-/scratch/lj2634/envs/isaaclab232_v2/bin/python}"
HIGH="/scratch/lj2634/runs/go2_soccer_v4d_joint/v4d_staged_joint_20260803_223518/full/logs/high_model_final.pt"
LOW="/scratch/lj2634/runs/go2_soccer_v4d_joint/v4d_staged_joint_20260803_223518/full/logs/low_model_final.pt"

for FILE in "$EXP/train_r876.py" "$EXP/run_r876_curriculum_high_repair.sbatch" \
  "$EXP/check_r876_latest.sh" "$HIGH" "$LOW"; do
  [[ -s "$FILE" ]] || {
    echo "R876_PREFLIGHT_FATAL=missing:$FILE" >&2
    exit 21
  }
done

[[ "$(sha256sum "$HIGH" | awk '{print $1}')" == \
  "1a05964f6330b532fe944d24028c67b880664eee3218a984cf17db6bf8010397" ]]
[[ "$(sha256sum "$LOW" | awk '{print $1}')" == \
  "52acb55160f4d4db0ac5c3da41aadbc5383a90555c64ab41851ba7383989996a" ]]

"$PY" -m py_compile "$EXP/train_r876.py"
bash -n "$EXP/run_r876_curriculum_high_repair.sbatch"
bash -n "$EXP/check_r876_latest.sh"

"$PY" - "$EXP/train_r876.py" <<'PY'
import ast
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
ast.parse(text)
required = (
    "def _r876_phase(",
    "def _r876_sample_easy(",
    "def _r876_sample_full(",
    "def _r876_sample_reset(",
    "[R876_CURRICULUM_STAGE]",
    '"curriculum_phase": curriculum_phase',
    "R8.76 changed the frozen low-level policy.",
    "[R876_LOW_LOAD] exact_checkpoint=true",
    "R876 phase produced no nonzero high-level reward",
    "R876 phase produced no goal event",
)
missing = [marker for marker in required if marker not in text]
if missing:
    raise SystemExit(f"R876_PREFLIGHT_FATAL=missing_markers:{missing}")
print("R876_STATIC_CURRICULUM_CONTRACT_OK")
PY

echo "R876_PARENT_HASHES_OK"
echo "R876_PREFLIGHT_OK"
