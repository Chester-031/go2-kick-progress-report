#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
EXP="$PROJECT/experiments/r8_75_v4d_owner_attribution"
PY="${ISAAC_PYTHON:-/scratch/lj2634/envs/isaaclab232_v2/bin/python}"
ACCEPTED_HIGH="/scratch/lj2634/runs/go2_soccer_v4d_joint/v4d_staged_joint_20260803_223518/full/logs/high_model_final.pt"
ACCEPTED_LOW="/scratch/lj2634/runs/go2_soccer_v4d_joint/v4d_staged_joint_20260803_223518/full/logs/low_model_final.pt"
CANDIDATE_HIGH="/scratch/lj2634/runs/go2_soccer_r8_74b_v4d_fresh_no_curriculum_fix/fresh_15834260_20260816_083308/logs/high_model_final.pt"

for FILE in "$EXP/diagnose_r875.py" "$EXP/run_r875_owner_attribution.sbatch" \
  "$ACCEPTED_HIGH" "$ACCEPTED_LOW" "$CANDIDATE_HIGH"; do
  [[ -s "$FILE" ]] || {
    echo "R875_PREFLIGHT_FATAL=missing:$FILE" >&2
    exit 21
  }
done

"$PY" -m py_compile "$EXP/diagnose_r875.py"

check_sha() {
  local file="$1" expected="$2" actual
  actual="$(sha256sum "$file" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || {
    echo "R875_PREFLIGHT_FATAL=sha_mismatch:$file" >&2
    exit 22
  }
}

check_sha "$ACCEPTED_HIGH" "1a05964f6330b532fe944d24028c67b880664eee3218a984cf17db6bf8010397"
check_sha "$ACCEPTED_LOW" "52acb55160f4d4db0ac5c3da41aadbc5383a90555c64ab41851ba7383989996a"
check_sha "$CANDIDATE_HIGH" "12b596f7627feaa4a5b557499ba681c042daa9323cce54ffa1339d4b9db4df3f"

"$PY" - "$EXP/diagnose_r875.py" <<'PY'
import ast
import sys
from pathlib import Path

path = Path(sys.argv[1])
tree = ast.parse(path.read_text(encoding="utf-8"))
functions = {
    node.name
    for node in ast.walk(tree)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
}
required = {
    "run_r875_owner_attribution",
    "_r875_scripted_actions",
    "_r875_align_and_verify_triplicates",
    "_r875_primitive_assessment",
}
missing = sorted(required - functions)
if missing:
    raise SystemExit(f"R875_PREFLIGHT_FATAL=missing_functions:{missing}")
text = path.read_text(encoding="utf-8")
for marker in (
    "R875_REQUIRED_ENVS = R875_POLICY_ENV_COUNT + len(R875_PRIMITIVE_CASES)",
    "training=false optimizer=false backward=false",
    "parameter_deltas",
    "r875_trace.csv.gz",
):
    if marker not in text:
        raise SystemExit(f"R875_PREFLIGHT_FATAL=missing_marker:{marker}")
print("R875_STATIC_CONTRACT_OK")
PY

bash -n "$EXP/run_r875_owner_attribution.sbatch"
bash -n "$EXP/check_r875_latest.sh"
echo "R875_CHECKPOINT_HASHES_OK"
echo "R875_PREFLIGHT_OK"
