#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
R859_ROOT="/scratch/lj2634/runs/go2_soccer_r8_59_parent_freeze"
R862_ROOT="/scratch/lj2634/runs/go2_soccer_r8_62_v4d_audit"
EXP="$PROJECT/experiments/r8_62_v4d_audit"
PYTHON_BIN="${ISAAC_PYTHON:-/scratch/lj2634/envs/isaaclab232_v2/bin/python}"

POINTER="$R859_ROOT/latest_r859_freeze.txt"
[[ -s "$POINTER" ]] || { echo "R862_FATAL missing=$POINTER"; exit 20; }
FREEZE="$(tr -d '\r\n' < "$POINTER")"
SOURCE="$FREEZE/source/train_go2_soccer_v4d_joint.snapshot.py"
CONTRACT="$FREEZE/integration_parent_contract.json"
[[ -s "$SOURCE" && -s "$CONTRACT" ]] || { echo "R862_FATAL frozen_v4d_source_or_contract_missing"; exit 21; }
[[ -x "$PYTHON_BIN" ]] || { echo "R862_FATAL python_not_executable=$PYTHON_BIN"; exit 22; }

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$R862_ROOT/v4d_audit_${STAMP}"
mkdir -p "$OUT/source" "$OUT/evidence"
cp -a "$SOURCE" "$OUT/source/"
cp -a "$CONTRACT" "$OUT/evidence/"
cp -a "$EXP/inspect_v4d_contract.py" "$OUT/source/"

"$PYTHON_BIN" "$OUT/source/inspect_v4d_contract.py" \
  --source "$OUT/source/train_go2_soccer_v4d_joint.snapshot.py" \
  --contract "$OUT/evidence/integration_parent_contract.json" \
  --out "$OUT/evidence/v4d_audit.json"

{
  printf 'version=r8_62_v4d_audit\ntraining=false\nparent_mutation=false\n'
  printf 'r859_freeze=%s\n' "$FREEZE"
  printf 'source_copy=%s\n' "$OUT/source/train_go2_soccer_v4d_joint.snapshot.py"
  printf 'purpose=understand_before_patch\n'
} > "$OUT/evidence/R862_STATUS.txt"
(cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
chmod -R a-w "$OUT"
mkdir -p "$R862_ROOT" "$EXP"
printf '%s\n' "$OUT" > "$R862_ROOT/latest_v4d_audit.txt"
printf '%s\n' "$OUT" > "$EXP/latest_v4d_audit.txt"
echo "R862_V4D_AUDIT_READY"
echo "AUDIT_RUN=$OUT"
