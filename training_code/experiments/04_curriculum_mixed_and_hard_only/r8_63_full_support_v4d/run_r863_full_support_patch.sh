#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
R862_ROOT="/scratch/lj2634/runs/go2_soccer_r8_62_v4d_audit"
R863_ROOT="/scratch/lj2634/runs/go2_soccer_r8_63_full_support_v4d"
EXP="$PROJECT/experiments/r8_63_full_support_v4d"
PYTHON_BIN="${ISAAC_PYTHON:-/scratch/lj2634/envs/isaaclab232_v2/bin/python}"

POINTER="$R862_ROOT/latest_v4d_audit.txt"
[[ -s "$POINTER" ]] || { echo "R863_FATAL missing=$POINTER"; exit 20; }
AUDIT="$(tr -d '\r\n' < "$POINTER")"
SOURCE="$AUDIT/source/train_go2_soccer_v4d_joint.snapshot.py"
[[ -s "$SOURCE" ]] || { echo "R863_FATAL missing_source=$SOURCE"; exit 21; }
[[ -x "$PYTHON_BIN" ]] || { echo "R863_FATAL python_not_executable=$PYTHON_BIN"; exit 22; }

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$R863_ROOT/full_support_${STAMP}"
mkdir -p "$OUT/source" "$OUT/evidence"
cp -a "$SOURCE" "$OUT/source/source_v4d_full_support.py"
cp -a "$EXP/patch_full_support_v4d.py" "$OUT/source/"

"$PYTHON_BIN" "$OUT/source/patch_full_support_v4d.py" \
  --source "$OUT/source/source_v4d_full_support.py" \
  --out "$OUT/source/train_v4d_full_support.py"

cp -a "$OUT/source/train_v4d_full_support.patch_report.json" "$OUT/evidence/"
{
  printf 'version=r8_63_full_support_v4d_candidate\ntraining=false\nparent_mutation=false\n'
  printf 'v4d_parent_audit=%s\n' "$AUDIT"
  printf 'purpose=remove_progressive_reset_curriculum_before_coverage_scan\n'
} > "$OUT/evidence/R863_STATUS.txt"
(cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
chmod -R a-w "$OUT"
mkdir -p "$R863_ROOT" "$EXP"
printf '%s\n' "$OUT" > "$R863_ROOT/latest_r863_candidate.txt"
printf '%s\n' "$OUT" > "$EXP/latest_r863_candidate.txt"
echo "R863_FULL_SUPPORT_CANDIDATE_READY"
echo "CANDIDATE=$OUT"
