#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
R863_ROOT="/scratch/lj2634/runs/go2_soccer_r8_63_full_support_v4d"
R864_ROOT="/scratch/lj2634/runs/go2_soccer_r8_64_v4d_coverage"
EXP="$PROJECT/experiments/r8_64_v4d_coverage_scan"
PYTHON_BIN="${ISAAC_PYTHON:-python}"
POINTER="$R863_ROOT/latest_r863_candidate.txt"
[[ -s "$POINTER" ]] || { echo "R864_FATAL missing=$POINTER"; exit 20; }
R863_RUN="$(tr -d '\r\n' < "$POINTER")"
SOURCE="$R863_RUN/source/train_v4d_full_support.py"
REPORT="$R863_RUN/source/train_v4d_full_support.patch_report.json"
[[ -s "$SOURCE" && -s "$REPORT" ]] || { echo "R864_FATAL missing_r863_artifacts"; exit 21; }
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$R864_ROOT/coverage_${STAMP}"
mkdir -p "$OUT/evidence"
cp -a "$SOURCE" "$OUT/evidence/source_v4d_full_support.py"
cp -a "$REPORT" "$OUT/evidence/patch_report.json"
"$PYTHON_BIN" "$EXP/coverage_probe.py" --source "$SOURCE" --report "$OUT/evidence/v4d_coverage.json" --samples "${R864_SAMPLES:-200000}" --seed "${R864_SEED:-864}" --min-distance 0.75
printf 'version=r8_64_v4d_coverage_scan\ntraining=false\ncheckpoint_frozen=true\nr863_candidate=%s\npurpose=verify_full_support_reset_distribution_before_policy_training\n' "$R863_RUN" > "$OUT/evidence/R864_STATUS.txt"
(cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
mkdir -p "$R864_ROOT" "$EXP"
printf '%s\n' "$OUT" > "$R864_ROOT/latest_r864.txt"
printf '%s\n' "$OUT" > "$EXP/latest_r864.txt"
echo "R864_V4D_COVERAGE_SCAN_OK"
echo "RUN=$OUT"
echo "TRAINING=FALSE"
echo "VERDICT=R864_READY_FOR_ISAAC_BEHAVIOR_SCAN"
