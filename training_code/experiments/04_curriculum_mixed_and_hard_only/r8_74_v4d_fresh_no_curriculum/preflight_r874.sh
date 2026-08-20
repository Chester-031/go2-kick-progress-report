#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
R863_ROOT="/scratch/lj2634/runs/go2_soccer_r8_63_full_support_v4d"
PY="${ISAAC_PYTHON:-/scratch/lj2634/envs/isaaclab232_v2/bin/python}"
LOW="${R874_LOW_CHECKPOINT:-/scratch/lj2634/runs/go2_soccer_v4d_joint/v4d_staged_joint_20260803_223518/full/logs/low_model_final.pt}"
TMP="$(mktemp -d /tmp/r874_preflight.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT
cd "$PROJECT"
export PYTHONPATH="$PROJECT/source/isaac_go2_low_level_rl:$PROJECT/source:${PYTHONPATH:-}"
[[ -s "$R863_ROOT/latest_r863_candidate.txt" ]] || { echo "R874_PREFLIGHT_PARENT_POINTER=FAIL"; exit 2; }
[[ -f "$LOW" ]] || { echo "R874_PREFLIGHT_LOW_CHECKPOINT=FAIL path=$LOW"; exit 2; }
BASE="$(tr -d '\r\n' < "$R863_ROOT/latest_r863_candidate.txt")"
SRC="$BASE/source/train_v4d_full_support.py"
[[ -f "$SRC" ]] || { echo "R874_PREFLIGHT_SOURCE=FAIL path=$SRC"; exit 2; }
"$PY" "$PROJECT/experiments/r8_74_v4d_fresh_no_curriculum/patch_headless_train.py" --source "$SRC" --out "$TMP/headless.py"
"$PY" "$PROJECT/experiments/r8_74_v4d_fresh_no_curriculum/patch_fresh_no_curriculum.py" --source "$TMP/headless.py" --out "$TMP/train.py"
"$PY" -m py_compile "$TMP/train.py"
echo "R874_PREFLIGHT_OK"
echo "FRESH_HIGH_POLICY=TRUE"
echo "LOW_POLICY=FROZEN_WARM_START"
echo "PROGRESSIVE_CURRICULUM=FALSE"
echo "LOW_CHECKPOINT=$LOW"
