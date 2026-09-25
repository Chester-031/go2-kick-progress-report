#!/usr/bin/env bash
#SBATCH --job-name=go2_soccer_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

set -Eeuo pipefail

if [[ $# -lt 7 ]]; then
  echo "usage: sbatch evaluate.sbatch PYTHON LOW_CHECKPOINT LOW_SHA256 HIGH_CHECKPOINT HIGH_SHA256 SPLIT OUTPUT_DIR [extra eval args...]" >&2
  exit 2
fi

PYTHON_BIN=$1
LOW_CHECKPOINT=$2
LOW_SHA256=$3
HIGH_CHECKPOINT=$4
HIGH_SHA256=$5
SPLIT=$6
OUTPUT_DIR=$7
shift 7

PACKAGE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$PACKAGE_ROOT/src:$PACKAGE_ROOT/runtime_project/isaac_go2_low_level_rl${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

exec "$PYTHON_BIN" -m robot_soccer.evaluate \
  --headless \
  --device cuda:0 \
  --low-checkpoint "$LOW_CHECKPOINT" \
  --low-checkpoint-sha256 "$LOW_SHA256" \
  --checkpoint "$HIGH_CHECKPOINT" \
  --checkpoint-sha256 "$HIGH_SHA256" \
  --split "$SPLIT" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
