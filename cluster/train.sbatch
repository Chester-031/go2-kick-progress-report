#!/usr/bin/env bash
#SBATCH --job-name=go2_adr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

# ADR training from fresh high-policy weights.  Extra arguments go to
# robot_soccer.train, e.g. --iterations 600, --initial-checkpoint PATH,
# --resume PATH or any --adr-* setting.

set -Eeuo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: sbatch train.sbatch PYTHON LOW_CHECKPOINT LOW_SHA256 OUTPUT_DIR [extra train args...]" >&2
  exit 2
fi

PYTHON_BIN=$1
LOW_CHECKPOINT=$2
LOW_SHA256=$3
OUTPUT_DIR=$4
shift 4

PACKAGE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$PACKAGE_ROOT/src:$PACKAGE_ROOT/runtime_project/isaac_go2_low_level_rl${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

exec "$PYTHON_BIN" -m robot_soccer.train \
  --headless \
  --device cuda:0 \
  --low-checkpoint "$LOW_CHECKPOINT" \
  --low-checkpoint-sha256 "$LOW_SHA256" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
