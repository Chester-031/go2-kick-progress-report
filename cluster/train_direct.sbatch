#!/usr/bin/env bash
#SBATCH --job-name=go2_direct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Train one policy straight from the Go2's joints with ADR.
#   sbatch train_direct.sbatch PYTHON CONTROL OUTPUT_DIR [extra train args...]
# CONTROL is position or velocity.  PYTHON is the Isaac Lab python; set
# EXTRA_PYTHONPATH for packages layered on top of it (rsl_rl 3.1.2,
# tensordict >= 0.7).  Extra arguments go to robot_soccer.train, e.g.
# --iterations 3000, --num-envs 4096, --resume PATH or any --adr-* setting.

set -Eeuo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: sbatch train_direct.sbatch PYTHON CONTROL OUTPUT_DIR [extra train args...]" >&2
  exit 2
fi

PYTHON_BIN=$1
CONTROL=$2
OUTPUT_DIR=$3
shift 3

PACKAGE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$PACKAGE_ROOT/src:$PACKAGE_ROOT/runtime_project/isaac_go2_low_level_rl${EXTRA_PYTHONPATH:+:$EXTRA_PYTHONPATH}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA=YES

exec "$PYTHON_BIN" -m robot_soccer.train \
  --headless \
  --device cuda:0 \
  --control "$CONTROL" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
