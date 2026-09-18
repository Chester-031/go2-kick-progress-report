#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

for argument in "$@"; do
  if [[ "$argument" == "--help" || "$argument" == "-h" ]]; then
    exec python3 "$SCRIPT_DIR/record_current_policy.py" "$@"
  fi
done

exec /home/ubuntu/IsaacLab_232/isaaclab.sh -p \
  "$SCRIPT_DIR/record_current_policy.py" \
  --headless \
  --enable_cameras \
  "$@"
