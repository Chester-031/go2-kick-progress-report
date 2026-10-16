#!/usr/bin/env bash
#SBATCH --job-name=go2_render6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00

# Render recorded episodes (scripts/record_trajectories.py) on the Go2
# model with Isaac Sim 6's path tracer, then encode one captioned MP4.
# Submit from the repository root (or set REPO), with absolute paths:
#   sbatch --export=ALL,SIF=/abs/isaac-sim-6.2.0.sif,NPZ=/abs/traj.npz,OUT=/abs/videos,PYTHON=ISAAC_LAB_PYTHON \
#       cluster/render_isaac6.sbatch
# Isaac Sim 4.5 (the training stack) segfaults in its RTX renderer on NVIDIA
# driver 610, so rendering runs in the Isaac Sim 6.2 container
# (apptainer pull isaac-sim-6.2.0.sif docker://nvcr.io/nvidia/isaac-sim:6.2.0).
# PYTHON is any python with numpy, opencv and imageio[ffmpeg] (the Isaac Lab
# one has them); it picks the episodes and encodes the video.
# Optional: NAME (reel name), EPISODES (default 4), PICK (best | sample),
# SAMPLES (path-tracing samples per pixel, default 64), TITLE, ROBOT_USD.
# On Torch, H100 nodes crashed at Kit startup; A100 and L40S render.

set -Eeuo pipefail
: "${SIF:?}" "${NPZ:?}" "${OUT:?}" "${PYTHON:?}"
# sbatch runs a copy of this file from the Slurm spool, so the repository
# is where sbatch was called from, not next to the running script.
PACKAGE_ROOT=${REPO:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}
if [[ ! -f "$PACKAGE_ROOT/scripts/replay_isaac6.py" ]]; then
  echo "no scripts/replay_isaac6.py under $PACKAGE_ROOT: submit from the repository root or set REPO" >&2
  exit 2
fi
mkdir -p "$OUT"
NAME=${NAME:-$(basename "$NPZ" .npz)}
STATE=$OUT/.isaac6-state
mkdir -p "$STATE"/kit-cache "$STATE"/kit-data "$STATE"/kit-logs "$STATE"/home
KEYS=$("$PYTHON" "$PACKAGE_ROOT/scripts/encode_replay.py" --trajectories "$NPZ" --list \
  --episodes "${EPISODES:-4}" --pick "${PICK:-best}")
KEY_ARGS=()
for key in $KEYS; do KEY_ARGS+=(--key "$key"); done
echo "episodes: $KEYS"

export APPTAINERENV_ACCEPT_EULA=Y APPTAINERENV_PRIVACY_CONSENT=Y APPTAINERENV_OMNI_KIT_ACCEPT_EULA=YES
apptainer exec --nv --writable-tmpfs --home "$STATE/home" \
  -B "$PACKAGE_ROOT" -B "$(cd "$(dirname "$NPZ")" && pwd)" -B "$OUT" \
  -B "$STATE/kit-cache:/isaac-sim/kit/cache" -B "$STATE/kit-data:/isaac-sim/kit/data" \
  -B "$STATE/kit-logs:/isaac-sim/kit/logs" \
  "$SIF" /isaac-sim/python.sh "$PACKAGE_ROOT/scripts/replay_isaac6.py" --trajectories "$NPZ" \
  --robot-usd "${ROBOT_USD:-https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd}" \
  --out-dir "$OUT/frames-$NAME" --samples "${SAMPLES:-64}" "${KEY_ARGS[@]}"

"$PYTHON" "$PACKAGE_ROOT/scripts/encode_replay.py" --trajectories "$NPZ" --frames "$OUT/frames-$NAME" \
  --out "$OUT/$NAME.mp4" --episodes "${EPISODES:-4}" --pick "${PICK:-best}" --title "${TITLE:-}"
