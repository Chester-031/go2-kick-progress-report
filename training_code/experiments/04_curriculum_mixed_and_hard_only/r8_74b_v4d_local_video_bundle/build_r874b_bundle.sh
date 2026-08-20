#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
EXP="$PROJECT/experiments/r8_74b_v4d_local_video_bundle"
VIDEO_ROOT="/scratch/lj2634/video_handoffs"

HIGH_RUN="${R874B_HIGH_RUN:-/scratch/lj2634/runs/go2_soccer_r8_74b_v4d_fresh_no_curriculum_fix/fresh_15834260_20260816_083308}"
PLAYER="${R874B_PLAYER:-$HIGH_RUN/source/train.py}"
HIGH="${R874B_HIGH:-$HIGH_RUN/logs/high_model_final.pt}"
LOW="${R874B_LOW:-/scratch/lj2634/runs/go2_soccer_v4d_joint/v4d_staged_joint_20260803_223518/full/logs/low_model_final.pt}"

for file in "$PLAYER" "$HIGH" "$LOW" "$EXP/render_local.sh" "$EXP/check_local.sh"; do
  [[ -s "$file" ]] || {
    echo "R874B_BUNDLE_FATAL=missing:$file" >&2
    exit 21
  }
done

python3 - "$PLAYER" <<'PY'
from pathlib import Path
import sys

text = Path(sys.argv[1]).read_text(encoding="utf-8")
required = (
    "run_v4d_evaluation",
    "--trained_low_checkpoint",
    "--eval_video",
    "--eval_summary",
)
missing = [item for item in required if item not in text]
if missing:
    raise SystemExit(f"R874B_BUNDLE_FATAL=player_contract_missing:{missing}")
print("R874B_PLAYER_CONTRACT=OK")
PY

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$VIDEO_ROOT/r874b_v4d_local_${STAMP}"
B="$OUT/bundle"

mkdir -p "$B/player" "$B/models" "$B/videos" "$B/logs" "$B/runtime"

install -m 0644 "$PLAYER" "$B/player/train_v4d_current.py"
install -m 0644 "$HIGH" "$B/models/high_model_final.pt"
install -m 0644 "$LOW" "$B/models/low_model_final.pt"
install -m 0755 "$EXP/render_local.sh" "$B/render_local.sh"
install -m 0755 "$EXP/check_local.sh" "$B/check_local.sh"
install -m 0644 "$EXP/README.txt" "$B/README.txt"

{
  echo "version=r8_74b_v4d_local_video_bundle_v1"
  echo "training=false"
  echo "cluster_rendering=false"
  echo "source=$PLAYER"
  echo "high_checkpoint=$HIGH"
  echo "low_checkpoint=$LOW"
  echo "seed=86501"
  echo "eval_steps=900"
  echo "eval_fps=15"
  echo "expected_local_project=/home/ubuntu/下载/go2_soccer_phaseA_results/project/Isaac-Go2-Low-Level-RL"
} > "$B/manifest.txt"

(
  cd "$B"
  find . -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum > SHA256SUMS
)

ARCHIVE="$OUT/r874b_v4d_local_video_bundle_${STAMP}.tgz"
tar -czf "$ARCHIVE" -C "$OUT" bundle
printf '%s\n' "$ARCHIVE" > "$VIDEO_ROOT/latest_r874b_v4d_bundle.txt"

echo "R874B_V4D_BUNDLE_READY"
echo "ARCHIVE=$ARCHIVE"
sha256sum "$ARCHIVE"
echo "VERDICT=R874B_BUNDLE_READY_FOR_DTN_DOWNLOAD"
