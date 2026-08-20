#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"
EXP="$PROJECT/experiments/r8_77_v4d_checkpoint_gate_local_video"
GATE_ROOT="/scratch/lj2634/runs/go2_soccer_r8_77_v4d_checkpoint_gate"
GATE_RUN="$(tr -d '\r\n' < "$GATE_ROOT/latest_r877.txt")"
R876_ROOT="/scratch/lj2634/runs/go2_soccer_r8_76_v4d_curriculum_high_repair"
R876_RUN="$(tr -d '\r\n' < "$R876_ROOT/latest_r876.txt")"
VIDEO_ROOT="/scratch/lj2634/video_handoffs"
PY="${ISAAC_PYTHON:-/scratch/lj2634/envs/isaaclab232_v2/bin/python}"

SELECTION="$GATE_RUN/results/r877_checkpoint_selection.json"
SELECTED="$(tr -d '\r\n' < "$GATE_RUN/results/SELECTED_HIGH_PATH.txt")"
PLAYER_SOURCE="$R876_RUN/source/train_r876.py"
LOW="/scratch/lj2634/runs/go2_soccer_v4d_joint/v4d_staged_joint_20260803_223518/full/logs/low_model_final.pt"

for FILE in "$GATE_RUN/R877_CHECKPOINT_GATE_OK" "$SELECTION" "$SELECTED" \
  "$PLAYER_SOURCE" "$LOW" "$EXP/patch_r876_eval_player.py" \
  "$EXP/render_local.sh" "$EXP/check_local.sh" "$EXP/EXPECTED_BEHAVIOR.txt"; do
  [[ -s "$FILE" || -f "$FILE" ]] || {
    echo "R877_BUNDLE_FATAL=missing:$FILE" >&2
    exit 21
  }
done

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$VIDEO_ROOT/r877_v4d_selected_${STAMP}"
B="$OUT/bundle"
mkdir -p "$B"/{player,models,selection,videos,logs,runtime}

install -m 0644 "$PLAYER_SOURCE" "$B/player/train_r876_video.py"
"$PY" "$EXP/patch_r876_eval_player.py" --source "$B/player/train_r876_video.py"
"$PY" -m py_compile "$B/player/train_r876_video.py"
grep -Fq '[R876_EVAL_DISTRIBUTION]' "$B/player/train_r876_video.py"

install -m 0644 "$SELECTED" "$B/models/high_model_selected.pt"
install -m 0644 "$LOW" "$B/models/low_model_final.pt"
install -m 0644 "$SELECTION" "$B/selection/r877_checkpoint_selection.json"
install -m 0644 "$EXP/EXPECTED_BEHAVIOR.txt" "$B/EXPECTED_BEHAVIOR.txt"
install -m 0755 "$EXP/render_local.sh" "$B/render_local.sh"
install -m 0755 "$EXP/check_local.sh" "$B/check_local.sh"

"$PY" - "$SELECTION" "$B/manifest.txt" "$SELECTED" "$LOW" <<'PY'
import json, sys
from pathlib import Path
d=json.load(open(sys.argv[1], encoding="utf-8"))
Path(sys.argv[2]).write_text("\n".join([
    "version=r8_77_v4d_selected_local_video_v1",
    "training=false",
    "cluster_rendering=false",
    "evaluation_distribution=stationary_full_support_no_curriculum",
    f"selected_label={d['selected_label']}",
    f"selection_status={d['selection_status']}",
    f"source_high_checkpoint={sys.argv[3]}",
    f"source_low_checkpoint={sys.argv[4]}",
    "seed=87701",
    "eval_steps=1200",
    "eval_fps=15",
    "video_count=1",
]) + "\n", encoding="utf-8")
PY

(
  cd "$B"
  find . -type f ! -name SHA256SUMS -print0 | sort -z \
    | xargs -0 sha256sum > SHA256SUMS
  sha256sum -c SHA256SUMS
)

ARCHIVE="$OUT/r877_v4d_selected_local_video_${STAMP}.tgz"
tar -czf "$ARCHIVE" -C "$OUT" bundle
printf '%s\n' "$ARCHIVE" > "$VIDEO_ROOT/latest_r877_v4d_bundle.txt"

echo "R877_LOCAL_VIDEO_BUNDLE_READY"
echo "ARCHIVE=$ARCHIVE"
sha256sum "$ARCHIVE"
echo "VERDICT=R877_BUNDLE_READY_FOR_LOCAL_DOWNLOAD"
