#!/usr/bin/env bash
set -Eeuo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${PROJECT:-/home/ubuntu/下载/go2_soccer_phaseA_results/project/Isaac-Go2-Low-Level-RL}"
CONDA_ENV="${CONDA_ENV:-isaaclab232}"
PLAYER="$BASE/player/train_r856_hold_bias.py"
LOW="$BASE/ballfree/artifacts/r821_lift_base.pt"
mkdir -p "$BASE/videos" "$BASE/local_logs"

for item in '0.75:low' '1.0:mid' '1.25:high'; do
  strength="${item%%:*}"; label="${item##*:}"
  video="$BASE/videos/R856_${label}_${strength}_kick.mp4"
  log="$BASE/local_logs/${label}.log"
  R856_VIDEO_ONLY=1 \
  R856_FREEZE_DIR="$BASE/ballfree" \
  R856_KICK="$BASE/models/kick64_final.pt" \
  R856_R851_SUMMARY="$BASE/evidence/r851_summary.json" \
  R856_BODY_JSON="$BASE/evidence/body_names.json" \
  R851_BODY_JSON="$BASE/evidence/body_names.json" \
  R856_RECOVERY="$BASE/models/recovery64_final.pt" \
  R856_WINNER="$BASE/evidence/r856_hold_winner.json" \
  R856_VIDEO_STRENGTH="$strength" \
  PYTHONPATH="$PROJECT/source/isaac_go2_low_level_rl:${PYTHONPATH:-}" \
  conda run --no-capture-output -n "$CONDA_ENV" python "$PLAYER" \
    --task Isaac-Go2-Soccer-Milestone2-v0 \
    --num_envs 1 --device cuda:0 --headless --enable_cameras \
    --max_iterations 1 --seed 42 --run_name "r856_video_${label}" \
    --low_level_checkpoint "$LOW" --log_dir "$BASE/local_logs/${label}_run" \
    --eval_video "$video" 2>&1 | tee "$log"
  test -s "$video"; test -s "${video}.ready"; test -s "${video%.mp4}.json"; grep -qF R856_EVAL_VIDEO_READY "$log"
done

sha256sum "$BASE"/videos/*.mp4 "$BASE"/videos/*.json > "$BASE/videos/SHA256SUMS"
echo R856_LOCAL_VIDEOS_READY
echo "VIDEOS=$BASE/videos"
xdg-open "$BASE/videos" >/dev/null 2>&1 || true
