#!/usr/bin/env bash
set -Eeuo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${PROJECT:-/home/ubuntu/下载/go2_soccer_phaseA_results/project/Isaac-Go2-Low-Level-RL}"
CONDA_ENV="${CONDA_ENV:-isaaclab232}"
SEED="${R874B_VIDEO_SEED:-86501}"
EVAL_STEPS="${R874B_EVAL_STEPS:-900}"
FPS="${R874B_EVAL_FPS:-15}"
TIMEOUT_MINUTES="${R874B_TIMEOUT_MINUTES:-30}"

PLAYER="$BASE/player/train_v4d_current.py"
HIGH="$BASE/models/high_model_final.pt"
LOW="$BASE/models/low_model_final.pt"
VIDEO="$BASE/videos/R874b_V4D_seed_${SEED}.mp4"
SUMMARY="$BASE/videos/R874b_V4D_seed_${SEED}.json"
TRAJECTORY="$BASE/videos/R874b_V4D_seed_${SEED}.csv"
READY="$VIDEO.ready"
LOG="$BASE/logs/render_seed_${SEED}.log"
STAGE="$BASE/logs/current_stage.txt"
RUN_LOG_DIR="$BASE/logs/run_seed_${SEED}"

for file in "$PLAYER" "$HIGH" "$LOW"; do
  [[ -s "$file" ]] || {
    echo "R874B_LOCAL_FATAL=missing:$file" >&2
    exit 21
  }
done
[[ -d "$PROJECT/source/isaac_go2_low_level_rl" ]] || {
  echo "R874B_LOCAL_FATAL=project_source_missing:$PROJECT" >&2
  exit 22
}

mkdir -p "$BASE/videos" "$BASE/logs" "$BASE/runtime/home" \
  "$BASE/runtime/cache" "$BASE/runtime/config" "$BASE/runtime/data" \
  "$BASE/runtime/tmp" "$BASE/runtime/mpl" "$RUN_LOG_DIR"

rm -f "$VIDEO" "$SUMMARY" "$TRAJECTORY" "$READY" "$LOG" "$STAGE"

echo "PRECHECK" > "$STAGE"
echo "============================================================"
echo " R8.74b V4D SINGLE-SEED LOCAL RENDER"
echo "============================================================"
echo "BASE=$BASE"
echo "PROJECT=$PROJECT"
echo "CONDA_ENV=$CONDA_ENV"
echo "SEED=$SEED EVAL_STEPS=$EVAL_STEPS FPS=$FPS"

conda run --no-capture-output -n "$CONDA_ENV" python - <<'PY'
import torch
print("LOCAL_TORCH_CUDA_AVAILABLE=", torch.cuda.is_available(), sep="")
if not torch.cuda.is_available():
    raise SystemExit("R874B_LOCAL_FATAL=cuda_unavailable")
print("LOCAL_GPU=", torch.cuda.get_device_name(0), sep="")
PY

(
  cd "$BASE"
  sha256sum -c SHA256SUMS
)

echo "ISAAC_LAUNCH" > "$STAGE"
START="$(date +%s)"

export PYTHONPATH="$PROJECT/source/isaac_go2_low_level_rl:$PROJECT/source:${PYTHONPATH:-}"
export HOME="$BASE/runtime/home"
export XDG_CACHE_HOME="$BASE/runtime/cache"
export XDG_CONFIG_HOME="$BASE/runtime/config"
export XDG_DATA_HOME="$BASE/runtime/data"
export TMPDIR="$BASE/runtime/tmp"
export MPLCONFIGDIR="$BASE/runtime/mpl"
export PYTHONUNBUFFERED=1
export QT_QPA_PLATFORM=offscreen
export R873_HEADLESS_STRIP=1
export V4D_HIGH_CHECKPOINT="$HIGH"

set +e
timeout --foreground --kill-after=30s "${TIMEOUT_MINUTES}m" \
  conda run --no-capture-output -n "$CONDA_ENV" \
  python "$PLAYER" \
    --headless \
    --enable_cameras \
    --num_envs 1 \
    --seed "$SEED" \
    --log_dir "$RUN_LOG_DIR" \
    --run_name "r874b_v4d_video_seed_${SEED}" \
    --eval_video "$VIDEO" \
    --eval_steps "$EVAL_STEPS" \
    --eval_fps "$FPS" \
    --trajectory_csv "$TRAJECTORY" \
    --eval_summary "$SUMMARY" \
    --low_level_checkpoint "$LOW" \
    --trained_low_checkpoint "$LOW" \
  > >(tee "$LOG") 2>&1 &
PID=$!

while kill -0 "$PID" 2>/dev/null; do
  NOW="$(date +%s)"
  ELAPSED="$((NOW - START))"
  LOG_SIZE="$(stat -c %s "$LOG" 2>/dev/null || echo 0)"
  VIDEO_SIZE="$(stat -c %s "$VIDEO" 2>/dev/null || echo 0)"

  if grep -qF 'V4D_EVAL_VIDEO_OK' "$LOG" 2>/dev/null; then
    CURRENT="VIDEO_FINALIZED"
  elif grep -qE 'V4D_EVAL|evaluation' "$LOG" 2>/dev/null; then
    CURRENT="EVALUATION_LOOP"
  elif grep -qF 'Completed setting up the environment' "$LOG" 2>/dev/null; then
    CURRENT="ENVIRONMENT_READY"
  elif grep -qF '[V4D_TASK_REGISTRATION_OK]' "$LOG" 2>/dev/null; then
    CURRENT="SCENE_CREATION"
  elif grep -qF '[PHASEA_BOOT]' "$LOG" 2>/dev/null; then
    CURRENT="TASK_IMPORT"
  else
    CURRENT="ISAAC_APP_STARTUP"
  fi

  printf '%s\n' "$CURRENT" > "$STAGE"
  echo "[R874B_WATCH] elapsed=${ELAPSED}s stage=$CURRENT log_bytes=$LOG_SIZE video_bytes=$VIDEO_SIZE"
  sleep 30
done

wait "$PID"
RC=$?
set -e

echo "PROCESS_EXIT rc=$RC" > "$STAGE"
echo "R874B_RENDER_RC=$RC"

if [[ "$RC" -eq 124 || "$RC" -eq 137 ]]; then
  echo "R874B_LOCAL_FATAL=timeout"
  bash "$BASE/check_local.sh" || true
  exit "$RC"
fi

if [[ "$RC" -ne 0 ]]; then
  echo "R874B_LOCAL_FATAL=process_exit_$RC"
  bash "$BASE/check_local.sh" || true
  exit "$RC"
fi

grep -qF 'V4D_EVAL_VIDEO_OK' "$LOG" || {
  echo "R874B_LOCAL_FATAL=missing_success_marker"
  bash "$BASE/check_local.sh" || true
  exit 31
}

[[ -s "$VIDEO" && -s "$SUMMARY" && -s "$TRAJECTORY" ]] || {
  echo "R874B_LOCAL_FATAL=missing_output_artifact"
  bash "$BASE/check_local.sh" || true
  exit 32
}

printf '%s\n' "seed=$SEED" "video=$VIDEO" > "$READY"
echo "COMPLETED" > "$STAGE"
echo "R874B_LOCAL_RENDER_OK"
bash "$BASE/check_local.sh"
