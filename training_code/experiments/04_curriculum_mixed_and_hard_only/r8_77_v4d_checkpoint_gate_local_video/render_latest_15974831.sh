#!/usr/bin/env bash
set -Eeuo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${PROJECT:-/home/ubuntu/下载/go2_soccer_phaseA_results/project/Isaac-Go2-Low-Level-RL}"
CONDA_ENV="${CONDA_ENV:-isaaclab232}"
SEED="${R877_VIDEO_SEED:-87701}"
STEPS="${R877_EVAL_STEPS:-1200}"
FPS="${R877_EVAL_FPS:-15}"
TIMEOUT_MINUTES="${R877_TIMEOUT_MINUTES:-40}"

PLAYER="$BASE/player/train_r876_video.py"
HIGH="/scratch/lj2634/runs/go2_soccer_r8_98_restore_native_command_probe/probe_15974831_20260818_170115/logs/high_model_final.pt"
LOW="/scratch/lj2634/runs/go2_soccer_r8_98_restore_native_command_probe/probe_15974831_20260818_170115/logs/low_model_final.pt"
VIDEO="$BASE/videos/R877_V4D_selected_seed_${SEED}.mp4"
SUMMARY="$BASE/videos/R877_V4D_selected_seed_${SEED}.json"
TRAJECTORY="$BASE/videos/R877_V4D_selected_seed_${SEED}.csv"
LOG="$BASE/logs/render_seed_${SEED}.log"
READY="$VIDEO.ready"
RUN_LOG="$BASE/logs/run_seed_${SEED}"

for FILE in "$PLAYER" "$HIGH" "$LOW" "$BASE/selection/r877_checkpoint_selection.json"; do
  [[ -s "$FILE" ]] || { echo "R877_LOCAL_FATAL=missing:$FILE"; exit 21; }
done
[[ -d "$PROJECT/source/isaac_go2_low_level_rl" ]] || {
  echo "R877_LOCAL_FATAL=project_source_missing:$PROJECT"; exit 22;
}
mkdir -p "$BASE"/{videos,logs,runtime/home,runtime/cache,runtime/config,runtime/data,runtime/tmp,runtime/mpl} "$RUN_LOG"

for FILE in "$VIDEO" "$SUMMARY" "$TRAJECTORY" "$LOG" "$READY"; do
  [[ ! -e "$FILE" ]] || unlink "$FILE"
done

echo "===== SELECTED CHECKPOINT ====="
python - "$BASE/selection/r877_checkpoint_selection.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding="utf-8"))
print("selected_label="+d["selected_label"])
print("selection_status="+d["selection_status"])
print("selected_metrics="+json.dumps(d["selected_metrics"],sort_keys=True))
PY

conda run --no-capture-output -n "$CONDA_ENV" python - <<'PY'
import torch
print("LOCAL_TORCH_CUDA_AVAILABLE=", torch.cuda.is_available(), sep="")
if not torch.cuda.is_available():
    raise SystemExit("R877_LOCAL_FATAL=cuda_unavailable")
print("LOCAL_GPU=", torch.cuda.get_device_name(0), sep="")
PY

(cd "$BASE" && sha256sum -c SHA256SUMS)

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
export R876_EVAL_FULL_SUPPORT=1
export R876_EVAL_CURRICULUM_ITERATION=300

set +e
timeout --foreground --kill-after=30s "${TIMEOUT_MINUTES}m" \
  conda run --no-capture-output -n "$CONDA_ENV" \
  python "$PLAYER" \
    --headless --enable_cameras --num_envs 1 --seed "$SEED" \
    --log_dir "$RUN_LOG" --run_name "r877_selected_seed_${SEED}" \
    --eval_video "$VIDEO" --eval_steps "$STEPS" --eval_fps "$FPS" \
    --trajectory_csv "$TRAJECTORY" --eval_summary "$SUMMARY" \
    --low_level_checkpoint "$LOW" --trained_low_checkpoint "$LOW" \
  > >(tee "$LOG") 2>&1
RC=$?
set -e
echo "R877_RENDER_RC=$RC"
[[ "$RC" -eq 0 ]] || { bash "$BASE/check_local.sh" || true; exit "$RC"; }
grep -qF '[R876_EVAL_DISTRIBUTION] phase=full_support' "$LOG"
grep -qF 'V4D_EVAL_VIDEO_OK' "$LOG"
[[ -s "$VIDEO" && -s "$SUMMARY" && -s "$TRAJECTORY" ]]
printf 'seed=%s\nvideo=%s\n' "$SEED" "$VIDEO" > "$READY"
bash "$BASE/check_local.sh"
