#!/usr/bin/env bash
set -euo pipefail

PROJECT="${SOURCE_REPO:-$HOME/Isaac-Go2-Soccer-kick-anchored}"
OUTPUT_DIR="$PROJECT/outputs/kick_behavior_local/videos/k2a3_evidence"
VIDEO="$OUTPUT_DIR/world_anchored_motion_prior.mp4"
PREVIEW="$OUTPUT_DIR/world_anchored_motion_prior_preview.png"
LOG="/tmp/k2a3_video_probe.log"
SUMMARY="$PROJECT/outputs/kick_behavior_local/k2a3_video_execution_probe.json"

export DISPLAY="${DISPLAY:-:0}"
export PYTHONPATH="$PROJECT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "$OUTPUT_DIR"
rm -f "$VIDEO" "$PREVIEW" "$LOG" "$SUMMARY"

for command_name in ffmpeg ffprobe xdotool; do
    command -v "$command_name" >/dev/null 2>&1 || {
        echo "[K2A3 VIDEO] Missing command: $command_name"
        echo "Install with: sudo apt-get install ffmpeg xdotool"
        exit 2
    }
done

PROBE="$(
    find "$PROJECT/scripts/kick_behavior" \
        -maxdepth 1 \
        -type f \
        \( -name 'probe_k2a3_policy_execution.py' -o -name '*k2a3*execution*probe*.py' \) \
        | sort \
        | head -n 1
)"

CKPT="$(
    find "$PROJECT/outputs/kick_behavior_training/go2_anchored_swing_kick_prior" \
        -type f \
        -name 'model_*.pt' \
        -printf '%T@ %p\n' \
        | sort -n \
        | tail -n 1 \
        | cut -d' ' -f2-
)"

RUNNER="$PROJECT/scripts/reporting/run_k2a3_probe_gui.py"

echo "[K2A3 VIDEO] probe=$PROBE"
echo "[K2A3 VIDEO] checkpoint=$CKPT"

test -f "$PROBE"
test -f "$CKPT"
test -f "$RUNNER"

cd "$PROJECT"

python "$RUNNER" \
    --target-script "$PROBE" \
    --eye-offset "0.82,-0.82,0.10" \
    --target-offset "0.08,-0.03,-0.20" \
    --resolution "1280,720" \
    --min-step-seconds 0.05 \
    -- \
    --checkpoint "$CKPT" \
    --steps 280 \
    --log-every 5 \
    --output "$SUMMARY" \
    >"$LOG" 2>&1 &

SIM_PID=$!

cleanup() {
    if kill -0 "$SIM_PID" 2>/dev/null; then
        kill -INT "$SIM_PID" 2>/dev/null || true
        sleep 2
        kill -TERM "$SIM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

WINDOW_ID=""
for _ in $(seq 1 240); do
    if ! kill -0 "$SIM_PID" 2>/dev/null; then
        echo "[K2A3 VIDEO] Probe exited before recording."
        tail -n 120 "$LOG"
        exit 3
    fi

    WINDOW_ID="$(
        xdotool search --onlyvisible --name 'Isaac Sim' 2>/dev/null \
        | tail -n 1 || true
    )"

    if [[ -n "$WINDOW_ID" ]] \
        && grep -Eq '\[K2A3 EXEC\].*env_step_complete' "$LOG"; then
        break
    fi
    sleep 0.25
done

if [[ -z "$WINDOW_ID" ]]; then
    echo "[K2A3 VIDEO] Isaac Sim window not found."
    tail -n 120 "$LOG"
    exit 4
fi

if ! grep -Eq '\[K2A3 EXEC\].*env_step_complete' "$LOG"; then
    echo "[K2A3 VIDEO] Policy motion did not begin."
    tail -n 120 "$LOG"
    exit 5
fi

xdotool windowactivate --sync "$WINDOW_ID" 2>/dev/null || true
xdotool windowraise "$WINDOW_ID" 2>/dev/null || true
sleep 0.5

eval "$(xdotool getwindowgeometry --shell "$WINDOW_ID")"

X="${X:-0}"
Y="${Y:-0}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"

(( X < 0 )) && X=0
(( Y < 0 )) && Y=0
WIDTH=$(( WIDTH / 2 * 2 ))
HEIGHT=$(( HEIGHT / 2 * 2 ))

echo "[K2A3 VIDEO] recording=${WIDTH}x${HEIGHT}+${X}+${Y}"

ffmpeg \
    -hide_banner \
    -loglevel warning \
    -y \
    -f x11grab \
    -draw_mouse 0 \
    -framerate 30 \
    -video_size "${WIDTH}x${HEIGHT}" \
    -i "${DISPLAY}+${X},${Y}" \
    -t 10 \
    -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
    -c:v libx264 \
    -preset veryfast \
    -crf 19 \
    -pix_fmt yuv420p \
    -movflags +faststart \
    "$VIDEO"

ffmpeg \
    -hide_banner \
    -loglevel error \
    -y \
    -ss 3 \
    -i "$VIDEO" \
    -frames:v 1 \
    "$PREVIEW"

wait "$SIM_PID" || true
trap - EXIT INT TERM

echo "[K2A3 VIDEO] created=$VIDEO"
echo "[K2A3 VIDEO] preview=$PREVIEW"
echo "[K2A3 VIDEO] summary=$SUMMARY"

ls -lh "$VIDEO" "$PREVIEW" "$SUMMARY"

ffprobe \
    -v error \
    -show_entries stream=width,height,r_frame_rate \
    -show_entries format=duration,size \
    -of default=noprint_wrappers=1 \
    "$VIDEO"

grep -E '\[K2A3 EXEC RESULT\]' "$LOG" | tail -n 3 || true
