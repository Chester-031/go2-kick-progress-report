#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT="/scratch/lj2634/Isaac-Go2-Low-Level-RL"; EXP="$PROJECT/experiments/r8_56_hold_bias_video"; VIDEO_ROOT="/scratch/lj2634/video_handoffs"; RUN="$(tr -d '\r\n' < /scratch/lj2634/runs/go2_soccer_r8_56_hold_bias/latest_r856.txt)"; SUMMARY="$RUN/output/r856_summary.json"
[[ -s "$SUMMARY" ]]||{ echo VIDEO_BUNDLE_FATAL=no_r856_summary >&2; exit 32; }
python - "$SUMMARY" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
if not d.get('accepted'): raise SystemExit('VIDEO_BUNDLE_FATAL=r856_not_accepted')
print('R856_NUMERIC_GATE=ACCEPTED')
PY
R855="$(tr -d '\r\n' < /scratch/lj2634/runs/go2_soccer_r8_55_touchdown_hold/latest_r855.txt)"; R851="$(tr -d '\r\n' < /scratch/lj2634/runs/go2_soccer_r8_51_kick64_true_contact/latest_r851.txt)"; R850="$(tr -d '\r\n' < "$PROJECT/experiments/r8_50_real_contact_truth/latest_r850.txt")"; FREEZE="/scratch/lj2634/runs/go2_soccer_kick_freeze/ballfree_r841_grace6_20260810_185244"; STAMP="$(date +%Y%m%d_%H%M%S)"; OUT="$VIDEO_ROOT/r856_hold_video_${STAMP}"; B="$OUT/bundle"; mkdir -p "$B/player" "$B/models" "$B/evidence" "$B/ballfree/artifacts" "$B/videos" "$B/local_logs"
cp "$RUN/train_r856_hold_bias.py" "$B/player/train_r856_hold_bias.py"; cp "$R855/output/recovery64_final.pt" "$B/models/recovery64_final.pt"; cp "$R851/output/kick64_final.pt" "$B/models/kick64_final.pt"; cp "$R851/output/r851_summary.json" "$B/evidence/r851_summary.json"; cp "$R850/body_names.json" "$B/evidence/body_names.json"; cp "$RUN/output/r856_hold_winner.json" "$B/evidence/r856_hold_winner.json"; cp "$SUMMARY" "$B/evidence/r856_summary.json"; cp "$EXP/render_r856_local.sh" "$B/render_local.sh"; cp "$EXP/check_r856_local_videos.sh" "$B/check_local_videos.sh"; chmod +x "$B/render_local.sh" "$B/check_local_videos.sh"
for f in r821_lift_base.pt r823_mode1_lift_head.pt r815_swing_base.pt r835_swing_stability_u32.pt r827_recovery_mode1_head.pt r839_recovery_vector_u32.pt; do cp "$FREEZE/artifacts/$f" "$B/ballfree/artifacts/$f"; done
{
 echo 'R8.56 exact numeric-winner video bundle'; echo "source_run=$RUN"; echo 'local_conda=isaaclab232'; echo 'local_project=/home/ubuntu/下载/go2_soccer_phaseA_results/project/Isaac-Go2-Low-Level-RL'; echo 'seed=42'; echo 'geometry=R851 accepted top geometry'; echo 'videos=0.75,1.0,1.25';
} > "$B/manifest.txt"; (cd "$B" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS); ARCHIVE="$OUT/r856_hold_video_bundle_${STAMP}.tgz"; tar -czf "$ARCHIVE" -C "$OUT" bundle; printf '%s\n' "$ARCHIVE" > "$VIDEO_ROOT/latest_r856_video_bundle.txt"; echo R856_VIDEO_BUNDLE_READY; echo "ARCHIVE=$ARCHIVE"; sha256sum "$ARCHIVE"; echo "DOWNLOAD_WITH=ssh lj2634@dtn.torch.hpc.nyu.edu 'cat $ARCHIVE' > r856_hold_video_bundle.tgz"
