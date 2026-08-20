#!/usr/bin/env bash
set -Eeuo pipefail
POINTER="/scratch/lj2634/video_handoffs/latest_r856_video_bundle.txt"
[[ -s "$POINTER" ]]||{ echo VERDICT=VIDEO_BUNDLE_NOT_READY; exit 2; }
ARCHIVE="$(tr -d '\r\n' < "$POINTER")"; echo "ARCHIVE=$ARCHIVE"; stat --printf='SIZE=%s bytes MTIME=%y\n' "$ARCHIVE"; echo "SHA256=$(sha256sum "$ARCHIVE"|awk '{print $1}')"; echo '===== CONTENTS ====='; tar -tzf "$ARCHIVE"; for f in bundle/player/train_r856_hold_bias.py bundle/models/recovery64_final.pt bundle/models/kick64_final.pt bundle/evidence/r856_hold_winner.json bundle/evidence/r856_summary.json bundle/render_local.sh bundle/check_local_videos.sh bundle/SHA256SUMS; do tar -tzf "$ARCHIVE"|grep -qx "$f"||{ echo "MISSING=$f"; exit 3; }; done; echo VERDICT=VIDEO_BUNDLE_READY_FOR_DTN_DOWNLOAD
