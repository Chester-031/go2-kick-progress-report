#!/usr/bin/env bash
set -Eeuo pipefail

POINTER="/scratch/lj2634/video_handoffs/latest_r874b_v4d_bundle.txt"
[[ -s "$POINTER" ]] || {
  echo "VERDICT=R874B_BUNDLE_POINTER_MISSING"
  exit 2
}

ARCHIVE="$(tr -d '\r\n' < "$POINTER")"
[[ -s "$ARCHIVE" ]] || {
  echo "VERDICT=R874B_BUNDLE_ARCHIVE_MISSING"
  exit 3
}

echo "============================================================"
echo " R8.74b V4D LOCAL-VIDEO BUNDLE CHECK"
echo "============================================================"
echo "ARCHIVE=$ARCHIVE"
stat --printf='SIZE=%s bytes MTIME=%y\n' "$ARCHIVE"
sha256sum "$ARCHIVE"

for file in \
  bundle/player/train_v4d_current.py \
  bundle/models/high_model_final.pt \
  bundle/models/low_model_final.pt \
  bundle/render_local.sh \
  bundle/check_local.sh \
  bundle/manifest.txt \
  bundle/SHA256SUMS
do
  tar -tzf "$ARCHIVE" "$file" >/dev/null
  echo "READY $file"
done

echo "VERDICT=R874B_BUNDLE_READY_FOR_DTN_DOWNLOAD"
