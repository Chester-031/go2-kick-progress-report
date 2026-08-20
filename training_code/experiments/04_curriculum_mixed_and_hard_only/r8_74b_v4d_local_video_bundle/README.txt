R8.74b V4D current-policy local video bundle adapter

Purpose
  Package the accepted current V4D source and checkpoints on Torch, then
  render one diagnostic video in the already-working local Isaac Sim setup.

This deliberately does NOT render on the cluster.  The cluster step only
copies immutable inputs, writes SHA256SUMS, and creates a tar archive.

Cluster usage
  cd /scratch/lj2634/Isaac-Go2-Low-Level-RL
  bash experiments/r8_74b_v4d_local_video_bundle/build_r874b_bundle.sh
  bash experiments/r8_74b_v4d_local_video_bundle/check_r874b_bundle.sh

Local download
  Read the archive path from:
    /scratch/lj2634/video_handoffs/latest_r874b_v4d_bundle.txt
  Download that archive through the Torch DTN.

Local rendering
  Extract the archive, cd into bundle, then run:
    PROJECT=/home/ubuntu/下载/go2_soccer_phaseA_results/project/Isaac-Go2-Low-Level-RL \
    CONDA_ENV=isaaclab232 \
    bash render_local.sh

Local checking
    bash check_local.sh

Expected final result
  VERDICT=R874B_LOCAL_VIDEO_READY_FOR_REVIEW

The first render intentionally uses one seed.  Do not expand to eight seeds
until this single-run gate succeeds.
