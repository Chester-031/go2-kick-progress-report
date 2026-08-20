R8.77 checkpoint gate and selected local video

1. Run preflight and the short, evaluation-only metric gate:
     bash experiments/r8_77_v4d_checkpoint_gate_local_video/preflight_r877.sh
     sbatch experiments/r8_77_v4d_checkpoint_gate_local_video/run_r877_checkpoint_gate.sbatch
     bash experiments/r8_77_v4d_checkpoint_gate_local_video/check_r877_latest.sh

2. Only after VERDICT=R877_READY_TO_BUILD_LOCAL_VIDEO_BUNDLE:
     bash experiments/r8_77_v4d_checkpoint_gate_local_video/build_r877_local_bundle.sh

3. Download the archive named in:
     /scratch/lj2634/video_handoffs/latest_r877_v4d_bundle.txt

4. Extract it locally, enter bundle, and run:
     PROJECT=/home/ubuntu/下载/go2_soccer_phaseA_results/project/Isaac-Go2-Low-Level-RL \
     CONDA_ENV=isaaclab232 bash render_local.sh

5. Check at any time:
     bash check_local.sh

The gate evaluates high_model_199.pt, high_model_449_joint_finetune.pt and
high_model_final.pt on the same deterministic R8.75 cases with the same exact
accepted low checkpoint. It produces no cluster video. The bundle contains
only the selected high checkpoint and forces local rendering onto stationary
full-support resets. Read EXPECTED_BEHAVIOR.txt before reviewing the video.
