# R8.57 balanced randomized hold-controller scan

R8.57 corrects the invalid R8.56 fixed environment-to-case assignment. Every
repeat uses a shuffled complete permutation of all 98 candidates, so every
candidate receives exactly the same number of episodes and rotates across
parallel environments. A runtime balance assertion prevents unequal counts.

It compares static latched targets and live Recovery targets with the same FL
thigh/calf grid. Winners must pass two independent confirmation seeds. The
checker reports `recovery3 | touchdown`, directly measuring contact retention.

```bash
cd /scratch/lj2634/Isaac-Go2-Low-Level-RL
mkdir -p experiments
tar -xzf /scratch/lj2634/r8_57_balanced_hold_scan_20260812.tgz -C experiments/
bash experiments/r8_57_balanced_hold_scan/preflight_r857.sh

R857_NUM_ENVS=128 \
R857_EPISODES_PER_CASE=64 \
R857_CONFIRM_EPISODES=512 \
sbatch experiments/r8_57_balanced_hold_scan/run_r857_scan.sbatch
```

```bash
bash experiments/r8_57_balanced_hold_scan/check_r857_latest.sh
```

Do not build or render video unless the checker reports
`COMPLETED_READY_TO_BUILD_VIDEO`.

After that verdict:

```bash
bash experiments/r8_57_balanced_hold_scan/build_r857_video_bundle.sh
bash experiments/r8_57_balanced_hold_scan/check_r857_video_bundle.sh
```

Download the archive printed by the builder through
`dtn.torch.hpc.nyu.edu`, extract it on Ubuntu, then run `bash render_local.sh`
and `bash check_local_videos.sh`. The exact accepted static or dynamic
controller mode is preserved in the winner JSON and video player.
