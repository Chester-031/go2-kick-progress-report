# R8.56 FL hold-bias scan and exact-policy video gate

R8.56 freezes the complete R8.55 network chain. It resolves the actual FL hip,
thigh and calf action indices from runtime joint names, scans a bounded 7x7
thigh/calf joint-target bias grid only after valid touchdown, and confirms one
winner independently for each requested strength.

No router or high-level PPO is started here. If all numerical gates pass, the
bundle builder packages the exact winner, source, checkpoints and evidence for
the previously verified local Ubuntu video route (`isaaclab232`, RGB-array,
imageio, `--enable_cameras`, `--eval_video`).

## Numeric scan

```bash
cd /scratch/lj2634/Isaac-Go2-Low-Level-RL
mkdir -p experiments
tar -xzf /scratch/lj2634/r8_56_hold_bias_video_20260811.tgz -C experiments/
bash experiments/r8_56_hold_bias_video/preflight_r856.sh

R856_NUM_ENVS=128 \
R856_EPISODES_PER_CASE=64 \
R856_CONFIRM_EPISODES=1024 \
sbatch experiments/r8_56_hold_bias_video/run_r856_scan.sbatch
```

No-job-ID numeric checker:

```bash
bash experiments/r8_56_hold_bias_video/check_r856_latest.sh
```

## Build and verify the video handoff

Run only after the checker reports `COMPLETED_READY_TO_BUILD_VIDEO`:

```bash
bash experiments/r8_56_hold_bias_video/build_r856_video_bundle.sh
bash experiments/r8_56_hold_bias_video/check_r856_video_bundle.sh
```

Download through the verified DTN host using the exact archive path printed by
the builder. Extract locally and run:

```bash
tar -xzf r856_hold_video_bundle.tgz
cd bundle
bash render_local.sh
bash check_local_videos.sh
```

Human review must confirm visible FL-ball impact, distinct monotonic ball travel
across 0.75/1.0/1.25, stable touchdown, no body-ball collision, and no obvious
post-kick instability before any router work begins.
