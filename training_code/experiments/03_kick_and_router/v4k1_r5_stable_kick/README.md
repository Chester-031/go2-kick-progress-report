# Go2 Soccer V4K1-R5 — Stable-Stance Isolated Delta-Pose Kick

R5 replaces the rejected reset-fall solution. It trains one left-front-foot kick while preserving the existing project direction:

- no high-level policy;
- no soccer goal or own-goal reward;
- direct planar Delta Pose context;
- 70-D teacher-free low-level observation;
- 12 joint-position residual actions;
- V4D low policy warm start;
- router integration deferred until video acceptance.

## Physical reset contract

Every episode now follows this order:

1. Reset Go2 with the ball parked far away.
2. Disable learned actions and wait for stable ground support.
3. Require low vertical/horizontal speed, small roll/pitch error, valid base height, and at least three supporting feet for consecutive steps.
4. Place the ball from the actual selected front-foot position.
5. Keep actions disabled during a short ball-arming window.
6. Start the kick phase only after both robot and ball are stable.

A reset timeout or ball motion before activation ends the episode as a failure.

## Valid Kick contract

A contact is valid only when all conditions hold:

- Kick phase is active;
- selected left front foot lifted and swung forward;
- Go2 is stable at contact;
- selected foot is the valid contact source;
- wrong foot and body are not crowding the ball;
- ball did not move before selected-foot contact.

A successful Kick additionally requires a short contact, directional speed and displacement, low base translation, no violation, and stable post-kick recovery.

## Submission chain

- Preflight: 32 envs × 3 PPO iterations, including the physical stable-reset guard.
- Pilot: 128 envs × 150 iterations.
- Full: 512 envs × 600 iterations.

The installer runs a collect-all package audit before copying or submitting anything, then repeats the audit on the installed source. Any error prevents GPU submission.

```bash
cd /scratch/lj2634
rm -rf go2_soccer_v4k1_r5_stable_kick
tar -xzf go2_soccer_v4k1_r5_stable_kick.tgz
cd go2_soccer_v4k1_r5_stable_kick
chmod +x *.sh *.sbatch
bash install_submit_v4k1_r5_stable_kick.sh
```

Monitor:

```bash
bash monitor_v4k1_r5_stable_kick.sh
```

Review after completion:

```bash
bash review_v4k1_r5_stable_kick.sh
```
