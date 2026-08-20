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


## R5.1 validated settle-budget repair

The reset diagnostic showed that the 75-step settle limit repeatedly timed out environments 6, 15, and 29, independently of seed. Extending only the settle limit to 150 steps produced 32/32 activation with zero settle timeouts and zero arming violations. R5.1 therefore changes only the validated settle budget, strengthens the preflight diagnostics, and uses fast process exit after success or fatal error to prevent Isaac Sim shutdown hangs. Soft arming remains disabled because it produced no diagnostic gain.


## R5.1 validated settle-budget repair

The completed reset diagnostic isolated the failure to the 75-step settling
budget. Environments 6, 15, and 29 timed out under both tested seeds. Soft
arming produced no improvement. Increasing only `settle_max_steps` to 150
produced 32/32 activation, zero settle timeouts, and zero arming violations.

R5.1 keeps strict arming unchanged, strengthens the reset guard diagnostics,
uses a 512-environment exact-scale preflight requiring 100% activation, and
uses direct process exit after success or fatal failure to avoid Isaac Sim
shutdown hangs.


## R5.2 temporal and metric repair

R5.1 correctly repaired the settling limit, and its 512-environment reset
guard passed. The later training smoke test nevertheless used only 60 rollout
steps per one-iteration curriculum stage. Every stage reset the environment,
so the rollout ended before the full settle, arm, kick, and completion path
could be observed.

The final guard also used `stable_activation_fraction`, which is a per-step
occupancy average and is not an episode success rate. R5.2 uses a 240-step
exact-scale preflight, requires all preflight environments to complete an
episode, and evaluates stable activation and invalid contact with the
episode-level fields already produced by training.


## R5.3 causal contact-discovery repair

R5.2 proved that stable reset and temporal coverage worked, but Pilot learned
no valid contact. The first-strike causality test was ordered incorrectly: ball
motion produced by the current action could be labeled as pre-contact motion.
Geometric proximity before a valid strike also accumulated contact duration and
created a permanent violation, while standing still retained positive reward.

R5.3 evaluates pre-contact motion from the before-action state, requires a
deliberate foot velocity toward the ball, starts contact duration only at the
first valid contact, removes the positive standing baseline in contact discovery,
and reports lift/swing/motion/contact rates separately.


## R5.3 comprehensive causal-contact repair

R5.2 passed the 512-environment stable-reset guard. Its Pilot failed because
the contact-causality implementation invalidated the first real ball response:
the code examined the post-action ball velocity as "precontact motion" before
accepting the selected-foot contact. It also converted harmless geometric
near-contact into a permanent violation and counted such proximity toward
contact duration. The stage-1 reward still paid a positive standing baseline.

R5.3 measures precontact motion from the before-action state, permits the
first causal selected-foot strike, requires deliberate foot velocity toward
the ball, and treats coarse wrong-foot/body proximity as a collision only
when it causes a simultaneous ball response. It reports lift, swing, motion
readiness, geometric reach, valid contact, passive response, precontact
motion, body response and wrong-foot response separately.

The installer runs the base validator and a second collect-all validator both
before and after installation. The collect-all validator checks Python and
shell syntax, train/play identity, AST decorator count, causal source order,
stage-1 reward semantics, submit-job argument arity, stage schedule sums,
runtime thresholds, warm starts and synthetic causal-contact predicates.
