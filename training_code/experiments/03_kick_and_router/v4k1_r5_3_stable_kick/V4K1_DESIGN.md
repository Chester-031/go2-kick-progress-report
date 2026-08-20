# V4K1-R5 Stable-Stance Kick Design

## Rejected R3 behavior

The earlier policy received Kick success when Go2 fell from its reset pose and a front leg happened to strike the ball. High contact, impact, speed, and displacement metrics therefore did not prove an active standing kick.

R5 rejects that mechanism at the environment and metric levels rather than attempting to repair it with looser thresholds or more PPO iterations.

## Locked project interface

- source policy: accepted V4D teacher-free direct Delta-Pose low policy;
- high-level soccer policy: absent;
- soccer goal reward: absent;
- low action: 12 joint-position residuals;
- low observation: 70 dimensions;
- legacy velocity-command columns: always zero;
- Kick context: phase, selected leg, direction, and strength;
- trained leg in R5: left front leg only.

The right leg is deliberately deferred until one physically valid left-leg Kick passes video review.

## State machine

### Settle

The ball is parked away from Go2. Actor output is ignored and zero residual actions are applied. Stability requires:

- world vertical speed below 0.05 m/s;
- horizontal speed below 0.12 m/s;
- roll/pitch angular speed below 0.18 rad/s;
- upright error below 0.012;
- base height within 0.22–0.45 m;
- at least three supporting feet.

### Place and arm

After consecutive stable steps, the ball is placed using the actual selected front-foot world position. A short arming window again applies zero actions. Ball motion or robot instability before activation invalidates the episode.

### Active Kick

The policy receives the same Delta-Pose and Kick context as before. Valid selected-foot contact requires prior lift and forward swing, stable support at contact, no wrong-foot or body collision, and no pre-contact ball motion.

### Recovery

A successful episode requires the selected foot to return, the base to remain upright, and the robot to recover stable support after impact.

## Anti-exploit acceptance

The final model cannot pass only by moving the ball. It must satisfy:

- stable activation rate;
- low violation rate;
- valid selected-foot contact rate;
- impact rate after valid contact;
- short-contact Kick success;
- base displacement limit;
- post-kick recovery;
- visual confirmation that the robot starts on the ground and performs a discrete front-leg strike.


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
