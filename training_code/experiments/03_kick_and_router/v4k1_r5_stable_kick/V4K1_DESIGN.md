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
