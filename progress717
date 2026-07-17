# Current Progress and Outstanding Issues for the Unitree Go2 Soccer Project

**Author:** Lishengqi Jiang  
**Date:** 07/17/2026  

---

## 1. Current Overall Status

The project has moved beyond the ball-free kick-motion-prior stage and now has a low-level policy that can produce repeatable physical ball motion through simulated foot–ball contact.

The most important current conclusion is:

> The K2C kick checkpoint can physically move the ball when it is evaluated as a closed-loop policy in its native Isaac Lab task, but this policy has not yet been correctly integrated into the complete 1v0 soccer environment.

The current system should therefore be described as having:

- validated locomotion and ball-approach behavior;
- a usable right-front-leg physical kick baseline;
- audited K2C contact metrics and playback tools;
- simulator-state-based ball localization;
- an incomplete transition between locomotion control and the closed-loop K2C policy;
- no completed end-to-end physical 1v0 goal-scoring result yet;
- no completed Franklin PPO long-training run on the cluster yet.

---

## 2. Progress Since the Previous Report

### 2.1 K2C physical metrics were implemented and tested

The K2C logging path was revised so that physical contact and episode-level results can be audited rather than inferred from total PPO reward.

The revised implementation covers:

- episode-level accumulation;
- done-mask handling;
- first-contact phase classification;
- right-front-foot contact attribution;
- wrong-body contact detection;
- forward and backward ball-motion measurements;
- target-direction projection;
- base-contact and recovery measurements;
- logging through the RSL-RL-compatible interface.

The associated unit test suite completed successfully:

```text
19 passed, 1 warning
```

This confirms the pure metric and aggregation logic, although simulator playback remains necessary because unit tests cannot verify physical realism.

### 2.2 Contact behavior was inspected through playback

The K2C playback path records contact forces, first-contact phase, ball velocity, ball displacement, base contact, and episode classification.

One audited episode recorded:

```text
first contact phase: approximately 0.529
first contact body: designated kick foot
early contact: false
strike contact: true
wrong-body contact: false
backward-ball event: false
base contact: false
```

The ball reached approximately:

```text
forward velocity: 0.68 m/s
forward displacement: 0.38 m during the displayed interval
```

This playback showed that the right-front foot could make a directional strike without a detected wrong-body or backward-drag event.

The earlier episode-level `success` flag remained zero under an older evaluation rule, which revealed that different scripts were using different success definitions. A unified release-level acceptance rule is still required.

### 2.3 Several reward and trajectory variants were evaluated

Multiple short K2C training and playback experiments were used to distinguish:

- clean strike-like contact;
- long push-like contact;
- premature contact;
- backward dragging;
- contact release;
- post-contact recovery;
- base instability.

These experiments showed that a high mean reward alone was not enough to establish a useful kick. Video inspection was adopted as a required step after each meaningful training or integration test.

The project also reached a practical decision regarding motion quality:

> The current kick should be frozen as a functional baseline rather than delaying the entire project to make the leg motion more natural.

The remaining naturalness, retraction, and recovery improvements can be revisited after the complete soccer pipeline is operational.

### 2.4 Open-loop action-macro integration was tested and rejected

An action macro was extracted from the kick behavior and replayed inside the 1v0 soccer controller. This approach did not preserve the behavior of the trained K2C policy.

Observed failures included:

- sudden joint-target discontinuities when switching from locomotion;
- backward body motion and large whole-body oscillation;
- occasional side falls;
- repeated macro triggering;
- stopping near the ball without making contact;
- continuing forward after passing the ball;
- failure to reacquire a ball that moved behind the robot.

Additional gating based on foot distance, body stability, tilt, velocity, and settling time did not solve the core problem.

The root cause was that the trained K2C behavior is a closed-loop policy. Its action at each step depends on its current 60-dimensional observation. Replaying a prerecorded sequence removes this feedback and starts the motion from states that differ from the original K2C training distribution.

The open-loop macro route has therefore been discontinued.

### 2.5 True closed-loop K2C evaluation succeeded

The actual K2C checkpoint was then evaluated by rebuilding its observation and calling the policy at every low-level control step.

The verified checkpoint was:

```text
model_1516.pt
```

The evaluation explicitly confirmed:

```text
policy inference at every step: enabled
frozen action macro: disabled
direct ball-velocity write: disabled
```

Three consecutive single-environment episodes completed successfully:

| Measurement | Result |
|---|---:|
| Completed episodes | 3 |
| Successful episodes | 3 |
| Success rate | 1.000 |
| Peak forward ball velocity | 0.721–0.840 m/s |
| Maximum forward ball displacement | 0.787–0.899 m |
| Minimum right-front-foot-to-ball distance | 0.116 m |
| Maximum body tilt | 8.3° |
| Maximum recorded base angular speed | approximately 3.07 rad/s |

This result establishes that the checkpoint can generate real ball motion through the physical simulation when used in the control setting for which it was trained.

It also establishes that the previous 1v0 failures were integration failures rather than proof that the K2C checkpoint was incapable of kicking.

### 2.6 Ball localization and scripted approach behavior are available

The current simulator can read:

- robot world pose;
- ball world pose;
- robot-relative ball position;
- ball velocity;
- goal position and direction.

The scripted 1v0 baseline can use this simulator state to:

- locate the ball;
- approach it;
- move toward a point behind it;
- orient the robot approximately toward the opposing goal.

This is privileged simulator-state localization, not vision.

The current project does not yet contain a complete perception pipeline of the form:

```text
camera image
-> ball detection
-> distance and bearing estimation
-> tracking
-> lost-ball search
```

Using simulator state is appropriate for the current control-integration stage because it isolates locomotion, positioning, skill switching, and physical contact from perception errors.

---

## 3. Current System Capability

| Component | Current Status |
|---|---|
| Go2 locomotion policy | Working |
| Forward, backward, and turning commands | Validated |
| Scripted approach to a stationary ball | Working |
| Positioning behind the ball | Partially working |
| Goal-direction alignment | Partially working |
| Physical right-front-leg kick in native K2C task | Working |
| K2C metrics and unit tests | Working |
| K2C playback and physical audit | Working |
| Open-loop kick macro in soccer environment | Rejected |
| Closed-loop K2C policy inside soccer environment | Not yet integrated |
| Franklin 86-dimensional observation contract | Defined, final physical-state audit incomplete |
| Franklin 13-action interface | Defined |
| Actions 0–4 to locomotion | Connected |
| Actions 5–12 to true closed-loop kick | Not completed |
| Complete physical 1v0 goal-scoring demonstration | Not completed |
| Franklin PPO long training on cluster | Not started |
| Camera-based ball search | Not implemented |

---

## 4. Main Outstanding Problems

### 4.1 Closed-loop skill integration is the primary blocker

The next implementation must run the K2C policy as a genuine sub-policy rather than converting it into a fixed joint-action sequence.

The required control path is:

```text
soccer locomotion controller
-> reach a valid kick-ready state
-> construct the native K2C observation
-> call model_1516.pt at every low-level step
-> apply its 12 joint actions with the correct scaling and joint order
-> detect completion or failure
-> return control to locomotion
```

The soccer environment must reproduce the K2C policy inputs consistently, including:

- joint position and velocity terms;
- projected gravity and body motion;
- ball position and velocity in the robot frame;
- motion phase;
- previous action or observation history where required;
- normalization and action scaling;
- reset state and low-level controller state.

A mismatch in any of these fields can make the checkpoint behave differently from the validated native-task playback.

### 4.2 Locomotion-to-kick transition state is not standardized

The kick policy was validated from the initial states produced by its own task. In the soccer pipeline, the robot reaches the ball while controlled by the locomotion policy.

The two policies may disagree on:

- nominal joint posture;
- gait phase;
- previous action;
- observation history;
- base velocity;
- support-foot loading;
- body height and tilt.

A direct switch can therefore create an out-of-distribution initial condition.

A kick-ready transition must define measurable entry conditions such as:

```text
right-front-foot geometry within the trained range
low base linear velocity
low yaw and roll/pitch angular velocity
acceptable body tilt and base height
stable support contact
short settling period
valid ball bearing and lateral offset
```

However, the transition should not replace feedback control with a prerecorded macro. Its purpose is only to place the closed-loop K2C policy inside a reasonable initial-state region.

### 4.3 The kick policy has limited demonstrated robustness

The current successful validation uses the native K2C setup and a relatively narrow initial ball configuration.

It has not yet been established that the checkpoint remains reliable under:

- errors in ball distance;
- errors in ball lateral offset;
- different robot yaw;
- residual locomotion velocity;
- different support configurations;
- field-surface differences;
- repeated kicks in the same episode;
- ball motion before the kick begins.

The first integration should therefore preserve a narrow trigger region and use the scripted controller to position the robot carefully. Wider randomization should follow only after the native closed-loop policy works in the soccer scene.

### 4.4 Success criteria are not yet unified

Different scripts have used different meanings of `success`, including combinations of:

- strike-phase contact;
- positive ball velocity;
- forward displacement;
- contact release;
- clean-strike classification;
- recovery success.

This produced cases in which playback visibly moved the ball forward but an older episode flag still reported `success=0`.

The project needs one release-level definition, for example:

```text
designated-foot strike contact
and peak forward ball velocity above threshold
and forward displacement above threshold
and no wrong-body contact
and no excessive backward motion
and no base contact
```

Recovery may be reported separately or included as an additional requirement, but the definition must be identical in local validation, cluster training, checkpoint release, and report figures.

### 4.5 Batch validation and formal freezing are still pending

The three-episode closed-loop result is strong evidence, but it is not yet the final release gate.

The planned formal freeze requires:

```text
8 environments × 3 episodes = 24 completed episodes
```

The release procedure should preserve:

- checkpoint;
- YAML configuration;
- policy runner;
- code revision;
- working-tree state;
- SHA256 checksums;
- batch metrics;
- validation log;
- representative playback video.

Until this batch gate is run and recorded, `model_1516.pt` should be treated as the current candidate baseline rather than an immutable final release.

### 4.6 End-to-end goal scoring has not been demonstrated

The project has separately demonstrated:

```text
locomotion can approach the ball
K2C can physically kick the ball
```

It has not yet demonstrated the complete sequence:

```text
spawn in own half
-> find the ball
-> move behind the ball
-> align with the goal
-> switch to closed-loop K2C
-> physically strike the ball
-> reacquire the ball if necessary
-> score
```

The current kick produces approximately 0.7–0.8 m/s peak forward ball speed in the verified trials. Depending on field friction and goal distance, one kick may not be sufficient from midfield. The first end-to-end baseline should allow repeated approach-and-kick cycles rather than requiring a single long-range shot.

### 4.7 The Franklin high-level training path is not ready for a long cluster run

The Franklin-compatible 86-dimensional observation and 13-action interface remain the intended high-level contract, but several items must be completed before long training:

- connect actions 5–12 to the true closed-loop K2C skill;
- ensure actions 0–4 and kick actions use consistent high-level timing;
- validate all dynamic observation fields against physical state;
- remove or bypass any remaining command-based ball-state proxy in the official path;
- define action masking or failed-kick semantics;
- verify reset behavior across both low-level policies;
- confirm that ball motion is never assigned directly;
- complete a local multi-episode end-to-end test.

Starting a cluster PPO run before these items are complete would mainly train against integration defects rather than the intended soccer task.

### 4.8 Perception remains outside the current implementation

The robot currently knows the ball location through simulator truth. It does not yet actively scan with a camera.

This is acceptable for the first 1v0 control baseline, but the report should clearly distinguish:

```text
simulator-state localization
```

from:

```text
visual ball detection and search
```

A visual perception module should be treated as a later extension rather than a prerequisite for the present physical-control milestone.

---

## 5. Immediate Next Steps

The recommended immediate sequence is:

```text
1. Run the 24-episode native K2C batch validation.
2. Freeze model_1516.pt together with its config, runner, code state, and checksums.
3. Implement a reusable native K2C observation-and-inference adapter.
4. Integrate the true closed-loop policy into the soccer scene.
5. Add a controlled locomotion-to-kick transition and post-kick return.
6. Run one-environment GUI playback and inspect the complete episode.
7. Test repeated approach–kick–reacquire cycles.
8. Demonstrate at least one physical goal without direct ball-velocity assignment.
9. Validate the full 86-dimensional Franklin observation.
10. Only then prepare the high-level PPO cluster training run.
```

The first end-to-end acceptance gate should require:

- the robot starts away from the ball;
- ball and robot positions are not identical at reset;
- ball localization comes from the current allowed simulator-state input;
- locomotion reaches a valid position behind the ball;
- the true K2C checkpoint is evaluated every low-level step;
- no prerecorded action macro is used;
- no direct ball-velocity assignment is used;
- the robot remains upright;
- the ball moves in the goal direction;
- the controller can reacquire the ball after an incomplete shot;
- the complete GUI playback is reviewed before scaling up.

---

## 6. Current Conclusion

The project has achieved a meaningful low-level milestone: a closed-loop Unitree Go2 policy can physically strike the ball in Isaac Lab and produce repeatable forward motion without directly modifying ball velocity.

The main problem is no longer whether the kick checkpoint can move the ball. The main problem is integrating that checkpoint into the soccer controller while preserving its observation, feedback, action scaling, and valid initial-state distribution.

The project is therefore ready to proceed to closed-loop skill integration and end-to-end scripted 1v0 validation, but it is not yet ready for the final Franklin PPO long-training run.
