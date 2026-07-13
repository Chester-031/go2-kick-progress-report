# Learning a Physical Contact Kick Skill for a Unitree Go2 Soccer Agent

**Project:** Isaac Go2 Soccer  
**Report Type:** Research Progress Report  
**Current Stage:** Low-Level Physical Kick Skill Learning  
**Author:** Alex Walker  
**Date:** July 2026  

---

## 1. Executive Summary

The long-term goal of this project is to train a Unitree Go2 quadruped to autonomously locate a soccer ball, approach it, position itself appropriately, align the ball with a desired target direction, and perform a physical kick in Isaac Lab.

The system follows a hierarchical control structure:

1. A high-level soccer policy selects discrete locomotion and kick actions.
2. A pretrained low-level locomotion policy executes walking and turning.
3. A separately trained low-level kick policy executes the whole-body kicking motion.
4. Isaac Lab and PhysX simulate the robot, ball, contacts, and field dynamics.

The current work focuses only on the low-level physical kick skill. At this stage, a kick does not mean scoring a goal. It means producing a stable physical behavior in which one foot leaves the ground, swings forward, contacts the ball, transfers momentum to it, and returns to a stable stance.

The main difficulty identified so far was reward exploitation. An earlier policy appeared to generate a strong forward kick according to its original body-relative tracking metric. However, world-frame diagnostics showed that the kicking foot barely moved. Instead, the robot lowered and translated its torso around a mostly planted foot.

To address this, I replaced the body-relative reference with a trajectory anchored to the robot’s initial world pose. The revised policy now produces genuine world-frame foot lifting and forward motion.

The latest measured behavior achieved:

| Metric | Current Result |
|---|---:|
| Maximum policy action magnitude | 2.625 |
| Maximum joint-position change | 0.756 rad |
| Maximum world-frame foot displacement | 0.170 m |
| Forward foot displacement | 0.167 m |
| Foot lift height | 0.100 m |
| Maximum total root displacement | 0.185 m |
| Base-contact termination | 0 |
| Invalid or non-finite actions | 0 |

The foot now performs a real swing rather than remaining planted. However, the robot still relies on substantial whole-body displacement. The current objective is therefore to strengthen the strike while reducing unnecessary base translation and excessive lowering before reintroducing the soccer ball.

---

## 2. Long-Term System Architecture

The final soccer system is intended to use hierarchical reinforcement learning.

```text
High-level soccer policy
Franklin-style observation: 86 dimensions
Discrete action space: 13 actions
                  |
                  v
+-----------------------------------------+
| Actions 0–4                             |
| Pretrained Unitree Go2 locomotion       |
|                                         |
| 0: Stop                                 |
| 1: Move forward                         |
| 2: Move backward                        |
| 3: Turn left                            |
| 4: Turn right                           |
+-----------------------------------------+
                  |
                  |
+-----------------------------------------+
| Actions 5–12                            |
| Strength-conditioned physical kick      |
|                                         |
| Low-level policy outputs                |
| 12 joint-position targets               |
+-----------------------------------------+
                  |
                  v
        Isaac Lab / PhysX simulation
                  |
                  v
       Physical robot–ball interaction
```

The high-level policy will eventually learn:

- where the ball is relative to the robot;
- how to approach the ball;
- how to move behind or around the ball;
- how to align the robot, ball, and target direction;
- when to initiate a kick;
- which kick strength to select.

The current report concerns only the low-level kick component.

---

## 3. Locomotion Status

The locomotion actions use an existing pretrained Unitree Go2 velocity policy.

The robot has been tested with the ball placed in several relative positions:

- directly in front;
- front-left;
- front-right;
- directly behind;
- rear-left;
- rear-right.

A simple closed-loop controller was used only for validation. It turned the robot toward the ball, moved forward when aligned, and stopped when it reached the required distance.

The robot successfully approached the ball in all tested regions. Rightward rotation appeared slower than leftward rotation, but both directions were functional.

These tests demonstrate that the current locomotion primitives are sufficient for reaching a ball from different relative positions. They do not yet demonstrate full autonomous soccer behavior because the validation controller was given the ball position directly.

---

## 4. Low-Level Kick Reinforcement-Learning Setup

| Component | Current Setting |
|---|---|
| Simulator | Isaac Lab / Isaac Sim / PhysX |
| Reinforcement-learning library | RSL-RL |
| Algorithm | PPO |
| Robot | Unitree Go2 |
| Current observation dimension | 60 |
| Current action dimension | 12 |
| Action type | Joint-position targets |
| Kick leg | Front-right leg |
| Support legs | Front-left, rear-left, rear-right |
| Episode length | 70 environment steps |
| Parallel environments | 128–256 locally |
| Initialization | Warm-start from an earlier motion-prior policy |
| Current task | Ball-free whole-body swing motion |
| Next task | Physical contact with a Size 3 soccer ball |

The policy is trained without a ball during the current motion-prior stage.

This separation was introduced because direct contact training requires several difficult behaviors to emerge simultaneously:

1. shifting weight onto the support legs;
2. lifting the kicking foot;
3. generating sufficient forward reach;
4. accelerating the foot;
5. maintaining balance;
6. intersecting the ball;
7. recovering after the motion.

If all of these are trained together from the beginning, the contact reward is extremely sparse and the cause of failure becomes difficult to diagnose.

---

## 5. Definition of the Kick Skill

The low-level kick policy is not required to aim at a goal.

A successful low-level kick should:

1. start from a stable four-leg stance;
2. transfer weight onto three support legs;
3. lift the front-right foot;
4. move the foot forward with sufficient velocity;
5. physically contact the ball;
6. transfer momentum to the ball;
7. retract the front-right leg;
8. recover to a stable stance.

The environment must never directly assign velocity to the ball during an episode. All ball motion must be generated by PhysX contact.

### Proposed Acceptance Criteria

| Metric | Target |
|---|---:|
| Forward foot displacement | At least 0.18 m |
| Foot lift height | At least 0.10 m |
| Peak foot speed | At least 0.80 m/s |
| Swing-foot air fraction | At least 0.75 |
| Base vertical drop | No more than 0.08 m |
| Base horizontal drift | No more than 0.07 m |
| Base-contact termination | 0 |
| Contact by non-kicking bodies | Approximately 0 |
| Ball motion | Caused only by physical contact |

Some body movement is expected. A quadruped normally needs to transfer its center of mass when lifting one leg. The goal is not to keep the torso completely fixed, but to prevent the torso from replacing the kicking leg as the main source of motion.

---

## 6. Why the Task Has Proven Difficult

### 6.1 Reward Exploitation

The first major problem was that the original policy optimized the reward without learning the intended physical behavior.

The original motion-prior objective measured the kicking-foot position relative to the robot’s current body frame. According to this metric, the policy achieved approximately 0.219 m of forward foot motion.

However, later world-frame measurements showed that the actual forward movement of the foot was only approximately 0.013 m.

| Measurement | Earlier Result |
|---|---:|
| Reported body-relative forward motion | 0.219 m |
| Measured world-frame forward motion | Approximately 0.013 m |
| Maximum base drop | 0.228 m |
| Physical foot–ball contact | None |

The robot lowered and translated its torso while leaving the kicking foot almost stationary on the ground. Relative to the moving torso, the foot appeared to move forward and upward.

This was a clear example of reward exploitation: the policy optimized the mathematical proxy but did not produce the desired physical skill.

### 6.2 Whole-Body Coordination

A quadruped kick cannot be treated as an isolated leg motion.

Lifting one front leg changes the support polygon and requires coordinated behavior from:

- the three support legs;
- the torso;
- the hips and shoulders;
- the center-of-mass trajectory;
- the body orientation.

This creates a reward-design trade-off:

- weak stability penalties allow excessive crouching and torso translation;
- very strong stability penalties can prevent the weight transfer required to lift and accelerate the leg;
- large foot-trajectory rewards can encourage unstable or unnatural joint motion;
- large support rewards can encourage the kicking foot to remain on the ground.

### 6.3 Sparse Physical Contact

Before a foot–ball contact reward can occur, the policy must already produce the correct:

- foot height;
- forward reach;
- lateral alignment;
- timing;
- velocity;
- body posture.

If the foot misses the ball by even a small distance, the contact reward remains zero.

Direct contact learning therefore provides little information about whether the failure was caused by reach, height, timing, alignment, balance, or velocity.

### 6.4 Misleading Simulator Signals

Several early observations were misleading:

- periodic ball velocity was initially caused by the ball falling to the ground, not by the robot kicking it;
- an early contact-sensor configuration did not isolate the correct kicking foot;
- ball placement based on reset-time body poses sometimes used stale robot-state information;
- runtime repositioning of the rigid ball blocked in the current simulation setup;
- a moving ball did not necessarily imply that the kicking foot had contacted it;
- a high trajectory reward did not necessarily imply meaningful world-frame foot motion.

The debugging process therefore required separate measurements of:

- kicking-foot contact force;
- contact force from other robot bodies;
- ball absolute velocity;
- foot–ball distance;
- world-frame foot position;
- joint movement;
- base translation;
- base height;
- episode termination state.

---

## 7. Experiment 1: Original Body-Relative Motion Prior

### Objective

Learn a stable whole-body motion that follows a predefined front-right-foot trajectory.

### Initial Interpretation

The learned movement appeared visually plausible. The robot lowered its body, shifted its weight, and moved the front-right leg.

The original tracking metric also reported substantial forward foot motion.

### Diagnostic Result

World-frame measurements showed that most of the apparent foot movement came from torso motion.

The front-right foot remained close to its original world position while the body moved around it.

### Conclusion

The original policy was not a valid physical kick prior.

The failure was not simply insufficient training. The reward definition allowed a behavior that was mathematically successful but physically incorrect.

---

## 8. Experiment 2: Fixed-Ball Contact Probes

A Size 3 soccer ball was placed at several fixed positions along the expected front-right-foot path.

The ball parameters were:

| Property | Value |
|---|---:|
| Radius | 0.0889 m |
| Mass | Approximately 0.318 kg |

The tested positions included:

| Ball X Position | Minimum Foot–Ball Distance | Kick Force | Forward Ball Motion |
|---:|---:|---:|---:|
| 0.445 m | 0.241 m | 0 N | 0 m |
| 0.475 m | 0.271 m | 0 N | 0 m |
| 0.505 m | 0.301 m | 0 N | 0 m |

No real foot contact occurred.

Moving the ball closer did not solve the problem because the underlying foot motion was not physically valid. The policy was mostly moving the torso while the foot remained near the ground.

### Conclusion

Ball-contact training was paused. The motion prior needed to be corrected before additional contact optimization would be meaningful.

---

## 9. Experiment 3: Ground-Anchored Motion Prior

The revised stage, referred to as K2A3, uses a trajectory anchored to the robot’s initial world pose.

### Main Changes

The revised objective:

- disables the original body-relative foot-tracking rewards;
- tracks the front-right foot in a fixed world-anchored reference frame;
- tracks front-right-foot velocity during the strike phase;
- rewards the front-right foot for leaving the ground;
- penalizes front-right-foot ground contact during the swing;
- rewards the three support feet for maintaining contact;
- rewards an upright body orientation;
- penalizes excessive base horizontal drift;
- penalizes excessive base-height change;
- rewards recovery near the end of the motion.

### Representative Training Metrics

| Reward or Termination Term | Representative Value |
|---|---:|
| Anchored foot tracking | 5.1106 |
| Anchored strike tracking | 0.0952 |
| Anchored velocity tracking | 0.7512 |
| Swing-foot air reward | 1.9221 |
| Swing-foot ground-contact penalty | -0.0109 |
| Base horizontal-drift penalty | -0.1123 |
| Excessive base-height penalty | -0.1865 |
| Base-contact termination | 0 |
| Time-out termination | 1.0 |

The older body-relative tracking terms were disabled and remained at zero.

### Execution-Probe Result

A separate execution probe was written to verify that:

- the checkpoint generated non-zero actions;
- the actions were finite;
- the environment executed each action;
- the simulated joint positions changed;
- the front-right foot moved in world coordinates.

The probe measured:

| Metric | Result |
|---|---:|
| Maximum policy action magnitude | 2.625 |
| Maximum joint-position change | 0.756 rad |
| Maximum total foot displacement | 0.170 m |
| Maximum forward foot displacement | 0.167 m |
| Maximum foot height | 0.100 m |
| Maximum total root displacement | 0.185 m |
| Environment execution | Successful |
| Non-finite actions | None |

Example intermediate measurements included approximately:

| Motion Stage | Forward Foot Movement | Foot Height |
|---|---:|---:|
| Mid-swing sample | 0.145 m | 0.043 m |
| Later swing sample | 0.089 m | 0.077 m |
| Maximum over the episode | 0.167 m | 0.100 m |

### Interpretation

K2A3 corrected the complete failure mode of the earlier policy.

The front-right foot now genuinely leaves the ground and moves forward in world coordinates. The policy is no longer relying entirely on moving the torso around a planted foot.

However, the maximum total root displacement is still approximately 0.185 m. This value combines vertical lowering and horizontal translation, so the next diagnostic separates:

- base vertical drop;
- base horizontal drift;
- foot speed;
- genuine foot displacement.

The current policy is therefore promising but not yet ready for final ball-contact learning.

---

## 10. Current Status

### Demonstrated Successfully

- The locomotion policy can approach the ball from multiple relative directions.
- The low-level kick checkpoint produces non-zero actions.
- The 12-dimensional action pipeline changes the simulated robot joints.
- The front-right foot performs genuine world-frame lifting.
- The front-right foot performs genuine forward movement.
- The motion completes without base-contact termination.
- The original body-relative reward-exploitation mechanism has been identified.
- The new world-anchored objective substantially reduces the earlier failure.

### Not Yet Demonstrated

- Reliable contact between the front-right foot and the ball;
- significant forward ball velocity generated by the foot;
- recovery after physical ball impact;
- kick-strength conditioning;
- reliable kick execution from small variations in ball position;
- integration into high-level actions 5–12;
- autonomous alignment of the robot, ball, and goal;
- final autonomous 1v0 training.

---

## 11. Current Hypothesis

The current limiting factor is the balance between strike strength and whole-body stability.

The policy can now generate a real foot swing. However, the robot may still rely on too much base translation or vertical lowering.

My current hypothesis is that the next successful policy needs:

1. stronger strike-phase foot-position tracking;
2. stronger strike-phase foot-velocity tracking;
3. a moderate increase in the swing-foot air reward;
4. stronger penalties for unnecessary base horizontal drift;
5. stronger penalties for excessive body lowering;
6. a stronger recovery objective;
7. a lower learning rate for refinement;
8. lower exploration entropy during fine-tuning.

The base should still be allowed to perform a physically reasonable center-of-mass shift. The purpose of the penalties is not to freeze the body, but to ensure that the kicking foot is the main source of the strike.

---

## 12. Immediate Next Experiment

The next stage is a short stability and strike refinement of K2A3.

The new evaluation will measure base motion as separate components:

- base vertical drop;
- base horizontal displacement;
- front-right-foot speed;
- front-right-foot forward displacement;
- front-right-foot lift height.

### Acceptance Criteria

| Metric | Required Result |
|---|---:|
| Forward foot displacement | At least 0.18 m |
| Foot lift height | At least 0.10 m |
| Peak foot speed | At least 0.80 m/s |
| Base vertical drop | No more than 0.08 m |
| Base horizontal drift | No more than 0.07 m |
| Base-contact termination | 0 |

If these criteria are met, the motion prior will be considered sufficiently stable to resume physical contact training.

---

## 13. Planned Physical Contact Stages

### 13.1 Fixed-Position Contact Learning

The first physical-contact task will use a fixed Size 3 ball position.

The policy will be rewarded for:

- front-right-foot contact with the ball;
- forward ball velocity;
- meaningful ball displacement;
- appropriate foot velocity before impact;
- stable recovery after impact.

The policy will be penalized for:

- the torso contacting the ball;
- another leg contacting the ball;
- excessive lateral ball velocity;
- excessive vertical ball velocity;
- falling;
- base contact;
- joint-limit violations;
- excessive body translation.

No goal or scoring reward will be included.

The purpose of this stage is only to learn a reliable physical kick.

### 13.2 Ball-Position Robustness

After contact is reliable, the ball position will be randomized over a small region.

The policy observation will then include ball-relative information so that the policy can adjust the foot trajectory.

Possible inputs include:

- ball position relative to the robot;
- ball position relative to the kicking foot;
- ball velocity;
- desired kick direction;
- requested kick strength;
- motion phase.

### 13.3 Strength-Conditioned Kick

The intended final design is one continuous kick policy rather than eight completely separate policies.

The policy will receive a requested kick-strength input.

The eight high-level kick actions will map to different strength values:

```text
Action 5  -> Weakest kick
Action 6  -> Slightly stronger kick
Action 7  -> Stronger kick
Action 8  -> Stronger kick
Action 9  -> Stronger kick
Action 10 -> Stronger kick
Action 11 -> Stronger kick
Action 12 -> Strongest kick
```

This allows all kick actions to share the same balance and contact skill while producing different ball speeds.

---

## 14. Final High-Level Soccer Training

After the locomotion and kick skills are validated, both low-level policies will be frozen.

The high-level policy will use the existing Franklin-style interface:

| Component | Planned Value |
|---|---:|
| Observation dimension | 86 |
| Discrete actions | 13 |
| Locomotion actions | 0–4 |
| Physical kick actions | 5–12 |
| Main task | Autonomous 1v0 soccer |

The high-level policy must learn to:

1. infer the relative ball position from its observation;
2. turn toward the ball;
3. approach it;
4. move to an appropriate side of the ball;
5. align the robot, ball, and goal direction;
6. choose a kick strength;
7. execute the physical kick;
8. score using only simulated physical interaction.

This alignment behavior cannot be solved by the low-level kick policy alone.

The high-level policy must understand both:

- the relative location of the ball;
- the relative location of the target goal.

A successful kick generally requires the robot to reach a state in which the robot heading, the ball, and the intended target direction are sufficiently aligned.

The current movement actions should theoretically support this behavior because the robot can:

- move forward;
- move backward;
- turn left;
- turn right;
- stop.

However, the action set does not contain a lateral stepping action. Repositioning around the ball therefore requires combinations of turning and forward or backward motion.

Whether the high-level policy can reliably learn this behavior from sparse scoring rewards remains an open experimental question.

---

## 15. Questions for Feedback

I would especially appreciate feedback on the following questions.

1. Is a motion-prior-first curriculum appropriate for learning this kick, or would residual control over a pretrained standing or locomotion policy be more effective?

2. Is a world-anchored task-space foot trajectory a reasonable objective, or would a demonstration-based imitation objective be preferable?

3. How should necessary center-of-mass transfer be distinguished from excessive torso translation or crouching?

4. Are the current acceptance metrics sufficient to distinguish a genuine kick from reward exploitation?

5. Would it be useful to measure support-leg force distribution or center-of-pressure movement explicitly?

6. Should the policy receive ball-relative observations during the first contact-learning stage, or should deterministic contact be learned before introducing ball-position conditioning?

7. Is it preferable to train one continuous strength-conditioned kick policy or several discrete kick primitives?

8. Should the kick policy be trained as a complete 12-joint policy, or as a residual policy on top of a pretrained standing controller?

9. Would a reference trajectory generated through inverse kinematics or motion optimization provide a better starting point than the manually designed foot trajectory?

10. For the final high-level task, is the current discrete movement set sufficient for moving behind and aligning with the ball, or would a lateral movement action significantly improve learnability?

11. Is sparse goal reward likely to be sufficient after freezing the locomotion and kick skills, or would a curriculum or temporary shaping reward be advisable?

12. Are there additional diagnostics that would make the current failure analysis more convincing?

---

## 16. Reproducibility Information

The current experiments use:

```text
Simulator: Isaac Lab / Isaac Sim / PhysX
RL library: RSL-RL
Algorithm: PPO
Robot: Unitree Go2
Policy observation: 60 dimensions
Policy action: 12 joint-position targets
Episode length: 70 environment steps
Parallel environments: 128–256
Kick leg: Front-right
```

The current motion-prior experiment produces:

```text
Maximum joint-position change: 0.756 rad
Maximum total foot displacement: 0.170 m
Maximum forward foot displacement: 0.167 m
Maximum foot height: 0.100 m
Maximum root displacement: 0.185 m
Base-contact termination: 0
```

Training checkpoints and large output files are not included in the report repository.

---

## 17. Limitations of the Current Results

The reported measurements come from simulation and have not been tested on a physical Unitree Go2.

The current motion prior:

- uses one selected kicking leg;
- uses a fixed reference motion;
- does not currently observe a ball;
- does not currently account for impact uncertainty;
- does not currently adapt its contact point;
- has not yet been evaluated under domain randomization;
- has not yet been integrated with the high-level soccer policy.

The current results should therefore be interpreted as progress toward a usable simulation skill, not as a complete soccer system.

---

## 18. Summary

The project initially appeared to have a kick motion, but detailed world-frame diagnostics showed that the policy had exploited a body-relative reward. The robot lowered and translated its torso while leaving the kicking foot mostly planted.

A revised world-anchored objective has now produced genuine front-right-foot lifting and forward movement.

The latest policy reaches approximately:

- 0.167 m of forward foot displacement;
- 0.100 m of foot lift;
- 0.170 m of total foot displacement;
- 0.756 rad of joint-position change;
- no base-contact termination.

The remaining issue is excessive whole-body displacement. The next experiment will refine strike velocity and stability before returning to physical ball contact.

The most important feedback I am seeking concerns:

- the choice of motion-prior curriculum;
- the balance between stability and necessary weight transfer;
- the world-anchored reward design;
- the transition to contact-based learning;
- the sufficiency of the high-level discrete action space.

---

## 19. Suggested Media

The following supporting materials can be added later:

- K2A2 reward-exploitation video;
- K2A3 world-anchored swing video;
- training-metrics screenshot;
- execution-probe screenshot;
- comparison of body-relative and world-frame foot motion;
- final contact-learning video.

