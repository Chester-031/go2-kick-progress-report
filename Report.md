# Building a Franklin-Compatible Soccer Environment for Unitree Go2 in Isaac Lab

**Project:** Isaac-Go2-Soccer  
**Report Type:** Research Progress Report  
**Current Stage:** Environment Integration and Physical Kick Skill Development  
**Author:** Alex Walker  
**Date:** July 2026  

---

## 1. Project Overview

The long-term goal of this project is to train a Unitree Go2 quadruped to autonomously play a simplified 1v0 soccer task in Isaac Lab.

The robot should eventually be able to:

1. observe the ball and field state;
2. locate the ball without being directly given a scripted route;
3. approach the ball using pretrained locomotion skills;
4. reposition itself relative to the ball and goal;
5. select an appropriate kick action;
6. physically contact the ball;
7. score using only simulated robot–ball interaction.

The project is based on an existing Franklin-style soccer environment. The original Franklin code already defines the high-level reinforcement-learning interface, including:

- an 86-dimensional observation;
- a 13-action discrete action space;
- soccer event and reward logic;
- high-level episode timing;
- goal and field-state representations.

The main engineering problem is therefore not to redesign the high-level task from scratch. It is to transfer the Franklin interface into an Isaac Lab environment in which actions are executed by a physically simulated Unitree Go2.

The resulting system is hierarchical:

```text
Franklin-style high-level soccer policy
                |
                v
        13 discrete actions
                |
      +---------+----------+
      |                    |
      v                    v
Pretrained Go2        Learned physical
locomotion policy     kick policy
      |                    |
      +---------+----------+
                |
                v
        12 Go2 joint targets
                |
                v
       Isaac Lab / PhysX
                |
                v
     Physical robot and ball state
                |
                v
Franklin-compatible observation and reward
```

---

## 2. Transition from Franklin to Isaac Lab

The Franklin code provides a useful high-level soccer abstraction, but it does not directly solve the low-level control problem of a quadruped robot.

The migration strategy is therefore to preserve the Franklin task interface while replacing its abstract execution backend with an Isaac Lab physical simulation.

### Components retained from Franklin

The following concepts are preserved:

- observation dimension and feature ordering;
- discrete action semantics;
- team and goal representation;
- game-time representation;
- episode-step accounting;
- sparse soccer reward;
- high-level policy interface.

The intended high-level observation remains:

```text
Observation dimension: 86
```

The intended high-level action space remains:

| Action | Meaning |
|---:|---|
| 0 | Stop |
| 1 | Move forward |
| 2 | Move backward |
| 3 | Turn left |
| 4 | Turn right |
| 5–12 | Kick with different requested strengths |

The soccer reward remains intentionally sparse:

| Event | Reward |
|---|---:|
| Team A scores | +1 |
| Team B scores | -1 |
| Other transitions | 0 |

### Components replaced or extended

The Franklin environment previously treated actions as high-level state transitions. In Isaac Lab, each high-level action must be connected to a real physical controller.

The new backend must provide:

- a Unitree Go2 articulation;
- pretrained low-level locomotion;
- physical joint commands;
- a rigid soccer ball;
- collision and contact sensing;
- field boundaries and goals;
- real robot position, orientation, and velocity;
- real ball position and velocity;
- episode reset of all physical and policy states.

The main design principle is:

> Franklin defines what the soccer agent observes and decides, while Isaac Lab defines how those decisions are physically executed.

---

## 3. Environment Architecture

The environment is divided into several layers so that the Franklin logic does not depend directly on Isaac Lab implementation details.

### 3.1 High-level RL environment

The high-level environment exposes the Franklin-compatible interface:

```text
observation: 86-dimensional vector
action: one integer from 0 to 12
reward: soccer event reward
termination: episode or task condition
```

The high-level policy does not directly output joint positions.

Instead, it selects a locomotion or kick primitive.

### 3.2 Action execution layer

The action execution layer translates the 13 discrete actions into low-level controllers.

```text
Actions 0–4
    -> pretrained Go2 locomotion policy

Actions 5–12
    -> physical kick policy
```

For locomotion actions, the high-level command is converted into a body-frame velocity command:

```text
[forward velocity, lateral velocity, yaw rate]
```

The pretrained locomotion policy then converts this command and the robot state into 12 joint actions.

For kick actions, the intended design is:

```text
kick policy(
    robot state,
    ball-relative state,
    requested kick strength,
    motion phase
)
-> 12 joint-position targets
```

### 3.3 Isaac Lab physical backend

The physical backend is responsible for:

- stepping the simulation;
- applying joint targets;
- maintaining contact sensors;
- reading robot root state;
- reading joint states;
- reading ball rigid-body state;
- detecting goals and out-of-bounds events;
- resetting robot, ball, controllers, and histories.

The ball must never be assigned a velocity as the result of a high-level kick action.

All official kick behavior must satisfy:

```text
robot joint motion
-> foot–ball contact
-> PhysX impulse
-> ball motion
```

### 3.4 Observation and event adapters

The simulation state must be converted back into the Franklin observation format.

This adapter is responsible for calculating features such as:

- robot position;
- robot orientation;
- robot linear and angular velocity;
- ball position relative to the robot;
- ball velocity;
- goal-relative information;
- field-relative information;
- high-level time remaining;
- previous action or task state where required.

This separation makes it possible to retain the Franklin policy definition even though the underlying environment has changed from an abstract simulator to a quadruped physics simulator.

---

## 4. Observation, Action, Reward, and Timing Design

### 4.1 Observation design

The final high-level policy is expected to use the existing 86-dimensional Franklin observation.

A major requirement is that all dynamic features come from real simulation state.

For example:

```text
robot yaw
robot velocity
ball-relative position
ball-relative velocity
```

must be derived from the Isaac Lab articulation and rigid-body tensors.

They must not be inferred only from the command that was sent to the robot.

This distinction is important because the low-level locomotion policy may:

- move more slowly than commanded;
- turn along an arc;
- experience slip;
- overshoot;
- temporarily lose balance;
- respond asymmetrically to left and right commands.

The high-level policy must observe what physically happened, not what the controller was asked to do.

### 4.2 Action duration

Each Franklin action is a high-level decision, not a single physics step.

A locomotion action should remain active for a short interval, currently designed around approximately:

```text
0.2 seconds per high-level decision
```

During this interval:

1. the high-level velocity command remains fixed;
2. the low-level locomotion policy is evaluated repeatedly;
3. joint commands are applied;
4. the simulator advances;
5. the resulting state is returned to the high-level environment.

A kick action is different because it represents a complete motion primitive:

```text
prepare
-> transfer weight
-> lift leg
-> strike
-> retract
-> recover
```

Its duration may therefore be longer than an ordinary locomotion decision.

### 4.3 Reward design

The final 1v0 task is intended to preserve the Franklin sparse reward.

This means the high-level policy will not necessarily receive dense rewards for:

- approaching the ball;
- facing the ball;
- aligning with the goal;
- selecting a kick action.

The benefit is that the final policy remains consistent with the original task definition.

The risk is that discovering a successful sequence may be difficult:

```text
locate ball
-> approach ball
-> move behind ball
-> align with goal
-> select kick
-> physically contact ball
-> score
```

A curriculum or temporary shaping reward may therefore be needed if sparse exploration produces too few goals.

---

## 5. Locomotion Integration and Validation

The low-level locomotion component uses a pretrained Unitree Go2 velocity policy in an Isaac Lab manager-based environment.

The locomotion policy already handles:

- robot joint ordering;
- default joint pose;
- action scaling;
- low-level observation construction;
- velocity-command tracking;
- balance;
- simulation decimation;
- joint control.

This avoids rebuilding the complete locomotion controller inside the original standalone soccer scene.

### 5.1 Validated discrete actions

The following high-level actions have been connected to locomotion commands:

| Action | Low-Level Intent |
|---:|---|
| 0 | Stand or stop |
| 1 | Move forward |
| 2 | Move backward |
| 3 | Turn left |
| 4 | Turn right |

The first test used a predetermined sequence of actions. This confirmed that:

- forward commands moved the robot forward;
- backward commands moved it backward;
- positive yaw commands turned it left;
- negative yaw commands turned it right;
- the robot remained physically simulated and did not teleport.

However, a predetermined sequence does not prove that the action set can support soccer navigation.

### 5.2 Ball-approach validation

A simple non-learning controller was therefore created to test closed-loop reachability.

The controller used the current robot pose and fixed ball position:

```text
if ball bearing is too far left:
    turn left

if ball bearing is too far right:
    turn right

if ball is approximately ahead:
    move forward

if the robot reaches the target distance:
    stop
```

The ball was tested in several relative positions:

- directly ahead;
- front-left;
- front-right;
- directly behind;
- rear-left;
- rear-right.

The robot was able to turn and approach the ball in all tested regions.

The tests also revealed that rightward turning was slower than leftward turning. This asymmetry may later affect the high-level policy but does not prevent the robot from reaching the ball.

### 5.3 Remaining locomotion question

The current locomotion tests demonstrate navigation toward a stationary ball.

They do not demonstrate dribbling.

When the robot walks near a ball, it may:

- contact the ball with the wrong foot;
- push it with a calf or body link;
- step over it;
- trap it between the legs;
- lose it while turning.

For the first autonomous 1v0 system, the intended strategy is therefore:

```text
approach the ball
-> stop near the ball
-> reposition and align
-> execute a separate kick primitive
```

Continuous dribbling can be developed later as an additional low-level skill.

---

## 6. Reinforcement-Learning and Training Pipeline

The low-level skill policies use PPO through RSL-RL.

### Current low-level configuration

| Component | Current Setting |
|---|---|
| Simulator | Isaac Lab / Isaac Sim / PhysX |
| Robot | Unitree Go2 |
| RL library | RSL-RL |
| Algorithm | PPO |
| Policy action | 12 joint-position targets |
| Kick-motion observation | 60 dimensions |
| Episode length | Approximately 70 control steps |
| Parallel environments | 128–256 locally |
| Device | CUDA GPU |
| Initialization | Warm-start from previous skill checkpoint |

The high-level Franklin policy will be trained only after the locomotion and kick skills are available.

The intended final training structure is:

```text
pretrained locomotion policy: frozen
physical kick policy: frozen
Franklin high-level policy: trainable
```

The high-level actor will learn only the selection and sequencing of skills.

### Current development workflow

The project currently uses separate Git branches and worktrees for major stages, including:

```text
locomotion validation
physical contact environment
motion-prior training
ground-anchored kick motion
contact-kick training
```

Training outputs and model checkpoints are kept outside Git.

Each major skill stage is expected to provide:

- a task configuration;
- an RSL-RL configuration;
- a training entry point;
- a deterministic playback or evaluation entry point;
- quantitative summary output;
- a checkpoint suitable for the next curriculum stage.

---

## 7. Physical Kick Experiment Design

The physical kick is being trained as a separate low-level skill.

A kick is defined as physical contact behavior, not as scoring a goal.

The policy should learn:

1. stable initial stance;
2. weight transfer to three support legs;
3. lifting the front-right foot;
4. forward foot acceleration;
5. foot–ball contact;
6. transfer of momentum;
7. leg retraction;
8. recovery to a stable stance.

### 7.1 Motion-prior stage

The first stage is trained without a ball.

The purpose is to learn a stable swing motion before adding sparse contact feedback.

The kick policy outputs all 12 joint targets. The other three legs are not frozen because they must actively maintain balance.

A piecewise foot trajectory is used:

```text
settle
-> lift
-> forward strike
-> retract
-> recover
```

The original trajectory was measured relative to the robot body.

This produced reward exploitation: the robot lowered and translated its torso while the foot remained nearly fixed in world coordinates.

The revised objective anchors the trajectory to the initial world pose.

The new rewards include:

- world-frame foot-position tracking;
- world-frame foot-velocity tracking;
- strike-phase tracking;
- swing-foot air reward;
- swing-foot ground-contact penalty;
- support-foot contact reward;
- upright-body reward;
- base horizontal-drift penalty;
- excessive height-change penalty;
- recovery reward.

### 7.2 Planned contact stage

After the motion prior is accepted, a Size 3 soccer ball will be placed along the measured physical foot path.

The first contact stage will use:

```text
fixed ball position
fixed forward kick direction
fixed maximum strength
```

The reward will include:

- front-right-foot contact;
- forward ball speed;
- forward ball displacement;
- stable recovery.

Penalties will include:

- body contact with the ball;
- contact by another leg;
- lateral or vertical ball motion;
- excessive body translation;
- falling or base contact.

The task will not contain a goal or scoring reward.

After reliable contact is learned, later stages will add:

- ball-position randomization;
- ball-relative observation;
- requested kick direction;
- requested kick strength;
- mapping to high-level actions 5–12.

---

## 8. Kick Difficulties and Current Progress

The most important difficulty was that the original reward did not measure the intended physical behavior.

### 8.1 Reward exploitation

The earlier policy reported approximately:

```text
body-relative forward foot motion: 0.219 m
```

World-frame diagnostics showed:

```text
actual forward foot motion: approximately 0.013 m
maximum base drop: approximately 0.228 m
```

The robot had learned to move its body around a mostly planted foot.

This was not a weak kick. It was an incorrect reward proxy.

### 8.2 Ground-anchored correction

The corrected policy uses an initial world-frame anchor.

A separate execution probe verified:

| Metric | Current Result |
|---|---:|
| Maximum action magnitude | 2.625 |
| Maximum joint-position change | 0.756 rad |
| Maximum total foot displacement | 0.170 m |
| Maximum forward foot displacement | 0.167 m |
| Maximum foot lift height | 0.100 m |
| Maximum total root displacement | 0.185 m |
| Base-contact termination | 0 |
| Non-finite actions | None |

The new policy therefore produces real joint motion and real front-right-foot movement.

The current remaining issue is that whole-body displacement is still relatively large.

The next refinement separates:

- base vertical drop;
- base horizontal drift;
- foot speed;
- foot forward displacement;
- foot lift height.

### Current acceptance targets

| Metric | Target |
|---|---:|
| Forward foot displacement | ≥ 0.18 m |
| Foot lift height | ≥ 0.10 m |
| Peak foot speed | ≥ 0.80 m/s |
| Base vertical drop | ≤ 0.08 m |
| Base horizontal drift | ≤ 0.07 m |
| Base-contact termination | 0 |

Physical ball-contact training will resume only after the swing motion satisfies these conditions.

---

## 9. Current Status and Next Steps

### Completed

- Preserved the Franklin 86-dimensional observation contract.
- Preserved the Franklin 13-action high-level interface.
- Preserved sparse soccer event rewards.
- Connected actions 0–4 to pretrained Go2 locomotion.
- Validated forward, backward, left-turn, and right-turn execution.
- Validated approach behavior with the ball in multiple relative positions.
- Built an Isaac Lab physical ball and contact-sensor environment.
- Removed the invalid direct-ball-velocity kick behavior.
- Built a full-body 12-joint kick-motion training environment.
- Identified body-relative reward exploitation.
- Replaced the original objective with world-anchored foot tracking.
- Verified real joint and world-frame foot motion.

### In progress

- Reducing unnecessary base translation and lowering;
- increasing physical strike reach;
- measuring real foot speed;
- preparing a reliable fixed-position contact task.

### Remaining stages

```text
1. Complete stable world-anchored swing motion
2. Train fixed-position physical contact
3. Add ball-position robustness
4. Add continuous strength conditioning
5. Map the kick policy to actions 5–12
6. Integrate locomotion and kick skills into one environment
7. Validate the Franklin 86-dimensional observation dynamically
8. Train the final 1v0 high-level policy
9. Evaluate the final checkpoint
10. Reproduce locally and record a video
```

The final high-level policy must learn a more difficult behavior than simply reaching the ball.

Before kicking, it may need to move so that:

```text
robot heading
ball position
goal direction
```

are sufficiently aligned.

The current action space should theoretically support this because the robot can move forward, backward, turn left, turn right, and stop.

However, the lack of a lateral stepping action may make repositioning around the ball less efficient.

---

## 10. Questions for Feedback

I would especially appreciate feedback on the following questions:

1. Is preserving the Franklin observation, action, and reward interface while replacing only the execution backend a reasonable system design?

2. Are there important differences between an abstract soccer environment and a physical quadruped environment that should lead to changes in the high-level observation?

3. Should the high-level policy directly observe the robot’s joint or stability state, or should those details remain hidden behind the low-level skills?

4. Is the current division between pretrained locomotion, separately trained kicking, and high-level soccer control appropriate?

5. Is the current five-action locomotion set sufficient for moving behind and aligning with the ball, or would a lateral movement action significantly improve learnability?

6. Is a motion-prior-first curriculum appropriate for the kick, or would residual control over a pretrained standing policy be more effective?

7. Are world-anchored foot tracking and explicit base-motion penalties appropriate ways to avoid reward exploitation?

8. Should ball-relative observations be introduced during the first contact stage, or only after deterministic contact has been learned?

9. Is one strength-conditioned kick policy preferable to several separate kick policies?

10. Is sparse Franklin goal reward likely to be sufficient for the final high-level task after freezing the low-level skills, or would a temporary curriculum or shaping reward be necessary?

11. Are the current validation stages sufficient before beginning long high-level training?

12. What additional experiments or measurements would make the environment design and current progress easier to evaluate?
