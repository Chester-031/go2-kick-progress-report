# Go2 Robot Soccer — Stage C Training Logic

This directory is a small, reader-facing projection of the no-Kick Stage C
training method. It explains the policy hierarchy, task logic, reset design,
continuation semantics, and PPO parameters without reproducing the production
trainer.

It is deliberately not runnable. There is no command-line entry point,
simulator launcher, deployment wrapper, artifact writer, or reporting stack.
The Python files use only constants, immutable data classes, and pure
functions so the scientific design can be reviewed without operational noise.

## Stage C in one paragraph

Stage C continued the same recurrent high-level policy and Adam optimizer from
iteration 220, then trained formal iterations 221–300. The high policy observes
70 values and emits a continuous three-axis velocity command `(vx, vy, wz)` at
10 Hz. A frozen low-level locomotion policy converts the held command to 12
joint actions at 50 Hz. There is no learned Kick action, Kick-strength output,
Kick reward, behavior router, waypoint controller, or hand-authored turn/push
state machine.

## Exact formal-training parameters

| Item | Stage C value |
| --- | ---: |
| Parent state | iteration 220 model **and** Adam state |
| Formal iterations | 221–300 (80 iterations) |
| Parallel environments | 4,096 |
| High-policy steps per environment | 200 |
| Transitions per iteration | 819,200 |
| Nominal transitions per minibatch | 204,800 |
| Formal transitions | 65,536,000 |
| PPO epochs / minibatches | 5 / 4 |
| Optimizer steps per iteration | 20 |
| Added formal optimizer steps | 1,600 |
| Parent / final cumulative Adam step | 4,400 / 6,000 |
| Learning rate | `3e-4`, fixed |
| Discount / GAE lambda | `0.997` / `0.98` |
| Entropy coefficient | `0.03` |
| Milestone iterations | 240, 260, 280, 300 |

Here `4` is `num_mini_batches`, not `batch_size=4`; the
819,200-transition rollout is partitioned into four nominal
204,800-transition minibatches per epoch.

A disposable iteration-221 canary was a process-integrity gate. Formal
training started only after a fresh restoration of the iteration-220 model and
Adam state, so canary updates were not ancestors of the formal trajectory.

## Three coequal task axes

Every reset is identified by three primary axes. They are peers in analysis;
none is treated as a subordinate diagnostic.

| Axis | Support | Stage C allocation |
| --- | --- | --- |
| Absolute orientation error | 0°, 45°, 90°, 180° | uniform |
| Robot–ball distance | 0.8 m, 1.6 m | uniform |
| Ball–goal distance | 2, 2.5, 3, 3.5, 4, 4.5, 5, 6 m | weights 2, 4, 3, 3, 5, 5, 5, 5 |

Only the ball–goal axis is reweighted. Its 32 allocation slots assign 20
slots, or 62.5%, to 4–6 m. Goal side, facing sign, lateral sign, and orbit sign
remain balanced. The schedule is fixed and non-adaptive: it does not inspect
rewards, outcomes, policy state, optimizer state, or wall-clock time.

## Policy and task logic

- Observation: the frozen low policy's 48-D input plus 22 entity/task
  features, for 70 dimensions total.
- High policy: one-layer GRU with hidden size 256, followed by actor and critic
  MLPs `[256, 256, 128]` with ELU activations.
- High action: three latent values mapped through `tanh`, physical axis ranges,
  the qualified convex command hull, and coupled rate limiting.
- Hierarchy: one high action is held for five 50 Hz low-policy ticks, producing
  a 10 Hz high-policy decision rate.
- Reward: normalized reduction in
  `robot → behind-ball staging target + ball → selected goal` path length,
  plus `+1` for a strict selected-goal terminal and `-1` for an own goal.
- Termination: goal, own goal, boundary, fall, hard joint-limit violation, or
  the 20 s finite horizon. Unsafe same-tick events take precedence over a
  requested goal.

The staging target is reward geometry, not a commanded waypoint. Navigation,
alignment, ball transport, and pushing remain learned closed-loop behavior.

## Scope limits

Stage C trained only robot–ball distances 0.8 m and 1.6 m. It therefore does
not establish coverage for arbitrary robot–ball spacing. The 70-D observation
reserves opponent position, velocity, and presence slots, but Stage C supplied
zeros in those slots. This is a single-robot ball-to-goal task, not learned 1v1
play. Full-field resets, live opponents, and self-play are future extensions,
not Stage C results.

## Pending A0019 optimizer screen — not a result

A0019 is a pending directional learning-rate × minibatch screen. Its declared
arms are `(1e-5, 4)`, `(1e-4, 4)`, `(3e-4, 4)` control, `(1e-3, 4)`,
`(1e-2, 4)`, `(3e-4, 2)`, `(3e-4, 8)`, and `(1.5e-4, 8)`, where each pair is
`(learning rate, minibatches)`. It is not a full factorial and has produced no
result represented in this package. Stage C parameters above remain the only
reported training method here.

## Reading order

1. `code/parameters.py`
2. `code/reset_strategy.py`
3. `code/policy_architecture.py`
4. `code/task_logic.py`
5. `code/training_outline.py`
6. `code/analysis_axes.py`

`SOURCE_MAP.md` describes how each projection relates to the frozen source.
`EXPORT_OMISSIONS.md` states what was intentionally excluded.
