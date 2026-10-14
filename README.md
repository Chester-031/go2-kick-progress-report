# Go2 Robot Soccer — ADR Training

No-Kick high-level training for the Unitree Go2 soccer task in Isaac Lab,
with automatic domain randomization (ADR) choosing where every episode
starts. This branch starts from the Stage B training code
(`codex/stage-b-training-2026-10-03`). It replaces the fixed
distance-then-orientation curriculum with ADR, removes the wrapper's main
sources of overhead, and adds standard training plots.

## Task (unchanged from Stage B)

- A recurrent high policy (one GRU layer of 256, actor/critic MLPs
  `[256, 256, 128]`) observes 70 values. It emits a `(vx, vy, wz)` command
  at 10 Hz. A frozen 50 Hz low policy executes the command for five ticks.
- Reward: normalized progress along
  `robot -> behind-ball staging target -> goal`, `+1` for a goal, `-1` for
  an own goal.
- Episodes end on a goal, own goal, fall, boundary, joint hard limit or the
  20 s horizon. An unsafe event on the same tick as a goal cancels the goal.
- PPO: 4096 envs × 200 steps per iteration, lr `3e-4` fixed, entropy
  `0.03`, gamma `0.997`, lambda `0.98`, 5 epochs × 4 minibatches.

## Automatic domain randomization (`src/robot_soccer/adr.py`)

ADR follows OpenAI, *Solving Rubik's Cube with a Robot Hand* (2019), §5.
Each knob has its own range. The range starts as a point at an easy value
and moves according to probe results:

| Knob | Limits | Start | Step |
|---|---|---|---|
| ball distance from the goal | 0.5 – 6.0 m | 1.0 m | 0.25 m |
| robot distance from the ball | 0.5 – 2.3 m | 0.8 m | 0.1 m |
| robot facing error to the ball (0 = looking at it, 180 = looking away) | 0 – 180° | 0° | 10° |

Episodes come in three kinds:

- **Training** (75% of episodes): every knob is drawn uniformly from its
  current range.
- **Probes** (20%): one knob is pinned at one edge of its range. Each edge
  collects 256 results. At ≥ 0.6 goal rate the edge moves one step out; at
  ≤ 0.3 it moves one step in. Both edges of every knob can move.
- **Fixed difficulty** (5%): drawn at difficulty d ∈ {0.25, 0.5, 1}. Each
  knob spans from its start value d of the way to its limits. This goal
  rate compares across runs and over training. The training goal rate
  doesn't, because its task keeps changing.

The rest of the start geometry is the same at every difficulty:

- which goal is attacked;
- the ball's lateral angle around the goal centre, ≤ 1°;
- the robot's position around the ball: beside and behind it, 122–136° from
  the ball-to-goal direction;
- which way the robot turns away from the ball.

The knob limits keep every combination inside the legal field; a test
checks this.

All fractions, thresholds, the buffer size and the evaluation difficulties
are `--adr-*` flags of `robot_soccer.train`.

## Speed

The wrapper's own cost per high-level step was measured on a fake scene
with toy physics, at 4096 envs on an L40S. It fell from 52.7 ms to 13.3 ms,
and is now flat in env count, so PhysX dominates. The three changes:

- **Command projection.** Commands are projected onto the qualified hull
  exactly, in one vectorized pass (`commands.CommandSurface`). The old
  Frank-Wolfe solver ran up to 64 iterations twice per step, with a host
  sync in each iteration, and still missed the true projection by up to
  0.012 in command units.
- **Resets.** Resets are sampled as one batch and written with tensor ops.
  Before, each env was a Python loop with scalar GPU writes and re-checks.
- **Episode records.** Finished episodes are gathered once per step into
  columns, in two host copies. Before, each terminal env took about 30
  `.item()` calls inside the tick loop. Per-tick safety checks are now one
  deferred check per step.

## Run

Requirements: Isaac Lab 2.3.2 (or a compatible Isaac Sim), with the
following installed into that environment:

- PyTorch from Isaac Lab;
- `rsl_rl` 3.1.2;
- `tensordict`.

You also need the qualified frozen low checkpoint, which is not committed.

```bash
export PYTHONPATH="$PWD/src:$PWD/runtime_project/isaac_go2_low_level_rl"
python -m robot_soccer.train --headless --device cuda:0 \
  --low-checkpoint /path/to/qualified_low.pt \
  --low-checkpoint-sha256 f46d3b760c43664350aac8099ae5bbd32bc1207f97c8814f5d21c9b531c3d96c \
  --output-dir /path/to/new_run --iterations 300
```

`cluster/train.sbatch PYTHON LOW_CHECKPOINT LOW_SHA256 OUTPUT_DIR [args]` is
the Slurm entry point. There are three ways to start a run:

- **Fresh weights**: the default.
- **`--initial-checkpoint PATH`**: loads the high-policy weights only.
- **`--resume PATH`**: restores the weights, optimizer, iteration and ADR
  ranges. Write the resumed run to a new output directory.

Each iteration prints the overall goal rate, the fixed-difficulty goal
rates, the ADR ranges and the throughput. The output directory holds:

- `run_info.json`;
- `metrics.jsonl`, one record per iteration;
- TensorBoard scalars;
- checkpoints every `--checkpoint-interval` iterations;
- `training_summary.json`.

## Plots

```bash
python scripts/plot_training.py RUN_DIR [RUN_DIR ...] --out PLOTS [--labels a,b]
```

The script reads only `run_info.json` and `metrics.jsonl`, and needs only
matplotlib and numpy. It overlays several runs (seeds or arms) and writes:

| Figure | Shows |
|---|---|
| `01_goal_rates` | goal rate at each fixed difficulty (the comparable score), and for training and probe episodes |
| `02_adr_ranges` | each knob's range over training, with its limits and start value |
| `03_adr_probes` | probe goal rate at every range edge against the move thresholds, with each edge move marked |
| `04_adr_health` | ADR entropy, share of the limits covered, cumulative edge moves |
| `05_outcomes` | how episodes end: goal, own goal, fall, out of bounds, joint limit, timeout |
| `06_knob_heatmaps` | goal rate, and where episodes started, over each pair of knobs, late in training |
| `07_goal_by_knob` | goal rate along each knob, early vs late in training |
| `08_episode_timing` | episode length, time to goal, time to first contact, ball progress, contact and unsafe rates |
| `09_ppo` | PPO losses and mean episode return |
| `10_throughput` | transitions per second and how each iteration's time splits between environment, policy and update |

## Tests

These CPU tests don't start Isaac Sim. They need `torch`, `tensordict`,
`scipy`, `matplotlib` and `rsl-rl-lib==3.1.2`.

```bash
PYTHONPATH=src:. python -m unittest discover -s tests -t .
```

- `tests/fake_isaac.py` stands in for the Isaac Lab scene with toy physics.
- `test_environment_characterization` compares every finished episode of a
  scripted rollout with a recorded fixture.
- `test_train_loop` runs the whole training loop (RSL-RL PPO, ADR,
  metrics, checkpoints, resume) on the fake scene.
- `scripts/benchmark_environment.py` times the wrapper on the fake scene.
- `tests/synthetic_run.py` writes a plausible run directory for the plots.
