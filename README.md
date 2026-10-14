# Go2 Robot Soccer — ADR Training

Unitree Go2 soccer in Isaac Lab, trained with automatic domain
randomization (ADR) choosing where every episode starts. The main path
trains **one policy straight from the 12 joints**: walking, finding the ball
and scoring are learned together, with no low-level locomotion policy.
This branch started from the Stage B code (`codex/stage-b-training-2026-10-03`).
Stage B's 10 Hz command hierarchy over a frozen low-level policy is kept as
the `hierarchical` control.

## Controls (`--control`)

| Control | Policy output | Rate | Scene |
|---|---|---|---|
| `position` (default) | 12 joint position targets, ±0.25 rad around standing, stock Go2 PD (Kp 25, Kd 0.5) | 50 Hz | `direct_env_cfg.py` |
| `velocity` | 12 joint velocity targets, 6 rad/s per unit, Kd 1.0, plus the stock Kp 25 spring to the standing pose | 50 Hz | `direct_env_cfg.py` |
| `hierarchical` | `(vx, vy, wz)` command, projected onto the qualified command hull, held for 5 ticks for a frozen low policy | 10 Hz | Stage B (Isaac Lab 2.3.2 + `qualified_low.pt`) |

The direct scene is Isaac Lab's stock flat Go2 locomotion config (present
since Isaac Lab 2.0) plus the Stage B soccer field: walls, two goals and a
0.43 kg ball with rolling resistance. Zero actions leave the robot standing
in both direct arms.

The velocity arm needs the spring toward the standing pose. Without it, a
fresh policy collapses into the joint limits within half a second. The
velocity gains are a first guess, not tuned.

## Reward

`+1` goal, `-1` own goal, plus two other terms:

- **Potential-based shaping** (`shaping.py`). This is Ng, Harada & Russell
  (1999): F = γΦ(s′) − Φ(s), using PPO's γ, with Φ = 0 after every terminal
  transition, timeout included. So every episode's discounted shaping is
  −Φ(start), and the optimal policy is unchanged. Φ adds up four potentials:
  - the ball's distance to the goal centre;
  - the robot's distance to the ball;
  - the base height's distance from 0.28 m;
  - the tilt (1 − cos tilt).
- **The stock motion regularizers** (direct controls only). These are joint
  torque, joint acceleration and action rate, at the stock Go2 weights. They
  are not potential-based: they deliberately bias toward smooth, low-effort
  motion.

Episodes end on a goal, own goal, fall (trunk contact), boundary, joint hard
limit, or after 20 s. An unsafe event on the same tick as a goal cancels the
goal.

## Automatic domain randomization (`src/robot_soccer/adr.py`)

ADR follows OpenAI, *Solving Rubik's Cube with a Robot Hand* (2019), §5.
Each knob's range starts as a point at a start so easy that random flailing
scores. In the Isaac smoke test, random actions scored in 42% (position)
and 72% (velocity) of finished episodes. Steps grow linearly from the
small step at the start to the large step at the limit: tiny moves while
the robot learns to stand and push, large ones once it can play.

| Knob | Limits | Start | Step near start → near limit |
|---|---|---|---|
| ball distance from the goal centre | 0.15 – 6.0 m | 0.25 m | 0.02 → 0.5 m |
| robot distance from the ball | 0.4 – 2.3 m | 0.45 m (nose at the ball) | 0.02 → 0.25 m |
| robot facing error to the ball (0 = looking at it) | 0 – 180° | 0° | 2 → 20° |
| robot position around the ball (0 = straight behind, 180 = between ball and goal) | 0 – 180° | 0° | 2 → 20° |

Episodes come in three kinds:

- **Training** (75% of episodes): every knob is drawn from its current
  range.
- **Probes** (20%): one knob is pinned at one edge of its range. Each edge
  collects 256 results. At ≥ 0.6 goal rate the edge moves out one step; at
  ≤ 0.3 it moves in.
- **Fixed difficulty** (5%): drawn at difficulty 0.25, 0.5 or 1. Each knob
  spans from its start d of the way to its limits. This goal rate compares
  across runs.

Starts that don't fit on the field are redrawn, keeping a probe's pinned
value. A probe that can't be placed at all becomes a training episode, and
the metrics count it. Every fraction, threshold and buffer size is an
`--adr-*` flag.

## Speed

On a fake scene with near-zero physics, the environment's own cost per
policy step (4096 envs, L40S) fell from 52.7 ms to 13.3 ms:

- **Command projection:** commands are projected onto the hull exactly in
  one pass, instead of Frank-Wolfe with a host sync per iteration. Frank-Wolfe
  also missed the true projection by up to 0.012.
- **Resets and episode records:** resets and finished-episode records are
  vectorized, instead of per-env Python loops with `.item()` reads.

In Isaac with 4096 envs, a direct-control step takes 23–35 ms.

## Run

The direct controls run on Isaac Lab 2.0.2 or later. Add `rsl_rl` 3.1.2
and `tensordict` ≥ 0.7. On the cluster these are installed with
`pip install --no-deps --target DIR rsl-rl-lib==3.1.2 tensordict==0.7.2 orjson`
and put on `EXTRA_PYTHONPATH`.

```bash
sbatch cluster/train_direct.sbatch PYTHON position /path/to/run   # or velocity
# directly:
python -m robot_soccer.train --headless --device cuda:0 --control position --output-dir RUN
```

The direct defaults (`config.DIRECT_JOINT_TRAINING`) are:

- an MLP actor-critic `[512, 256, 128]`;
- 24 steps per env per iteration;
- 3000 iterations (≈ 295M steps at 4096 envs);
- adaptive learning rate 1e-3, entropy 0.01;
- γ 0.995 (also the shaping discount), λ 0.95.

`scripts/smoke_isaac.py --headless --control position` builds the real
scene and steps it with zero and random actions before a long job. The
`hierarchical` control needs `--low-checkpoint` and `--low-checkpoint-sha256`
and the Stage B scene (`cluster/train.sbatch`).

There are three ways to start a run:

- **Fresh weights**: the default.
- **`--initial-checkpoint PATH`**: loads the weights only.
- **`--resume PATH`**: restores the weights, optimizer, iteration and ADR
  ranges. Write the resumed run to a new directory.

The output directory holds `run_info.json`, `metrics.jsonl` (one record per
iteration), TensorBoard scalars, checkpoints and `training_summary.json`.

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
| `09_ppo` | PPO losses, mean episode return, and the shaping return per potential |
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
