# Robot Soccer: Reference + Full Training Core

This directory is the concise runnable version of the no-Kick training code.
It keeps the environment, policy, PPO loop, fixed evaluation, and the metrics
needed to diagnose training. Deployment-specific orchestration and raw rollout
dumps are intentionally outside this package.

## What this code runs

- One recurrent high policy for both conditions; there is no condition router.
- A stationary reset distribution from the first rollout: 75% Reference and
  25% Full-distribution cases.
- A 70-D observation and a 3-D `(vx, vy, wz)` high-level action.
- A one-layer GRU with hidden size 256 and actor/critic MLPs
  `[256, 256, 128]`.
- A frozen 50 Hz low policy. The high action is held for five low ticks, so the
  high policy runs at 10 Hz.
- A 20 s finite horizon (200 high-policy steps), without timeout bootstrap.
- Task reward: normalized progress along
  `robot -> behind-ball staging target -> selected goal`, plus `+1` for a
  strict goal and `-1` for an own goal.
- Safety-first same-tick termination: unsafe outcomes reject a simultaneous
  goal.
- PPO defaults: 4096 environments, 200 rollout steps, 10 iterations, learning
  rate `3e-4`, entropy coefficient `0.03`, gamma `0.997`, lambda `0.98`, five
  epochs, and four minibatches.

The reported experiment used a pre-action high-policy checkpoint with a fresh
optimizer. It did **not** warm-start from a Reference-trained checkpoint and it
did **not** use a staged curriculum. A future Reference-to-Full transfer test
must therefore be implemented as a separate experiment, not silently folded
into this reproduction.

## Layout

```text
src/robot_soccer/       behavior, environment, PPO runtime, train/evaluate CLIs
runtime_project/        Isaac Lab task registration and scene configuration
cluster/                minimal Slurm launch examples
tests/                  focused behavior and logging tests
```

The sibling `robot_soccer_reference_full_v1` directory is the byte-preserving
research snapshot. This directory is a curated equivalent intended for code
review; it is not a byte-identical provenance artifact.

## Requirements

- Isaac Lab 2.3.2 / compatible Isaac Sim installation
- PyTorch from that Isaac Lab environment
- `rsl_rl` 3.1.2
- `tensordict`
- TensorBoard is optional; JSONL metrics are always written
- the qualified frozen-low checkpoint
- the pre-action high-policy initialization checkpoint

Checkpoints are runtime inputs and are intentionally not committed.
The training hashes below bind the qualified low policy and the reported
pre-action initialization. For a different experiment, compute and pass that
experiment's own hashes.

Install the two Python packages into the active Isaac Lab environment with
`python -m pip install -e . --no-deps`, or use the supplied Slurm wrappers,
which set both source roots on `PYTHONPATH`.

## Training

On the Cluster, use the environment's direct Python rather than the local
`isaaclab.sh` playback launcher:

```bash
export PYTHONPATH="$PWD/src:$PWD/runtime_project/isaac_go2_low_level_rl"
python -m robot_soccer.train \
  --headless \
  --device cuda:0 \
  --low-checkpoint /path/to/qualified_low.pt \
  --low-checkpoint-sha256 f46d3b760c43664350aac8099ae5bbd32bc1207f97c8814f5d21c9b531c3d96c \
  --initial-checkpoint /path/to/pre_action_high.pt \
  --initial-checkpoint-sha256 ef4637dedd43f3d632d37ff228b56f002320abca3e1a81d5e8ce80f526678d79 \
  --output-dir /path/to/run
```

The command prints one line per PPO iteration. Detailed values go to
`metrics.jsonl` and, when available, `tensorboard/`. Milestone checkpoints are
written at iterations 1, 3, 5, and 10. Output directories must be new or empty
so results from separate runs cannot be mixed.

## Fixed evaluation

Use the same checkpoint and run each split read-only:

```bash
HIGH_CHECKPOINT=/path/to/model_iteration_10.pt
HIGH_SHA256=$(sha256sum -- "$HIGH_CHECKPOINT" | awk '{print $1}')

python -m robot_soccer.evaluate --headless --device cuda:0 \
  --low-checkpoint /path/to/qualified_low.pt \
  --low-checkpoint-sha256 f46d3b760c43664350aac8099ae5bbd32bc1207f97c8814f5d21c9b531c3d96c \
  --checkpoint "$HIGH_CHECKPOINT" \
  --checkpoint-sha256 "$HIGH_SHA256" \
  --split reference --output-dir /path/to/eval/reference

python -m robot_soccer.evaluate --headless --device cuda:0 \
  --low-checkpoint /path/to/qualified_low.pt \
  --low-checkpoint-sha256 f46d3b760c43664350aac8099ae5bbd32bc1207f97c8814f5d21c9b531c3d96c \
  --checkpoint "$HIGH_CHECKPOINT" \
  --checkpoint-sha256 "$HIGH_SHA256" \
  --split full --output-dir /path/to/eval/full

python -m robot_soccer.evaluate --headless --device cuda:0 \
  --low-checkpoint /path/to/qualified_low.pt \
  --low-checkpoint-sha256 f46d3b760c43664350aac8099ae5bbd32bc1207f97c8814f5d21c9b531c3d96c \
  --checkpoint "$HIGH_CHECKPOINT" \
  --checkpoint-sha256 "$HIGH_SHA256" \
  --split hard --output-dir /path/to/eval/hard
```

The archived reported iteration-10 checkpoint has SHA-256
`cacf5684670d621faf9c51efdde7ee0364bda2183936554dab337cc69a21d746`.

Each evaluation writes one episode JSONL file and one compact summary. It
checks that neither the high policy nor frozen low policy changes and performs
zero optimizer steps.

## Metrics for advisor review

The training logger uses cumulative completed episodes as the TensorBoard
x-axis and records:

- mean episode return;
- goal and contact rates, overall and by reset condition;
- goal/contact rates binned by initial robot-ball distance;
- goal/contact rates binned by initial ball-goal distance;
- goal/contact rates binned by initial facing error;
- the combined facing-error × robot-ball-distance bins.

These are the minimal diagnostics needed to distinguish failure to approach or
turn from failure after contact.

## Tests

The CPU-only tests do not start Isaac Sim:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Run an Isaac smoke test in the actual Isaac Lab environment before launching a
long training job.
