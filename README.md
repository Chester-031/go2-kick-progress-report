# Go2 Robot Soccer — Stage B Training

This branch contains only the current no-Kick Stage B training path. It omits
the evaluation/recovery pipeline, historical experiments, generated evidence,
videos, and example outputs.

## Training design

- recurrent 70-D observation / 3-D velocity-command policy;
- frozen low-level locomotion policy at 50 Hz;
- high-level policy at 10 Hz with a 20 s episode horizon;
- 4,096 parallel environments and 200 rollout steps per PPO iteration;
- 30 PPO iterations, five epochs and four minibatches;
- checkpoints at iterations 5, 10, 15, and 30;
- no learned kick action and no kick-strength reward.

The reset curriculum is fixed before training and never adapts to results:

| Iterations | Facing angles | Robot–ball distance | Ball–goal distance |
|---|---|---|---|
| 1–5 | 45° | 0.8, 1.6 m | 2 m |
| 6–10 | 45° | 0.8, 1.6 m | 2, 4 m |
| 11–15 | 45° | 0.8, 1.6 m | 2, 4, 6 m |
| 16–30 | 0°, 45°, 90°, 180° | 0.8, 1.6 m | 2, 4, 6 m |

Changing phases does not reset active episodes, the optimizer, recurrent
state, or the simulator. New resets use the newly active phase.

## Run

Use the direct Python executable from the Isaac Lab environment on the
Cluster:

```bash
export PYTHONPATH="$PWD/src:$PWD/runtime_project/isaac_go2_low_level_rl"
python -m robot_soccer.train \
  --headless --device cuda:0 \
  --low-checkpoint /path/to/qualified_low.pt \
  --low-checkpoint-sha256 LOW_SHA256 \
  --initial-checkpoint /path/to/pre_action_high.pt \
  --initial-checkpoint-sha256 INITIAL_SHA256 \
  --output-dir /path/to/new_run
```

`cluster/train.sbatch` is the equivalent minimal Slurm entry point. Runtime
checkpoints are inputs and are not included in this repository.
