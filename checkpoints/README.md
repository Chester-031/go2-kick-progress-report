# Checkpoints

## `go2_soccer_position_it16800.pt`

The best direct-joint soccer policy so far: the `position` control at PPO
iteration 16,800. It walks, finds the ball and scores from anywhere on the
field, with no low-level locomotion policy.

| | |
|---|---|
| SHA-256 | `de9500a5139d3e1edb38da36d49d0aa25edaab3d2dbdd5ebded80a6829505133` |
| Control | `position` (12 joint position targets, ±0.25 rad around standing, 50 Hz) |
| Network | MLP actor and critic `[512, 256, 128]`, 70 observations → 12 actions |
| Environment | this branch's direct scene: falls do not end episodes; episodes end on a goal, own goal, boundary or the 20 s timeout |
| Steps | 16,800 iterations × 24 steps × 4096 envs ≈ 1.65 B env steps |
| Contents | `model_state_dict`, `optimizer_state_dict`, `iter`, `infos`, `adr_state` (so `--resume` continues the curriculum) |

Goal rate on the fixed-difficulty episodes during the 500 iterations before
this checkpoint (difficulty d: every knob drawn from its start value out to
d of the way to its limits; 1 is the whole field):

| Difficulty 0.25 | Difficulty 0.5 | Difficulty 1 |
|---|---|---|
| 0.996 (n = 3746) | 0.994 (n = 3612) | **0.921** (n = 3574, ±0.009) |

At this point ADR trains on 98% of the full task: every knob is at its
limits except robot-to-ball, at 0.4–7.4 m of 0.4–8.0 m. Mean time to a goal
is 3.2 s. The robot touches the ground with its trunk in 92% of episodes
but spends only 0.3 s per episode there: brief contacts during head-down
pushes, not lying down. These rates were measured while training, on the
policy of the preceding iterations, not by a separate evaluation of this
file.

### How it was trained

One run, seed 83001, 4096 envs, with the PPO settings of Optuna trial 34
(`scripts/tune_optuna.py`, study `ppo-control-v1`):

```bash
python -m robot_soccer.train --headless --device cuda:0 --control position --num-envs 4096 --seed 83001 \
  --gamma 0.9968 --lam 0.955 --learning-rate 1.13e-3 --schedule adaptive --desired-kl 0.0104 \
  --entropy-coef 0.00186 --steps-per-env 24 --num-mini-batches 4 --num-learning-epochs 4 --clip-param 0.171 \
  --iterations 6000 --output-dir RUN
```

then resumed twice with the same flags (`--resume RUN/model_iteration_6000.pt
--iterations 12000`, then from iteration 12,000 with `--iterations 20000`).
ADR used the defaults: thresholds 0.2 / 0.4, robot-to-ball as the lead knob
until 4 m.

### Use

```bash
# Record episodes headless (Isaac Lab 2.0.2), then render them (cluster/render_isaac6.sbatch).
python scripts/record_trajectories.py --headless --control position \
  --checkpoint checkpoints/go2_soccer_position_it16800.pt --difficulty 1 --episodes 10 --out traj.npz
# Keep training.
python -m robot_soccer.train --headless --control position --resume checkpoints/go2_soccer_position_it16800.pt \
  --iterations 20000 --output-dir RUN2 [same PPO flags as above]
```
