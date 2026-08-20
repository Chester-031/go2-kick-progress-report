# Isaac-Go2-Low-Level-RL

A clean, independent Isaac Lab 2.3.2 project for training a Unitree Go2
low-level locomotion policy with PPO from a randomly initialized policy.

## Scope

The first-stage policy receives:

- commanded forward velocity `vx`
- commanded lateral velocity `vy`
- commanded yaw rate `wz`
- robot proprioception and the previous action

It outputs 12 relative joint-position targets.

The task trains:

- stable standing
- forward and backward locomotion
- lateral locomotion
- yaw turning
- velocity tracking
- posture stability
- recovery from periodic velocity pushes

This repository does **not** include official locomotion checkpoints, soccer
code, kick code, Franklin code, old project copies, debug scripts, videos,
assets, logs, or trained checkpoints.

## Repository layout

```text
Isaac-Go2-Low-Level-RL/
├── README.md
├── .gitignore
├── init_git_remote.sh
├── cluster/
│   └── train_go2_low_level.sbatch
├── scripts/
│   └── rsl_rl/
│       └── train.py
└── source/
    └── isaac_go2_low_level_rl/
        ├── setup.py
        └── isaac_go2_low_level_rl/
            └── tasks/
                └── manager_based/
                    └── locomotion/
                        └── velocity/
                            ├── go2_low_level_env_cfg.py
                            └── agents/
                                └── rsl_rl_ppo_cfg.py
```

The required `__init__.py` package-registration files are also included.

## Requirements

- Isaac Sim compatible with Isaac Lab 2.3.2
- Isaac Lab 2.3.2
- `isaaclab_rl`
- `isaaclab_tasks`
- `isaaclab_assets`
- `rsl-rl-lib` compatible with the Isaac Lab 2.3.2 environment

No official Go2 policy or checkpoint is loaded.

## Local setup

```bash
cd /home/ubuntu
tar -xzf /home/ubuntu/下载/Isaac-Go2-Low-Level-RL-mainline-v1-refresh.tar.gz
cd /home/ubuntu/Isaac-Go2-Low-Level-RL

export PYTHONPATH="$PWD/source/isaac_go2_low_level_rl:${PYTHONPATH:-}"
```

Optional editable installation:

```bash
python -m pip install -e source/isaac_go2_low_level_rl --no-deps
```

## Create the private GitHub repository

The intended repository is:

```text
Chester-031/Isaac-Go2-Low-Level-RL
```

Run:

```bash
./init_git_remote.sh
```

When GitHub CLI is unavailable or not logged in, create an empty private
repository named `Isaac-Go2-Low-Level-RL`, then run the commands printed by
the script.

## Cluster location

```text
/scratch/lj2634/Isaac-Go2-Low-Level-RL
```

Clone after the first push:

```bash
ssh lj2634@login.torch.hpc.nyu.edu
cd /scratch/lj2634
git clone git@github.com:Chester-031/Isaac-Go2-Low-Level-RL.git
```

## First launch through the final training entry point

This is not a separate smoke-test program. It uses the same task, PPO
configuration, training script, and Slurm script as the long run.

```bash
cd /scratch/lj2634/Isaac-Go2-Low-Level-RL

NUM_ENVS=64 \
MAX_ITERATIONS=5 \
sbatch cluster/train_go2_low_level.sbatch
```

After the job enters PPO iterations and creates its normal RSL-RL output,
start the long run:

```bash
NUM_ENVS=2048 \
MAX_ITERATIONS=1500 \
sbatch cluster/train_go2_low_level.sbatch
```

Useful overrides:

```bash
ISAACLAB_PYTHON=/scratch/lj2634/envs/isaaclab232_v2/bin/python \
NUM_ENVS=2048 \
MAX_ITERATIONS=1500 \
SEED=42 \
DEVICE=cuda:0 \
sbatch cluster/train_go2_low_level.sbatch
```

Training logs are written under:

```text
logs/rsl_rl/go2_low_level/
```

## Custom task identity

```text
Isaac-Go2-Low-Level-Flat-v0
```

The task is registered by this project. It does not invoke the official
`Isaac-Velocity-Flat-Unitree-Go2-v0` task and does not load any official
policy weights.
