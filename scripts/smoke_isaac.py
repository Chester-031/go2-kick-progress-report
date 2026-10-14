"""Build the real Isaac Lab soccer scene for one control and step it.

A check that the scene, the soccer environment and ADR fit together on a
given Isaac Lab install before a long training job.  Zero actions should
leave the robot standing (position control holds the standing pose);
random actions should knock it over.  Prints episode outcomes, base height
and uprightness, ball contact and the time per policy step.

    python scripts/smoke_isaac.py --headless --control position --num-envs 64
"""

from __future__ import annotations

import argparse
import collections
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", choices=("position", "velocity"), default="position")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=300)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app

    import torch

    from robot_soccer.adr import AutomaticDomainRandomization
    from robot_soccer.environment import OUTCOMES
    from robot_soccer.runtime import make_environment

    adr = AutomaticDomainRandomization(seed=0)
    env, adapter, _ = make_environment(
        control=args.control, num_envs=args.num_envs, seed=0, device=args.device, reset_sampler=adr
    )
    print(f"[smoke] control={args.control} num_envs={env.num_envs} actions={env.num_actions} "
          f"ticks_per_action={env.low_steps_per_high_action} horizon={env.max_episode_length} "
          f"observation={tuple(adapter.get_observations()['policy'].shape)}", flush=True)
    generator = torch.Generator(device=env.device).manual_seed(0)
    for label in ("zero", "random"):
        outcomes: collections.Counter[str] = collections.Counter()
        contacts = 0
        heights = []
        ups = []
        rewards = []
        started = time.perf_counter()
        for _ in range(args.steps):
            if label == "zero":
                action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
            else:
                action = torch.randn(env.num_envs, env.num_actions, device=env.device, generator=generator)
            _, reward, _, extras = env.step(action)
            rewards.append(float(reward.mean()))
            heights.append(float(env.robot.data.root_pos_w[:, 2].mean()))
            ups.append(float((-env.robot.data.projected_gravity_b[:, 2]).mean()))
            episodes = extras["episodes"]
            if episodes:
                outcomes.update(OUTCOMES[code] for code in episodes["outcome"].tolist())
                contacts += int(episodes["contact_seen"].sum())
        elapsed = time.perf_counter() - started
        print(f"[smoke] actions={label} ms_per_step={1000 * elapsed / args.steps:.1f} "
              f"mean_reward={sum(rewards) / len(rewards):.4f} base_height first/last={heights[0]:.3f}/{heights[-1]:.3f} "
              f"up first/last={ups[0]:.3f}/{ups[-1]:.3f} episodes={sum(outcomes.values())} "
              f"contacts={contacts} outcomes={dict(outcomes)}", flush=True)
    env.close()
    app.close()


if __name__ == "__main__":
    main()
