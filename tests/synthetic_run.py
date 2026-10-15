"""Write a plausible training output directory without a simulator.

Drives the real ``AutomaticDomainRandomization`` and the real
``TrainingMetrics`` writer with synthetic finished episodes from a policy
whose skill grows over training: an episode scores with probability
``sigmoid(8 (skill - hardness))``, where hardness grows with every knob.
Used to test and visually check ``scripts/plot_training.py``.

    PYTHONPATH=src:. python tests/synthetic_run.py OUT_DIR [--iterations 200] [--seed 0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from robot_soccer.adr import PARAMETERS, AutomaticDomainRandomization
from robot_soccer.metrics import OUTCOME_NAMES, TrainingMetrics


def synthetic_episodes(adr: AutomaticDomainRandomization, count: int, skill: float,
                       generator: torch.Generator) -> dict[str, torch.Tensor]:
    batch = adr(count)
    values = batch.tags["adr_values"].double()
    minimum = torch.tensor([parameter.minimum for parameter in PARAMETERS], dtype=torch.float64)
    maximum = torch.tensor([parameter.maximum for parameter in PARAMETERS], dtype=torch.float64)
    hardness = ((values - minimum) / (maximum - minimum) * torch.tensor([0.4, 0.15, 0.15, 0.15, 0.15], dtype=torch.float64)).sum(1)
    goal = torch.rand(count, generator=generator, dtype=torch.float64) < torch.sigmoid(8.0 * (skill - hardness))
    # Non-goals end as own goal, boundary or timeout.
    other = torch.multinomial(torch.tensor([0.04, 0.26, 0.70]), count, replacement=True,
                              generator=generator) + 1
    outcome = torch.where(goal, torch.zeros_like(other), other)
    contact = goal | (torch.rand(count, generator=generator) < 0.3 + 0.5 * skill)
    ticks = torch.where(goal, (200 + 600 * hardness).long(), torch.randint(100, 1000, (count,), generator=generator))
    episodes = {
        "outcome": outcome,
        "goal": goal,
        "own_goal": outcome == OUTCOME_NAMES.index("own_goal"),
        "contact_seen": contact,
        "unsafe_pre_goal": (outcome >= 1) & (outcome <= 2),
        "falls": torch.randint(0, 3, (count,), generator=generator),
        "seconds_on_ground": 2.0 * torch.rand(count, generator=generator),
        "max_joint_overshoot_rad": 0.01 * torch.rand(count, generator=generator),
        "shaping_return": 0.3 * goal.double() + 0.05 * torch.rand(count, generator=generator),
        "shaping_ball_goal": 0.25 * goal.double(),
        "shaping_robot_ball": 0.05 * torch.rand(count, generator=generator, dtype=torch.float64),
        "shaping_base_height": -0.02 * torch.rand(count, generator=generator, dtype=torch.float64),
        "shaping_upright": -0.01 * torch.rand(count, generator=generator, dtype=torch.float64),
        "episode_native_ticks": ticks,
        "first_true_contact_native_tick": torch.where(contact, ticks // 3, torch.full_like(ticks, -1)),
        "requested_goal_axis_ball_displacement_m": torch.where(goal, values[:, 0], 0.2 * values[:, 0]).float(),
    }
    episodes.update(batch.tags)
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--episodes-per-iteration", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    generator = torch.Generator().manual_seed(args.seed)
    adr = AutomaticDomainRandomization(seed=args.seed)
    metrics = TrainingMetrics(
        args.output,
        run_info={"arguments": {"adr_low_threshold": "0.3", "adr_high_threshold": "0.6"}, "synthetic": True},
        tensorboard=False,
    )
    for iteration in range(1, args.iterations + 1):
        skill = 0.15 + 0.75 * iteration / args.iterations + 0.03 * args.seed
        episodes = synthetic_episodes(adr, args.episodes_per_iteration, skill, generator)
        adr.record(episodes)
        metrics.write(
            iteration=iteration,
            episodes=episodes,
            losses={"value_function": 0.5 / iteration ** 0.3, "surrogate": -0.01, "entropy": 4.0 - skill},
            adr=adr.take_period_metrics(),
            timing={"rollout_seconds": 30.0, "env_step_seconds": 26.0, "policy_seconds": 4.0,
                    "update_seconds": 6.0, "transitions_per_second": 819_200 / 36.0},
        )
    metrics.close()


if __name__ == "__main__":
    main()
