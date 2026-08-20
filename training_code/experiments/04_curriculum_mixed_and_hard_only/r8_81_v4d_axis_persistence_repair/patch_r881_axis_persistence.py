#!/usr/bin/env python3
"""Patch R8.79 with axis-specific persistence rewards and fresh target observations."""

from __future__ import annotations

import argparse
from pathlib import Path


def once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"R881_PATCH_FATAL={label}_count:{count}")
    return text.replace(old, new, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = args.source.read_text(encoding="utf-8")

    wrong_old = '''        wrong_direction = (
            wrong_direction * requested_axis.float()
        ).sum(dim=1)
'''
    wrong_new = wrong_old + '''
        # R8.81: total speed is not directional success. Require a minimum
        # signed response on every requested axis, so backward cannot hide in
        # lateral drift and yaw cannot collapse into a static stance.
        signed_response = actual_velocity * torch.sign(requested_velocity)
        response_threshold = torch.tensor(
            [0.055, 0.055, 0.18], device=self.device
        )
        axis_persistence_failure = (
            requested_axis & (signed_response < response_threshold)
        ).float().sum(dim=1)
'''
    text = once(text, wrong_old, wrong_new, "axis_failure_definition")
    text = once(
        text,
        "            - 0.45 * wrong_direction\n",
        "            - 0.45 * wrong_direction\n"
        "            - 1.20 * axis_persistence_failure\n",
        "axis_failure_weight",
    )

    stall_old = '''        motion_requested = (~stand_mask) & (after_xy > 0.08)
        sustained_stall = motion_requested & (robot_speed < 0.04)
        reward = reward - 2.0 * sustained_stall.float()
'''
    stall_new = '''        linear_requested = (~stand_mask) & (after_xy > 0.08)
        angular_speed = after_state["robot_angular_velocity_body"][:, 2].abs()
        angular_requested = (~stand_mask) & (after_error[2].abs() > 0.08)
        sustained_stall = (
            (linear_requested & (robot_speed < 0.04))
            | (angular_requested & (angular_speed < 0.08))
        )
        reward = reward - 2.0 * sustained_stall.float()
'''
    text = once(text, stall_old, stall_new, "yaw_aware_stall")

    text = once(
        text, "            [0.0, 0.0, 0.85],\n",
        "            [0.0, 0.0, 1.00],\n", "turn_left_strength",
    )
    text = once(
        text, "            [0.0, 0.0, -0.85],\n",
        "            [0.0, 0.0, -1.00],\n", "turn_right_strength",
    )

    eval_old = '''            env.begin_high_action(high_actions)
            for low_index in range(env.low_level_steps_per_action):
'''
    eval_new = '''            env.begin_high_action(high_actions)
            # Refresh after installing the new Delta Pose target. The inherited
            # loop otherwise gives the first low substep the previous target.
            low_obs = low_adapter.get_observations().to(low_runner.device)
            for low_index in range(env.low_level_steps_per_action):
'''
    text = once(text, eval_old, eval_new, "evaluation_target_refresh")
    train_old = '''                env.begin_high_action(actions)
                for low_index in range(env.low_level_steps_per_action):
'''
    train_new = '''                env.begin_high_action(actions)
                low_obs = low_adapter.get_observations().to(low_runner.device)
                for low_index in range(env.low_level_steps_per_action):
'''
    text = once(text, train_old, train_new, "training_target_refresh")

    text = text.replace("R879_", "R881_")
    text = text.replace("r879_", "r881_")
    text = text.replace("R8.79", "R8.81")
    text = text.replace("r8_79", "r8_81")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print("R881_AXIS_PERSISTENCE_PATCH_OK")


if __name__ == "__main__":
    main()
