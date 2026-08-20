#!/usr/bin/env python3
"""Convert the completed R8.78 trainer into an isolated directional R8.79 repair."""

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"R879_PATCH_FATAL={label}_count:{count}")
    return text.replace(old, new, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = args.source.read_text(encoding="utf-8")

    required_parent = (
        "def run_r878_low_repair(",
        "sustained_stall = motion_requested & (robot_speed < 0.04)",
        "R878_LOW_REPAIR_ACCEPTED",
    )
    missing = [item for item in required_parent if item not in text]
    if missing:
        raise SystemExit(f"R879_PATCH_FATAL=parent_markers_missing:{missing}")

    push_old = '''        push_mask = (
            (state["robot_ball_distance"] < 0.95)
            & (state["robot_ball_x_body"] > 0.03)
            & (state["ball_goal_alignment"] > 0.55)
            & (action[:, 0] > -0.20)
        )
'''
    push_new = push_old + '''        # R8.79 primitive repair is a locomotion unit test. A nearby
        # curriculum ball must not silently rewrite lateral/turn commands as push.
        if os.environ.get("R879_FORCE_LOCOMOTION_PRIMITIVES") == "1":
            push_mask = torch.zeros_like(push_mask)
'''
    text = replace_once(text, push_old, push_new, "push_isolation")

    velocity_old = '''        velocity_track = torch.exp(
            -linear_error.square().sum(dim=1) / (0.45 ** 2)
            -angular_error.square() / (0.70 ** 2)
        )
'''
    velocity_new = velocity_old + '''
        # R8.79: the old isotropic term is too broad and lets forward motion
        # score well for backward, lateral and yaw requests. Track the actual
        # signed reference command per axis and penalize reversed components.
        actual_velocity = torch.cat(
            (
                after_state["robot_velocity_body"][:, :2],
                after_state["robot_angular_velocity_body"][:, 2:3],
            ),
            dim=1,
        )
        requested_velocity = self.last_reference_command
        directional_error = actual_velocity - requested_velocity
        directional_velocity_track = torch.exp(
            -directional_error[:, 0].square() / (0.20 ** 2)
            -directional_error[:, 1].square() / (0.16 ** 2)
            -directional_error[:, 2].square() / (0.32 ** 2)
        )
        requested_axis = requested_velocity.abs() > 0.04
        wrong_direction = torch.relu(
            -actual_velocity * torch.sign(requested_velocity)
        )
        wrong_direction = (
            wrong_direction
            / torch.tensor([0.30, 0.25, 0.60], device=self.device)
        )
        wrong_direction = (
            wrong_direction * requested_axis.float()
        ).sum(dim=1)
'''
    text = replace_once(text, velocity_old, velocity_new, "directional_reward")
    text = replace_once(
        text,
        "            + 0.45 * velocity_track\n",
        "            + 0.45 * velocity_track\n"
        "            + 0.90 * directional_velocity_track\n"
        "            - 0.45 * wrong_direction\n",
        "directional_reward_weight",
    )

    text = replace_once(
        text,
        "            [-0.75, 0.0, 0.0],\n",
        "            [-1.00, 0.0, 0.0],\n",
        "backward_primitive_strength",
    )

    # Keep the proven R8.78 control flow, but give this continuation distinct
    # markers, filenames and environment switches.
    text = text.replace("R878_", "R879_")
    text = text.replace("r878_", "r879_")
    text = text.replace("R8.78", "R8.79")
    text = text.replace("r8_78", "r8_79")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print("R879_DIRECTIONAL_BALANCE_PATCH_OK")


if __name__ == "__main__":
    main()
