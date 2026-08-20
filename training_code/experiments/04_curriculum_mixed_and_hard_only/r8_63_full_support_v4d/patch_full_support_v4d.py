#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FULL_SUPPORT_RESET = '''        count = int(env_ids.numel())

        # R8.63 full-support reset: no progress-dependent difficulty.
        # Robot and ball are sampled independently across the playable
        # interior. Wall-margin cases are evaluated separately.
        field_x_min = -3.0
        field_x_max = 2.4
        field_y_min = -1.9
        field_y_max = 1.9
        min_robot_ball_distance = 0.75

        ball_x_local = field_x_min + (
            field_x_max - field_x_min
        ) * self._rand((count,))
        ball_y_local = field_y_min + (
            field_y_max - field_y_min
        ) * self._rand((count,))

        robot_x_local = field_x_min + (
            field_x_max - field_x_min
        ) * self._rand((count,))
        robot_y_local = field_y_min + (
            field_y_max - field_y_min
        ) * self._rand((count,))

        # Keep the initial state physically separable without imposing a
        # preferred robot-ball side or heading.
        for _ in range(8):
            distance = torch.sqrt(
                (robot_x_local - ball_x_local).square()
                + (robot_y_local - ball_y_local).square()
                + 1.0e-8
            )
            invalid = distance < min_robot_ball_distance
            if not invalid.any():
                break
            robot_x_local[invalid] = field_x_min + (
                field_x_max - field_x_min
            ) * self._rand((int(invalid.sum().item()),))
            robot_y_local[invalid] = field_y_min + (
                field_y_max - field_y_min
            ) * self._rand((int(invalid.sum().item()),))

        robot_yaw = (
            2.0 * self._rand((count,)) - 1.0
        ) * torch.pi
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    source = Path(a.source).resolve()
    text = source.read_text(encoding="utf-8")

    high_old = '''        difficulty = min(
            1.0,
            self.total_resets
            / max(1.0, float(self.num_envs * 300)),
        )
        difficulty_tensor = torch.full(
            (self.num_envs, 1),
            difficulty,
            device=self.device,
        )
'''
    high_new = '''        # Keep the 17D interface, but remove difficulty as a training
        # curriculum signal. The final-support policy always sees 1.0.
        difficulty_tensor = torch.ones(
            (self.num_envs, 1),
            device=self.device,
        )
'''
    if text.count(high_old) != 1:
        raise RuntimeError(f"high difficulty block count={text.count(high_old)}")
    text = text.replace(high_old, high_new, 1)

    start = text.find("        count = int(env_ids.numel())\n        difficulty = min(")
    end_marker = "        origins = self.env_origins[env_ids]"
    end = text.find(end_marker, start)
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError("reset difficulty block not found")
    if text.count("        count = int(env_ids.numel())\n        difficulty = min(") != 1:
        raise RuntimeError("reset difficulty block is not unique")
    text = text[:start] + FULL_SUPPORT_RESET + "\n" + text[end:]

    out = Path(a.out)
    out.write_text(text, encoding="utf-8")
    report = {
        "version": "r8_63_full_support_v4d_candidate",
        "training": False,
        "parent_mutation": False,
        "source_parent": str(source),
        "high_observation_dim": 17,
        "low_observation_dim": 62,
        "low_action_dim": 12,
        "progressive_difficulty_removed": True,
        "difficulty_observation_fixed": 1.0,
        "reset_distribution": {
            "robot_x": [-3.0, 2.4],
            "robot_y": [-1.9, 1.9],
            "ball_x": [-3.0, 2.4],
            "ball_y": [-1.9, 1.9],
            "robot_yaw": [-3.141592653589793, 3.141592653589793],
            "min_robot_ball_distance": 0.75,
            "wall_margin_evaluation": "separate_diagnostic_set",
        },
        "reward_contract": {"goal": 1, "own_goal": -1, "other": 0},
        "low_step_contract": "unchanged_single_low_env_step",
        "next_step": "frozen_checkpoint_full_support_coverage_scan",
    }
    report_path = out.with_suffix(".patch_report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("R863_FULL_SUPPORT_PATCH_OK")
    print("TRAINING=FALSE")
    print("PARENT_MUTATION=FALSE")
    print(f"PATCHED_SOURCE={out}")
    print(f"PATCH_REPORT={report_path}")


if __name__ == "__main__":
    main()
