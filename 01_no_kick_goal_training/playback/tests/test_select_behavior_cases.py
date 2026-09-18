from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from playback.select_behavior_cases import (
    ROW_FIELDS,
    ROW_SCHEMA,
    SelectionError,
    load_rows,
    select_cases,
)


def row(assignment: int, **updates):
    value = {
        "schema": ROW_SCHEMA,
        "evaluation_assignment": assignment,
        "counts_for_evaluation": True,
        "hard_goal_facing_ball_behind": False,
        "bucket": "front_near",
        "goal_side": 1,
        "sample_serial": -1,
        "item_seed": -1,
        "reset_robot_xy_m": [0.0, 0.0],
        "reset_robot_yaw_rad": 0.0,
        "reset_ball_xy_m": [1.0, 0.0],
        "outcome": "timeout",
        "goal": False,
        "own_goal": False,
        "boundary": False,
        "fall": False,
        "joint_hard_limit": False,
        "timeout": True,
        "unsafe_pre_goal": False,
        "requested_goal_line_reached": False,
        "contact_seen": False,
        "first_true_contact_native_tick": None,
        "maximum_contact_force_n": 0.0,
        "terminal_contact_force_n": 0.0,
        "terminal_base_contact_force_n": 0.0,
        "terminal_joint_hard_limit_excess_rad": 0.0,
        "native_safe_guard_changed_elements": 0,
        "native_safe_guard_max_correction_rad": 0.0,
        "episode_high_ticks": 200,
        "episode_native_ticks": 1000,
        "terminal_robot_xy_m": [0.0, 0.0],
        "terminal_ball_xy_m": [1.0, 0.0],
        "initial_l_m": 1.0,
        "final_l_m": 1.0,
        "cumulative_progress_reward": 0.0,
        "requested_goal_axis_ball_displacement_m": 0.0,
        "abs_lateral_ball_drift_m": 0.0,
    }
    value.update(updates)
    if value["contact_seen"] and value["first_true_contact_native_tick"] is None:
        value["first_true_contact_native_tick"] = 10
    return value


def contact(assignment: int, axis: float, lateral: float, hard: bool = False):
    return row(
        assignment,
        contact_seen=True,
        maximum_contact_force_n=2.0,
        requested_goal_axis_ball_displacement_m=axis,
        abs_lateral_ball_drift_m=lateral,
        hard_goal_facing_ball_behind=hard,
    )


class BehaviorCaseSelectionTests(unittest.TestCase):
    def make_rows(self):
        rows = [row(assignment) for assignment in range(288)]
        rows[17] = row(
            17,
            outcome="own_goal",
            own_goal=True,
            timeout=False,
            unsafe_pre_goal=True,
            contact_seen=True,
            maximum_contact_force_n=2.0,
        )
        rows[2] = row(
            2,
            outcome="own_goal",
            own_goal=True,
            timeout=False,
            unsafe_pre_goal=True,
            contact_seen=True,
            maximum_contact_force_n=2.0,
        )
        rows[10] = contact(10, 0.1, 0.1, True)
        rows[11] = contact(11, 0.2, 0.2, True)
        rows[12] = contact(12, 0.3, 0.3, True)
        rows[20] = contact(20, -3.0, 0.4)
        rows[21] = contact(21, -2.0, 0.5)
        rows[22] = contact(22, -1.0, 0.6)
        rows[30] = contact(30, 0.4, 0.7)
        rows[31] = contact(31, 0.5, 0.8)
        rows[32] = contact(32, 0.6, 0.9)
        rows[40] = row(
            40,
            outcome="goal",
            goal=True,
            timeout=False,
            requested_goal_line_reached=True,
            contact_seen=True,
            maximum_contact_force_n=2.0,
            requested_goal_axis_ball_displacement_m=0.45,
            abs_lateral_ball_drift_m=100.0,
        )
        return rows

    def test_selects_five_distinct_deterministic_cases(self):
        result = select_cases(self.make_rows(), "a" * 64)
        self.assertEqual(
            [case["label"] for case in result["cases"]],
            ["own_goal", "hard_contact", "wrong_way_contact", "high_lateral_drift", "modest_forward_push"],
        )
        self.assertEqual(result["selected_assignments"], [2, 11, 21, 30, 12])
        self.assertEqual(len(set(result["selected_assignments"])), 5)

    def test_load_rows_rejects_field_drift(self):
        rows = self.make_rows()
        rows[0]["unexpected"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")
            with self.assertRaisesRegex(SelectionError, "field set changed"):
                load_rows(path)

    def test_load_rows_accepts_exact_schema_and_rejects_duplicate_assignment(self):
        rows = self.make_rows()
        self.assertEqual(set(rows[0]), ROW_FIELDS)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")
            loaded, digest = load_rows(path)
            self.assertEqual(len(loaded), 288)
            self.assertEqual(len(digest), 64)
            rows[-1]["evaluation_assignment"] = 0
            path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")
            with self.assertRaisesRegex(SelectionError, "exact-once"):
                load_rows(path)


if __name__ == "__main__":
    unittest.main()
