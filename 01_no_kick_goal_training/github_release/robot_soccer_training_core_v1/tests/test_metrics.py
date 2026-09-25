from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest


PACKAGE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))

from robot_soccer.metrics import TrainingMetrics, summarize_episodes


def event(
    *,
    outcome: str,
    regime: str,
    contact: bool,
    robot=(0.0, 0.0),
    ball=(1.0, 0.0),
    yaw=0.0,
    side=1,
    progress=0.1,
):
    return {
        "outcome": outcome,
        "goal": outcome == "goal",
        "own_goal": outcome == "own_goal",
        "contact_seen": contact,
        "unsafe_pre_goal": outcome in {"own_goal", "fall", "boundary", "joint_hard_limit"},
        "reset_regime": regime,
        "reset_robot_xy_m": list(robot),
        "reset_robot_yaw_rad": yaw,
        "reset_ball_xy_m": list(ball),
        "goal_side": side,
        "cumulative_progress_reward": progress,
    }


class MetricsTests(unittest.TestCase):
    def test_episode_summary_separates_return_goal_and_contact(self):
        rows = [
            event(outcome="goal", regime="REFERENCE", contact=True),
            event(outcome="timeout", regime="FULL", contact=False, yaw=3.141592653589793),
            event(outcome="own_goal", regime="FULL", contact=True, progress=-0.2),
        ]
        summary = summarize_episodes(rows)
        self.assertEqual(summary["completed_episodes"], 3)
        self.assertEqual(summary["goal_count"], 1)
        self.assertEqual(summary["contact_count"], 2)
        self.assertAlmostEqual(summary["mean_episode_return"], (1.1 + 0.1 - 1.2) / 3)
        self.assertEqual(summary["by_regime"]["REFERENCE"]["goals"], 1)
        self.assertEqual(summary["by_regime"]["FULL"]["episodes"], 2)
        self.assertTrue(any(key.startswith("facing_x_robot_ball/") for key in summary["geometry_bins"]))

    def test_jsonl_uses_cumulative_completed_episodes(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = TrainingMetrics(directory, tensorboard=False)
            logger.write(
                iteration=1,
                events=[event(outcome="goal", regime="REFERENCE", contact=True)],
                losses={"surrogate": 0.25},
            )
            logger.write(
                iteration=2,
                events=[
                    event(outcome="timeout", regime="FULL", contact=False),
                    event(outcome="goal", regime="REFERENCE", contact=True),
                ],
                losses={"surrogate": 0.10},
            )
            logger.close()
            records = [
                json.loads(line)
                for line in (pathlib.Path(directory) / "metrics.jsonl").read_text().splitlines()
            ]
            self.assertEqual([row["cumulative_completed_episodes"] for row in records], [1, 3])


if __name__ == "__main__":
    unittest.main()
