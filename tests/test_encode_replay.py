"""Which recorded episodes a replay reel shows."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("encode_replay", ROOT / "scripts" / "encode_replay.py")
encode = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(encode)


def episode(key: str, difficulty: float, outcome: str, distance: float) -> dict:
    return {"key": key, "difficulty": difficulty, "outcome": outcome,
            "knobs": {"robot_ball_distance_m": distance, "ball_goal_distance_m": 1.0}}


EPISODES = [
    episode("d0.5_e0", 0.5, "goal", 9.0),
    episode("d1_e0", 1.0, "timeout", 7.0),
    episode("d1_e1", 1.0, "goal", 2.0),
    episode("d1_e2", 1.0, "goal", 6.0),
    episode("d1_e3", 1.0, "goal", 4.0),
]


class ChooseTest(unittest.TestCase):
    def test_best_takes_the_longest_goals_at_the_hardest_difficulty(self) -> None:
        self.assertEqual([e["key"] for e in encode.choose(EPISODES, 2, "best")], ["d1_e2", "d1_e3"])

    def test_first_is_the_unbiased_sample_at_the_hardest_difficulty(self) -> None:
        """record_trajectories keeps each slot's first episode, so the first
        recorded episodes are a fair sample, failures included."""

        self.assertEqual([e["key"] for e in encode.choose(EPISODES, 2, "first")], ["d1_e0", "d1_e1"])


if __name__ == "__main__":
    unittest.main()
