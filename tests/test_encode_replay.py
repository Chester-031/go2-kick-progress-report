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

    def test_sample_draws_any_hardest_difficulty_episode(self) -> None:
        """Recorded keys are numbered in the order episodes finished (quick
        goals first, 20 s timeouts last), so the first keys are not a fair
        sample.  The recorded set is (each slot's first episode), so a
        random subset of it is: across seeds every episode at the hardest
        difficulty, the slow ones included, gets drawn."""

        drawn = set()
        for seed in range(50):
            chosen = encode.choose(EPISODES, 2, "sample", seed=seed)
            self.assertEqual(len(chosen), 2)
            self.assertTrue(all(e["difficulty"] == 1.0 for e in chosen))
            drawn |= {e["key"] for e in chosen}
        self.assertEqual(drawn, {"d1_e0", "d1_e1", "d1_e2", "d1_e3"})


if __name__ == "__main__":
    unittest.main()
