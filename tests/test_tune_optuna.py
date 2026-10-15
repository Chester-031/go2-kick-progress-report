"""The Optuna tuning worker's pure parts: the search space, how a trial
becomes training arguments, and the score."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import optuna

from tests.fake_isaac import install_isaaclab_stub

install_isaaclab_stub()

from robot_soccer import train  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("tune_optuna", ROOT / "scripts" / "tune_optuna.py")
tune = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tune)


def record(iteration: int, rates: dict[str, tuple[int, float]]) -> dict:
    return {
        "iteration": iteration,
        "evaluation": {d: {"episodes": n, "goal_rate": r} for d, (n, r) in rates.items()},
    }


class SearchSpaceTest(unittest.TestCase):
    def test_baselines_lie_inside_the_search_space(self) -> None:
        for baseline in tune.BASELINE_TRIALS:
            params = tune.suggest(optuna.trial.FixedTrial(baseline))
            self.assertIn(params["control"], ("position", "velocity"))
            self.assertAlmostEqual(params["gamma"], 0.998)

    def test_a_trial_becomes_arguments_train_accepts_with_a_fixed_step_budget(self) -> None:
        baseline = dict(tune.BASELINE_TRIALS[0])
        baseline["steps_per_env"] = 48
        params = tune.suggest(optuna.trial.FixedTrial(baseline))
        arguments = tune.train_arguments(params, budget_steps=150_000_000, num_envs=4096, output=Path("out"))
        parsed = train._parser().parse_args(arguments)
        self.assertEqual(parsed.control, params["control"])
        self.assertEqual(parsed.steps_per_env, 48)
        # 150M steps at 4096 envs x 48 steps per iteration.
        self.assertEqual(parsed.iterations, 763)
        self.assertAlmostEqual(parsed.gamma, params["gamma"])
        self.assertEqual(parsed.schedule, params["schedule"])
        self.assertEqual(parsed.checkpoint_interval, parsed.iterations)

    def test_target_kl_is_only_searched_with_the_adaptive_schedule(self) -> None:
        fixed = dict(tune.BASELINE_TRIALS[0], schedule="fixed")
        fixed.pop("desired_kl")
        params = tune.suggest(optuna.trial.FixedTrial(fixed))
        self.assertNotIn("desired_kl", params)
        self.assertNotIn("--desired-kl", tune.train_arguments(params, 1_000_000, 4096, Path("o")))


class ScoreTest(unittest.TestCase):
    def test_score_averages_the_fixed_difficulties_over_the_last_tenth(self) -> None:
        records = [record(i, {"0.25": (10, 0.0), "0.5": (10, 0.0), "1": (10, 0.0)}) for i in range(1, 91)]
        records += [record(i, {"0.25": (10, 0.9), "0.5": (20, 0.6), "1": (10, 0.3)}) for i in range(91, 101)]
        # Each difficulty pooled over the last ten records, then averaged.
        self.assertAlmostEqual(tune.score(records), (0.9 + 0.6 + 0.3) / 3)

    def test_no_evaluation_episodes_scores_zero(self) -> None:
        self.assertEqual(tune.score([{"iteration": 1}]), 0.0)


if __name__ == "__main__":
    unittest.main()
