"""The plot script renders every figure from real-format training records."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FIGURES = (
    "00_overview.png", "01_goal_rates.png", "02_adr_ranges.png", "03_adr_probes.png", "04_adr_health.png",
    "05_outcomes.png", "06_knob_heatmaps.png", "07_goal_by_knob.png", "08_episode_timing.png",
    "09_ppo.png", "10_throughput.png",
)


class PlotTrainingTest(unittest.TestCase):
    def test_every_figure_from_two_runs(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            runs = []
            for seed in (0, 1):
                run = Path(root) / f"seed{seed}"
                subprocess.run(
                    [sys.executable, str(ROOT / "tests" / "synthetic_run.py"), str(run), "--seed", str(seed),
                     "--iterations", "40", "--episodes-per-iteration", "800"],
                    check=True, cwd=ROOT,
                )
                runs.append(str(run))
            out = Path(root) / "plots"
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "plot_training.py"), *runs, "--out", str(out)],
                check=True, cwd=ROOT,
            )
            for name in FIGURES:
                self.assertGreater((out / name).stat().st_size, 10_000, name)


def load_plot_module():
    spec = importlib.util.spec_from_file_location("plot_training", ROOT / "scripts" / "plot_training.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PooledGoalRateTest(unittest.TestCase):
    def test_bins_weight_iterations_by_their_episodes(self) -> None:
        """A fixed-difficulty goal rate per bin is goals over episodes in the
        bin, not the mean of per-iteration rates: one lucky 1-episode
        iteration must not count as much as a 9-episode one."""

        plot = load_plot_module()
        records = [
            {"iteration": 1, "evaluation": {"1": {"episodes": 1, "goal_rate": 1.0}}},
            {"iteration": 2, "evaluation": {"1": {"episodes": 9, "goal_rate": 0.0}}},
            {"iteration": 3, "evaluation": {"1": {"episodes": 4, "goal_rate": 0.5}}},
            {"iteration": 4},
        ]
        run = {"records": records, "iteration": [1.0, 2.0, 3.0, 4.0]}
        centres, rate, low, high = plot.pooled_goal_rate(run, ("evaluation", "1"), bin_width=2)
        self.assertEqual(list(centres), [1.5, 3.5])
        self.assertAlmostEqual(rate[0], 0.1)
        self.assertAlmostEqual(rate[1], 0.5)
        self.assertTrue(low[0] < 0.1 < high[0] and low[1] < 0.5 < high[1])


if __name__ == "__main__":
    unittest.main()
