"""The plot script renders every figure from real-format training records."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FIGURES = (
    "01_goal_rates.png", "02_adr_ranges.png", "03_adr_probes.png", "04_adr_health.png",
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


if __name__ == "__main__":
    unittest.main()
