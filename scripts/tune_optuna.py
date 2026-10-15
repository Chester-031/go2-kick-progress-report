"""Optuna worker: tune the PPO hyperparameters and the control.

Several copies run at once (a Slurm array), sharing one Optuna study
through a journal file.  Each copy asks the study for a trial, trains it as
a fresh Isaac Lab process (``robot_soccer.train``), reports the score as the
run goes so the median pruner can stop weak trials, and repeats until the
study has ``--trials`` trials.  Every trial gets the same number of
environment steps (``--budget-steps``), so settings with bigger batches get
fewer, larger PPO updates rather than more experience.

Searched (``suggest``): the control (position or velocity joint targets),
the discount gamma (also the shaping discount), GAE lambda, learning rate
and schedule (with the target KL when adaptive), entropy coefficient,
steps per env per iteration (with 4096 envs, the batch size), minibatches,
epochs and the PPO clip.  The first trials are the current settings for
each control (``BASELINE_TRIALS``).

Score (``score``): the goal rate of the fixed-difficulty episodes (ADR
difficulty 0.25, 0.5 and 1, the same tasks in every trial), pooled over the
last tenth of the run's iterations and averaged over the difficulties.
Training episodes are not used: their task depends on how far the trial's
own ADR got.

    python scripts/tune_optuna.py --storage study/journal.log --study-name go2 \
        --trials 40 --budget-steps 150000000 --python ISAAC_PYTHON --out-dir study/trials
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

DIFFICULTIES = ("0.25", "0.5", "1")
# The settings before tuning (gamma 0.998, Isaac Lab's stock Go2 PPO), one
# per control.
_BASELINE = {
    "one_minus_gamma": 0.002,
    "lam": 0.95,
    "learning_rate": 1.0e-3,
    "schedule": "adaptive",
    "desired_kl": 0.01,
    "entropy_coef": 0.01,
    "steps_per_env": 24,
    "num_mini_batches": 4,
    "num_learning_epochs": 5,
    "clip_param": 0.2,
}
BASELINE_TRIALS = ({"control": "position", **_BASELINE}, {"control": "velocity", **_BASELINE})


def suggest(trial: optuna.trial.BaseTrial) -> dict[str, Any]:
    """The trial's control and PPO settings (training-flag names)."""

    params: dict[str, Any] = {
        "control": trial.suggest_categorical("control", ["position", "velocity"]),
        # Searched as 1 - gamma on a log scale: 0.99 .. 0.999.
        "gamma": 1.0 - trial.suggest_float("one_minus_gamma", 1.0e-3, 1.0e-2, log=True),
        "lam": trial.suggest_float("lam", 0.9, 0.99),
        "learning_rate": trial.suggest_float("learning_rate", 1.0e-4, 3.0e-3, log=True),
        "schedule": trial.suggest_categorical("schedule", ["fixed", "adaptive"]),
        "entropy_coef": trial.suggest_float("entropy_coef", 1.0e-4, 3.0e-2, log=True),
        "steps_per_env": trial.suggest_categorical("steps_per_env", [16, 24, 32, 48, 64]),
        "num_mini_batches": trial.suggest_categorical("num_mini_batches", [2, 4, 8]),
        "num_learning_epochs": trial.suggest_int("num_learning_epochs", 3, 8),
        "clip_param": trial.suggest_float("clip_param", 0.1, 0.3),
    }
    if params["schedule"] == "adaptive":
        params["desired_kl"] = trial.suggest_float("desired_kl", 0.005, 0.03, log=True)
    return params


def train_arguments(params: dict[str, Any], budget_steps: int, num_envs: int, output: Path) -> list[str]:
    """``robot_soccer.train`` arguments for one trial: ``budget_steps``
    environment steps, one checkpoint at the end."""

    iterations = math.ceil(budget_steps / (num_envs * params["steps_per_env"]))
    arguments = [
        "--output-dir", str(output),
        "--num-envs", str(num_envs),
        "--iterations", str(iterations),
        "--checkpoint-interval", str(iterations),
        "--no-tensorboard",
    ]
    for name, value in params.items():
        arguments += ["--" + name.replace("_", "-"), str(value)]
    return arguments


def score(records: list[dict[str, Any]], fraction: float = 0.1) -> float:
    """Mean over the fixed difficulties of the goal rate pooled over the last
    ``fraction`` of ``records`` (0 when there are no such episodes)."""

    if not records:
        return 0.0
    window = records[-max(1, int(round(fraction * len(records)))):]
    rates = []
    for difficulty in DIFFICULTIES:
        episodes = goals = 0.0
        for record in window:
            entry = record.get("evaluation", {}).get(difficulty)
            if entry and entry["episodes"]:
                episodes += entry["episodes"]
                goals += entry["episodes"] * entry["goal_rate"]
        if episodes:
            rates.append(goals / episodes)
    return sum(rates) / len(rates) if rates else 0.0


class MetricsReader:
    """Reads a growing metrics.jsonl without re-reading what it has seen."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.records: list[dict[str, Any]] = []

    def update(self) -> list[dict[str, Any]]:
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as stream:
                stream.seek(self.offset)
                for line in stream:
                    if not line.endswith("\n"):
                        break
                    self.offset += len(line.encode("utf-8"))
                    self.records.append(json.loads(line))
        return self.records


def run_trial(trial: optuna.trial.Trial, args: argparse.Namespace) -> None:
    params = suggest(trial)
    output = args.out_dir / f"trial_{trial.number:03d}"
    command = [args.python, "-m", "robot_soccer.train", "--headless", "--device", "cuda:0",
               *train_arguments(params, args.budget_steps, args.num_envs, output)]
    trial.set_user_attr("output", str(output))
    trial.set_user_attr("params", params)
    print(f"[tune] trial {trial.number}: {json.dumps(params)}", flush=True)
    log = (args.out_dir / f"trial_{trial.number:03d}.log").open("w", encoding="utf-8")
    # The training process may need a different module path than this
    # worker (another Python, with its own packages): TRAIN_PYTHONPATH.
    environment = dict(os.environ)
    if "TRAIN_PYTHONPATH" in environment:
        environment["PYTHONPATH"] = environment["TRAIN_PYTHONPATH"]
    process = subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=environment
    )
    reader = MetricsReader(output / "metrics.jsonl")
    steps_per_iteration = args.num_envs * params["steps_per_env"]
    pruned = False
    while process.poll() is None:
        time.sleep(args.poll_seconds)
        records = reader.update()
        if not records:
            continue
        step = int(records[-1]["iteration"] * steps_per_iteration / 1.0e6)
        trial.report(score(records), step)
        if trial.should_prune():
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=120)
            pruned = True
            break
    log.close()
    records = reader.update()
    if pruned:
        print(f"[tune] trial {trial.number} pruned at {score(records):.3f}", flush=True)
        args.study.tell(trial, state=optuna.trial.TrialState.PRUNED)
    elif process.returncode != 0 or not records:
        print(f"[tune] trial {trial.number} failed (exit {process.returncode})", flush=True)
        args.study.tell(trial, state=optuna.trial.TrialState.FAIL)
    else:
        value = score(records)
        print(f"[tune] trial {trial.number} score {value:.4f}", flush=True)
        args.study.tell(trial, value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--storage", type=Path, required=True, help="Optuna journal file shared by the workers")
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--budget-steps", type=int, default=150_000_000)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--python", required=True, help="the Isaac Lab python that runs robot_soccer.train")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(JournalFileBackend(str(args.storage)))
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed, multivariate=True, group=True, constant_liar=True),
        # Scores near the start of a run say little (every trial is near 0
        # at full difficulty), so prune only past 40% of the budget.
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=int(0.4 * args.budget_steps / 1e6)),
    )
    args.study = study
    if os.environ.get("SLURM_ARRAY_TASK_ID", "0") == "0" and not study.trials:
        for baseline in BASELINE_TRIALS:
            study.enqueue_trial(baseline)
    else:
        time.sleep(30)
    while True:
        started = [t for t in study.get_trials(deepcopy=False) if t.state != optuna.trial.TrialState.WAITING]
        if len(started) >= args.trials:
            break
        run_trial(study.ask(), args)
    print("[tune] study has enough trials; worker done", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
