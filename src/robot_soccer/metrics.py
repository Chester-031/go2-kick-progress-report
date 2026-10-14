"""Per-iteration training records: one JSON line per PPO iteration.

The environment reports finished episodes as columnar batches (field name ->
CPU tensor, one row per episode, carrying the ADR tags of their start).
``concatenate_episodes`` joins one iteration's batches and
``summarize_episodes`` reduces them to rates, outcome counts, the goal rate
of each fixed evaluation difficulty and a sparse grid of episodes over the
three ADR knobs.  ``TrainingMetrics`` appends that summary, the ADR state,
the PPO losses and the timing to ``metrics.jsonl``, writes the static run
description to ``run_info.json`` once, and mirrors the headline numbers to
TensorBoard.  ``scripts/plot_training.py`` reads only those two files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .adr import EPISODE_KINDS, EVALUATION, PARAMETERS
from .rewards import GOAL_CENTER_X_M

OUTCOME_NAMES = ("goal", "own_goal", "fall", "boundary", "joint_hard_limit", "timeout")
# Cell edges of the knob grid, per knob in ADR PARAMETERS order.
KNOB_GRID_EDGES = (
    tuple(0.5 + 0.5 * index for index in range(12)),  # ball-goal 0.5..6.0 m
    tuple(0.5 + 0.3 * index for index in range(7)),  # robot-ball 0.5..2.3 m
    tuple(30.0 * index for index in range(7)),  # facing error 0..180 deg
)
NATIVE_TICK_SECONDS = 0.02


def concatenate_episodes(batches: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Join columnar episode batches (empty batches are skipped)."""

    filled = [batch for batch in batches if batch]
    if not filled:
        return {}
    return {name: torch.cat([batch[name] for batch in filled]) for name in filled[0]}


def initial_geometry(episodes: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Each episode's start measured from the reset poses as the three task
    axes: robot-ball distance (m), ball-goal distance (m) and the robot's
    facing error to the ball (deg, 0 = facing the ball, 180 = facing away).
    These are the ADR knobs as realized in the simulator."""

    robot = episodes["reset_robot_xy_m"].double()
    ball = episodes["reset_ball_xy_m"].double()
    yaw = episodes["reset_robot_yaw_rad"].double()
    side = episodes["goal_side"].double()
    to_ball = ball - robot
    bearing = torch.atan2(to_ball[:, 1], to_ball[:, 0])
    facing = torch.atan2(torch.sin(bearing - yaw), torch.cos(bearing - yaw)).abs()
    return {
        "robot_ball_m": torch.linalg.vector_norm(to_ball, dim=1),
        "ball_goal_m": torch.hypot(side * GOAL_CENTER_X_M - ball[:, 0], -ball[:, 1]),
        "facing_deg": torch.rad2deg(facing),
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _knob_grid(episodes: Mapping[str, torch.Tensor]) -> list[list[int]]:
    """Sparse ``[i, j, k, episodes, goals, contacts]`` rows over the grid
    cells (KNOB_GRID_EDGES) of each episode's requested knob values."""

    values = episodes["adr_values"].double()
    cells = []
    for axis, edges in enumerate(KNOB_GRID_EDGES):
        inner = torch.tensor(edges[1:-1], dtype=torch.float64)
        cells.append(torch.bucketize(values[:, axis], inner, right=True))
    shape = tuple(len(edges) - 1 for edges in KNOB_GRID_EDGES)
    flat = (cells[0] * shape[1] + cells[1]) * shape[2] + cells[2]
    size = shape[0] * shape[1] * shape[2]
    counts = torch.bincount(flat, minlength=size)
    goals = torch.bincount(flat, weights=episodes["goal"].double(), minlength=size)
    contacts = torch.bincount(flat, weights=episodes["contact_seen"].double(), minlength=size)
    rows = []
    for cell in torch.nonzero(counts).flatten().tolist():
        rows.append(
            [
                cell // (shape[1] * shape[2]),
                (cell // shape[2]) % shape[1],
                cell % shape[2],
                int(counts[cell]),
                int(goals[cell]),
                int(contacts[cell]),
            ]
        )
    return rows


def summarize_episodes(episodes: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Reduce one iteration's finished episodes to the record fields.

    Rates are over the episodes that finished this iteration; a rate is
    ``None`` when no episode of that kind finished.
    """

    count = int(episodes["outcome"].numel()) if episodes else 0
    if count == 0:
        return {"completed_episodes": 0}
    goal = episodes["goal"]
    contact = episodes["contact_seen"]
    unsafe = episodes["unsafe_pre_goal"]
    progress = episodes["cumulative_progress_reward"].double()
    returns = progress + goal.double() - episodes["own_goal"].double()
    if not bool(torch.isfinite(returns).all()):
        raise ValueError("episode return is non-finite")
    outcomes = torch.bincount(episodes["outcome"], minlength=len(OUTCOME_NAMES))
    seconds = episodes["episode_native_ticks"].double() * NATIVE_TICK_SECONDS
    first_contact = episodes["first_true_contact_native_tick"]

    by_kind = {}
    for index, name in enumerate(EPISODE_KINDS):
        rows = episodes["adr_kind"] == index
        total = int(rows.sum())
        by_kind[name] = {
            "episodes": total,
            "goal_rate": _rate(int(goal[rows].sum()), total),
            "contact_rate": _rate(int(contact[rows].sum()), total),
            "unsafe_rate": _rate(int(unsafe[rows].sum()), total),
        }
    evaluation = {}
    difficulty = episodes["adr_difficulty"]
    for value in sorted({float(item) for item in difficulty[episodes["adr_kind"] == EVALUATION].tolist()}):
        rows = (episodes["adr_kind"] == EVALUATION) & (difficulty == value)
        total = int(rows.sum())
        evaluation[f"{value:g}"] = {
            "episodes": total,
            "goal_rate": _rate(int(goal[rows].sum()), total),
            "contact_rate": _rate(int(contact[rows].sum()), total),
        }

    goal_count = int(goal.sum())
    contact_count = int(contact.sum())
    return {
        "completed_episodes": count,
        "mean_episode_return": float(returns.mean()),
        "mean_cumulative_progress_reward": float(progress.mean()),
        "goal_count": goal_count,
        "goal_rate": goal_count / count,
        "contact_count": contact_count,
        "contact_rate": contact_count / count,
        "unsafe_rate": int(unsafe.sum()) / count,
        "outcomes": {name: int(outcomes[index]) for index, name in enumerate(OUTCOME_NAMES)},
        "by_kind": by_kind,
        "evaluation": evaluation,
        "episode_seconds": {
            "mean": float(seconds.mean()),
            "mean_goal": float(seconds[goal].mean()) if goal_count else None,
            "mean_first_contact": (
                float(first_contact[contact].double().mean()) * NATIVE_TICK_SECONDS if contact_count else None
            ),
        },
        "mean_goal_axis_ball_displacement_m": float(episodes["requested_goal_axis_ball_displacement_m"].double().mean()),
        "knob_grid": _knob_grid(episodes),
    }


class TrainingMetrics:
    """Write one JSON line (and TensorBoard scalars) per PPO iteration."""

    def __init__(
        self,
        output: str | Path,
        *,
        run_info: Mapping[str, Any],
        tensorboard: bool = True,
        completed_episodes: int = 0,
    ) -> None:
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.output / "metrics.jsonl"
        self.completed_episodes = int(completed_episodes)
        info = {
            **run_info,
            "adr_parameters": [vars(parameter) for parameter in PARAMETERS],
            "knob_grid_edges": [list(edges) for edges in KNOB_GRID_EDGES],
            "outcome_names": list(OUTCOME_NAMES),
        }
        (self.output / "run_info.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.writer = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=str(self.output / "tensorboard"))
            except ImportError:
                self.writer = None

    def write(
        self,
        *,
        iteration: int,
        episodes: Mapping[str, torch.Tensor],
        losses: Mapping[str, Any],
        adr: Mapping[str, Any],
        timing: Mapping[str, float],
    ) -> dict[str, Any]:
        summary = summarize_episodes(episodes)
        self.completed_episodes += int(summary["completed_episodes"])

        def scalar(value: Any) -> float:
            return float(value.detach().cpu().item()) if hasattr(value, "detach") else float(value)

        record = {
            "iteration": int(iteration),
            "cumulative_completed_episodes": self.completed_episodes,
            **summary,
            "adr": dict(adr),
            "timing": dict(timing),
            "losses": {name: scalar(value) for name, value in losses.items()},
        }
        with self.jsonl.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

        if self.writer is not None:
            step = int(iteration)
            for name in ("goal_rate", "contact_rate", "unsafe_rate", "mean_episode_return"):
                if name in record:
                    self.writer.add_scalar(f"train/{name}", record[name], step)
            for kind, values in record.get("by_kind", {}).items():
                if values["goal_rate"] is not None:
                    self.writer.add_scalar(f"kind/{kind}_goal_rate", values["goal_rate"], step)
            for difficulty, values in record.get("evaluation", {}).items():
                if values["goal_rate"] is not None:
                    self.writer.add_scalar(f"evaluation/difficulty_{difficulty}_goal_rate", values["goal_rate"], step)
            self.writer.add_scalar("adr/entropy", record["adr"]["entropy"], step)
            for knob, values in record["adr"]["knobs"].items():
                self.writer.add_scalar(f"adr/{knob}/low", values["low"], step)
                self.writer.add_scalar(f"adr/{knob}/high", values["high"], step)
            for name, value in record["timing"].items():
                self.writer.add_scalar(f"timing/{name}", value, step)
            for name, value in record["losses"].items():
                self.writer.add_scalar(f"loss/{name}", value, step)
            self.writer.flush()
        return record

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
