"""Per-iteration episode summaries for training curves and geometry diagnostics.

The environment reports finished episodes as columnar batches (field name ->
CPU tensor, one row per episode).  ``concatenate_episodes`` joins one
iteration's batches and ``summarize_episodes`` reduces them to the compact
record ``TrainingMetrics`` appends to ``metrics.jsonl``; the plotting script
reads only that file.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .rewards import GOAL_CENTER_X_M

ROBOT_BALL_DISTANCE_BINS_M = (0.0, 1.0, 1.5, 2.0, 2.5, math.inf)
BALL_GOAL_DISTANCE_BINS_M = (0.0, 2.0, 4.0, 6.0, 8.0, math.inf)
FACING_ERROR_BINS_DEG = (0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.000001)
OUTCOME_NAMES = ("goal", "own_goal", "fall", "boundary", "joint_hard_limit", "timeout")


def concatenate_episodes(batches: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Join columnar episode batches (empty batches are skipped)."""

    filled = [batch for batch in batches if batch]
    if not filled:
        return {}
    return {name: torch.cat([batch[name] for batch in filled]) for name in filled[0]}


def initial_geometry(episodes: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Each episode's start as the three task axes: robot-ball distance (m),
    ball-goal distance (m) and the robot's facing error to the ball (deg,
    0 = facing the ball, 180 = facing straight away)."""

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


def _bin_labels(values: torch.Tensor, edges: Sequence[float], unit: str) -> list[str]:
    names = []
    for lower, upper in zip(edges, edges[1:]):
        rendered_upper = "inf" if math.isinf(upper) else f"{upper:g}"
        names.append(f"[{lower:g},{rendered_upper}){unit}")
    index = torch.bucketize(values, torch.tensor(edges[1:-1], dtype=values.dtype), right=True)
    if bool(((values < edges[0]) | (values >= edges[-1])).any()):
        raise ValueError("a value is outside the declared bins")
    return [names[i] for i in index.tolist()]


def summarize_episodes(episodes: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Reduce one iteration's finished episodes to rates, outcome counts and
    goal/contact counts by start geometry."""

    count = int(episodes["outcome"].numel()) if episodes else 0
    if count == 0:
        return {
            "completed_episodes": 0,
            "mean_episode_return": 0.0,
            "mean_cumulative_progress_reward": 0.0,
            "goal_count": 0,
            "goal_rate": 0.0,
            "contact_count": 0,
            "contact_rate": 0.0,
            "unsafe_count": 0,
            "unsafe_rate": 0.0,
            "outcomes": {},
            "geometry_bins": {},
        }
    goal = episodes["goal"]
    contact = episodes["contact_seen"]
    unsafe = episodes["unsafe_pre_goal"]
    progress = episodes["cumulative_progress_reward"].double()
    returns = progress + goal.double() - episodes["own_goal"].double()
    if not bool(torch.isfinite(returns).all()):
        raise ValueError("episode return is non-finite")
    outcomes = torch.bincount(episodes["outcome"], minlength=len(OUTCOME_NAMES))

    geometry = initial_geometry(episodes)
    robot_ball = _bin_labels(geometry["robot_ball_m"], ROBOT_BALL_DISTANCE_BINS_M, "m")
    ball_goal = _bin_labels(geometry["ball_goal_m"], BALL_GOAL_DISTANCE_BINS_M, "m")
    facing = _bin_labels(geometry["facing_deg"], FACING_ERROR_BINS_DEG, "deg")
    bins: dict[str, dict[str, int]] = {}
    for index, (goal_flag, contact_flag) in enumerate(zip(goal.tolist(), contact.tolist())):
        for label in (
            "robot_ball/" + robot_ball[index],
            "ball_goal/" + ball_goal[index],
            "facing/" + facing[index],
            "facing_x_robot_ball/" + facing[index] + "/" + robot_ball[index],
        ):
            item = bins.setdefault(label, {"episodes": 0, "goals": 0, "contacts": 0})
            item["episodes"] += 1
            item["goals"] += int(goal_flag)
            item["contacts"] += int(contact_flag)

    goal_count = int(goal.sum())
    contact_count = int(contact.sum())
    unsafe_count = int(unsafe.sum())
    return {
        "completed_episodes": count,
        "mean_episode_return": float(returns.mean()),
        "mean_cumulative_progress_reward": float(progress.mean()),
        "goal_count": goal_count,
        "goal_rate": goal_count / count,
        "contact_count": contact_count,
        "contact_rate": contact_count / count,
        "unsafe_count": unsafe_count,
        "unsafe_rate": unsafe_count / count,
        "outcomes": {
            name: int(outcomes[index]) for index, name in enumerate(OUTCOME_NAMES) if int(outcomes[index])
        },
        "geometry_bins": dict(sorted(bins.items())),
    }


class TrainingMetrics:
    """Write one compact JSON/TensorBoard record per PPO iteration."""

    def __init__(self, output: str | Path, *, tensorboard: bool = True) -> None:
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.output / "metrics.jsonl"
        self.completed_episodes = 0
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
    ) -> dict[str, Any]:
        summary = summarize_episodes(episodes)
        self.completed_episodes += int(summary["completed_episodes"])

        def scalar(value: Any) -> float:
            return float(value.detach().cpu().item()) if hasattr(value, "detach") else float(value)

        record = {
            "iteration": int(iteration),
            "cumulative_completed_episodes": self.completed_episodes,
            **summary,
            "losses": {name: scalar(value) for name, value in losses.items()},
        }
        with self.jsonl.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

        if self.writer is not None:
            step = self.completed_episodes
            self.writer.add_scalar("train/episode_return", record["mean_episode_return"], step)
            self.writer.add_scalar("train/goal_rate", record["goal_rate"], step)
            self.writer.add_scalar("train/contact_rate", record["contact_rate"], step)
            for label, values in record["geometry_bins"].items():
                if values["episodes"]:
                    self.writer.add_scalar(
                        "geometry/" + label + "/goal_rate",
                        values["goals"] / values["episodes"],
                        step,
                    )
                    self.writer.add_scalar(
                        "geometry/" + label + "/contact_rate",
                        values["contacts"] / values["episodes"],
                        step,
                    )
            self.writer.flush()
        return record

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
