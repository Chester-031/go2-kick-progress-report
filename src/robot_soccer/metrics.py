"""Compact episode metrics for training curves and geometry diagnostics."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROBOT_BALL_DISTANCE_BINS_M = (0.0, 1.0, 1.5, 2.0, 2.5, math.inf)
BALL_GOAL_DISTANCE_BINS_M = (0.0, 2.0, 4.0, 6.0, 8.0, math.inf)
FACING_ERROR_BINS_DEG = (0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.000001)


def _bin_label(value: float, edges: Sequence[float], unit: str) -> str:
    for lower, upper in zip(edges, edges[1:], strict=True):
        if lower <= value < upper:
            rendered_upper = "inf" if math.isinf(upper) else f"{upper:g}"
            return f"[{lower:g},{rendered_upper}){unit}"
    raise ValueError(f"value {value} is outside declared bins")


def _initial_geometry(event: Mapping[str, Any]) -> tuple[float, float, float]:
    robot_x, robot_y = (float(value) for value in event["reset_robot_xy_m"])
    ball_x, ball_y = (float(value) for value in event["reset_ball_xy_m"])
    yaw = float(event["reset_robot_yaw_rad"])
    side = int(event["goal_side"])
    robot_ball = math.hypot(ball_x - robot_x, ball_y - robot_y)
    ball_goal = math.hypot(side * 4.110000133514404 - ball_x, -ball_y)
    target_yaw = math.atan2(ball_y - robot_y, ball_x - robot_x)
    facing = abs(math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw)))
    return robot_ball, ball_goal, math.degrees(facing)


def summarize_episodes(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize completed episodes without retaining rollout tensors."""

    rows = list(events)
    by_regime: dict[str, dict[str, int]] = defaultdict(
        lambda: {"episodes": 0, "goals": 0, "contacts": 0, "unsafe": 0}
    )
    bins: dict[str, dict[str, int]] = defaultdict(
        lambda: {"episodes": 0, "goals": 0, "contacts": 0}
    )
    returns: list[float] = []
    outcomes: dict[str, int] = defaultdict(int)

    for event in rows:
        goal = bool(event["goal"])
        own_goal = bool(event["own_goal"])
        contact = bool(event["contact_seen"])
        unsafe = bool(event["unsafe_pre_goal"])
        regime = str(event["reset_regime"])
        task_return = float(event["cumulative_progress_reward"]) + float(goal) - float(own_goal)
        if not math.isfinite(task_return):
            raise ValueError("episode return is non-finite")
        returns.append(task_return)
        outcomes[str(event["outcome"])] += 1
        summary = by_regime[regime]
        summary["episodes"] += 1
        summary["goals"] += int(goal)
        summary["contacts"] += int(contact)
        summary["unsafe"] += int(unsafe)

        robot_ball, ball_goal, facing = _initial_geometry(event)
        labels = (
            "robot_ball/" + _bin_label(robot_ball, ROBOT_BALL_DISTANCE_BINS_M, "m"),
            "ball_goal/" + _bin_label(ball_goal, BALL_GOAL_DISTANCE_BINS_M, "m"),
            "facing/" + _bin_label(facing, FACING_ERROR_BINS_DEG, "deg"),
            "facing_x_robot_ball/"
            + _bin_label(facing, FACING_ERROR_BINS_DEG, "deg")
            + "/"
            + _bin_label(robot_ball, ROBOT_BALL_DISTANCE_BINS_M, "m"),
        )
        for label in labels:
            item = bins[label]
            item["episodes"] += 1
            item["goals"] += int(goal)
            item["contacts"] += int(contact)

    count = len(rows)
    goal_count = sum(int(row["goal"]) for row in rows)
    contact_count = sum(int(row["contact_seen"]) for row in rows)
    unsafe_count = sum(int(row["unsafe_pre_goal"]) for row in rows)
    mean_progress = (
        sum(float(row["cumulative_progress_reward"]) for row in rows) / count
        if count
        else 0.0
    )
    result = {
        "completed_episodes": count,
        "mean_episode_return": sum(returns) / count if count else 0.0,
        "mean_cumulative_progress_reward": mean_progress,
        "goal_count": goal_count,
        "goal_rate": goal_count / count if count else 0.0,
        "contact_count": contact_count,
        "contact_rate": contact_count / count if count else 0.0,
        "unsafe_count": unsafe_count,
        "unsafe_rate": unsafe_count / count if count else 0.0,
        "outcomes": dict(sorted(outcomes.items())),
        "by_regime": {},
        "geometry_bins": dict(sorted(bins.items())),
    }
    for regime, values in sorted(by_regime.items()):
        episodes = values["episodes"]
        result["by_regime"][regime] = {
            **values,
            "goal_rate": values["goals"] / episodes if episodes else 0.0,
            "contact_rate": values["contacts"] / episodes if episodes else 0.0,
        }
    return result


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
        events: Sequence[Mapping[str, Any]],
        losses: Mapping[str, Any],
    ) -> dict[str, Any]:
        summary = summarize_episodes(events)
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
            for regime, values in record["by_regime"].items():
                key = regime.lower()
                self.writer.add_scalar(f"train/{key}_goal_rate", values["goal_rate"], step)
                self.writer.add_scalar(f"train/{key}_contact_rate", values["contact_rate"], step)
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
