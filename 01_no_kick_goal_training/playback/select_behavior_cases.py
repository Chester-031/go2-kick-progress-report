#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import statistics
import sys
from typing import Any, Callable


ROW_SCHEMA = "sparse_high_task_progress_canary_v1/episode_row/v1"
SELECTION_SCHEMA = "no_kick_playback/behavior_case_selection"
ROW_COUNT = 288
ROW_FIELDS = frozenset({
    "schema",
    "evaluation_assignment",
    "counts_for_evaluation",
    "hard_goal_facing_ball_behind",
    "bucket",
    "goal_side",
    "sample_serial",
    "item_seed",
    "reset_robot_xy_m",
    "reset_robot_yaw_rad",
    "reset_ball_xy_m",
    "outcome",
    "goal",
    "own_goal",
    "boundary",
    "fall",
    "joint_hard_limit",
    "timeout",
    "unsafe_pre_goal",
    "requested_goal_line_reached",
    "contact_seen",
    "first_true_contact_native_tick",
    "maximum_contact_force_n",
    "terminal_contact_force_n",
    "terminal_base_contact_force_n",
    "terminal_joint_hard_limit_excess_rad",
    "native_safe_guard_changed_elements",
    "native_safe_guard_max_correction_rad",
    "episode_high_ticks",
    "episode_native_ticks",
    "terminal_robot_xy_m",
    "terminal_ball_xy_m",
    "initial_l_m",
    "final_l_m",
    "cumulative_progress_reward",
    "requested_goal_axis_ball_displacement_m",
    "abs_lateral_ball_drift_m",
})
BOOLEAN_FIELDS = (
    "counts_for_evaluation",
    "hard_goal_facing_ball_behind",
    "goal",
    "own_goal",
    "boundary",
    "fall",
    "joint_hard_limit",
    "timeout",
    "unsafe_pre_goal",
    "requested_goal_line_reached",
    "contact_seen",
)
FINITE_FIELDS = (
    "reset_robot_yaw_rad",
    "maximum_contact_force_n",
    "terminal_contact_force_n",
    "terminal_base_contact_force_n",
    "terminal_joint_hard_limit_excess_rad",
    "native_safe_guard_max_correction_rad",
    "initial_l_m",
    "final_l_m",
    "cumulative_progress_reward",
    "requested_goal_axis_ball_displacement_m",
    "abs_lateral_ball_drift_m",
)
VECTOR_FIELDS = (
    "reset_robot_xy_m",
    "reset_ball_xy_m",
    "terminal_robot_xy_m",
    "terminal_ball_xy_m",
)
OUTCOMES = ("goal", "own_goal", "fall", "boundary", "joint_hard_limit", "timeout")


class SelectionError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SelectionError(message)


def _finite(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label} must be finite",
    )
    return float(value)


def _integer(value: Any, label: str, minimum: int | None = None) -> int:
    _require(type(value) is int, f"{label} must be an integer")
    if minimum is not None:
        _require(value >= minimum, f"{label} is below {minimum}")
    return int(value)


def _vector(value: Any, label: str) -> list[float]:
    _require(isinstance(value, list) and len(value) == 2, f"{label} must have length two")
    return [_finite(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonfinite_constant(value: str) -> None:
    raise SelectionError(f"non-finite JSON constant: {value}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_row(row: Any, line_number: int) -> dict[str, Any]:
    label = f"line {line_number}"
    _require(isinstance(row, dict), f"{label} is not an object")
    _require(set(row) == ROW_FIELDS, f"{label} field set changed")
    _require(row["schema"] == ROW_SCHEMA, f"{label} schema changed")
    assignment = _integer(row["evaluation_assignment"], f"{label}.evaluation_assignment")
    _require(0 <= assignment < ROW_COUNT, f"{label} assignment out of range")
    for key in BOOLEAN_FIELDS:
        _require(type(row[key]) is bool, f"{label}.{key} must be Boolean")
    _require(row["counts_for_evaluation"] is True, f"{label} is not an evaluation case")
    side = _integer(row["goal_side"], f"{label}.goal_side")
    _require(side in (-1, 1), f"{label}.goal_side is invalid")
    _require(
        _integer(row["sample_serial"], f"{label}.sample_serial") == -1,
        f"{label}.sample_serial must equal -1 for fixed evaluation",
    )
    _require(
        _integer(row["item_seed"], f"{label}.item_seed") == -1,
        f"{label}.item_seed must equal -1 for fixed evaluation",
    )
    _integer(row["native_safe_guard_changed_elements"], f"{label}.native_safe_guard_changed_elements", 0)
    _integer(row["episode_high_ticks"], f"{label}.episode_high_ticks", 1)
    _integer(row["episode_native_ticks"], f"{label}.episode_native_ticks", 1)
    first_contact = row["first_true_contact_native_tick"]
    if first_contact is not None:
        _integer(first_contact, f"{label}.first_true_contact_native_tick", 0)
    _require(
        (first_contact is not None) is row["contact_seen"],
        f"{label} contact tick and latch disagree",
    )
    _require(
        isinstance(row["bucket"], str)
        and row["bucket"].split("_") in [
            [direction, distance]
            for direction in ("front", "back", "left", "right")
            for distance in ("near", "far")
        ],
        f"{label}.bucket is invalid",
    )
    for key in FINITE_FIELDS:
        _finite(row[key], f"{label}.{key}")
    for key in VECTOR_FIELDS:
        _vector(row[key], f"{label}.{key}")
    _require(row["maximum_contact_force_n"] >= 0, f"{label} has negative contact force")
    _require(row["terminal_contact_force_n"] >= 0, f"{label} has negative terminal contact force")
    _require(row["terminal_base_contact_force_n"] >= 0, f"{label} has negative base contact force")
    _require(row["terminal_joint_hard_limit_excess_rad"] >= 0, f"{label} has negative hard-limit excess")
    _require(row["native_safe_guard_max_correction_rad"] >= 0, f"{label} has negative guard correction")
    unsafe = row["own_goal"] or row["boundary"] or row["fall"] or row["joint_hard_limit"]
    _require(row["unsafe_pre_goal"] is unsafe, f"{label} unsafe flag changed")
    active_outcomes = [name for name in OUTCOMES if row[name]]
    _require(bool(active_outcomes), f"{label} has no terminal outcome")
    expected_outcome = active_outcomes[0]
    _require(row["outcome"] == expected_outcome, f"{label} outcome precedence changed")
    _require(not (row["goal"] and unsafe), f"{label} goal overlaps unsafe outcome")
    return row


def load_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
    _require(not path.is_symlink(), f"rows path is a symlink: {path}")
    resolved = path.resolve(strict=True)
    metadata = resolved.stat()
    _require(stat.S_ISREG(metadata.st_mode), f"rows path is not a regular file: {path}")
    rows: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            _require(bool(line.strip()), f"line {line_number} is empty")
            try:
                value = json.loads(
                    line,
                    object_pairs_hook=_unique_object,
                    parse_constant=_nonfinite_constant,
                )
            except (json.JSONDecodeError, SelectionError) as error:
                raise SelectionError(f"line {line_number} is invalid JSON: {error}") from error
            rows.append(_validate_row(value, line_number))
    _require(len(rows) == ROW_COUNT, f"row count must equal {ROW_COUNT}")
    assignments = [row["evaluation_assignment"] for row in rows]
    _require(sorted(assignments) == list(range(ROW_COUNT)), "assignments are not exact-once")
    return rows, _sha256(resolved)


def _linear_quantile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    _require(bool(ordered), "quantile population is empty")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _choose(
    rows: list[dict[str, Any]],
    used: set[int],
    label: str,
    criterion: str,
    predicate: Callable[[dict[str, Any]], bool],
    metric: str | None,
    target: Callable[[list[float]], float] | None,
) -> dict[str, Any]:
    eligible = [row for row in rows if predicate(row)]
    _require(bool(eligible), f"no eligible rows for {label}")
    if metric is None:
        ranked = sorted(eligible, key=lambda row: row["evaluation_assignment"])
        target_value = None
    else:
        values = [float(row[metric]) for row in eligible]
        _require(target is not None, f"target function is missing for {label}")
        target_value = target(values)
        ranked = sorted(
            eligible,
            key=lambda row: (
                abs(float(row[metric]) - target_value),
                row["evaluation_assignment"],
            ),
        )
    selected = next(
        (row for row in ranked if row["evaluation_assignment"] not in used),
        None,
    )
    _require(selected is not None, f"no unused eligible row for {label}")
    used.add(selected["evaluation_assignment"])
    return {
        "label": label,
        "criterion": criterion,
        "eligible_count": len(eligible),
        "target_value": target_value,
        "row": selected,
    }


def select_cases(rows: list[dict[str, Any]], source_sha256: str) -> dict[str, Any]:
    _require(
        isinstance(source_sha256, str)
        and len(source_sha256) == 64
        and all(character in "0123456789abcdef" for character in source_sha256),
        "source SHA256 is invalid",
    )
    used: set[int] = set()
    safe_contact = lambda row: (
        row["contact_seen"]
        and not row["unsafe_pre_goal"]
        and row["outcome"] == "timeout"
    )
    cases = [
        _choose(
            rows,
            used,
            "own_goal",
            "minimum evaluation assignment among own-goal outcomes",
            lambda row: row["outcome"] == "own_goal",
            None,
            None,
        ),
        _choose(
            rows,
            used,
            "hard_contact",
            "safe hard-case contact nearest the median requested-goal-axis displacement",
            lambda row: safe_contact(row) and row["hard_goal_facing_ball_behind"],
            "requested_goal_axis_ball_displacement_m",
            statistics.median,
        ),
        _choose(
            rows,
            used,
            "wrong_way_contact",
            "safe negative-axis contact nearest the median requested-goal-axis displacement",
            lambda row: safe_contact(row) and row["requested_goal_axis_ball_displacement_m"] < 0.0,
            "requested_goal_axis_ball_displacement_m",
            statistics.median,
        ),
        _choose(
            rows,
            used,
            "high_lateral_drift",
            "safe contact nearest the linear 75th percentile of absolute lateral drift",
            safe_contact,
            "abs_lateral_ball_drift_m",
            lambda values: _linear_quantile(values, 0.75),
        ),
        _choose(
            rows,
            used,
            "modest_forward_push",
            "safe contact in the open 0-to-1-metre axis interval nearest its median",
            lambda row: safe_contact(row)
            and 0.0 < row["requested_goal_axis_ball_displacement_m"] < 1.0,
            "requested_goal_axis_ball_displacement_m",
            statistics.median,
        ),
    ]
    assignments = [case["row"]["evaluation_assignment"] for case in cases]
    _require(len(assignments) == len(set(assignments)), "selected assignments are not unique")
    return {
        "schema": SELECTION_SCHEMA,
        "source_episode_rows_sha256": source_sha256,
        "source_row_count": len(rows),
        "selected_case_count": len(cases),
        "selected_assignments": assignments,
        "cases": cases,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rows", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        rows, source_sha256 = load_rows(args.rows)
        result = select_cases(rows, source_sha256)
        if args.output is None:
            json.dump(result, sys.stdout, indent=2, sort_keys=True, allow_nan=False)
            sys.stdout.write("\n")
        else:
            _write_json(args.output, result)
            print(f"SELECTION_OUTPUT={args.output.resolve()}")
            print(f"SELECTION_SHA256={_sha256(args.output.resolve())}")
        return 0
    except (OSError, SelectionError) as error:
        print(f"SELECTION_ERROR={error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
