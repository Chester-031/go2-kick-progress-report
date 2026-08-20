#!/usr/bin/env python3
"""Reassess R8.75 primitive traces using sustained late-window behavior."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


MOTION_CASES = ("forward", "backward", "left", "right")
TURN_CASES = ("turn_left", "turn_right")
WINDOWS = {
    "early": range(0, 30),
    "middle": range(90, 120),
    "late": range(210, 240),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def average(values: list[float]) -> float:
    return mean(values) if values else 0.0


def main() -> None:
    args = parse_args()
    selected: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with gzip.open(args.trace, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["mode"] != "scripted_primitive" or row["low_step"] != "4":
                continue
            high_step = int(row["high_step"])
            for window, indices in WINDOWS.items():
                if high_step in indices:
                    selected[row["case"]][window].append(row)
                    break

    cases: dict[str, Any] = {}
    failed: list[str] = []
    for case, windows in sorted(selected.items()):
        case_result: dict[str, Any] = {}
        for window, rows in windows.items():
            low_norm = []
            low_delta = []
            previous_action: list[float] | None = None
            for row in rows:
                action = [float(row[f"low_action_applied_{i:02d}"]) for i in range(12)]
                low_norm.append(math.sqrt(sum(value * value for value in action)))
                if previous_action is not None:
                    low_delta.append(
                        math.sqrt(
                            sum((a - b) ** 2 for a, b in zip(action, previous_action))
                        )
                    )
                previous_action = action
            case_result[window] = {
                "robot_speed_mean": average([float(r["robot_speed"]) for r in rows]),
                "pose_error_mean": average([float(r["pose_error_xy"]) for r in rows]),
                "pose_yaw_error_mean": average([float(r["pose_error_yaw"]) for r in rows]),
                "upright_error_mean": average([float(r["upright_error"]) for r in rows]),
                "low_reward_mean": average([float(r["low_reward"]) for r in rows]),
                "low_action_norm_mean": average(low_norm),
                "low_action_change_mean": average(low_delta),
                "low_clip_fraction": average([float(r["low_clip"]) for r in rows]),
                "sample_count": len(rows),
            }

        early = case_result.get("early", {})
        late = case_result.get("late", {})
        if case in MOTION_CASES:
            passes = (
                late.get("robot_speed_mean", 0.0) >= 0.06
                and late.get("pose_error_mean", 1.0) <= 0.30
            )
        elif case in TURN_CASES:
            # The legacy trace lacks angular velocity. Persistent low-action
            # change plus bounded yaw error is the conservative proxy here;
            # the repair job performs the actual angular-speed gate.
            passes = (
                late.get("low_action_change_mean", 0.0) >= 1.0e-4
                and late.get("pose_yaw_error_mean", 1.0) <= 0.40
            )
        elif case == "stand":
            passes = late.get("robot_speed_mean", 1.0) <= 0.04
        else:
            passes = late.get("upright_error_mean", 1.0) <= 0.03
        case_result["sustained_pass"] = passes
        case_result["early_to_late_speed_ratio"] = (
            late.get("robot_speed_mean", 0.0)
            / max(early.get("robot_speed_mean", 0.0), 1.0e-8)
        )
        if not passes:
            failed.append(case)
        cases[case] = case_result

    motion_passes = sum(cases[name]["sustained_pass"] for name in MOTION_CASES)
    verdict = (
        "ACCEPTED_LOW_SUSTAINED_FAILURE"
        if motion_passes < len(MOTION_CASES)
        else "LEGACY_TRACE_SUSTAINED_MOTION_NOT_REJECTED"
    )
    result = {
        "version": "r8_78_sustained_trace_reassessment_v1",
        "trace": str(args.trace),
        "windows": {name: [min(value), max(value)] for name, value in WINDOWS.items()},
        "cases": cases,
        "failed_cases": failed,
        "motion_pass_count": motion_passes,
        "motion_total": len(MOTION_CASES),
        "verdict": verdict,
        "legacy_gate_invalidated": verdict == "ACCEPTED_LOW_SUSTAINED_FAILURE",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("R878_TRACE_RESULT=" + json.dumps(result, sort_keys=True))
    print(f"R878_TRACE_VERDICT={verdict}")


if __name__ == "__main__":
    main()
