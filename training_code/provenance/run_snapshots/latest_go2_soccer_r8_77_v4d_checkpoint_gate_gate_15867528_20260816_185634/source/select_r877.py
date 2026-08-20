#!/usr/bin/env python3
"""Select one R8.76 checkpoint using the deterministic R8.75 cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LABELS = ("intermediate_199", "hard_mix_449", "full_support_final")
CANDIDATE_MODE = "r874b_high"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--candidate-paths", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def score(metrics: dict[str, float]) -> float:
    return (
        12.0 * metrics["goal_case_rate"]
        + 4.0 * metrics["contact_ball_rate"]
        + 3.0 * metrics["find_ball_rate"]
        - 8.0 * metrics["wall_stall_rate"]
        - 3.0 * metrics["stationary_rate"]
        + 0.5 * metrics["low_reward_mean"]
        - 20.0 * metrics["upright_error_mean"]
        - 2.0 * metrics["action_rate_mean"]
    )


def main() -> None:
    args = parse_args()
    paths = load_json(args.candidate_paths)
    summaries: dict[str, dict[str, Any]] = {}
    for label in LABELS:
        summary_path = args.results_root / label / "r875_summary.json"
        summaries[label] = load_json(summary_path)

    first = summaries[LABELS[0]]
    accepted = first["mode_summary"]["accepted_high"]
    scripted = first["mode_summary"]["scripted"]
    case_step = 1.0 / float(first["policy_case_count"])

    rows: list[dict[str, Any]] = []
    for label in LABELS:
        summary = summaries[label]
        if summary["primitive_pass_count"] < 6:
            raise RuntimeError(f"Low primitive gate failed for {label}")
        if any(
            float(item["max_abs"]) != 0.0
            for item in summary["parameter_deltas"].values()
        ):
            raise RuntimeError(f"Evaluation mutated parameters for {label}")
        triplicate = summary["triplicate_initial_state_check"]
        if max(float(value) for value in triplicate.values()) > 2.0e-6:
            raise RuntimeError(f"Triplicate reset mismatch for {label}: {triplicate}")

        metrics = summary["mode_summary"][CANDIDATE_MODE]
        eligible = (
            metrics["goal_case_rate"] + 1.0e-12 >= accepted["goal_case_rate"]
            and metrics["find_ball_rate"] + case_step + 1.0e-12
            >= accepted["find_ball_rate"]
            and metrics["contact_ball_rate"] + case_step + 1.0e-12
            >= accepted["contact_ball_rate"]
            and metrics["wall_stall_rate"]
            <= accepted["wall_stall_rate"] + case_step + 1.0e-12
            and metrics["stationary_rate"]
            <= accepted["stationary_rate"] + case_step + 1.0e-12
            and metrics["upright_error_mean"] <= 0.020
            and metrics["action_rate_mean"] <= 0.150
        )
        materially_improved = (
            metrics["goal_case_rate"] > accepted["goal_case_rate"] + 1.0e-12
            or metrics["wall_stall_rate"] + 1.0e-12
            < accepted["wall_stall_rate"]
            or metrics["find_ball_rate"]
            > accepted["find_ball_rate"] + 1.0e-12
            or metrics["contact_ball_rate"]
            > accepted["contact_ball_rate"] + 1.0e-12
        )
        rows.append(
            {
                "label": label,
                "checkpoint": paths[label],
                "metrics": metrics,
                "score": score(metrics),
                "eligible": eligible,
                "materially_improved": materially_improved,
                "source_verdict": summary["verdict"],
            }
        )

    eligible_rows = [row for row in rows if row["eligible"]]
    pool = eligible_rows if eligible_rows else rows
    selected = max(
        pool,
        key=lambda row: (
            row["score"],
            row["metrics"]["goal_case_rate"],
            -row["metrics"]["wall_stall_rate"],
            row["metrics"]["contact_ball_rate"],
        ),
    )
    promoted = bool(selected["eligible"] and selected["materially_improved"])
    status = "PROMOTED_FOR_VIDEO" if promoted else "DIAGNOSTIC_VIDEO_ONLY"

    output = {
        "version": "r8_77_v4d_checkpoint_gate_v1",
        "accepted_high": accepted,
        "scripted": scripted,
        "candidates": rows,
        "selected_label": selected["label"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_metrics": selected["metrics"],
        "selection_score": selected["score"],
        "selection_status": status,
        "promotion": promoted,
        "video_contract": "one_selected_checkpoint_only",
        "evaluation_distribution": "r875_deterministic_full_support_cases",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "r877_checkpoint_selection.json"
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "SELECTED_HIGH_PATH.txt").write_text(
        selected["checkpoint"] + "\n", encoding="utf-8"
    )
    (args.output_dir / "VERDICT.txt").write_text(
        f"VERDICT={status}\nSELECTED={selected['label']}\n",
        encoding="utf-8",
    )

    print("R877_SELECTION=" + json.dumps(output, sort_keys=True))
    print(f"R877_SELECTED_LABEL={selected['label']}")
    print(f"R877_SELECTED_HIGH={selected['checkpoint']}")
    print(f"R877_SELECTION_STATUS={status}")
    print("R877_CHECKPOINT_GATE_COMPLETE")


if __name__ == "__main__":
    main()
