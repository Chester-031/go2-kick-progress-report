#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = ("C_ball_straight", "D_policy_ball_v2")


def read_numeric_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = reader.fieldnames or []

    if not rows:
        raise RuntimeError(f"No rows found in {path}")

    result: dict[str, np.ndarray] = {}
    for field in fields:
        values: list[float] = []
        for row in rows:
            try:
                values.append(float(row[field]))
            except (TypeError, ValueError):
                break
        else:
            result[field] = np.asarray(values, dtype=np.float64)
    return result


def finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def mean(values: np.ndarray) -> float:
    values = finite(values)
    return float(np.mean(values)) if values.size else float("nan")


def rms(values: np.ndarray) -> float:
    values = finite(values)
    return float(np.sqrt(np.mean(values**2))) if values.size else float("nan")


def p95(values: np.ndarray) -> float:
    values = finite(values)
    return float(np.percentile(values, 95.0)) if values.size else float("nan")


def ratio(numerator: float, denominator: float) -> float:
    if (
        not math.isfinite(numerator)
        or not math.isfinite(denominator)
        or abs(denominator) < 1.0e-8
    ):
        return float("nan")
    return numerator / denominator


def command_flip_rate(values: np.ndarray, duration: float, deadband: float = 0.03) -> float:
    values = np.asarray(values, dtype=np.float64)
    signs = np.zeros(values.shape, dtype=np.int8)
    signs[values > deadband] = 1
    signs[values < -deadband] = -1
    nonzero = signs[signs != 0]
    if nonzero.size < 2 or duration <= 0.0:
        return 0.0
    flips = int(np.sum(nonzero[1:] != nonzero[:-1]))
    return float(flips / duration)


def high_command_stats(high: dict[str, np.ndarray]) -> dict[str, float]:
    steps = high["high_step"]
    duration = float(max(1.0, steps[-1] - steps[0] + 1.0) * 0.1)

    commands = np.stack(
        [
            high["command_vx"],
            high["command_vy"],
            high["command_wz"],
        ],
        axis=1,
    )
    deltas = np.linalg.norm(np.diff(commands, axis=0), axis=1)

    return {
        "high_command_vx_mean": mean(high["command_vx"]),
        "high_command_vy_rms": rms(high["command_vy"]),
        "high_command_wz_rms": rms(high["command_wz"]),
        "high_command_delta_mean": mean(deltas),
        "high_command_delta_p95": p95(deltas),
        "high_command_vy_flip_rate_hz": command_flip_rate(
            high["command_vy"], duration
        ),
        "high_command_wz_flip_rate_hz": command_flip_rate(
            high["command_wz"], duration
        ),
    }


def summarize(
    label: str,
    low: dict[str, np.ndarray],
    high: dict[str, np.ndarray],
) -> dict[str, float | str]:
    time = low["sim_time_s"]
    dt = float(np.median(np.diff(time))) if time.size > 1 else 0.02

    robot_ball = low["robot_ball_distance"]
    contact = np.isfinite(robot_ball) & (robot_ball < 0.62)

    ball_vx = low["ball_vx_world"]
    ball_vy = low["ball_vy_world"]

    ball_ax = np.gradient(ball_vx, dt)
    ball_ay = np.gradient(ball_vy, dt)
    ball_horizontal_acceleration = np.sqrt(ball_ax**2 + ball_ay**2)

    ball_goal = finite(low["ball_goal_distance"])
    ball_y = finite(low["ball_y"])

    summary: dict[str, float | str] = {
        "test": label,
        "duration_s": float(time[-1] - time[0]) if time.size > 1 else 0.0,
        "command_vx_mean_low_rate": mean(low["command_vx"]),
        "command_vy_rms_low_rate": rms(low["command_vy"]),
        "command_wz_rms_low_rate": rms(low["command_wz"]),
        "base_vx_mean": mean(low["base_vx_body"]),
        "base_vy_rms": rms(low["base_vy_body"]),
        "base_wz_rms": rms(low["base_wz_body"]),
        "roll_rms_deg": math.degrees(rms(low["base_roll"])),
        "pitch_rms_deg": math.degrees(rms(low["base_pitch"])),
        "low_action_delta_mean": mean(low["low_action_delta_norm"]),
        "low_action_delta_p95": p95(low["low_action_delta_norm"]),
        "joint_velocity_rms_mean": mean(low["joint_velocity_rms"]),
        "joint_acceleration_rms_mean": mean(low["joint_acceleration_rms"]),
        "torque_rms_mean": mean(low["applied_torque_rms"]),
        "foot_slip_mean": mean(low["mean_contact_foot_slip"]),
        "foot_slip_p95": p95(low["max_contact_foot_slip"]),
        "front_rear_calf_diff_mean": mean(
            low["front_rear_calf_abs_diff"]
        ),
        "front_rear_calf_diff_p95": p95(
            low["front_rear_calf_abs_diff"]
        ),
        "front_rear_thigh_diff_mean": mean(
            low["front_rear_thigh_abs_diff"]
        ),
        "contact_fraction": float(np.mean(contact)),
        "ball_vx_contact_mean": (
            mean(ball_vx[contact]) if np.any(contact) else float("nan")
        ),
        "ball_vy_contact_rms": (
            rms(ball_vy[contact]) if np.any(contact) else float("nan")
        ),
        "ball_horizontal_acceleration_rms": (
            rms(ball_horizontal_acceleration[contact])
            if np.any(contact)
            else float("nan")
        ),
        "ball_horizontal_acceleration_p95": (
            p95(ball_horizontal_acceleration[contact])
            if np.any(contact)
            else float("nan")
        ),
        "ball_goal_progress_m": (
            float(ball_goal[0] - ball_goal[-1])
            if ball_goal.size >= 2
            else float("nan")
        ),
        "ball_lateral_drift_m": (
            float(ball_y[-1] - ball_y[0])
            if ball_y.size >= 2
            else float("nan")
        ),
        "alignment_mean": mean(low["ball_goal_alignment"]),
        "ball_velocity_toward_goal_mean": mean(
            low["ball_velocity_toward_goal"]
        ),
    }
    summary.update(high_command_stats(high))
    return summary


def plot_pair(
    datasets: dict[str, dict[str, np.ndarray]],
    field: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    for label in LABELS:
        data = datasets[label]
        ax.plot(
            data["sim_time_s"],
            data[field],
            linewidth=1.0,
            label=label,
        )
    ax.set_title(title)
    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cd-root", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--ab-root", default="")
    args = parser.parse_args()

    cd_root = Path(args.cd_root).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    low_data = {
        label: read_numeric_csv(cd_root / label / "low_level.csv")
        for label in LABELS
    }
    high_data = {
        label: read_numeric_csv(cd_root / label / "high_level.csv")
        for label in LABELS
    }

    summaries = [
        summarize(label, low_data[label], high_data[label])
        for label in LABELS
    ]
    by_label = {str(row["test"]): row for row in summaries}
    c = by_label["C_ball_straight"]
    d = by_label["D_policy_ball_v2"]

    with (report_dir / "cd_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    plots = (
        (
            "base_vy_body",
            "Base lateral velocity (m/s)",
            "C vs D — Base Lateral Velocity",
            "cd_base_lateral_velocity.png",
        ),
        (
            "base_wz_body",
            "Base yaw rate (rad/s)",
            "C vs D — Base Yaw Rate",
            "cd_base_yaw_rate.png",
        ),
        (
            "low_action_delta_norm",
            "Low-level action delta norm",
            "C vs D — Low-Level Action Variation",
            "cd_low_action_delta.png",
        ),
        (
            "max_contact_foot_slip",
            "Maximum contacting-foot tangential speed (m/s)",
            "C vs D — Foot Slip Indicator",
            "cd_foot_slip.png",
        ),
        (
            "front_rear_calf_abs_diff",
            "Front–rear calf-angle difference (rad)",
            "C vs D — Front–Rear Calf Difference",
            "cd_front_rear_calf_difference.png",
        ),
        (
            "ball_vx_world",
            "Ball world-frame x velocity (m/s)",
            "C vs D — Ball Forward Velocity",
            "cd_ball_forward_velocity.png",
        ),
        (
            "ball_vy_world",
            "Ball world-frame lateral velocity (m/s)",
            "C vs D — Ball Lateral Velocity",
            "cd_ball_lateral_velocity.png",
        ),
    )
    for field, ylabel, title, filename in plots:
        plot_pair(
            low_data,
            field,
            ylabel,
            title,
            report_dir / filename,
        )

    ratios = {
        "D_over_C_base_vy_rms": ratio(
            float(d["base_vy_rms"]), float(c["base_vy_rms"])
        ),
        "D_over_C_base_wz_rms": ratio(
            float(d["base_wz_rms"]), float(c["base_wz_rms"])
        ),
        "D_over_C_foot_slip_mean": ratio(
            float(d["foot_slip_mean"]), float(c["foot_slip_mean"])
        ),
        "D_over_C_action_delta_mean": ratio(
            float(d["low_action_delta_mean"]),
            float(c["low_action_delta_mean"]),
        ),
        "D_over_C_joint_acceleration": ratio(
            float(d["joint_acceleration_rms_mean"]),
            float(c["joint_acceleration_rms_mean"]),
        ),
        "D_over_C_ball_lateral_velocity": ratio(
            float(d["ball_vy_contact_rms"]),
            float(c["ball_vy_contact_rms"]),
        ),
    }

    baseline = None
    if args.ab_root:
        ab_root = Path(args.ab_root).expanduser().resolve()
        a_path = ab_root / "A_no_ball_straight" / "summary.json"
        b_path = ab_root / "B_no_ball_turn" / "summary.json"
        if a_path.is_file() and b_path.is_file():
            baseline = {
                "A": json.loads(a_path.read_text(encoding="utf-8")),
                "B": json.loads(b_path.read_text(encoding="utf-8")),
            }

    findings: list[str] = []
    recommendations: list[str] = []

    c_unstable = (
        float(c["base_vy_rms"]) > 0.07
        or float(c["base_wz_rms"]) > 0.10
        or float(c["foot_slip_mean"]) > 0.14
        or float(c["joint_acceleration_rms_mean"]) > 70.0
    )
    d_adds_correction = (
        (
            math.isfinite(ratios["D_over_C_base_vy_rms"])
            and ratios["D_over_C_base_vy_rms"] > 1.25
        )
        or (
            math.isfinite(ratios["D_over_C_base_wz_rms"])
            and ratios["D_over_C_base_wz_rms"] > 1.25
        )
        or (
            math.isfinite(ratios["D_over_C_action_delta_mean"])
            and ratios["D_over_C_action_delta_mean"] > 1.20
        )
    ) and (
        float(d["high_command_wz_rms"]) > 0.08
        or float(d["high_command_vy_rms"]) > 0.05
        or float(d["high_command_delta_p95"]) > 0.10
    )

    if c_unstable:
        findings.append(
            "Test C is unstable even under a constant straight command. "
            "Ball contact therefore exposes or amplifies a low-level/contact "
            "robustness problem before high-level correction is considered."
        )
        recommendations.append(
            "Jointly fine-tune the low-level controller in the soccer "
            "environment, with ball-contact load, foot-slip, yaw/lateral "
            "stability, action-rate, and joint-acceleration objectives."
        )
    else:
        findings.append(
            "Test C remains comparatively stable under constant ball "
            "contact, so ball reaction alone is not the dominant failure."
        )

    if d_adds_correction:
        findings.append(
            "Test D is materially less stable than Test C while the "
            "high-level policy changes vy/wz. High-level over-correction "
            "is an additional contributor."
        )
        recommendations.append(
            "Add a Push regime with a small alignment deadband, bounded "
            "vy/wz, and command slew-rate limits during joint fine-tuning."
        )
    else:
        findings.append(
            "Test D is not materially worse than Test C. The high-level "
            "policy is unlikely to be the main source of the sway."
        )

    findings.append(
        "Ball horizontal acceleration is logged as a reaction-load proxy. "
        "It is not a direct isolated robot–ball contact force because it "
        "also contains ball–ground effects."
    )

    if baseline is not None:
        a = baseline["A"]
        findings.append(
            "The report also includes the earlier no-ball Test A values "
            "for context, but A used vx=0.45 while C uses vx=0.35, so this "
            "is not a perfectly matched causal comparison."
        )
        context = {
            "A_base_vy_rms": a.get("base_vy_rms"),
            "A_base_wz_tracking_rmse": a.get("base_wz_tracking_rmse"),
            "A_foot_slip_mean": a.get("foot_slip_mean"),
            "A_action_delta_mean": a.get("action_delta_mean"),
            "A_joint_acceleration_rms_mean": a.get(
                "joint_acceleration_rms_mean"
            ),
        }
    else:
        context = {}

    if not recommendations:
        recommendations.append(
            "Do not change Version 3 yet; inspect the C/D time-series "
            "plots and rerun a speed-matched no-ball control at vx=0.35."
        )

    report = [
        "# C/D Ball-Contact Diagnostic",
        "",
        "## Tests",
        "",
        "- **C:** aligned robot–ball–goal, fixed command "
        "`vx=0.35, vy=0, wz=0`.",
        "- **D:** the same initial geometry with the deterministic "
        "Version-2 high-level policy.",
        "",
        "## C/D ratios",
        "",
    ]
    report.extend(
        f"- {key}: `{value:.4f}`"
        for key, value in ratios.items()
    )
    if context:
        report.extend(["", "## Earlier no-ball context", ""])
        report.extend(
            f"- {key}: `{value}`"
            for key, value in context.items()
        )
    report.extend(["", "## Findings", ""])
    report.extend(f"- {item}" for item in findings)
    report.extend(["", "## Version-3 implications", ""])
    report.extend(
        f"{index}. {item}"
        for index, item in enumerate(recommendations, start=1)
    )
    report.append("")

    (report_dir / "cd_root_cause_assessment.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("===== C/D COMPACT SUMMARY =====")
    keys = (
        "base_vx_mean",
        "base_vy_rms",
        "base_wz_rms",
        "low_action_delta_mean",
        "joint_acceleration_rms_mean",
        "foot_slip_mean",
        "front_rear_calf_diff_mean",
        "contact_fraction",
        "ball_vx_contact_mean",
        "ball_vy_contact_rms",
        "ball_horizontal_acceleration_rms",
        "ball_goal_progress_m",
        "ball_lateral_drift_m",
        "high_command_vy_rms",
        "high_command_wz_rms",
        "high_command_delta_p95",
        "high_command_vy_flip_rate_hz",
        "high_command_wz_flip_rate_hz",
    )
    print(f"{'metric':42s} {'C_fixed':>14s} {'D_policy':>14s}")
    for key in keys:
        print(
            f"{key:42s} "
            f"{float(c[key]):14.6f} "
            f"{float(d[key]):14.6f}"
        )

    print()
    for key, value in ratios.items():
        print(f"{key}={value}")
    print(f"REPORT_DIR={report_dir}")
    print("GO2_SWAY_CD_ANALYSIS_OK")


if __name__ == "__main__":
    main()
