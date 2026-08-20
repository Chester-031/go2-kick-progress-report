"""Fixed-command low-level Go2 sway diagnostics: tests A and B."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose Go2 straight and combined-command sway."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-Go2-Low-Level-Round2-v0",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--test_label", required=True)

    parser.add_argument("--command_vx", type=float, required=True)
    parser.add_argument("--command_vy", type=float, required=True)
    parser.add_argument("--command_wz", type=float, required=True)

    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--stop_on_done", action="store_true")

    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = build_parser()
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

print("[SWAY_BOOT] AppLauncher returned.", flush=True)

import gymnasium as gym
import torch

import isaac_go2_low_level_rl.tasks  # noqa: F401

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import (
    load_cfg_from_registry,
    parse_env_cfg,
)
from rsl_rl.runners import OnPolicyRunner


LEGS = ("FL", "FR", "RL", "RR")


def instantiate_cfg(value):
    return value() if isinstance(value, type) else value


def extract_observation(value):
    if isinstance(value, tuple):
        return value[0]
    return value


def extract_action_tensor(value) -> torch.Tensor:
    if torch.is_tensor(value):
        return value

    for key in ("policy", "action", "actions"):
        try:
            candidate = value.get(key)
        except Exception:
            candidate = None

        if torch.is_tensor(candidate):
            return candidate

    raise RuntimeError(
        "Could not extract an action tensor from "
        f"{type(value)!r}."
    )


def normalize_step_result(result):
    if not isinstance(result, tuple):
        raise RuntimeError(
            f"Unexpected step return type: {type(result)!r}"
        )

    if len(result) == 4:
        obs, reward, dones, infos = result
        return obs, reward, dones, infos

    if len(result) == 5:
        obs, reward, terminated, truncated, infos = result
        dones = torch.logical_or(terminated, truncated)
        return obs, reward, dones, infos

    raise RuntimeError(
        f"Unexpected step return length: {len(result)}"
    )


def safe_name(value: str) -> str:
    return re.sub(
        r"[^A-Za-z0-9_]+",
        "_",
        value,
    ).strip("_")


def to_numpy(value) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=np.float64)

    if not torch.is_tensor(value):
        value = torch.as_tensor(value)

    return np.asarray(
        value.detach().reshape(-1).cpu().numpy(),
        dtype=np.float64,
    )


def vector_value(
    vector: np.ndarray,
    index: int,
    default: float = np.nan,
) -> float:
    try:
        return float(vector[index])
    except (IndexError, TypeError):
        return float(default)


def finite_rms(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]

    if not values.size:
        return float("nan")

    return float(np.sqrt(np.mean(values**2)))


def finite_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]

    if not values.size:
        return float("nan")

    return float(np.mean(values))


def finite_percentile(
    values: np.ndarray,
    percentile: float,
) -> float:
    values = values[np.isfinite(values)]

    if not values.size:
        return float("nan")

    return float(np.percentile(values, percentile))


def dominant_frequency(
    values: np.ndarray,
    sample_rate_hz: float,
) -> float:
    values = values[np.isfinite(values)]

    if values.size < 64:
        return float("nan")

    values = values - np.mean(values)

    spectrum = np.abs(np.fft.rfft(values)) ** 2
    frequencies = np.fft.rfftfreq(
        values.size,
        d=1.0 / sample_rate_hz,
    )

    valid = (
        (frequencies >= 0.2)
        & (frequencies <= 8.0)
    )

    indices = np.where(valid)[0]

    if not indices.size:
        return float("nan")

    selected = indices[
        int(np.argmax(spectrum[indices]))
    ]

    return float(frequencies[selected])


def quaternion_to_rpy(
    quaternion,
) -> tuple[float, float, float]:
    q = to_numpy(quaternion)

    if q.size < 4:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
        )

    # Isaac Lab quaternion order: w, x, y, z.
    w, x, y, z = [
        float(item)
        for item in q[:4]
    ]

    roll = math.atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )

    pitch_argument = 2.0 * (w * y - z * x)
    pitch_argument = max(
        -1.0,
        min(1.0, pitch_argument),
    )
    pitch = math.asin(pitch_argument)

    yaw = math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )

    return roll, pitch, yaw


def get_command_tensor(base_env) -> torch.Tensor:
    manager = base_env.command_manager

    if hasattr(manager, "get_term"):
        term = manager.get_term("base_velocity")
    elif hasattr(manager, "_terms"):
        term = manager._terms["base_velocity"]
    else:
        raise RuntimeError(
            "Cannot access the base_velocity command term."
        )

    for name in (
        "vel_command_b",
        "command",
        "_command",
        "command_b",
    ):
        value = getattr(term, name, None)

        if (
            torch.is_tensor(value)
            and value.ndim == 2
            and value.shape[1] >= 3
        ):
            print(
                "[SWAY_BOOT] "
                f"command_tensor={name} "
                f"shape={tuple(value.shape)}",
                flush=True,
            )
            return value

    raise RuntimeError(
        "No writable three-axis velocity-command tensor found."
    )


def set_command(
    tensor: torch.Tensor,
    command: tuple[float, float, float],
) -> None:
    tensor[:, :3] = torch.tensor(
        command,
        dtype=tensor.dtype,
        device=tensor.device,
    )


def first_tensor_attribute(
    source,
    names: tuple[str, ...],
):
    for name in names:
        value = getattr(source, name, None)

        if torch.is_tensor(value):
            return value

    return None


def joint_group_mean(
    values: np.ndarray,
    names: list[str],
    token: str,
    legs: tuple[str, ...],
) -> float:
    selected: list[float] = []

    for index, name in enumerate(names):
        upper = name.upper()

        if (
            token in upper
            and any(
                upper.startswith(leg)
                for leg in legs
            )
        ):
            value = vector_value(values, index)

            if np.isfinite(value):
                selected.append(value)

    if not selected:
        return float("nan")

    return float(np.mean(selected))


def main() -> None:
    result_dir = Path(
        args_cli.result_dir
    ).expanduser().resolve()

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = Path(
        args_cli.checkpoint
    ).expanduser().resolve()

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    if args_cli.steps <= args_cli.warmup_steps:
        raise ValueError(
            "steps must be greater than warmup_steps."
        )

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
    )

    try:
        env_cfg.observations.policy.enable_corruption = False
    except AttributeError:
        pass

    agent_cfg = instantiate_cfg(
        load_cfg_from_registry(
            args_cli.task,
            "rsl_rl_cfg_entry_point",
        )
    )
    agent_cfg.device = args_cli.device

    gym_env = gym.make(
        args_cli.task,
        cfg=env_cfg,
    )

    env = RslRlVecEnvWrapper(
        gym_env,
        clip_actions=agent_cfg.clip_actions,
    )

    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=None,
        device=agent_cfg.device,
    )

    runner.load(
        str(checkpoint),
        load_optimizer=False,
    )

    policy = runner.get_inference_policy(
        device=env.device,
    )

    base_env = env.unwrapped
    robot = base_env.scene["robot"]

    command_tensor = get_command_tensor(base_env)

    joint_names = [
        str(name)
        for name in robot.joint_names
    ]
    body_names = [
        str(name)
        for name in robot.body_names
    ]

    foot_body_indices: dict[str, int] = {}

    for leg in LEGS:
        matches = [
            index
            for index, name in enumerate(body_names)
            if name.upper().startswith(leg)
            and "FOOT" in name.upper()
        ]

        if matches:
            foot_body_indices[leg] = matches[0]

    contact_sensor = None
    contact_sensor_name = ""
    contact_body_names: list[str] = []

    try:
        sensor_items = list(
            base_env.scene.sensors.items()
        )
    except Exception:
        sensor_items = []

    for sensor_name, sensor in sensor_items:
        if hasattr(
            getattr(sensor, "data", None),
            "net_forces_w",
        ):
            contact_sensor = sensor
            contact_sensor_name = str(sensor_name)
            contact_body_names = [
                str(name)
                for name in getattr(
                    sensor,
                    "body_names",
                    [],
                )
            ]
            break

    contact_indices: dict[str, int] = {}

    for leg, body_index in foot_body_indices.items():
        body_name = body_names[body_index]

        if body_name in contact_body_names:
            contact_indices[leg] = (
                contact_body_names.index(body_name)
            )
        elif not contact_body_names:
            contact_indices[leg] = body_index

    step_dt = float(
        getattr(base_env, "step_dt", 0.02)
    )
    sample_rate_hz = 1.0 / step_dt

    command = (
        float(args_cli.command_vx),
        float(args_cli.command_vy),
        float(args_cli.command_wz),
    )

    metadata = {
        "test_label": args_cli.test_label,
        "task": args_cli.task,
        "checkpoint": str(checkpoint),
        "seed": args_cli.seed,
        "command": command,
        "steps_requested": args_cli.steps,
        "warmup_steps": args_cli.warmup_steps,
        "step_dt": step_dt,
        "sample_rate_hz": sample_rate_hz,
        "joint_names": joint_names,
        "body_names": body_names,
        "foot_body_indices": foot_body_indices,
        "contact_sensor": contact_sensor_name,
        "contact_body_names": contact_body_names,
    }

    (result_dir / "metadata.json").write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    reset_result = env.reset()
    observations = extract_observation(
        reset_result
    )

    rows: list[dict[str, object]] = []
    previous_action: np.ndarray | None = None
    reset_count = 0

    print(
        "[SWAY_TEST] "
        f"label={args_cli.test_label} "
        f"command={command} "
        f"steps={args_cli.steps} "
        f"warmup={args_cli.warmup_steps}",
        flush=True,
    )

    with torch.inference_mode():
        for step in range(args_cli.steps):
            set_command(
                command_tensor,
                command,
            )

            actions = policy(observations)
            action_tensor = extract_action_tensor(
                actions
            )
            action_vector = to_numpy(
                action_tensor[0]
            )

            result = env.step(actions)

            (
                observations,
                reward,
                dones,
                infos,
            ) = normalize_step_result(result)

            observations = extract_observation(
                observations
            )

            done = bool(
                torch.as_tensor(dones)
                .reshape(-1)[0]
                .item()
            )

            reset_count += int(done)

            robot_data = robot.data

            root_position = to_numpy(
                robot_data.root_pos_w[0]
            )
            roll, pitch, yaw = quaternion_to_rpy(
                robot_data.root_quat_w[0]
            )

            root_linear_body = first_tensor_attribute(
                robot_data,
                (
                    "root_lin_vel_b",
                    "root_com_lin_vel_b",
                    "root_link_lin_vel_b",
                ),
            )
            root_angular_body = first_tensor_attribute(
                robot_data,
                (
                    "root_ang_vel_b",
                    "root_com_ang_vel_b",
                    "root_link_ang_vel_b",
                ),
            )

            linear_velocity = to_numpy(
                root_linear_body[0]
            )
            angular_velocity = to_numpy(
                root_angular_body[0]
            )

            joint_position = to_numpy(
                robot_data.joint_pos[0]
            )
            joint_velocity = to_numpy(
                robot_data.joint_vel[0]
            )

            joint_acceleration_source = getattr(
                robot_data,
                "joint_acc",
                None,
            )
            joint_acceleration = (
                to_numpy(
                    joint_acceleration_source[0]
                )
                if joint_acceleration_source is not None
                else np.full(
                    joint_position.shape,
                    np.nan,
                )
            )

            torque_source = getattr(
                robot_data,
                "applied_torque",
                None,
            )

            if torque_source is None:
                torque_source = getattr(
                    robot_data,
                    "computed_torque",
                    None,
                )

            torque = (
                to_numpy(torque_source[0])
                if torque_source is not None
                else np.full(
                    joint_position.shape,
                    np.nan,
                )
            )

            if previous_action is None:
                action_delta = 0.0
            else:
                action_delta = float(
                    np.linalg.norm(
                        action_vector
                        - previous_action
                    )
                )

            previous_action = action_vector.copy()

            front_calf = joint_group_mean(
                joint_position,
                joint_names,
                "CALF",
                ("FL", "FR"),
            )
            rear_calf = joint_group_mean(
                joint_position,
                joint_names,
                "CALF",
                ("RL", "RR"),
            )
            front_thigh = joint_group_mean(
                joint_position,
                joint_names,
                "THIGH",
                ("FL", "FR"),
            )
            rear_thigh = joint_group_mean(
                joint_position,
                joint_names,
                "THIGH",
                ("RL", "RR"),
            )

            body_velocity_source = first_tensor_attribute(
                robot_data,
                (
                    "body_lin_vel_w",
                    "body_link_lin_vel_w",
                    "body_com_lin_vel_w",
                ),
            )

            contact_forces = None

            if contact_sensor is not None:
                try:
                    contact_forces = to_numpy(
                        contact_sensor.data.net_forces_w[0]
                    ).reshape(-1, 3)
                except Exception:
                    contact_forces = None

            foot_slips: list[float] = []
            front_contact_force = 0.0
            rear_contact_force = 0.0

            row: dict[str, object] = {
                "test_label": args_cli.test_label,
                "step": step,
                "time_s": step * step_dt,
                "command_vx": command[0],
                "command_vy": command[1],
                "command_wz": command[2],
                "base_x": vector_value(
                    root_position, 0
                ),
                "base_y": vector_value(
                    root_position, 1
                ),
                "base_z": vector_value(
                    root_position, 2
                ),
                "base_roll": roll,
                "base_pitch": pitch,
                "base_yaw": yaw,
                "base_vx_body": vector_value(
                    linear_velocity, 0
                ),
                "base_vy_body": vector_value(
                    linear_velocity, 1
                ),
                "base_vz_body": vector_value(
                    linear_velocity, 2
                ),
                "base_wx_body": vector_value(
                    angular_velocity, 0
                ),
                "base_wy_body": vector_value(
                    angular_velocity, 1
                ),
                "base_wz_body": vector_value(
                    angular_velocity, 2
                ),
                "reward": (
                    float(
                        torch.as_tensor(reward)
                        .reshape(-1)[0]
                        .item()
                    )
                ),
                "done": int(done),
                "action_norm": float(
                    np.linalg.norm(action_vector)
                ),
                "action_delta_norm": action_delta,
                "joint_velocity_rms": finite_rms(
                    joint_velocity
                ),
                "joint_acceleration_rms": finite_rms(
                    joint_acceleration
                ),
                "torque_rms": finite_rms(torque),
                "mechanical_power_abs_mean": (
                    finite_mean(
                        np.abs(
                            torque
                            * joint_velocity
                        )
                    )
                ),
                "front_calf_mean": front_calf,
                "rear_calf_mean": rear_calf,
                "front_rear_calf_abs_diff": (
                    abs(front_calf - rear_calf)
                    if np.isfinite(front_calf)
                    and np.isfinite(rear_calf)
                    else np.nan
                ),
                "front_thigh_mean": front_thigh,
                "rear_thigh_mean": rear_thigh,
                "front_rear_thigh_abs_diff": (
                    abs(front_thigh - rear_thigh)
                    if np.isfinite(front_thigh)
                    and np.isfinite(rear_thigh)
                    else np.nan
                ),
            }

            for joint_index, joint_name in enumerate(
                joint_names
            ):
                name = safe_name(joint_name)

                row[f"joint_pos_{name}"] = (
                    vector_value(
                        joint_position,
                        joint_index,
                    )
                )
                row[f"joint_vel_{name}"] = (
                    vector_value(
                        joint_velocity,
                        joint_index,
                    )
                )
                row[f"joint_acc_{name}"] = (
                    vector_value(
                        joint_acceleration,
                        joint_index,
                    )
                )
                row[f"joint_torque_{name}"] = (
                    vector_value(
                        torque,
                        joint_index,
                    )
                )
                row[f"action_{name}"] = (
                    vector_value(
                        action_vector,
                        joint_index,
                    )
                )

            for leg in LEGS:
                body_index = foot_body_indices.get(leg)

                foot_velocity = np.full(
                    3,
                    np.nan,
                )

                if (
                    body_index is not None
                    and body_velocity_source is not None
                ):
                    foot_velocity = to_numpy(
                        body_velocity_source[
                            0,
                            body_index,
                        ]
                    )

                foot_speed_xy = float(
                    np.linalg.norm(
                        foot_velocity[:2]
                    )
                )

                contact_force = np.full(
                    3,
                    np.nan,
                )

                contact_index = contact_indices.get(
                    leg
                )

                if (
                    contact_forces is not None
                    and contact_index is not None
                    and contact_index
                    < len(contact_forces)
                ):
                    contact_force = contact_forces[
                        contact_index
                    ]

                force_norm = float(
                    np.linalg.norm(contact_force)
                )

                contact_slip = (
                    foot_speed_xy
                    if np.isfinite(force_norm)
                    and force_norm > 5.0
                    else 0.0
                )

                foot_slips.append(contact_slip)

                if leg in ("FL", "FR"):
                    if np.isfinite(force_norm):
                        front_contact_force += force_norm
                else:
                    if np.isfinite(force_norm):
                        rear_contact_force += force_norm

                row[f"{leg}_foot_speed_xy"] = (
                    foot_speed_xy
                )
                row[f"{leg}_contact_force"] = (
                    force_norm
                )
                row[f"{leg}_contact_fz"] = (
                    vector_value(
                        contact_force,
                        2,
                    )
                )
                row[f"{leg}_contact_slip"] = (
                    contact_slip
                )

            row["mean_contact_foot_slip"] = float(
                np.mean(foot_slips)
            )
            row["max_contact_foot_slip"] = float(
                np.max(foot_slips)
            )
            row["front_contact_force_sum"] = (
                front_contact_force
            )
            row["rear_contact_force_sum"] = (
                rear_contact_force
            )
            row[
                "front_rear_contact_force_imbalance"
            ] = (
                abs(
                    front_contact_force
                    - rear_contact_force
                )
                / (
                    front_contact_force
                    + rear_contact_force
                    + 1.0e-8
                )
            )

            rows.append(row)

            if step % 50 == 0:
                print(
                    "[SWAY_PROGRESS] "
                    f"step={step}/{args_cli.steps} "
                    f"cmd=({command[0]:+.2f},"
                    f"{command[1]:+.2f},"
                    f"{command[2]:+.2f}) "
                    f"actual=("
                    f"{row['base_vx_body']:+.3f},"
                    f"{row['base_vy_body']:+.3f},"
                    f"{row['base_wz_body']:+.3f}) "
                    f"roll={math.degrees(roll):+.2f}deg "
                    f"pitch={math.degrees(pitch):+.2f}deg "
                    f"slip={row['mean_contact_foot_slip']:.4f} "
                    f"action_delta={action_delta:.4f} "
                    f"done={int(done)}",
                    flush=True,
                )

            if done and args_cli.stop_on_done:
                print(
                    "[SWAY_TEST] "
                    f"stopped_on_done_at_step={step}",
                    flush=True,
                )
                break

    env.close()

    if not rows:
        raise RuntimeError(
            "No diagnostic rows were recorded."
        )

    trajectory_path = (
        result_dir / "low_level_diagnostic.csv"
    )

    with trajectory_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    settled_rows = rows[
        min(
            args_cli.warmup_steps,
            max(0, len(rows) - 1),
        ):
    ]

    def column(name: str) -> np.ndarray:
        return np.asarray(
            [
                float(row[name])
                for row in settled_rows
            ],
            dtype=np.float64,
        )

    base_x = column("base_x")
    base_y = column("base_y")
    base_yaw = np.unwrap(
        column("base_yaw")
    )

    measured_vx = column("base_vx_body")
    measured_vy = column("base_vy_body")
    measured_wz = column("base_wz_body")

    summary = {
        "test_label": args_cli.test_label,
        "command": command,
        "steps_requested": args_cli.steps,
        "steps_completed": len(rows),
        "warmup_steps": args_cli.warmup_steps,
        "reset_count": reset_count,
        "contact_sensor_found": (
            contact_sensor is not None
        ),
        "contact_sensor": contact_sensor_name,
        "duration_s": (
            (len(rows) - 1) * step_dt
        ),
        "base_vx_mean": finite_mean(
            measured_vx
        ),
        "base_vx_tracking_rmse": finite_rms(
            measured_vx - command[0]
        ),
        "base_vy_mean": finite_mean(
            measured_vy
        ),
        "base_vy_rms": finite_rms(
            measured_vy
        ),
        "base_wz_mean": finite_mean(
            measured_wz
        ),
        "base_wz_tracking_rmse": finite_rms(
            measured_wz - command[2]
        ),
        "roll_rms_deg": math.degrees(
            finite_rms(
                column("base_roll")
            )
        ),
        "pitch_rms_deg": math.degrees(
            finite_rms(
                column("base_pitch")
            )
        ),
        "yaw_drift_deg": (
            math.degrees(
                float(
                    base_yaw[-1]
                    - base_yaw[0]
                )
            )
            if base_yaw.size >= 2
            else float("nan")
        ),
        "lateral_drift_m": (
            float(base_y[-1] - base_y[0])
            if base_y.size >= 2
            else float("nan")
        ),
        "forward_progress_m": (
            float(base_x[-1] - base_x[0])
            if base_x.size >= 2
            else float("nan")
        ),
        "lateral_oscillation_frequency_hz": (
            dominant_frequency(
                measured_vy,
                sample_rate_hz,
            )
        ),
        "yaw_oscillation_frequency_hz": (
            dominant_frequency(
                measured_wz - command[2],
                sample_rate_hz,
            )
        ),
        "action_delta_mean": finite_mean(
            column("action_delta_norm")
        ),
        "action_delta_p95": finite_percentile(
            column("action_delta_norm"),
            95.0,
        ),
        "joint_velocity_rms_mean": finite_mean(
            column("joint_velocity_rms")
        ),
        "joint_acceleration_rms_mean": finite_mean(
            column(
                "joint_acceleration_rms"
            )
        ),
        "torque_rms_mean": finite_mean(
            column("torque_rms")
        ),
        "mechanical_power_abs_mean": finite_mean(
            column(
                "mechanical_power_abs_mean"
            )
        ),
        "foot_slip_mean": finite_mean(
            column(
                "mean_contact_foot_slip"
            )
        ),
        "foot_slip_p95": finite_percentile(
            column(
                "max_contact_foot_slip"
            ),
            95.0,
        ),
        "front_rear_contact_force_imbalance_mean": (
            finite_mean(
                column(
                    "front_rear_contact_force_imbalance"
                )
            )
        ),
        "front_rear_calf_diff_mean": finite_mean(
            column(
                "front_rear_calf_abs_diff"
            )
        ),
        "front_rear_calf_diff_p95": finite_percentile(
            column(
                "front_rear_calf_abs_diff"
            ),
            95.0,
        ),
        "front_rear_thigh_diff_mean": finite_mean(
            column(
                "front_rear_thigh_abs_diff"
            )
        ),
    }

    summary_path = result_dir / "summary.json"

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print()
    print("===== SWAY DIAGNOSTIC SUMMARY =====")

    for key, value in summary.items():
        print(
            f"[SWAY_SUMMARY] {key}={value}",
            flush=True,
        )

    print(f"CSV={trajectory_path}")
    print(f"SUMMARY={summary_path}")
    print("GO2_SWAY_DIAGNOSTIC_COMPLETE")


try:
    main()
except BaseException as error:
    print(
        "[SWAY_FATAL] "
        f"{type(error).__name__}: {error}",
        flush=True,
    )
    raise
finally:
    print(
        "[SWAY_BOOT] closing_simulation_app",
        flush=True,
    )
    simulation_app.close()
