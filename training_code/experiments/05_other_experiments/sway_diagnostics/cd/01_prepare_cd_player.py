#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from pathlib import Path


MARKER = "# PHASEA_FIXED_COMMAND_DIAGNOSTIC_V1"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(source)

    text = source.read_text(encoding="utf-8")

    required = (
        "class HighLevelSoccerVecEnv:",
        "# PHASEA_DETERMINISTIC_VIDEO_EVAL_V2",
        "PHASEA_EVAL_VIDEO_OK",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(
            "The source is not the validated Phase-A player. "
            f"Missing markers: {missing}"
        )

    if MARKER in text:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8", newline="\n")
        print(f"DIAGNOSTIC_PATCH_ALREADY_PRESENT={output}")
        return

    cli_block = r"""
# PHASEA_FIXED_COMMAND_DIAGNOSTIC_V1
parser.add_argument("--diagnostic_dir", type=str, default="")
parser.add_argument("--diagnostic_label", type=str, default="")
parser.add_argument(
    "--diagnostic_mode",
    choices=("fixed_no_ball", "fixed_ball", "policy_ball"),
    default="policy_ball",
)
parser.add_argument("--fixed_vx", type=float, default=0.0)
parser.add_argument("--fixed_vy", type=float, default=0.0)
parser.add_argument("--fixed_wz", type=float, default=0.0)
parser.add_argument("--diagnostic_stop_on_done", action="store_true")
parser.add_argument("--diagnostic_skip_render", action="store_true")

"""
    text = replace_once(
        text,
        "AppLauncher.add_app_launcher_args(parser)",
        cli_block + "AppLauncher.add_app_launcher_args(parser)",
        "CLI insertion",
    )

    text = replace_once(
        text,
        """        self.vector_steps = 0

        self.cfg = SimpleNamespace(is_finite_horizon=False)
""",
        """        self.vector_steps = 0

        # PHASEA_FIXED_COMMAND_DIAGNOSTIC_V1
        self.diagnostic_hook = None

        self.cfg = SimpleNamespace(is_finite_horizon=False)
""",
        "diagnostic hook attribute",
    )

    text = replace_once(
        text,
        """            for _ in range(self.low_level_steps_per_action):
                self._set_command(command)
""",
        """            for low_step_index in range(
                self.low_level_steps_per_action
            ):
                self._set_command(command)
""",
        "low-level loop",
    )

    callback_anchor_line = (
        "                self._low_obs = "
        "extract_observation(low_obs)"
    )

    callback_count = text.count(callback_anchor_line)

    if callback_count != 1:
        nearby = [
            line
            for line in text.splitlines()
            if (
                "extract_observation(low_obs)" in line
                or "torch.is_tensor(dones)" in line
            )
        ]

        raise RuntimeError(
            "diagnostic callback line: expected exactly one "
            f"target, found {callback_count}. "
            f"Nearby source lines: {nearby[-20:]}"
        )

    callback_replacement = (
        callback_anchor_line
        + "\n\n"
        + "                if self.diagnostic_hook "
        + "is not None:\n"
        + "                    self.diagnostic_hook(\n"
        + "                        self,\n"
        + "                        low_step_index,\n"
        + "                        command,\n"
        + "                        low_actions,\n"
        + "                        dones,\n"
        + "                    )"
    )

    text = text.replace(
        callback_anchor_line,
        callback_replacement,
        1,
    )

    text = replace_once(
        text,
        """        import imageio.v2 as _imageio
        import numpy as _np
""",
        """        import csv as _csv
        import json as _json
        import math as _math

        import imageio.v2 as _imageio
        import numpy as _np
""",
        "evaluation imports",
    )

    setup_block = r"""        observations = runner_env.reset()

        _diag_enabled = bool(args_cli.diagnostic_dir)
        _diag_low_file = None
        _diag_high_file = None
        _diag_low_writer = None
        _diag_high_writer = None
        _diag_context = {
            "high_step": -1,
            "global_low_step": 0,
            "high_action": torch.zeros(
                high_env.num_envs,
                3,
                device=high_env.device,
            ),
            "previous_low_action": None,
        }

        def _diag_vector(value):
            if value is None:
                return _np.asarray([], dtype=_np.float64)
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            return _np.asarray(
                value.detach().reshape(-1).cpu().numpy(),
                dtype=_np.float64,
            )

        def _diag_at(vector, index, default=_np.nan):
            try:
                return float(vector[index])
            except (IndexError, TypeError):
                return float(default)

        def _diag_rpy(quaternion):
            q = _diag_vector(quaternion)
            if q.size < 4:
                return (_np.nan, _np.nan, _np.nan)
            w, x, y, z = [float(item) for item in q[:4]]
            roll = _math.atan2(
                2.0 * (w * x + y * z),
                1.0 - 2.0 * (x * x + y * y),
            )
            pitch_argument = 2.0 * (w * y - z * x)
            pitch = _math.asin(
                max(-1.0, min(1.0, pitch_argument))
            )
            yaw = _math.atan2(
                2.0 * (w * z + x * y),
                1.0 - 2.0 * (y * y + z * z),
            )
            return (roll, pitch, yaw)

        def _diag_action_tensor(value):
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
                "Cannot extract low-level action tensor from "
                f"{type(value)!r}"
            )

        def _diag_inverse_command(command):
            action = torch.zeros_like(command)
            action[:, 0] = torch.where(
                command[:, 0] >= 0.0,
                command[:, 0] / 0.65,
                command[:, 0] / 0.25,
            )
            action[:, 1] = command[:, 1] / 0.35
            action[:, 2] = command[:, 2] / 0.90
            return torch.clamp(action, -1.0, 1.0)

        if _diag_enabled:
            _diag_dir = _EvalPath(
                args_cli.diagnostic_dir
            ).expanduser().resolve()
            _diag_dir.mkdir(parents=True, exist_ok=True)
            _diag_label = (
                args_cli.diagnostic_label
                or args_cli.diagnostic_mode
            )

            _joint_names = [
                str(name)
                for name in high_env.robot.joint_names
            ]
            _body_names = [
                str(name)
                for name in high_env.robot.body_names
            ]

            _foot_body_indices = {}
            for _leg in ("FL", "FR", "RL", "RR"):
                _matches = [
                    index
                    for index, name in enumerate(_body_names)
                    if name.upper().startswith(_leg)
                    and "FOOT" in name.upper()
                ]
                if _matches:
                    _foot_body_indices[_leg] = _matches[0]

            _contact_sensor = None
            _contact_body_names = []
            try:
                _sensor_items = list(
                    high_env.base_env.scene.sensors.items()
                )
            except Exception:
                _sensor_items = []

            for _sensor_name, _sensor in _sensor_items:
                if hasattr(
                    getattr(_sensor, "data", None),
                    "net_forces_w",
                ):
                    _contact_sensor = _sensor
                    _contact_body_names = [
                        str(name)
                        for name in getattr(_sensor, "body_names", [])
                    ]
                    print(
                        "[PHASEA_DIAG] "
                        f"contact_sensor={_sensor_name}",
                        flush=True,
                    )
                    break

            _contact_indices = {}
            for _leg, _body_index in _foot_body_indices.items():
                _body_name = _body_names[_body_index]
                if _body_name in _contact_body_names:
                    _contact_indices[_leg] = (
                        _contact_body_names.index(_body_name)
                    )
                elif not _contact_body_names:
                    _contact_indices[_leg] = _body_index

            _ids = torch.arange(
                high_env.num_envs,
                device=high_env.device,
                dtype=torch.long,
            )
            _origins = high_env.env_origins[_ids]

            _robot_pose = torch.zeros(
                high_env.num_envs, 7, device=high_env.device
            )
            _robot_pose[:, 0] = _origins[:, 0] - 1.00
            _robot_pose[:, 1] = _origins[:, 1]
            _robot_pose[:, 2] = (
                high_env.robot.data.root_pos_w[:, 2]
            )
            _robot_pose[:, 3] = 1.0

            _ball_pose = torch.zeros(
                high_env.num_envs, 7, device=high_env.device
            )
            if args_cli.diagnostic_mode == "fixed_no_ball":
                _ball_pose[:, 0] = _origins[:, 0] - 2.50
                _ball_pose[:, 1] = _origins[:, 1] + 2.00
            else:
                _ball_pose[:, 0] = _origins[:, 0] - 0.25
                _ball_pose[:, 1] = _origins[:, 1]
            _ball_pose[:, 2] = high_env.ball.data.root_pos_w[:, 2]
            _ball_pose[:, 3] = 1.0

            _zero_velocity = torch.zeros(
                high_env.num_envs, 6, device=high_env.device
            )

            high_env.robot.write_root_pose_to_sim(
                _robot_pose, env_ids=_ids
            )
            high_env.robot.write_root_velocity_to_sim(
                _zero_velocity, env_ids=_ids
            )
            high_env.ball.write_root_pose_to_sim(
                _ball_pose, env_ids=_ids
            )
            high_env.ball.write_root_velocity_to_sim(
                _zero_velocity, env_ids=_ids
            )

            high_env.command_tensor[:, :3] = 0.0
            high_env.previous_action[:] = 0.0
            high_env.episode_length_buf[:] = 0
            high_env.reset_buf[:] = False
            high_env.rew_buf[:] = 0.0

            _pose_state = high_env._state()
            high_env.previous_ball_goal_distance = (
                _pose_state["ball_goal_distance"].clone()
            )
            high_env.previous_robot_ball_distance = (
                _pose_state["robot_ball_distance"].clone()
            )
            high_env._low_obs = extract_observation(
                high_env.low_env.get_observations()
            )
            try:
                observations = runner_env.get_observations()
            except Exception:
                observations = high_env.get_observations()

            _low_fields = [
                "label", "mode", "high_step", "low_substep",
                "global_low_step", "sim_time_s",
                "command_vx", "command_vy", "command_wz",
                "high_action_0", "high_action_1", "high_action_2",
                "low_action_norm", "low_action_delta_norm",
                "base_done_substep",
                "base_x", "base_y", "base_z",
                "base_roll", "base_pitch", "base_yaw",
                "base_vx_body", "base_vy_body", "base_vz_body",
                "base_wx_body", "base_wy_body", "base_wz_body",
                "ball_x", "ball_y", "ball_z",
                "ball_vx_world", "ball_vy_world", "ball_vz_world",
                "robot_ball_distance", "ball_goal_distance",
                "ball_goal_alignment", "ball_velocity_toward_goal",
                "front_calf_mean", "rear_calf_mean",
                "front_rear_calf_abs_diff",
                "front_thigh_mean", "rear_thigh_mean",
                "front_rear_thigh_abs_diff",
                "joint_velocity_rms",
                "joint_acceleration_rms",
                "applied_torque_rms",
                "mean_contact_foot_slip",
                "max_contact_foot_slip",
            ]
            for _leg in ("FL", "FR", "RL", "RR"):
                _low_fields.extend(
                    [
                        f"{_leg}_foot_speed_xy",
                        f"{_leg}_contact_force",
                        f"{_leg}_contact_fz",
                        f"{_leg}_contact_slip",
                    ]
                )

            _high_fields = [
                "label", "mode", "high_step", "reward", "done",
                "time_out", "command_vx", "command_vy", "command_wz",
                "high_action_0", "high_action_1", "high_action_2",
                "robot_ball_distance", "ball_goal_distance",
                "ball_goal_alignment", "ball_velocity_toward_goal",
            ]

            _diag_low_file = (
                _diag_dir / "low_level.csv"
            ).open("w", encoding="utf-8", newline="")
            _diag_high_file = (
                _diag_dir / "high_level.csv"
            ).open("w", encoding="utf-8", newline="")
            _diag_low_writer = _csv.DictWriter(
                _diag_low_file, fieldnames=_low_fields
            )
            _diag_high_writer = _csv.DictWriter(
                _diag_high_file, fieldnames=_high_fields
            )
            _diag_low_writer.writeheader()
            _diag_high_writer.writeheader()

            _diag_dt = float(
                getattr(high_env.base_env, "step_dt", 0.02)
            )

            (_diag_dir / "metadata.json").write_text(
                _json.dumps(
                    {
                        "label": _diag_label,
                        "mode": args_cli.diagnostic_mode,
                        "fixed_command": [
                            args_cli.fixed_vx,
                            args_cli.fixed_vy,
                            args_cli.fixed_wz,
                        ],
                        "low_level_step_dt": _diag_dt,
                        "low_level_steps_per_action": (
                            high_env.low_level_steps_per_action
                        ),
                        "joint_names": _joint_names,
                        "body_names": _body_names,
                        "foot_body_indices": _foot_body_indices,
                        "contact_body_names": _contact_body_names,
                        "initial_robot_local_xy": [-1.0, 0.0],
                        "initial_ball_local_xy": (
                            [-2.5, 2.0]
                            if args_cli.diagnostic_mode
                            == "fixed_no_ball"
                            else [-0.25, 0.0]
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            def _joint_group_mean(values, token, legs):
                vector = _diag_vector(values)
                selected = []
                for index, name in enumerate(_joint_names):
                    upper = name.upper()
                    if token in upper and any(
                        upper.startswith(leg) for leg in legs
                    ):
                        selected.append(_diag_at(vector, index))
                selected = [
                    value for value in selected
                    if _np.isfinite(value)
                ]
                return (
                    float(_np.mean(selected))
                    if selected
                    else _np.nan
                )

            def _diag_hook(
                diagnostic_env,
                low_step_index,
                command,
                low_actions,
                low_dones,
            ):
                robot_data = diagnostic_env.robot.data
                ball_data = diagnostic_env.ball.data
                state = diagnostic_env._state()

                low_action = _diag_vector(
                    _diag_action_tensor(low_actions)[0]
                )
                previous = _diag_context["previous_low_action"]
                delta_norm = (
                    0.0
                    if previous is None
                    else float(_np.linalg.norm(low_action - previous))
                )
                _diag_context["previous_low_action"] = (
                    low_action.copy()
                )

                command_vector = _diag_vector(command[0])
                high_action = _diag_vector(
                    _diag_context["high_action"][0]
                )
                base_pos = _diag_vector(
                    robot_data.root_pos_w[0]
                )
                roll, pitch, yaw = _diag_rpy(
                    robot_data.root_quat_w[0]
                )
                base_vel = _diag_vector(
                    robot_data.root_lin_vel_b[0]
                )
                base_ang = _diag_vector(
                    robot_data.root_ang_vel_b[0]
                )
                ball_pos = _diag_vector(
                    ball_data.root_pos_w[0]
                )
                ball_vel = _diag_vector(
                    ball_data.root_lin_vel_w[0]
                )

                joint_pos = _diag_vector(robot_data.joint_pos[0])
                joint_vel = _diag_vector(robot_data.joint_vel[0])

                joint_acc_source = getattr(
                    robot_data, "joint_acc", None
                )
                joint_acc = (
                    _diag_vector(joint_acc_source[0])
                    if joint_acc_source is not None
                    else _np.full(joint_pos.shape, _np.nan)
                )

                torque_source = getattr(
                    robot_data, "applied_torque", None
                )
                if torque_source is None:
                    torque_source = getattr(
                        robot_data, "computed_torque", None
                    )
                torque = (
                    _diag_vector(torque_source[0])
                    if torque_source is not None
                    else _np.full(joint_pos.shape, _np.nan)
                )

                front_calf = _joint_group_mean(
                    joint_pos, "CALF", ("FL", "FR")
                )
                rear_calf = _joint_group_mean(
                    joint_pos, "CALF", ("RL", "RR")
                )
                front_thigh = _joint_group_mean(
                    joint_pos, "THIGH", ("FL", "FR")
                )
                rear_thigh = _joint_group_mean(
                    joint_pos, "THIGH", ("RL", "RR")
                )

                row = {
                    "label": _diag_label,
                    "mode": args_cli.diagnostic_mode,
                    "high_step": _diag_context["high_step"],
                    "low_substep": int(low_step_index),
                    "global_low_step": (
                        _diag_context["global_low_step"]
                    ),
                    "sim_time_s": (
                        _diag_context["global_low_step"] * _diag_dt
                    ),
                    "command_vx": _diag_at(command_vector, 0),
                    "command_vy": _diag_at(command_vector, 1),
                    "command_wz": _diag_at(command_vector, 2),
                    "high_action_0": _diag_at(high_action, 0),
                    "high_action_1": _diag_at(high_action, 1),
                    "high_action_2": _diag_at(high_action, 2),
                    "low_action_norm": float(
                        _np.linalg.norm(low_action)
                    ),
                    "low_action_delta_norm": delta_norm,
                    "base_done_substep": int(
                        bool(
                            torch.as_tensor(low_dones)
                            .reshape(-1)[0].item()
                        )
                    ),
                    "base_x": _diag_at(base_pos, 0),
                    "base_y": _diag_at(base_pos, 1),
                    "base_z": _diag_at(base_pos, 2),
                    "base_roll": roll,
                    "base_pitch": pitch,
                    "base_yaw": yaw,
                    "base_vx_body": _diag_at(base_vel, 0),
                    "base_vy_body": _diag_at(base_vel, 1),
                    "base_vz_body": _diag_at(base_vel, 2),
                    "base_wx_body": _diag_at(base_ang, 0),
                    "base_wy_body": _diag_at(base_ang, 1),
                    "base_wz_body": _diag_at(base_ang, 2),
                    "ball_x": _diag_at(ball_pos, 0),
                    "ball_y": _diag_at(ball_pos, 1),
                    "ball_z": _diag_at(ball_pos, 2),
                    "ball_vx_world": _diag_at(ball_vel, 0),
                    "ball_vy_world": _diag_at(ball_vel, 1),
                    "ball_vz_world": _diag_at(ball_vel, 2),
                    "robot_ball_distance": float(
                        state["robot_ball_distance"][0].item()
                    ),
                    "ball_goal_distance": float(
                        state["ball_goal_distance"][0].item()
                    ),
                    "ball_goal_alignment": float(
                        state["ball_goal_alignment"][0].item()
                    ),
                    "ball_velocity_toward_goal": float(
                        state["ball_velocity_toward_goal"][0].item()
                    ),
                    "front_calf_mean": front_calf,
                    "rear_calf_mean": rear_calf,
                    "front_rear_calf_abs_diff": (
                        abs(front_calf - rear_calf)
                        if _np.isfinite(front_calf)
                        and _np.isfinite(rear_calf)
                        else _np.nan
                    ),
                    "front_thigh_mean": front_thigh,
                    "rear_thigh_mean": rear_thigh,
                    "front_rear_thigh_abs_diff": (
                        abs(front_thigh - rear_thigh)
                        if _np.isfinite(front_thigh)
                        and _np.isfinite(rear_thigh)
                        else _np.nan
                    ),
                    "joint_velocity_rms": float(
                        _np.sqrt(_np.nanmean(joint_vel ** 2))
                    ),
                    "joint_acceleration_rms": float(
                        _np.sqrt(_np.nanmean(joint_acc ** 2))
                    ),
                    "applied_torque_rms": float(
                        _np.sqrt(_np.nanmean(torque ** 2))
                    ),
                }

                body_vel_source = getattr(
                    robot_data, "body_lin_vel_w", None
                )
                contact_forces = None
                if _contact_sensor is not None:
                    try:
                        contact_forces = _diag_vector(
                            _contact_sensor.data.net_forces_w[0]
                        ).reshape(-1, 3)
                    except Exception:
                        contact_forces = None

                contact_slips = []
                for leg in ("FL", "FR", "RL", "RR"):
                    body_index = _foot_body_indices.get(leg)
                    foot_velocity = _np.full(3, _np.nan)
                    if (
                        body_index is not None
                        and body_vel_source is not None
                    ):
                        foot_velocity = _diag_vector(
                            body_vel_source[0, body_index]
                        )
                    foot_speed = float(
                        _np.linalg.norm(foot_velocity[:2])
                    )

                    force = _np.full(3, _np.nan)
                    contact_index = _contact_indices.get(leg)
                    if (
                        contact_forces is not None
                        and contact_index is not None
                        and contact_index < len(contact_forces)
                    ):
                        force = contact_forces[contact_index]
                    force_norm = float(_np.linalg.norm(force))
                    slip = (
                        foot_speed
                        if _np.isfinite(force_norm)
                        and force_norm > 5.0
                        else 0.0
                    )
                    contact_slips.append(slip)
                    row[f"{leg}_foot_speed_xy"] = foot_speed
                    row[f"{leg}_contact_force"] = force_norm
                    row[f"{leg}_contact_fz"] = _diag_at(force, 2)
                    row[f"{leg}_contact_slip"] = slip

                row["mean_contact_foot_slip"] = float(
                    _np.mean(contact_slips)
                )
                row["max_contact_foot_slip"] = float(
                    _np.max(contact_slips)
                )

                _diag_low_writer.writerow(row)
                _diag_context["global_low_step"] += 1
                if _diag_context["global_low_step"] % 50 == 0:
                    _diag_low_file.flush()

            high_env.diagnostic_hook = _diag_hook

            print(
                "[PHASEA_DIAG] "
                f"label={_diag_label} "
                f"mode={args_cli.diagnostic_mode} "
                f"directory={_diag_dir}",
                flush=True,
            )
"""
    text = replace_once(
        text,
        "        observations = runner_env.reset()\n",
        setup_block,
        "evaluation setup",
    )

    text = replace_once(
        text,
        """                with torch.no_grad():
                    actions = policy(
                        observations
                    )

                (
""",
        """                if (
                    _diag_enabled
                    and args_cli.diagnostic_mode != "policy_ball"
                ):
                    _fixed_command = torch.tensor(
                        [
                            args_cli.fixed_vx,
                            args_cli.fixed_vy,
                            args_cli.fixed_wz,
                        ],
                        dtype=torch.float32,
                        device=high_env.device,
                    ).repeat(high_env.num_envs, 1)
                    actions = _diag_inverse_command(_fixed_command)
                else:
                    with torch.no_grad():
                        actions = policy(observations)

                if _diag_enabled:
                    _diag_context["high_step"] = step_index
                    _diag_context["high_action"] = (
                        actions.detach().clone()
                    )

                (
""",
        "fixed command action selection",
    )

    text = replace_once(
        text,
        """                done_total += int(
                    dones.sum()
                )

                frame = render_env.render()
""",
        """                done_total += int(
                    dones.sum()
                )

                if _diag_enabled:
                    _state = high_env._state()
                    _action = _diag_vector(actions[0])
                    _command = _diag_vector(
                        high_env._action_to_command(actions)[0]
                    )
                    _time_out = 0.0
                    try:
                        _time_out = float(
                            extras["time_outs"]
                            .reshape(-1)[0].item()
                        )
                    except Exception:
                        pass
                    _diag_high_writer.writerow(
                        {
                            "label": _diag_label,
                            "mode": args_cli.diagnostic_mode,
                            "high_step": step_index,
                            "reward": float(
                                rewards.reshape(-1)[0].item()
                            ),
                            "done": int(
                                bool(
                                    dones.reshape(-1)[0].item()
                                )
                            ),
                            "time_out": _time_out,
                            "command_vx": _diag_at(_command, 0),
                            "command_vy": _diag_at(_command, 1),
                            "command_wz": _diag_at(_command, 2),
                            "high_action_0": _diag_at(_action, 0),
                            "high_action_1": _diag_at(_action, 1),
                            "high_action_2": _diag_at(_action, 2),
                            "robot_ball_distance": float(
                                _state[
                                    "robot_ball_distance"
                                ][0].item()
                            ),
                            "ball_goal_distance": float(
                                _state[
                                    "ball_goal_distance"
                                ][0].item()
                            ),
                            "ball_goal_alignment": float(
                                _state[
                                    "ball_goal_alignment"
                                ][0].item()
                            ),
                            "ball_velocity_toward_goal": float(
                                _state[
                                    "ball_velocity_toward_goal"
                                ][0].item()
                            ),
                        }
                    )
                    _diag_high_file.flush()

                if (
                    _diag_enabled
                    and args_cli.diagnostic_skip_render
                ):
                    frame = _np.zeros(
                        (16, 16, 3),
                        dtype=_np.uint8,
                    )
                else:
                    frame = render_env.render()
""",
        "high-level record",
    )

    text = replace_once(
        text,
        """                writer.append_data(frame)
                frame_count += 1

                if (
""",
        """                writer.append_data(frame)
                frame_count += 1

                if (
                    _diag_enabled
                    and args_cli.diagnostic_stop_on_done
                    and bool(dones.any())
                ):
                    print(
                        "[PHASEA_DIAG] "
                        f"stop_on_done_at_high_step={step_index}",
                        flush=True,
                    )
                    break

                if (
""",
        "stop on done",
    )

    text = replace_once(
        text,
        """        finally:
            writer.close()

        if not eval_video.is_file():
""",
        """        finally:
            writer.close()
            if _diag_enabled:
                high_env.diagnostic_hook = None
                if _diag_low_file is not None:
                    _diag_low_file.flush()
                    _diag_low_file.close()
                if _diag_high_file is not None:
                    _diag_high_file.flush()
                    _diag_high_file.close()

        if not eval_video.is_file():
""",
        "diagnostic file close",
    )

    ast.parse(text)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8", newline="\n")

    print(f"DIAGNOSTIC_PLAYER_CREATED={output}")
    print("PHASEA_FIXED_COMMAND_DIAGNOSTIC_PATCH_OK")


if __name__ == "__main__":
    main()
