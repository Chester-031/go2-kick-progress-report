from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--source", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

source = Path(args.source).resolve()
output = Path(args.output).resolve()
text = source.read_text(encoding="utf-8")

for marker in (
    "class HighLevelSoccerVecEnv:",
    "# PHASEA_DETERMINISTIC_VIDEO_EVAL_V2",
    "PHASEA_EVAL_VIDEO_OK",
):
    if marker not in text:
        raise RuntimeError(f"Missing source marker: {marker}")

render_pattern = re.compile(
    r"(?m)^(?P<indent>[ \t]*)"
    r"frame[ \t]*=[ \t]*render_env\.render\(\)[ \t]*$"
)
render_matches = list(render_pattern.finditer(text))

if len(render_matches) != 1:
    raise RuntimeError(
        "Expected one render call, found "
        f"{len(render_matches)}"
    )

match = render_matches[0]
indent = match.group("indent")
text = (
    text[:match.start()]
    + indent
    + "frame = _np.zeros((16, 16, 3), dtype=_np.uint8)"
    + text[match.end():]
)

patch = r'''
# PHASEA_CD_DIRECT_DIAGNOSTIC_V2
import atexit as _cd_atexit
import csv as _cd_csv
import json as _cd_json
import math as _cd_math
import os as _cd_os
from pathlib import Path as _CDPath

import numpy as _cd_np


_CD_DIR = _cd_os.environ.get("PHASEA_CD_DIAG_DIR", "")
_CD_LABEL = _cd_os.environ.get("PHASEA_CD_DIAG_LABEL", "")
_CD_MODE = _cd_os.environ.get("PHASEA_CD_DIAG_MODE", "policy_ball")
_CD_FIXED = (
    float(_cd_os.environ.get("PHASEA_CD_FIXED_VX", "0")),
    float(_cd_os.environ.get("PHASEA_CD_FIXED_VY", "0")),
    float(_cd_os.environ.get("PHASEA_CD_FIXED_WZ", "0")),
)


def _cd_vec(value):
    if value is None:
        return _cd_np.asarray([], dtype=_cd_np.float64)
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return _cd_np.asarray(
        value.detach().reshape(-1).cpu().numpy(),
        dtype=_cd_np.float64,
    )


def _cd_at(vector, index):
    try:
        return float(vector[index])
    except Exception:
        return float("nan")


def _cd_rms(vector):
    vector = _cd_np.asarray(vector, dtype=_cd_np.float64)
    vector = vector[_cd_np.isfinite(vector)]
    if not vector.size:
        return float("nan")
    return float(_cd_np.sqrt(_cd_np.mean(vector**2)))


def _cd_rpy(quaternion):
    q = _cd_vec(quaternion)
    if q.size < 4:
        return (float("nan"),) * 3
    w, x, y, z = [float(item) for item in q[:4]]
    roll = _cd_math.atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )
    pitch_arg = max(
        -1.0,
        min(1.0, 2.0 * (w * y - z * x)),
    )
    pitch = _cd_math.asin(pitch_arg)
    yaw = _cd_math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    return roll, pitch, yaw


def _cd_action(value):
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
        f"Cannot extract action tensor from {type(value)!r}"
    )


def _cd_first_tensor(source, names):
    for name in names:
        value = getattr(source, name, None)
        if torch.is_tensor(value):
            return value
    return None


def _cd_group_mean(values, names, token, legs):
    vector = _cd_vec(values)
    selected = []
    for index, name in enumerate(names):
        upper = name.upper()
        if token in upper and any(
            upper.startswith(leg) for leg in legs
        ):
            value = _cd_at(vector, index)
            if _cd_np.isfinite(value):
                selected.append(value)
    return (
        float(_cd_np.mean(selected))
        if selected
        else float("nan")
    )


if _CD_DIR:
    _cd_orig_init = HighLevelSoccerVecEnv.__init__
    _cd_orig_reset = HighLevelSoccerVecEnv.reset
    _cd_orig_convert = HighLevelSoccerVecEnv._action_to_command
    _cd_orig_step = HighLevelSoccerVecEnv.step

    def _cd_close(self):
        for name in ("_cd_low_file", "_cd_high_file"):
            handle = getattr(self, name, None)
            if handle is not None and not handle.closed:
                handle.flush()
                handle.close()

    def _cd_place(self):
        ids = torch.arange(
            self.num_envs,
            device=self.device,
            dtype=torch.long,
        )
        origins = self.env_origins[ids]

        robot_pose = torch.zeros(
            self.num_envs, 7, device=self.device
        )
        robot_pose[:, 0] = origins[:, 0] - 1.0
        robot_pose[:, 1] = origins[:, 1]
        robot_pose[:, 2] = self.robot.data.root_pos_w[:, 2]
        robot_pose[:, 3] = 1.0

        ball_pose = torch.zeros(
            self.num_envs, 7, device=self.device
        )
        ball_pose[:, 0] = origins[:, 0] - 0.25
        ball_pose[:, 1] = origins[:, 1]
        ball_pose[:, 2] = self.ball.data.root_pos_w[:, 2]
        ball_pose[:, 3] = 1.0

        zero_velocity = torch.zeros(
            self.num_envs, 6, device=self.device
        )

        self.robot.write_root_pose_to_sim(
            robot_pose, env_ids=ids
        )
        self.robot.write_root_velocity_to_sim(
            zero_velocity, env_ids=ids
        )
        self.ball.write_root_pose_to_sim(
            ball_pose, env_ids=ids
        )
        self.ball.write_root_velocity_to_sim(
            zero_velocity, env_ids=ids
        )

        try:
            self.base_env.scene.write_data_to_sim()
            self.base_env.sim.forward()
        except Exception as error:
            print(
                "[CD_DIAG] scene flush warning: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )

        state = self._state()
        self.previous_ball_goal_distance = (
            state["ball_goal_distance"].clone()
        )
        self.previous_robot_ball_distance = (
            state["robot_ball_distance"].clone()
        )
        self._low_obs = extract_observation(
            self.low_env.get_observations()
        )

    def _cd_setup(self):
        directory = _CDPath(_CD_DIR).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)

        self._cd_high_index = 0
        self._cd_low_substep = 0
        self._cd_global_low = 0
        self._cd_prev_action = None
        self._cd_last_action = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._cd_last_command = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._cd_dt = float(
            getattr(self.base_env, "step_dt", 0.02)
        )

        self._cd_joint_names = [
            str(name) for name in self.robot.joint_names
        ]
        body_names = [
            str(name) for name in self.robot.body_names
        ]

        self._cd_foot_bodies = {}
        for leg in ("FL", "FR", "RL", "RR"):
            matches = [
                index
                for index, name in enumerate(body_names)
                if name.upper().startswith(leg)
                and "FOOT" in name.upper()
            ]
            if matches:
                self._cd_foot_bodies[leg] = matches[0]

        self._cd_sensor = None
        sensor_body_names = []
        try:
            sensors = list(
                self.base_env.scene.sensors.items()
            )
        except Exception:
            sensors = []

        for sensor_name, sensor in sensors:
            if hasattr(
                getattr(sensor, "data", None),
                "net_forces_w",
            ):
                self._cd_sensor = sensor
                sensor_body_names = [
                    str(name)
                    for name in getattr(sensor, "body_names", [])
                ]
                print(
                    f"[CD_DIAG] contact_sensor={sensor_name}",
                    flush=True,
                )
                break

        self._cd_contact_indices = {}
        for leg, body_index in self._cd_foot_bodies.items():
            body_name = body_names[body_index]
            if body_name in sensor_body_names:
                self._cd_contact_indices[leg] = (
                    sensor_body_names.index(body_name)
                )
            elif not sensor_body_names:
                self._cd_contact_indices[leg] = body_index

        low_fields = [
            "label", "mode", "high_step", "low_substep",
            "global_low_step", "sim_time_s",
            "command_vx", "command_vy", "command_wz",
            "high_action_0", "high_action_1", "high_action_2",
            "low_action_delta_norm", "base_done",
            "base_roll", "base_pitch", "base_yaw",
            "base_vx_body", "base_vy_body", "base_wz_body",
            "ball_x", "ball_y", "ball_vx_world", "ball_vy_world",
            "robot_ball_distance", "ball_goal_distance",
            "ball_goal_alignment", "ball_velocity_toward_goal",
            "joint_velocity_rms", "joint_acceleration_rms",
            "torque_rms", "front_rear_calf_abs_diff",
            "front_rear_thigh_abs_diff",
            "mean_contact_foot_slip", "max_contact_foot_slip",
            "front_rear_contact_force_imbalance",
        ]
        for leg in ("FL", "FR", "RL", "RR"):
            low_fields.extend(
                (
                    f"{leg}_contact_force",
                    f"{leg}_contact_slip",
                )
            )

        high_fields = [
            "label", "mode", "high_step", "reward", "done",
            "command_vx", "command_vy", "command_wz",
            "high_action_0", "high_action_1", "high_action_2",
            "robot_ball_distance", "ball_goal_distance",
            "ball_goal_alignment", "ball_velocity_toward_goal",
        ]

        self._cd_low_file = (
            directory / "low_level.csv"
        ).open("w", encoding="utf-8", newline="")
        self._cd_high_file = (
            directory / "high_level.csv"
        ).open("w", encoding="utf-8", newline="")
        self._cd_low_writer = _cd_csv.DictWriter(
            self._cd_low_file, fieldnames=low_fields
        )
        self._cd_high_writer = _cd_csv.DictWriter(
            self._cd_high_file, fieldnames=high_fields
        )
        self._cd_low_writer.writeheader()
        self._cd_high_writer.writeheader()

        (directory / "metadata.json").write_text(
            _cd_json.dumps(
                {
                    "label": _CD_LABEL,
                    "mode": _CD_MODE,
                    "fixed_command": _CD_FIXED,
                    "step_dt": self._cd_dt,
                    "low_level_steps_per_action": (
                        self.low_level_steps_per_action
                    ),
                    "initial_robot_local_xy": [-1.0, 0.0],
                    "initial_ball_local_xy": [-0.25, 0.0],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        original_low_step = self.low_env.step

        def wrapped_low_step(low_actions):
            result = original_low_step(low_actions)
            robot_data = self.robot.data
            ball_data = self.ball.data
            state = self._state()

            action = _cd_vec(_cd_action(low_actions)[0])
            action_delta = (
                0.0
                if self._cd_prev_action is None
                else float(
                    _cd_np.linalg.norm(
                        action - self._cd_prev_action
                    )
                )
            )
            self._cd_prev_action = action.copy()

            command = _cd_vec(self.command_tensor[0, :3])
            high_action = _cd_vec(self._cd_last_action[0])

            roll, pitch, yaw = _cd_rpy(
                robot_data.root_quat_w[0]
            )

            linear_source = _cd_first_tensor(
                robot_data,
                (
                    "root_lin_vel_b",
                    "root_com_lin_vel_b",
                    "root_link_lin_vel_b",
                ),
            )
            angular_source = _cd_first_tensor(
                robot_data,
                (
                    "root_ang_vel_b",
                    "root_com_ang_vel_b",
                    "root_link_ang_vel_b",
                ),
            )
            linear = _cd_vec(linear_source[0])
            angular = _cd_vec(angular_source[0])

            ball_position = _cd_vec(ball_data.root_pos_w[0])
            ball_velocity = _cd_vec(
                ball_data.root_lin_vel_w[0]
            )

            joint_position = _cd_vec(
                robot_data.joint_pos[0]
            )
            joint_velocity = _cd_vec(
                robot_data.joint_vel[0]
            )

            joint_acc_source = getattr(
                robot_data, "joint_acc", None
            )
            joint_acc = (
                _cd_vec(joint_acc_source[0])
                if joint_acc_source is not None
                else _cd_np.full(
                    joint_position.shape, _cd_np.nan
                )
            )

            torque_source = getattr(
                robot_data, "applied_torque", None
            )
            if torque_source is None:
                torque_source = getattr(
                    robot_data, "computed_torque", None
                )
            torque = (
                _cd_vec(torque_source[0])
                if torque_source is not None
                else _cd_np.full(
                    joint_position.shape, _cd_np.nan
                )
            )

            front_calf = _cd_group_mean(
                joint_position,
                self._cd_joint_names,
                "CALF",
                ("FL", "FR"),
            )
            rear_calf = _cd_group_mean(
                joint_position,
                self._cd_joint_names,
                "CALF",
                ("RL", "RR"),
            )
            front_thigh = _cd_group_mean(
                joint_position,
                self._cd_joint_names,
                "THIGH",
                ("FL", "FR"),
            )
            rear_thigh = _cd_group_mean(
                joint_position,
                self._cd_joint_names,
                "THIGH",
                ("RL", "RR"),
            )

            body_velocity_source = _cd_first_tensor(
                robot_data,
                (
                    "body_lin_vel_w",
                    "body_link_lin_vel_w",
                    "body_com_lin_vel_w",
                ),
            )

            forces = None
            if self._cd_sensor is not None:
                try:
                    forces = _cd_vec(
                        self._cd_sensor.data.net_forces_w[0]
                    ).reshape(-1, 3)
                except Exception:
                    forces = None

            slips = []
            front_force = 0.0
            rear_force = 0.0
            foot_values = {}

            for leg in ("FL", "FR", "RL", "RR"):
                body_index = self._cd_foot_bodies.get(leg)
                foot_velocity = _cd_np.full(
                    3, _cd_np.nan
                )
                if (
                    body_index is not None
                    and body_velocity_source is not None
                ):
                    foot_velocity = _cd_vec(
                        body_velocity_source[0, body_index]
                    )

                speed_xy = float(
                    _cd_np.linalg.norm(
                        foot_velocity[:2]
                    )
                )

                force = _cd_np.full(3, _cd_np.nan)
                contact_index = (
                    self._cd_contact_indices.get(leg)
                )
                if (
                    forces is not None
                    and contact_index is not None
                    and contact_index < len(forces)
                ):
                    force = forces[contact_index]

                force_norm = float(
                    _cd_np.linalg.norm(force)
                )
                slip = (
                    speed_xy
                    if _cd_np.isfinite(force_norm)
                    and force_norm > 5.0
                    else 0.0
                )
                slips.append(slip)

                if _cd_np.isfinite(force_norm):
                    if leg in ("FL", "FR"):
                        front_force += force_norm
                    else:
                        rear_force += force_norm

                foot_values[f"{leg}_contact_force"] = (
                    force_norm
                )
                foot_values[f"{leg}_contact_slip"] = slip

            dones = (
                torch.logical_or(result[2], result[3])
                if len(result) == 5
                else result[2]
            )

            row = {
                "label": _CD_LABEL,
                "mode": _CD_MODE,
                "high_step": self._cd_high_index,
                "low_substep": self._cd_low_substep,
                "global_low_step": self._cd_global_low,
                "sim_time_s": (
                    self._cd_global_low * self._cd_dt
                ),
                "command_vx": _cd_at(command, 0),
                "command_vy": _cd_at(command, 1),
                "command_wz": _cd_at(command, 2),
                "high_action_0": _cd_at(high_action, 0),
                "high_action_1": _cd_at(high_action, 1),
                "high_action_2": _cd_at(high_action, 2),
                "low_action_delta_norm": action_delta,
                "base_done": int(
                    bool(
                        torch.as_tensor(dones)
                        .reshape(-1)[0].item()
                    )
                ),
                "base_roll": roll,
                "base_pitch": pitch,
                "base_yaw": yaw,
                "base_vx_body": _cd_at(linear, 0),
                "base_vy_body": _cd_at(linear, 1),
                "base_wz_body": _cd_at(angular, 2),
                "ball_x": _cd_at(ball_position, 0),
                "ball_y": _cd_at(ball_position, 1),
                "ball_vx_world": _cd_at(ball_velocity, 0),
                "ball_vy_world": _cd_at(ball_velocity, 1),
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
                    state[
                        "ball_velocity_toward_goal"
                    ][0].item()
                ),
                "joint_velocity_rms": _cd_rms(
                    joint_velocity
                ),
                "joint_acceleration_rms": _cd_rms(
                    joint_acc
                ),
                "torque_rms": _cd_rms(torque),
                "front_rear_calf_abs_diff": (
                    abs(front_calf - rear_calf)
                    if _cd_np.isfinite(front_calf)
                    and _cd_np.isfinite(rear_calf)
                    else _cd_np.nan
                ),
                "front_rear_thigh_abs_diff": (
                    abs(front_thigh - rear_thigh)
                    if _cd_np.isfinite(front_thigh)
                    and _cd_np.isfinite(rear_thigh)
                    else _cd_np.nan
                ),
                "mean_contact_foot_slip": float(
                    _cd_np.mean(slips)
                ),
                "max_contact_foot_slip": float(
                    _cd_np.max(slips)
                ),
                "front_rear_contact_force_imbalance": (
                    abs(front_force - rear_force)
                    / (
                        front_force
                        + rear_force
                        + 1.0e-8
                    )
                ),
                **foot_values,
            }

            self._cd_low_writer.writerow(row)
            self._cd_low_substep += 1
            self._cd_global_low += 1

            if self._cd_global_low % 50 == 0:
                self._cd_low_file.flush()

            return result

        self.low_env.step = wrapped_low_step
        _cd_atexit.register(_cd_close, self)

        print(
            "[CD_DIAG] "
            f"label={_CD_LABEL} mode={_CD_MODE} "
            f"directory={directory}",
            flush=True,
        )

    def _cd_init(self, *args, **kwargs):
        _cd_orig_init(self, *args, **kwargs)
        _cd_setup(self)
        _cd_place(self)

    def _cd_reset(self, *args, **kwargs):
        result = _cd_orig_reset(self, *args, **kwargs)
        _cd_place(self)

        try:
            return self.get_observations()
        except Exception:
            return result

    def _cd_convert(self, action):
        self._cd_last_action = action.detach().clone()

        if _CD_MODE == "fixed_ball":
            command = torch.tensor(
                _CD_FIXED,
                dtype=torch.float32,
                device=self.device,
            ).repeat(self.num_envs, 1)
        else:
            command = _cd_orig_convert(self, action)

        self._cd_last_command = command.detach().clone()
        return command

    def _cd_step(self, action):
        self._cd_low_substep = 0
        result = _cd_orig_step(self, action)

        observations, rewards, dones, extras = result
        state = self._state()
        high_action = _cd_vec(self._cd_last_action[0])
        command = _cd_vec(self._cd_last_command[0])

        self._cd_high_writer.writerow(
            {
                "label": _CD_LABEL,
                "mode": _CD_MODE,
                "high_step": self._cd_high_index,
                "reward": float(
                    rewards.reshape(-1)[0].item()
                ),
                "done": int(
                    bool(
                        dones.reshape(-1)[0].item()
                    )
                ),
                "command_vx": _cd_at(command, 0),
                "command_vy": _cd_at(command, 1),
                "command_wz": _cd_at(command, 2),
                "high_action_0": _cd_at(high_action, 0),
                "high_action_1": _cd_at(high_action, 1),
                "high_action_2": _cd_at(high_action, 2),
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
                    state[
                        "ball_velocity_toward_goal"
                    ][0].item()
                ),
            }
        )

        self._cd_high_file.flush()
        self._cd_high_index += 1
        return result

    HighLevelSoccerVecEnv.__init__ = _cd_init
    HighLevelSoccerVecEnv.reset = _cd_reset
    HighLevelSoccerVecEnv._action_to_command = _cd_convert
    HighLevelSoccerVecEnv.step = _cd_step
'''

tree = ast.parse(text)

entry_candidates = []

for node in tree.body:
    if not isinstance(node, ast.Try):
        continue

    direct_main_calls = [
        statement
        for statement in node.body
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id == "main"
        )
    ]

    if direct_main_calls:
        entry_candidates.append(node)

if len(entry_candidates) != 1:
    tail = "\n".join(text.splitlines()[-80:])

    raise RuntimeError(
        "Could not uniquely locate the top-level try/main() "
        f"entry block; found {len(entry_candidates)}. "
        f"Source tail:\n{tail}"
    )

entry_node = entry_candidates[0]

line_offsets = []
offset = 0

for line in text.splitlines(keepends=True):
    line_offsets.append(offset)
    offset += len(line)

insert_at = line_offsets[entry_node.lineno - 1]

text = (
    text[:insert_at]
    + patch
    + "\n\n"
    + text[insert_at:]
)

ast.parse(text)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(
    text,
    encoding="utf-8",
    newline="\n",
)

print(f"CD_PLAYER_CREATED={output}")
print("PHASEA_CD_DIRECT_PATCH_OK")
