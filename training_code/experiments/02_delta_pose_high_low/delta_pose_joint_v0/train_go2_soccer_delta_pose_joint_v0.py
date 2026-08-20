# DELTA_POSE_JOINT_STANDALONE_V0
"""Joint high/low PPO with a controller-resolved delta-pose interface."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict

from isaaclab.app import AppLauncher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase-A high-level Go2 soccer PPO training."
    )

    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-Go2-Soccer-Milestone2-v0",
    )
    parser.add_argument(
        "--low_level_checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
    )

    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--max_iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=43)

    parser.add_argument(
        "--low_level_steps_per_action",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--high_level_episode_steps",
        type=int,
        default=300,
    )

    parser.add_argument("--goal_line_x", type=float, default=4.02)
    parser.add_argument("--goal_center_x", type=float, default=4.60)
    parser.add_argument("--goal_half_width", type=float, default=0.78)

    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = build_parser()
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

print("[PHASEA_BOOT] app_launcher_returned", flush=True)

import isaac_go2_low_level_rl.tasks  # noqa: E402,F401

# DELTA_POSE_SOCCER_REGISTRATION_IMPORT_V2
import isaac_go2_low_level_rl.tasks.manager_based.soccer  # noqa: E402,F401

if args_cli.task not in gym.registry:
    matching_tasks = sorted(
        task_id
        for task_id in gym.registry.keys()
        if "Go2-Soccer" in task_id
    )

    raise RuntimeError(
        "Soccer task import completed, but the requested "
        f"task is still absent: {args_cli.task}. "
        f"Registered Go2 soccer tasks={matching_tasks}"
    )

print(
    "[DELTA_TASK_REGISTRATION_OK] "
    f"task={args_cli.task}",
    flush=True,
)

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import (  # noqa: E402
    load_cfg_from_registry,
    parse_env_cfg,
)
from rsl_rl.runners import OnPolicyRunner  # noqa: E402


def instantiate_cfg(value: Any) -> Any:
    return value() if isinstance(value, type) else value


def extract_observation(value: Any) -> torch.Tensor:
    """Recursively extract the policy tensor from RSL-RL observations."""

    if isinstance(value, tuple):
        if not value:
            raise RuntimeError("Empty observation tuple.")
        value = value[0]

    while not torch.is_tensor(value):
        if hasattr(value, "keys") and hasattr(value, "__getitem__"):
            try:
                keys = list(value.keys())
            except TypeError:
                keys = list(value.keys(include_nested=False))

            if "policy" in keys:
                value = value["policy"]
            elif len(keys) == 1:
                value = value[keys[0]]
            else:
                raise RuntimeError(
                    "Unsupported observation mapping. "
                    f"keys={keys}"
                )
            continue

        raise RuntimeError(
            f"Expected tensor observation, received {type(value)}"
        )

    return value.detach().clone()




def normalize_step_result(result: Any) -> tuple[Any, Any, Any, Any]:
    if not isinstance(result, tuple):
        raise RuntimeError(
            f"Unexpected env.step result type: {type(result)}"
        )

    if len(result) == 4:
        return result

    if len(result) == 5:
        obs, reward, terminated, truncated, infos = result
        dones = torch.logical_or(terminated, truncated)
        return obs, reward, dones, infos

    raise RuntimeError(
        f"Unexpected env.step tuple length: {len(result)}"
    )


def writable_command_tensor(base_env: Any) -> torch.Tensor:
    manager = base_env.command_manager

    if hasattr(manager, "get_term"):
        term = manager.get_term("base_velocity")
    elif hasattr(manager, "_terms"):
        term = manager._terms["base_velocity"]
    else:
        raise RuntimeError("Cannot access the command manager term.")

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
                f"[PHASEA_BOOT] command_tensor={name} "
                f"shape={tuple(value.shape)}",
                flush=True,
            )
            return value

    candidates = sorted(
        name
        for name in dir(term)
        if "command" in name.lower()
    )

    raise RuntimeError(
        "No writable command tensor was found. "
        f"Candidates={candidates}"
    )


def yaw_from_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert wxyz quaternion tensors to yaw."""

    w = quaternion[:, 0]
    x = quaternion[:, 1]
    y = quaternion[:, 2]
    z = quaternion[:, 3]

    return torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y.square() + z.square()),
    )


def world_to_body(
    x_world: torch.Tensor,
    y_world: torch.Tensor,
    yaw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)

    x_body = cosine * x_world + sine * y_world
    y_body = -sine * x_world + cosine * y_world

    return x_body, y_body



# DELTA_POSE_JOINT_V0
def normalize_done_tensor(
    value: Any,
    *,
    num_envs: int,
    device: torch.device | str,
) -> torch.Tensor:
    tensor = torch.as_tensor(
        value,
        device=device,
        dtype=torch.bool,
    )

    if tensor.ndim == 0:
        return tensor.expand(num_envs).clone()

    if tensor.shape[0] != num_envs:
        raise RuntimeError(
            "Termination batch mismatch. "
            f"expected={num_envs} "
            f"received={tuple(tensor.shape)}"
        )

    if tensor.ndim == 1:
        return tensor

    return tensor.reshape(num_envs, -1).any(dim=1)


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(
        torch.sin(angle),
        torch.cos(angle),
    )


class RslRlHighLevelEnvAdapter:
    """Expose the high-level environment using RSL-RL 3.x semantics."""

    def __init__(self, env: "HighLevelSoccerVecEnv") -> None:
        self.env = env
        self.num_envs = env.num_envs
        self.num_actions = env.num_actions
        self.device = env.device
        self.max_episode_length = env.max_episode_length
        self.cfg = getattr(env, "cfg", {})

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.env.episode_length_buf = value

    def _pack(self, value: Any) -> TensorDict:
        observation = extract_observation(value)

        expected = (
            self.num_envs,
            self.env.num_obs,
        )

        if observation.shape != expected:
            raise RuntimeError(
                "High-level observation shape mismatch. "
                f"expected={expected} "
                f"received={tuple(observation.shape)}"
            )

        if not torch.isfinite(observation).all():
            raise RuntimeError(
                "High-level observations contain non-finite values."
            )

        return TensorDict(
            {"policy": observation},
            batch_size=[self.num_envs],
            device=observation.device,
        )

    @torch.inference_mode(False)
    def get_observations(self) -> TensorDict:
        return self._pack(
            self.env.get_observations()
        )

    @torch.inference_mode(False)
    def reset(self) -> TensorDict:
        return self._pack(
            self.env.reset()
        )

    @torch.inference_mode(False)
    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[
        TensorDict,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        safe_actions = actions.detach().clone()

        observations, rewards, dones, extras = (
            self.env.step(safe_actions)
        )

        return (
            self._pack(observations),
            rewards.detach().clone(),
            dones.detach().clone(),
            extras,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


class HighLevelSoccerVecEnv:
    """High-level delta-pose soccer environment over a trainable low-level policy."""

    def __init__(
        self,
        *,
        low_env: RslRlVecEnvWrapper,
        low_policy: Any,
        low_level_steps_per_action: int,
        episode_steps: int,
        goal_line_x: float,
        goal_center_x: float,
        goal_half_width: float,
        seed: int,
    ) -> None:
        self.low_env = low_env
        self.base_env = low_env.unwrapped
        self.low_policy = low_policy

        # When set, the low-level PPO policy is trained jointly.
        self.joint_low_runner = None
        self.joint_low_obs = None
        self.joint_low_reward_sum = 0.0
        self.joint_low_reward_count = 0


        self.device = low_env.device
        self.num_envs = low_env.num_envs
        self.num_actions = 3
        self.num_privileged_obs = None

        self.low_level_steps_per_action = (
            low_level_steps_per_action
        )
        self.max_episode_length = episode_steps

        self.goal_line_x = goal_line_x
        self.goal_center_x = goal_center_x
        self.goal_half_width = goal_half_width

        self.robot = self.base_env.scene["robot"]
        self.ball = self.base_env.scene["ball"]

        self.env_origins = self.base_env.scene.env_origins
        self.command_tensor = writable_command_tensor(
            self.base_env
        )

        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

        self.episode_length_buf = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        self.reset_buf = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self.rew_buf = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )

        self.previous_action = torch.zeros(
            self.num_envs,
            self.num_actions,
            device=self.device,
        )

        self.target_x_world = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.target_y_world = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.target_yaw_world = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.last_pose_error_xy = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.last_pose_error_yaw = torch.zeros(
            self.num_envs,
            device=self.device,
        )

        self.previous_ball_goal_distance = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.previous_robot_ball_distance = torch.zeros(
            self.num_envs,
            device=self.device,
        )

        self.episode_reward_sum = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.episode_goal_progress_sum = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.episode_goal_count = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.episode_own_goal_count = torch.zeros(
            self.num_envs,
            device=self.device,
        )

        self.total_resets = 0
        self.vector_steps = 0

        self.cfg = SimpleNamespace(is_finite_horizon=False)

        self._low_obs = extract_observation(
            self.low_env.get_observations()
        )

        all_ids = torch.arange(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )

        self._reset_idx(all_ids)

        self.obs_buf = self._build_observation()
        self.num_obs = int(self.obs_buf.shape[1])

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_obs,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_actions,),
            dtype=np.float32,
        )

        print(
            f"[PHASEA_BOOT] high_level_num_obs={self.num_obs}",
            flush=True,
        )
        print(
            f"[PHASEA_BOOT] high_level_num_actions={self.num_actions}",
            flush=True,
        )

    def _rand(
        self,
        size: tuple[int, ...],
    ) -> torch.Tensor:
        return torch.rand(
            size,
            generator=self.generator,
            device=self.device,
        )

    def _state(self) -> dict[str, torch.Tensor]:
        robot_position = self.robot.data.root_pos_w
        robot_quaternion = self.robot.data.root_quat_w
        robot_yaw = yaw_from_quaternion(robot_quaternion)

        ball_position = self.ball.data.root_pos_w
        ball_velocity_world = self.ball.data.root_lin_vel_w

        origin_x = self.env_origins[:, 0]
        origin_y = self.env_origins[:, 1]

        goal_x = origin_x + self.goal_center_x
        goal_y = origin_y

        robot_to_ball_x_world = (
            ball_position[:, 0] - robot_position[:, 0]
        )
        robot_to_ball_y_world = (
            ball_position[:, 1] - robot_position[:, 1]
        )

        robot_to_goal_x_world = (
            goal_x - robot_position[:, 0]
        )
        robot_to_goal_y_world = (
            goal_y - robot_position[:, 1]
        )

        ball_to_goal_x_world = goal_x - ball_position[:, 0]
        ball_to_goal_y_world = goal_y - ball_position[:, 1]

        robot_ball_x_body, robot_ball_y_body = world_to_body(
            robot_to_ball_x_world,
            robot_to_ball_y_world,
            robot_yaw,
        )

        robot_goal_x_body, robot_goal_y_body = world_to_body(
            robot_to_goal_x_world,
            robot_to_goal_y_world,
            robot_yaw,
        )

        ball_velocity_x_body, ball_velocity_y_body = (
            world_to_body(
                ball_velocity_world[:, 0],
                ball_velocity_world[:, 1],
                robot_yaw,
            )
        )

        robot_ball_distance = torch.sqrt(
            robot_to_ball_x_world.square()
            + robot_to_ball_y_world.square()
            + 1.0e-8
        )

        ball_goal_distance = torch.sqrt(
            ball_to_goal_x_world.square()
            + ball_to_goal_y_world.square()
            + 1.0e-8
        )

        goal_direction_x = (
            ball_to_goal_x_world / ball_goal_distance
        )
        goal_direction_y = (
            ball_to_goal_y_world / ball_goal_distance
        )

        staging_x = (
            ball_position[:, 0] - 0.68 * goal_direction_x
        )
        staging_y = (
            ball_position[:, 1] - 0.68 * goal_direction_y
        )

        staging_distance = torch.sqrt(
            (robot_position[:, 0] - staging_x).square()
            + (robot_position[:, 1] - staging_y).square()
            + 1.0e-8
        )

        ball_velocity_toward_goal = (
            ball_velocity_world[:, 0] * goal_direction_x
            + ball_velocity_world[:, 1] * goal_direction_y
        )

        robot_velocity_body = self.robot.data.root_lin_vel_b
        robot_angular_velocity_body = (
            self.robot.data.root_ang_vel_b
        )

        ball_goal_alignment = (
            robot_to_ball_x_world * ball_to_goal_x_world
            + robot_to_ball_y_world * ball_to_goal_y_world
        ) / (
            robot_ball_distance * ball_goal_distance + 1.0e-6
        )

        return {
            "robot_position": robot_position,
            "robot_yaw": robot_yaw,
            "ball_position": ball_position,
            "ball_velocity_world": ball_velocity_world,
            "robot_velocity_body": robot_velocity_body,
            "robot_angular_velocity_body": (
                robot_angular_velocity_body
            ),
            "robot_ball_x_body": robot_ball_x_body,
            "robot_ball_y_body": robot_ball_y_body,
            "robot_goal_x_body": robot_goal_x_body,
            "robot_goal_y_body": robot_goal_y_body,
            "ball_velocity_x_body": ball_velocity_x_body,
            "ball_velocity_y_body": ball_velocity_y_body,
            "robot_ball_distance": robot_ball_distance,
            "ball_goal_distance": ball_goal_distance,
            "goal_direction_x": goal_direction_x,
            "goal_direction_y": goal_direction_y,
            "staging_distance": staging_distance,
            "ball_velocity_toward_goal": (
                ball_velocity_toward_goal
            ),
            "ball_goal_alignment": ball_goal_alignment,
            "origin_x": origin_x,
            "origin_y": origin_y,
        }

    def _build_observation(self) -> torch.Tensor:
        state = self._state()

        difficulty = min(
            1.0,
            self.total_resets
            / max(1.0, float(self.num_envs * 300)),
        )

        difficulty_tensor = torch.full(
            (self.num_envs, 1),
            difficulty,
            device=self.device,
        )

        observation = torch.cat(
            (
                state["robot_velocity_body"][:, :2] / 1.5,
                state["robot_angular_velocity_body"][:, 2:3]
                / 2.0,
                state["robot_ball_x_body"].unsqueeze(1) / 5.0,
                state["robot_ball_y_body"].unsqueeze(1) / 3.0,
                state["robot_goal_x_body"].unsqueeze(1) / 7.0,
                state["robot_goal_y_body"].unsqueeze(1) / 4.0,
                state["ball_velocity_x_body"].unsqueeze(1)
                / 3.0,
                state["ball_velocity_y_body"].unsqueeze(1)
                / 3.0,
                state["robot_ball_distance"].unsqueeze(1)
                / 6.0,
                state["ball_goal_distance"].unsqueeze(1)
                / 7.0,
                state["staging_distance"].unsqueeze(1) / 5.0,
                state["ball_goal_alignment"].unsqueeze(1),
                self.previous_action,
                difficulty_tensor,
            ),
            dim=1,
        )

        if not torch.isfinite(observation).all():
            raise RuntimeError(
                "Non-finite values detected in high-level observations."
            )

        return observation

    def _action_to_delta_pose(
        self,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Map high-level actions to short relative SE(2) subgoals.

        The scales are chosen so that the initial proportional-controller
        command matches the previous V3 velocity mapping. This provides a
        function-preserving bridge at the start of the experiment.
        """

        action = torch.clamp(action, -1.0, 1.0)
        delta = torch.empty_like(action)

        delta[:, 0] = torch.where(
            action[:, 0] >= 0.0,
            0.20 * action[:, 0],
            0.10 * action[:, 0],
        )
        delta[:, 1] = 0.12 * action[:, 1]
        delta[:, 2] = 0.20 * action[:, 2]

        return delta

    def _set_delta_target(
        self,
        delta_pose: torch.Tensor,
    ) -> None:
        state = self._state()
        yaw = state["robot_yaw"]

        cosine = torch.cos(yaw)
        sine = torch.sin(yaw)

        delta_x_world = (
            cosine * delta_pose[:, 0]
            - sine * delta_pose[:, 1]
        )
        delta_y_world = (
            sine * delta_pose[:, 0]
            + cosine * delta_pose[:, 1]
        )

        self.target_x_world = (
            state["robot_position"][:, 0]
            + delta_x_world
        )
        self.target_y_world = (
            state["robot_position"][:, 1]
            + delta_y_world
        )
        self.target_yaw_world = wrap_to_pi(
            yaw + delta_pose[:, 2]
        )

    def _pose_error(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self._state()

        error_x_world = (
            self.target_x_world
            - state["robot_position"][:, 0]
        )
        error_y_world = (
            self.target_y_world
            - state["robot_position"][:, 1]
        )

        error_x_body, error_y_body = world_to_body(
            error_x_world,
            error_y_world,
            state["robot_yaw"],
        )

        error_yaw = wrap_to_pi(
            self.target_yaw_world
            - state["robot_yaw"]
        )

        return (
            error_x_body,
            error_y_body,
            error_yaw,
        )

    def _pose_error_to_command(
        self,
        error_x_body: torch.Tensor,
        error_y_body: torch.Tensor,
        error_yaw: torch.Tensor,
    ) -> torch.Tensor:
        """Resolve an SE(2) pose error into the legacy velocity interface."""

        command = torch.empty(
            self.num_envs,
            3,
            device=self.device,
        )

        command[:, 0] = torch.where(
            error_x_body >= 0.0,
            3.25 * error_x_body,
            2.50 * error_x_body,
        )
        command[:, 1] = (
            2.9166666667 * error_y_body
        )
        command[:, 2] = 4.50 * error_yaw

        command[:, 0] = torch.clamp(
            command[:, 0],
            -0.25,
            0.65,
        )
        command[:, 1] = torch.clamp(
            command[:, 1],
            -0.35,
            0.35,
        )
        command[:, 2] = torch.clamp(
            command[:, 2],
            -0.90,
            0.90,
        )

        return command


    def _set_command(
        self,
        command: torch.Tensor,
    ) -> None:
        self.command_tensor[:, :3] = command

    def _reset_idx(
        self,
        env_ids: torch.Tensor,
    ) -> None:
        if env_ids.numel() == 0:
            return

        if not hasattr(self.base_env, "_reset_idx"):
            raise RuntimeError(
                "ManagerBased environment has no _reset_idx method."
            )

        self.base_env._reset_idx(env_ids)

        count = int(env_ids.numel())

        difficulty = min(
            1.0,
            self.total_resets
            / max(1.0, float(self.num_envs * 300)),
        )

        ball_x_local = (
            0.55
            + (1.65 + 0.35 * difficulty)
            * self._rand((count,))
        )
        ball_y_limit = 0.65 + 0.90 * difficulty
        ball_y_local = (
            2.0 * self._rand((count,)) - 1.0
        ) * ball_y_limit

        robot_x_offset = (
            1.0
            + (0.9 + 0.7 * difficulty)
            * self._rand((count,))
        )
        robot_y_offset_limit = 0.45 + 0.85 * difficulty
        robot_y_offset = (
            2.0 * self._rand((count,)) - 1.0
        ) * robot_y_offset_limit

        robot_x_local = ball_x_local - robot_x_offset
        robot_y_local = ball_y_local + robot_y_offset

        robot_x_local = torch.clamp(
            robot_x_local,
            -3.3,
            2.5,
        )
        robot_y_local = torch.clamp(
            robot_y_local,
            -2.1,
            2.1,
        )

        heading_to_ball = torch.atan2(
            ball_y_local - robot_y_local,
            ball_x_local - robot_x_local,
        )

        yaw_noise_limit = 0.55 + 2.20 * difficulty
        robot_yaw = heading_to_ball + (
            2.0 * self._rand((count,)) - 1.0
        ) * yaw_noise_limit

        origins = self.env_origins[env_ids]

        robot_z = self.robot.data.root_pos_w[env_ids, 2]
        ball_z = self.ball.data.root_pos_w[env_ids, 2]

        robot_pose = torch.zeros(
            count,
            7,
            device=self.device,
        )
        robot_pose[:, 0] = origins[:, 0] + robot_x_local
        robot_pose[:, 1] = origins[:, 1] + robot_y_local
        robot_pose[:, 2] = robot_z
        robot_pose[:, 3] = torch.cos(robot_yaw / 2.0)
        robot_pose[:, 6] = torch.sin(robot_yaw / 2.0)

        ball_pose = torch.zeros(
            count,
            7,
            device=self.device,
        )
        ball_pose[:, 0] = origins[:, 0] + ball_x_local
        ball_pose[:, 1] = origins[:, 1] + ball_y_local
        ball_pose[:, 2] = ball_z
        ball_pose[:, 3] = 1.0

        zero_velocity = torch.zeros(
            count,
            6,
            device=self.device,
        )

        self.robot.write_root_pose_to_sim(
            robot_pose,
            env_ids=env_ids,
        )
        self.robot.write_root_velocity_to_sim(
            zero_velocity,
            env_ids=env_ids,
        )

        self.ball.write_root_pose_to_sim(
            ball_pose,
            env_ids=env_ids,
        )
        self.ball.write_root_velocity_to_sim(
            zero_velocity,
            env_ids=env_ids,
        )

        self.command_tensor[env_ids, :3] = 0.0
        self.previous_action[env_ids] = 0.0

        self.episode_length_buf[env_ids] = 0
        self.episode_reward_sum[env_ids] = 0.0
        self.episode_goal_progress_sum[env_ids] = 0.0
        self.episode_goal_count[env_ids] = 0.0
        self.episode_own_goal_count[env_ids] = 0.0

        state = self._state()

        self.target_x_world[env_ids] = (
            state["robot_position"][env_ids, 0]
        )
        self.target_y_world[env_ids] = (
            state["robot_position"][env_ids, 1]
        )
        self.target_yaw_world[env_ids] = (
            state["robot_yaw"][env_ids]
        )
        self.last_pose_error_xy[env_ids] = 0.0
        self.last_pose_error_yaw[env_ids] = 0.0

        self.previous_ball_goal_distance[env_ids] = (
            state["ball_goal_distance"][env_ids]
        )
        self.previous_robot_ball_distance[env_ids] = (
            state["robot_ball_distance"][env_ids]
        )

        self.total_resets += count

        self._low_obs = extract_observation(
            self.low_env.get_observations()
        )

    def get_observations(self) -> torch.Tensor:
        self.obs_buf = self._build_observation()
        return self.obs_buf

    def get_privileged_observations(self) -> None:
        return None

    def reset(self) -> torch.Tensor:
        env_ids = torch.arange(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        self._reset_idx(env_ids)
        self.obs_buf = self._build_observation()
        return self.obs_buf

    @torch.inference_mode(False)
    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        if not torch.is_tensor(actions):
            raise RuntimeError(
                "High-level actions must be a tensor."
            )

        if actions.ndim == 1 and self.num_envs == 1:
            actions = actions.unsqueeze(0)

        expected_action_shape = (
            self.num_envs,
            self.num_actions,
        )

        if actions.shape != expected_action_shape:
            raise RuntimeError(
                "High-level action shape mismatch. "
                f"expected={expected_action_shape} "
                f"received={tuple(actions.shape)}"
            )

        actions = torch.clamp(
            actions.to(self.device),
            -1.0,
            1.0,
        )

        delta_pose = self._action_to_delta_pose(actions)
        self._set_delta_target(delta_pose)

        initial_error = self._pose_error()
        initial_error_xy = torch.sqrt(
            initial_error[0].square()
            + initial_error[1].square()
            + 1.0e-8
        )

        base_done = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )

        command_abs_sum = torch.zeros(
            3,
            device=self.device,
        )

        for low_index in range(
            self.low_level_steps_per_action
        ):
            (
                error_x_body,
                error_y_body,
                error_yaw,
            ) = self._pose_error()

            command = self._pose_error_to_command(
                error_x_body,
                error_y_body,
                error_yaw,
            )

            if base_done.any():
                command = command.clone()
                command[base_done] = 0.0

            command_abs_sum += command.abs().mean(dim=0)
            self._set_command(command)

            low_observations = (
                self.low_env.get_observations()
            )

            if self.joint_low_runner is None:
                low_tensor = extract_observation(
                    low_observations
                )
                low_policy_input = {
                    "policy": low_tensor,
                }

                with torch.no_grad():
                    low_actions = self.low_policy(
                        low_policy_input
                    )
            else:
                low_observations = low_observations.to(
                    self.device
                )

                with torch.no_grad():
                    low_actions = (
                        self.joint_low_runner.alg.act(
                            low_observations
                        )
                    )

            low_actions = low_actions.detach().clone()

            if not torch.isfinite(low_actions).all():
                raise RuntimeError(
                    "Low-level policy produced non-finite actions."
                )

            result = self.low_env.step(low_actions)

            (
                low_obs,
                low_rewards,
                low_dones,
                low_extras,
            ) = normalize_step_result(result)

            low_done_tensor = normalize_done_tensor(
                low_dones,
                num_envs=self.num_envs,
                device=self.device,
            )

            base_done |= low_done_tensor
            self._low_obs = extract_observation(low_obs)

            if self.joint_low_runner is not None:
                low_obs = low_obs.to(self.device)
                low_rewards = low_rewards.to(self.device)

                low_done_for_ppo = (
                    low_done_tensor.clone()
                    | base_done
                )

                # The high-level wrapper resets after a soccer terminal
                # event. Mark the final nested transition as terminal so
                # the low-level value function does not bootstrap across it.
                if (
                    low_index
                    == self.low_level_steps_per_action - 1
                ):
                    preview = self._state()

                    preview_goal = (
                        preview["ball_position"][:, 0]
                        >= (
                            preview["origin_x"]
                            + self.goal_line_x
                        )
                    ) & (
                        torch.abs(
                            preview["ball_position"][:, 1]
                            - preview["origin_y"]
                        )
                        <= self.goal_half_width
                    )

                    preview_own_goal = (
                        preview["ball_position"][:, 0]
                        <= (
                            preview["origin_x"]
                            - self.goal_line_x
                        )
                    ) & (
                        torch.abs(
                            preview["ball_position"][:, 1]
                            - preview["origin_y"]
                        )
                        <= self.goal_half_width
                    )

                    preview_timeout = (
                        self.episode_length_buf + 1
                        >= self.max_episode_length
                    )

                    low_done_for_ppo |= (
                        preview_goal
                        | preview_own_goal
                        | preview_timeout
                    )

                self.joint_low_runner.alg.process_env_step(
                    low_obs,
                    low_rewards,
                    low_done_for_ppo,
                    low_extras,
                )

                self.joint_low_obs = low_obs
                self.joint_low_reward_sum += float(
                    low_rewards.mean()
                )
                self.joint_low_reward_count += 1

        state = self._state()

        goal_progress = (
            self.previous_ball_goal_distance
            - state["ball_goal_distance"]
        )

        goal_scored = (
            state["ball_position"][:, 0]
            >= state["origin_x"] + self.goal_line_x
        ) & (
            torch.abs(
                state["ball_position"][:, 1]
                - state["origin_y"]
            )
            <= self.goal_half_width
        )

        own_goal = (
            state["ball_position"][:, 0]
            <= state["origin_x"] - self.goal_line_x
        ) & (
            torch.abs(
                state["ball_position"][:, 1]
                - state["origin_y"]
            )
            <= self.goal_half_width
        )

        # Final high-level objective: pure sparse soccer reward.
        reward = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        reward = reward + goal_scored.float()
        reward = reward - own_goal.float()

        self.episode_length_buf += 1

        time_out = (
            self.episode_length_buf
            >= self.max_episode_length
        )

        dones = (
            base_done
            | goal_scored
            | own_goal
            | time_out
        )

        final_error = self._pose_error()
        final_error_xy = torch.sqrt(
            final_error[0].square()
            + final_error[1].square()
            + 1.0e-8
        )
        final_error_yaw = final_error[2].abs()

        self.last_pose_error_xy = final_error_xy
        self.last_pose_error_yaw = final_error_yaw

        pose_reached = (
            (final_error_xy < 0.08)
            & (final_error_yaw < 0.12)
        )

        self.episode_reward_sum += reward
        self.episode_goal_progress_sum += goal_progress
        self.episode_goal_count += goal_scored.float()
        self.episode_own_goal_count += own_goal.float()

        infos: dict[str, Any] = {
            "time_outs": time_out.float(),
        }

        done_ids = torch.nonzero(
            dones,
            as_tuple=False,
        ).squeeze(-1)

        if done_ids.numel() > 0:
            infos["episode"] = {
                "reward": self.episode_reward_sum[
                    done_ids
                ].mean(),
                "goal_rate": self.episode_goal_count[
                    done_ids
                ].mean(),
                "own_goal_rate": self.episode_own_goal_count[
                    done_ids
                ].mean(),
                "ball_goal_progress": (
                    self.episode_goal_progress_sum[
                        done_ids
                    ].mean()
                ),
                "episode_length": self.episode_length_buf[
                    done_ids
                ].float().mean(),
            }

        self.previous_ball_goal_distance = (
            state["ball_goal_distance"].clone()
        )
        self.previous_robot_ball_distance = (
            state["robot_ball_distance"].clone()
        )
        self.previous_action = actions.clone()

        self.rew_buf = reward
        self.reset_buf = dones

        if done_ids.numel() > 0:
            self._reset_idx(done_ids)

        self.obs_buf = self._build_observation()

        if self.joint_low_runner is not None:
            self.joint_low_obs = (
                self.low_env.get_observations()
            ).to(self.device)

        self.vector_steps += 1

        if (
            self.vector_steps == 1
            or self.vector_steps % 100 == 0
        ):
            mean_command = (
                command_abs_sum
                / float(
                    self.low_level_steps_per_action
                )
            )

            low_reward_mean = (
                self.joint_low_reward_sum
                / max(
                    self.joint_low_reward_count,
                    1,
                )
                if self.joint_low_runner is not None
                else float("nan")
            )

            print(
                "[DELTA_POSE_PROGRESS] "
                f"vector_step={self.vector_steps} "
                f"initial_pose_error="
                f"{float(initial_error_xy.mean()):.5f} "
                f"final_pose_error="
                f"{float(final_error_xy.mean()):.5f} "
                f"final_yaw_error="
                f"{float(final_error_yaw.mean()):.5f} "
                f"pose_reached_fraction="
                f"{float(pose_reached.float().mean()):.5f} "
                f"mean_abs_command_vx="
                f"{float(mean_command[0]):.5f} "
                f"mean_abs_command_vy="
                f"{float(mean_command[1]):.5f} "
                f"mean_abs_command_wz="
                f"{float(mean_command[2]):.5f} "
                f"goal_fraction="
                f"{float(goal_scored.float().mean()):.5f} "
                f"own_goal_fraction="
                f"{float(own_goal.float().mean()):.5f} "
                f"low_reward_mean={low_reward_mean:.5f}",
                flush=True,
            )

        return (
            self.obs_buf,
            self.rew_buf,
            self.reset_buf,
            infos,
        )


    def close(self) -> None:
        self.low_env.close()


def create_high_runner_cfg(
    *,
    base_cfg: Any,
    run_name: str,
) -> dict[str, Any]:
    cfg = base_cfg.to_dict()

    cfg["num_steps_per_env"] = int(
        os.environ.get(
            "DELTA_HIGH_ROLLOUT_STEPS",
            "24",
        )
    )
    cfg["save_interval"] = 10_000
    cfg["experiment_name"] = (
        "go2_soccer_delta_pose_joint_v0"
    )
    cfg["run_name"] = run_name

    cfg["obs_groups"] = {
        "policy": ["policy"],
        "critic": ["policy"],
    }

    cfg["policy"]["init_noise_std"] = float(
        os.environ.get(
            "DELTA_HIGH_INIT_NOISE_STD",
            "0.05",
        )
    )
    cfg["policy"]["actor_hidden_dims"] = [256, 256, 128]
    cfg["policy"]["critic_hidden_dims"] = [256, 256, 128]
    cfg["policy"]["actor_obs_normalization"] = False
    cfg["policy"]["critic_obs_normalization"] = False

    cfg["algorithm"]["entropy_coef"] = float(
        os.environ.get(
            "DELTA_HIGH_ENTROPY_COEF",
            "0.0",
        )
    )
    cfg["algorithm"]["learning_rate"] = float(
        os.environ.get(
            "DELTA_HIGH_LEARNING_RATE",
            "0.00002",
        )
    )
    cfg["algorithm"]["clip_param"] = float(
        os.environ.get(
            "DELTA_HIGH_CLIP_PARAM",
            "0.08",
        )
    )
    cfg["algorithm"]["num_learning_epochs"] = int(
        os.environ.get(
            "DELTA_HIGH_EPOCHS",
            "3",
        )
    )
    cfg["algorithm"]["num_mini_batches"] = int(
        os.environ.get(
            "DELTA_HIGH_MINI_BATCHES",
            "4",
        )
    )
    cfg["algorithm"]["gamma"] = float(
        os.environ.get(
            "DELTA_HIGH_GAMMA",
            "0.995",
        )
    )
    cfg["algorithm"]["lam"] = 0.95
    cfg["algorithm"]["desired_kl"] = float(
        os.environ.get(
            "DELTA_HIGH_DESIRED_KL",
            "0.002",
        )
    )
    cfg["algorithm"]["schedule"] = "fixed"

    return cfg


def create_low_runner_cfg(
    *,
    base_cfg: Any,
    high_rollout_steps: int,
    low_steps_per_high_action: int,
) -> dict[str, Any]:
    cfg = base_cfg.to_dict()

    cfg["num_steps_per_env"] = (
        high_rollout_steps
        * low_steps_per_high_action
    )
    cfg["save_interval"] = 10_000

    cfg["policy"]["init_noise_std"] = float(
        os.environ.get(
            "DELTA_LOW_INIT_NOISE_STD",
            "0.05",
        )
    )

    cfg["algorithm"]["entropy_coef"] = float(
        os.environ.get(
            "DELTA_LOW_ENTROPY_COEF",
            "0.0",
        )
    )
    cfg["algorithm"]["learning_rate"] = float(
        os.environ.get(
            "DELTA_LOW_LEARNING_RATE",
            "0.000001",
        )
    )
    cfg["algorithm"]["num_learning_epochs"] = int(
        os.environ.get(
            "DELTA_LOW_EPOCHS",
            "2",
        )
    )
    cfg["algorithm"]["num_mini_batches"] = int(
        os.environ.get(
            "DELTA_LOW_MINI_BATCHES",
            "4",
        )
    )
    cfg["algorithm"]["desired_kl"] = float(
        os.environ.get(
            "DELTA_LOW_DESIRED_KL",
            "0.005",
        )
    )
    cfg["algorithm"]["schedule"] = "fixed"

    return cfg


def policy_module(runner: Any) -> Any:
    module = getattr(runner.alg, "policy", None)

    if module is None:
        module = getattr(
            runner.alg,
            "actor_critic",
            None,
        )

    if module is None:
        raise RuntimeError(
            "Could not locate policy module."
        )

    return module


def set_action_std(
    module: Any,
    value: float,
) -> bool:
    if hasattr(module, "set_action_std"):
        try:
            module.set_action_std(value)
            return True
        except Exception:
            pass

    for name in (
        "std",
        "_std",
        "log_std",
        "_log_std",
    ):
        tensor = getattr(module, name, None)

        if not torch.is_tensor(tensor):
            continue

        fill_value = (
            math.log(value)
            if "log_std" in name
            else value
        )

        with torch.no_grad():
            tensor.fill_(fill_value)

        return True

    return False


# DELTA_POSE_BASE_MODULE_LOAD_STATE_DICT_V1
def warm_start_policy_weights(
    *,
    runner: Any,
    checkpoint: Path,
    requested_std: float,
    label: str,
) -> None:
    try:
        payload = torch.load(
            checkpoint,
            map_location=args_cli.device,
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(
            checkpoint,
            map_location=args_cli.device,
        )

    source_state = None

    if isinstance(payload, dict):
        for key in (
            "model_state_dict",
            "policy_state_dict",
            "actor_critic_state_dict",
            "state_dict",
        ):
            candidate = payload.get(key)

            if isinstance(candidate, dict):
                source_state = candidate
                break

        if source_state is None and all(
            torch.is_tensor(value)
            for value in payload.values()
        ):
            source_state = payload

    if not isinstance(source_state, dict):
        raise RuntimeError(
            f"Could not locate state dictionary in {checkpoint}"
        )

    module = policy_module(runner)
    target_state = module.state_dict()

    def is_noise_parameter(name: str) -> bool:
        return name.rsplit(".", 1)[-1] in {
            "std",
            "log_std",
            "_std",
            "_log_std",
        }

    compatible = {
        key: value
        for key, value in source_state.items()
        if (
            key in target_state
            and target_state[key].shape == value.shape
            and not is_noise_parameter(key)
        )
    }

    if len(compatible) < 6:
        raise RuntimeError(
            f"Too few compatible parameters for {label}: "
            f"{len(compatible)}"
        )

    result = torch.nn.Module.load_state_dict(module, compatible, strict=False)

    std_reset = set_action_std(
        module,
        requested_std,
    )

    runner.current_learning_iteration = 0

    print(
        "[DELTA_WARM_START] "
        f"label={label} "
        f"checkpoint={checkpoint} "
        f"compatible_parameters={len(compatible)} "
        f"missing_keys={len(result.missing_keys)} "
        f"unexpected_keys={len(result.unexpected_keys)} "
        f"optimizer_loaded=false "
        f"action_std_reset={std_reset} "
        f"requested_std={requested_std}",
        flush=True,
    )


def snapshot_policy(
    runner: Any,
) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in (
            policy_module(runner).state_dict()
        ).items()
        if torch.is_tensor(value)
    }


def parameter_delta(
    before: dict[str, torch.Tensor],
    runner: Any,
) -> dict[str, float]:
    current = policy_module(runner).state_dict()

    square_sum = 0.0
    maximum = 0.0
    changed = 0

    for key, old in before.items():
        difference = (
            current[key].detach().cpu()
            - old
        )

        square_sum += float(
            difference.double().square().sum()
        )

        item_maximum = float(
            difference.abs().max()
        )
        maximum = max(
            maximum,
            item_maximum,
        )

        if item_maximum > 0.0:
            changed += 1

    return {
        "l2": square_sum ** 0.5,
        "max_abs": maximum,
        "changed_tensors": float(changed),
    }


def save_checkpoint(
    *,
    runner: Any,
    path: Path,
    iteration: int,
    role: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": (
                policy_module(runner).state_dict()
            ),
            "optimizer_state_dict": (
                runner.alg.optimizer.state_dict()
            ),
            "iter": iteration,
            "infos": {
                "role": role,
                "delta_pose_joint_v0": True,
            },
        },
        path,
    )


def run_joint_training(
    *,
    high_runner: Any,
    low_runner: Any,
    runner_env: RslRlHighLevelEnvAdapter,
    high_env: HighLevelSoccerVecEnv,
    iterations: int,
    log_dir: Path,
) -> None:
    high_before = snapshot_policy(high_runner)
    low_before = snapshot_policy(low_runner)

    high_runner.alg.policy.train()
    low_runner.alg.policy.train()
    high_env.joint_low_runner = low_runner

    high_observations = runner_env.reset().to(
        high_runner.device
    )
    high_env.joint_low_obs = (
        high_env.low_env.get_observations()
    ).to(low_runner.device)

    expected_low_steps = (
        high_runner.cfg["num_steps_per_env"]
        * high_env.low_level_steps_per_action
    )

    if (
        low_runner.cfg["num_steps_per_env"]
        != expected_low_steps
    ):
        raise RuntimeError(
            "Nested rollout size mismatch. "
            f"expected_low_steps={expected_low_steps} "
            f"configured_low_steps="
            f"{low_runner.cfg['num_steps_per_env']}"
        )

    records: list[dict[str, Any]] = []

    for iteration in range(iterations):
        high_reward_values: list[float] = []
        goal_values: list[float] = []
        own_goal_values: list[float] = []
        episode_length_values: list[float] = []

        high_env.joint_low_reward_sum = 0.0
        high_env.joint_low_reward_count = 0

        with torch.inference_mode():
            for _ in range(
                high_runner.cfg["num_steps_per_env"]
            ):
                high_actions = high_runner.alg.act(
                    high_observations
                )

                (
                    high_observations,
                    high_rewards,
                    high_dones,
                    high_extras,
                ) = runner_env.step(
                    high_actions.to(
                        runner_env.device
                    )
                )

                high_observations = (
                    high_observations.to(
                        high_runner.device
                    )
                )
                high_rewards = high_rewards.to(
                    high_runner.device
                )
                high_dones = high_dones.to(
                    high_runner.device
                )

                high_runner.alg.process_env_step(
                    high_observations,
                    high_rewards,
                    high_dones,
                    high_extras,
                )

                high_reward_values.append(
                    float(high_rewards.mean())
                )

                episode = high_extras.get("episode")

                if isinstance(episode, dict):
                    if "goal_rate" in episode:
                        goal_values.append(
                            float(episode["goal_rate"])
                        )
                    if "own_goal_rate" in episode:
                        own_goal_values.append(
                            float(
                                episode["own_goal_rate"]
                            )
                        )
                    if "episode_length" in episode:
                        episode_length_values.append(
                            float(
                                episode["episode_length"]
                            )
                        )

            high_runner.alg.compute_returns(
                high_observations
            )

            if high_env.joint_low_obs is None:
                raise RuntimeError(
                    "Low-level bootstrap observation is missing."
                )

            low_runner.alg.compute_returns(
                high_env.joint_low_obs
            )

        high_losses = high_runner.alg.update()
        low_losses = low_runner.alg.update()

        high_runner.current_learning_iteration = (
            iteration
        )
        low_runner.current_learning_iteration = (
            iteration
        )

        def mean(values: list[float]) -> float | None:
            return (
                sum(values) / len(values)
                if values
                else None
            )

        low_reward_mean = (
            high_env.joint_low_reward_sum
            / max(
                high_env.joint_low_reward_count,
                1,
            )
        )

        record = {
            "iteration": iteration,
            "high_sparse_reward": mean(
                high_reward_values
            ),
            "goal_rate": mean(goal_values),
            "own_goal_rate": mean(
                own_goal_values
            ),
            "episode_length": mean(
                episode_length_values
            ),
            "low_locomotion_reward": (
                low_reward_mean
            ),
            "mean_pose_error": float(
                high_env.last_pose_error_xy.mean()
            ),
            "mean_yaw_error": float(
                high_env.last_pose_error_yaw.mean()
            ),
            "high_losses": high_losses,
            "low_losses": low_losses,
        }

        records.append(record)

        print(
            "[DELTA_JOINT_ITER] "
            f"iteration={iteration}/{iterations} "
            f"high_sparse_reward="
            f"{record['high_sparse_reward']} "
            f"goal_rate={record['goal_rate']} "
            f"own_goal_rate={record['own_goal_rate']} "
            f"episode_length={record['episode_length']} "
            f"low_locomotion_reward="
            f"{low_reward_mean:.6f} "
            f"mean_pose_error="
            f"{record['mean_pose_error']:.6f} "
            f"mean_yaw_error="
            f"{record['mean_yaw_error']:.6f} "
            f"high_losses={high_losses} "
            f"low_losses={low_losses}",
            flush=True,
        )

        if (
            iteration == 0
            or (iteration + 1) % 10 == 0
        ):
            save_checkpoint(
                runner=high_runner,
                path=(
                    log_dir
                    / f"high_model_{iteration}.pt"
                ),
                iteration=iteration,
                role="high_level",
            )
            save_checkpoint(
                runner=low_runner,
                path=(
                    log_dir
                    / f"low_model_{iteration}.pt"
                ),
                iteration=iteration,
                role="low_level",
            )

    high_delta = parameter_delta(
        high_before,
        high_runner,
    )
    low_delta = parameter_delta(
        low_before,
        low_runner,
    )

    high_final = log_dir / "high_model_final.pt"
    low_final = log_dir / "low_model_final.pt"

    save_checkpoint(
        runner=high_runner,
        path=high_final,
        iteration=iterations - 1,
        role="high_level",
    )
    save_checkpoint(
        runner=low_runner,
        path=low_final,
        iteration=iterations - 1,
        role="low_level",
    )

    summary = {
        "version": "delta_pose_joint_v0",
        "delta_pose_resolution": (
            "proportional_controller_to_velocity_command"
        ),
        "high_reward": {
            "goal": 1,
            "own_goal": -1,
            "all_other": 0,
        },
        "low_reward": (
            "existing_round2_locomotion_reward"
        ),
        "high_parameter_delta": high_delta,
        "low_parameter_delta": low_delta,
        "records": records,
    }

    summary_path = log_dir / "delta_joint_summary.json"
    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(
        "[DELTA_PARAMETER_DELTA] "
        f"high_l2={high_delta['l2']:.9f} "
        f"high_max_abs="
        f"{high_delta['max_abs']:.9f} "
        f"low_l2={low_delta['l2']:.9f} "
        f"low_max_abs="
        f"{low_delta['max_abs']:.9f}",
        flush=True,
    )

    if high_delta["max_abs"] <= 0.0:
        raise RuntimeError(
            "High-level parameters did not update."
        )

    if low_delta["max_abs"] <= 0.0:
        raise RuntimeError(
            "Low-level parameters did not update."
        )

    (log_dir / "DELTA_POSE_JOINT_OK").touch()

    print(
        f"DELTA_HIGH_CHECKPOINT={high_final}",
        flush=True,
    )
    print(
        f"DELTA_LOW_CHECKPOINT={low_final}",
        flush=True,
    )
    print(
        f"DELTA_SUMMARY={summary_path}",
        flush=True,
    )
    print(
        "DELTA_POSE_JOINT_TRAINING_COMPLETE",
        flush=True,
    )


def main() -> None:
    if args_cli.num_envs <= 0:
        raise ValueError(
            "--num_envs must be positive."
        )

    if args_cli.max_iterations <= 0:
        raise ValueError(
            "--max_iterations must be positive."
        )

    low_checkpoint = Path(
        args_cli.low_level_checkpoint
    ).resolve()

    high_checkpoint_raw = os.environ.get(
        "DELTA_HIGH_CHECKPOINT",
        "",
    ).strip()

    if not high_checkpoint_raw:
        raise RuntimeError(
            "DELTA_HIGH_CHECKPOINT is required."
        )

    high_checkpoint = Path(
        high_checkpoint_raw
    ).resolve()
    log_dir = Path(args_cli.log_dir).resolve()

    if not low_checkpoint.is_file():
        raise FileNotFoundError(
            f"Low checkpoint not found: {low_checkpoint}"
        )

    if not high_checkpoint.is_file():
        raise FileNotFoundError(
            f"High checkpoint not found: {high_checkpoint}"
        )

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.manual_seed(args_cli.seed)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    env_cfg.episode_length_s = 40.0

    try:
        env_cfg.observations.policy.enable_corruption = False
    except AttributeError:
        pass

    try:
        env_cfg.events.push_robot = None
    except AttributeError:
        pass

    try:
        env_cfg.commands.base_velocity.resampling_time_range = (
            1000.0,
            1000.0,
        )
        env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    except AttributeError:
        pass

    low_agent_cfg = instantiate_cfg(
        load_cfg_from_registry(
            "Isaac-Go2-Low-Level-Round2-v0",
            "rsl_rl_cfg_entry_point",
        )
    )
    low_agent_cfg.device = args_cli.device

    gym_env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode=None,
    )

    low_env = RslRlVecEnvWrapper(
        gym_env,
        clip_actions=low_agent_cfg.clip_actions,
    )

    high_cfg = create_high_runner_cfg(
        base_cfg=low_agent_cfg,
        run_name=args_cli.run_name,
    )

    low_cfg = create_low_runner_cfg(
        base_cfg=low_agent_cfg,
        high_rollout_steps=(
            high_cfg["num_steps_per_env"]
        ),
        low_steps_per_high_action=(
            args_cli.low_level_steps_per_action
        ),
    )

    print(
        "[DELTA_CONTRACT] "
        "high_action=relative_SE2_delta_pose "
        "resolver=proportional_controller "
        "low_interface=velocity_conditioned "
        "high_and_low_trainable=true "
        "high_reward=goal_plus1_own_goal_minus1_other0 "
        "low_reward=round2_locomotion",
        flush=True,
    )

    low_runner = OnPolicyRunner(
        low_env,
        low_cfg,
        log_dir=None,
        device=low_agent_cfg.device,
    )

    low_runner.load(
        str(low_checkpoint),
        load_optimizer=False,
    )

    low_module = policy_module(low_runner)
    low_std = float(
        low_cfg["policy"]["init_noise_std"]
    )
    low_std_reset = set_action_std(
        low_module,
        low_std,
    )

    low_policy = low_runner.get_inference_policy(
        device=low_env.device,
    )

    high_env = HighLevelSoccerVecEnv(
        low_env=low_env,
        low_policy=low_policy,
        low_level_steps_per_action=(
            args_cli.low_level_steps_per_action
        ),
        episode_steps=(
            args_cli.high_level_episode_steps
        ),
        goal_line_x=args_cli.goal_line_x,
        goal_center_x=args_cli.goal_center_x,
        goal_half_width=args_cli.goal_half_width,
        seed=args_cli.seed,
    )

    runner_env = RslRlHighLevelEnvAdapter(
        high_env
    )

    runner_obs = runner_env.get_observations()

    print(
        "[DELTA_PREFLIGHT] "
        f"high_obs_shape="
        f"{tuple(runner_obs['policy'].shape)} "
        f"low_rollout_steps="
        f"{low_cfg['num_steps_per_env']} "
        f"low_std_reset={low_std_reset}",
        flush=True,
    )

    # Validate the controller-resolved delta-pose path before PPO.
    with torch.no_grad():
        zero_actions = torch.zeros(
            high_env.num_envs,
            high_env.num_actions,
            device=high_env.device,
        )

        for preflight_step in range(2):
            observations, rewards, dones, _ = (
                runner_env.step(zero_actions)
            )

            if not torch.isfinite(
                observations["policy"]
            ).all():
                raise RuntimeError(
                    "Preflight observations are non-finite."
                )

            if not torch.isfinite(rewards).all():
                raise RuntimeError(
                    "Preflight rewards are non-finite."
                )

            print(
                "[DELTA_PREFLIGHT] "
                f"step={preflight_step} "
                f"reward_mean="
                f"{float(rewards.mean()):.6f} "
                f"done_count={int(dones.sum())}",
                flush=True,
            )

    high_env.reset()

    high_runner = OnPolicyRunner(
        runner_env,
        high_cfg,
        log_dir=None,
        device=args_cli.device,
    )

    warm_start_policy_weights(
        runner=high_runner,
        checkpoint=high_checkpoint,
        requested_std=float(
            high_cfg["policy"]["init_noise_std"]
        ),
        label="high_v3_zero_sum",
    )

    contract = {
        "version": "delta_pose_joint_v0",
        "task": args_cli.task,
        "high_checkpoint": str(high_checkpoint),
        "low_checkpoint": str(low_checkpoint),
        "high_action": [
            "delta_x_body",
            "delta_y_body",
            "delta_yaw",
        ],
        "delta_pose_scales": {
            "delta_x_positive": 0.20,
            "delta_x_negative": 0.10,
            "delta_y": 0.12,
            "delta_yaw": 0.20,
        },
        "pose_controller": {
            "kx_positive": 3.25,
            "kx_negative": 2.50,
            "ky": 2.9166666667,
            "kyaw": 4.50,
            "velocity_limits": {
                "vx": [-0.25, 0.65],
                "vy": [-0.35, 0.35],
                "wz": [-0.90, 0.90],
            },
        },
        "high_reward": {
            "goal": 1,
            "own_goal": -1,
            "all_other": 0,
        },
        "low_reward": (
            "existing_round2_locomotion_reward"
        ),
        "high_cfg": high_cfg,
        "low_cfg": low_cfg,
    }

    (log_dir / "delta_pose_joint_contract.json").write_text(
        json.dumps(
            contract,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    run_joint_training(
        high_runner=high_runner,
        low_runner=low_runner,
        runner_env=runner_env,
        high_env=high_env,
        iterations=args_cli.max_iterations,
        log_dir=log_dir,
    )

    high_env.close()




try:
    main()
except BaseException as error:
    print(
        f"[PHASEA_FATAL] {type(error).__name__}: {error}",
        flush=True,
    )
    raise
finally:
    print(
        "[PHASEA_BOOT] closing_simulation_app",
        flush=True,
    )
    simulation_app.close()