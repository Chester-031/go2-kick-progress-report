# V4D_STAGED_ZERO_SUM_JOINT_INTEGRATION
"""V4D staged zero-sum high/low joint integration over direct Delta Pose."""

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
        description="V4D staged zero-sum high/low joint soccer training."
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

    parser.add_argument(
        "--trained_low_checkpoint",
        type=str,
        default="",
        help="Exact V4D low-level checkpoint used for evaluation.",
    )
    parser.add_argument(
        "--eval_video",
        type=str,
        default="",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=600,
    )
    parser.add_argument(
        "--eval_fps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--trajectory_csv",
        type=str,
        default="",
    )
    parser.add_argument(
        "--eval_summary",
        type=str,
        default="",
    )

    parser.add_argument("--kick_episode_steps", type=int, default=60)
    parser.add_argument("--eval_episodes", type=int, default=8)
    parser.add_argument("--kick_fixed_leg", type=str, default="alternate", choices=("alternate", "left", "right"))

    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = build_parser()
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

print("[PHASEA_BOOT] app_launcher_returned", flush=True)

import isaac_go2_low_level_rl.tasks  # noqa: E402,F401
import isaac_go2_low_level_rl.tasks.manager_based.soccer  # noqa: E402,F401

if args_cli.task not in gym.registry:
    matching_tasks = sorted(
        task_id
        for task_id in gym.registry.keys()
        if "Go2-Soccer" in task_id
    )
    raise RuntimeError(
        "Soccer task registration failed. "
        f"requested={args_cli.task} "
        f"registered={matching_tasks}"
    )

print(
    "[V4D_TASK_REGISTRATION_OK] "
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
    """Train a pose-conditioned low-level policy under a frozen V3 soccer policy.

    V4D deliberately merges two previously separate milestones:

    * pose-conditioned locomotion; and
    * continuous, contact-aware pushing.

    The high-level V3 policy is frozen and provides broad movement intent.
    A deterministic router converts that intent into one of three contexts:
    stand, locomotion, or push. The trainable low-level network directly
    observes pose error, target terminal velocity, remaining time, skill
    context, and ball-relative state. Its 12-dimensional output is applied
    directly as joint-position residuals.

    The legacy velocity reference is never exposed to the trainable actor.
    It may be written to the simulator command manager only for native-reward
    diagnostics. The actor always receives scrubbed legacy command slots and
    acts from direct delta-pose, skill, time, and ball context.
    """

    SKILL_STAND = 0
    SKILL_LOCOMOTION = 1
    SKILL_PUSH = 2
    NUM_SKILLS = 3
    SKILL_NAMES = (
        "stand",
        "locomotion",
        "push",
    )

    NATIVE_LOW_OBS_DIM = 48
    LOW_ACTION_DIM = 12

    # Round-2 observation layout and V4D-R1 direct-pose context indices.
    NATIVE_COMMAND_SLICE = slice(9, 12)
    CONTEXT_ERROR_X = 48
    CONTEXT_ERROR_Y = 49
    CONTEXT_SIN_YAW = 50
    CONTEXT_COS_YAW = 51
    CONTEXT_TERMINAL_VX = 52
    CONTEXT_TERMINAL_VY = 53
    CONTEXT_TERMINAL_WZ = 54
    CONTEXT_REMAINING = 55
    CONTEXT_SKILL_START = 56
    CONTEXT_BALL_X = 59
    CONTEXT_BALL_Y = 60
    CONTEXT_TEACHER_FLAG = 61

    def __init__(
        self,
        *,
        low_env: RslRlVecEnvWrapper,
        low_level_steps_per_action: int,
        episode_steps: int,
        goal_line_x: float,
        goal_center_x: float,
        goal_half_width: float,
        seed: int,
    ) -> None:
        self.low_env = low_env
        self.base_env = low_env.unwrapped

        self.device = low_env.device
        self.num_envs = low_env.num_envs
        self.num_actions = 3
        self.num_privileged_obs = None

        self.low_level_steps_per_action = int(
            low_level_steps_per_action
        )
        self.max_episode_length = int(episode_steps)
        self.max_low_episode_length = (
            self.max_episode_length
            * self.low_level_steps_per_action
        )

        self.goal_line_x = float(goal_line_x)
        self.goal_center_x = float(goal_center_x)
        self.goal_half_width = float(goal_half_width)

        self.robot = self.base_env.scene["robot"]
        self.ball = self.base_env.scene["ball"]
        self.env_origins = self.base_env.scene.env_origins
        self.command_tensor = writable_command_tensor(
            self.base_env
        )

        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

        self.cfg = SimpleNamespace(is_finite_horizon=False)

        self.episode_length_buf = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        self.low_episode_length_buf = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )

        self.previous_high_action = torch.zeros(
            self.num_envs,
            3,
            device=self.device,
        )
        self.previous_low_action = torch.zeros(
            self.num_envs,
            self.LOW_ACTION_DIM,
            device=self.device,
        )

        self.skill_id = torch.zeros(
            self.num_envs,
            dtype=torch.long,
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
        self.target_terminal_velocity = torch.zeros(
            self.num_envs,
            3,
            device=self.device,
        )
        self.push_direction_world = torch.zeros(
            self.num_envs,
            2,
            device=self.device,
        )
        self.target_height = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.target_elapsed_steps = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        self.target_duration_steps = torch.full(
            (self.num_envs,),
            self.low_level_steps_per_action,
            dtype=torch.long,
            device=self.device,
        )

        self.teacher_scale = 1.0
        self.last_reference_command = torch.zeros(
            self.num_envs,
            3,
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
        self.last_low_reward = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.last_upright_error = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.last_push_stall = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )

        self.base_done_accum = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self.pending_goal = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self.pending_own_goal = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self.pending_timeout = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
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

        self.total_resets = 0
        self.vector_steps = 0

        self._native_low_obs = extract_observation(
            self.low_env.get_observations()
        )

        if self._native_low_obs.shape[1] != self.NATIVE_LOW_OBS_DIM:
            raise RuntimeError(
                "V4D expects the accepted Round-2 48-dimensional "
                "native observation. "
                f"received={tuple(self._native_low_obs.shape)}"
            )

        all_ids = torch.arange(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        self._reset_idx(all_ids)

        self.high_obs_buf = self._build_high_observation()
        self.num_obs = int(self.high_obs_buf.shape[1])

        self.low_obs_buf = self._build_low_observation()
        self.low_num_obs = int(self.low_obs_buf.shape[1])
        self.low_num_actions = self.LOW_ACTION_DIM

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
            "[V4D_BOOT] "
            f"high_obs={self.num_obs} "
            f"high_actions={self.num_actions} "
            f"low_obs={self.low_num_obs} "
            f"low_actions={self.low_num_actions}",
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
        ball_velocity_x_body, ball_velocity_y_body = world_to_body(
            ball_velocity_world[:, 0],
            ball_velocity_world[:, 1],
            robot_yaw,
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
        robot_angular_velocity_body = self.robot.data.root_ang_vel_b

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

    def _build_high_observation(self) -> torch.Tensor:
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
                self.previous_high_action,
                difficulty_tensor,
            ),
            dim=1,
        )

        if not torch.isfinite(observation).all():
            raise RuntimeError(
                "V4D high-level observations are non-finite."
            )

        return observation

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
            self.target_yaw_world - state["robot_yaw"]
        )

        return error_x_body, error_y_body, error_yaw

    def _remaining_fraction(self) -> torch.Tensor:
        duration = torch.clamp(
            self.target_duration_steps,
            min=1,
        )
        remaining = (
            duration - self.target_elapsed_steps
        ).clamp(min=0)
        return remaining.float() / duration.float()

    def _reference_velocity(
        self,
    ) -> torch.Tensor:
        error_x, error_y, error_yaw = self._pose_error()

        command = torch.zeros(
            self.num_envs,
            3,
            device=self.device,
        )
        command[:, 0] = torch.where(
            error_x >= 0.0,
            2.2 * error_x,
            1.8 * error_x,
        )
        command[:, 1] = 2.0 * error_y
        command[:, 2] = 3.2 * error_yaw

        error_xy = torch.sqrt(
            error_x.square()
            + error_y.square()
            + 1.0e-8
        )
        near_target = torch.exp(
            -error_xy / 0.16
        ).unsqueeze(1)

        command = (
            (1.0 - near_target) * command
            + near_target * self.target_terminal_velocity
        )

        push_mask = self.skill_id == self.SKILL_PUSH
        command[push_mask, 0] = torch.maximum(
            command[push_mask, 0],
            self.target_terminal_velocity[push_mask, 0],
        )

        stand_mask = self.skill_id == self.SKILL_STAND
        command[stand_mask] = 0.0

        command[:, 0] = torch.clamp(
            command[:, 0],
            -0.30,
            0.70,
        )
        command[:, 1] = torch.clamp(
            command[:, 1],
            -0.40,
            0.40,
        )
        command[:, 2] = torch.clamp(
            command[:, 2],
            -1.00,
            1.00,
        )

        self.last_reference_command = command
        return command

    def _build_low_observation(self) -> torch.Tensor:
        native = extract_observation(
            self.low_env.get_observations()
        )

        if native.shape != (
            self.num_envs,
            self.NATIVE_LOW_OBS_DIM,
        ):
            raise RuntimeError(
                "Native low-level observation shape changed. "
                f"received={tuple(native.shape)}"
            )

        # V4D-R1: the actor never observes legacy velocity commands.
        native = native.clone()
        native[:, self.NATIVE_COMMAND_SLICE] = 0.0
        self._native_low_obs = native

        error_x, error_y, error_yaw = self._pose_error()
        remaining = self._remaining_fraction()
        one_hot = torch.nn.functional.one_hot(
            self.skill_id,
            num_classes=self.NUM_SKILLS,
        ).float()
        state = self._state()

        context = torch.cat(
            (
                (error_x / 0.80).unsqueeze(1),
                (error_y / 0.50).unsqueeze(1),
                torch.sin(error_yaw).unsqueeze(1),
                torch.cos(error_yaw).unsqueeze(1),
                self.target_terminal_velocity[:, 0:1] / 0.70,
                self.target_terminal_velocity[:, 1:2] / 0.40,
                self.target_terminal_velocity[:, 2:3] / 1.00,
                remaining.unsqueeze(1),
                one_hot,
                state["robot_ball_x_body"].unsqueeze(1) / 2.0,
                state["robot_ball_y_body"].unsqueeze(1) / 1.0,
                # Keep the 62-D shape, but never expose teacher presence.
                torch.zeros(
                    (self.num_envs, 1),
                    device=self.device,
                ),
            ),
            dim=1,
        )

        observation = torch.cat(
            (native, context),
            dim=1,
        )

        if not torch.isfinite(observation).all():
            raise RuntimeError(
                "V4D low-level observations are non-finite."
            )

        return observation

    def set_teacher_scale(
        self,
        value: float,
    ) -> None:
        self.teacher_scale = float(
            max(0.0, min(1.0, value))
        )

    def begin_high_action(
        self,
        action: torch.Tensor,
    ) -> dict[str, float]:
        if action.ndim == 1 and self.num_envs == 1:
            action = action.unsqueeze(0)

        expected = (self.num_envs, 3)
        if action.shape != expected:
            raise RuntimeError(
                "V4D high action shape mismatch. "
                f"expected={expected} "
                f"received={tuple(action.shape)}"
            )

        action = torch.clamp(
            action.to(self.device),
            -1.0,
            1.0,
        )
        state = self._state()

        skill = torch.full(
            (self.num_envs,),
            self.SKILL_LOCOMOTION,
            dtype=torch.long,
            device=self.device,
        )

        stand_mask = action.abs().amax(dim=1) < 0.05
        push_mask = (
            (state["robot_ball_distance"] < 0.95)
            & (state["robot_ball_x_body"] > 0.03)
            & (state["ball_goal_alignment"] > 0.55)
            & (action[:, 0] > -0.20)
        )

        skill[push_mask] = self.SKILL_PUSH
        skill[stand_mask] = self.SKILL_STAND
        self.skill_id = skill

        delta = torch.zeros(
            self.num_envs,
            3,
            device=self.device,
        )
        delta[:, 0] = torch.where(
            action[:, 0] >= 0.0,
            0.32 * action[:, 0],
            0.16 * action[:, 0],
        )
        delta[:, 1] = 0.18 * action[:, 1]
        delta[:, 2] = 0.30 * action[:, 2]

        terminal_velocity = torch.zeros(
            self.num_envs,
            3,
            device=self.device,
        )
        terminal_velocity[:, 0] = 0.12 * action[:, 0]
        terminal_velocity[:, 1] = 0.08 * action[:, 1]
        terminal_velocity[:, 2] = 0.15 * action[:, 2]

        desired_goal_yaw = torch.atan2(
            state["goal_direction_y"],
            state["goal_direction_x"],
        )
        push_yaw_delta = wrap_to_pi(
            desired_goal_yaw - state["robot_yaw"]
        )

        push_forward = torch.clamp(
            0.55 + 0.10 * torch.relu(action[:, 0]),
            0.55,
            0.65,
        )
        delta[push_mask, 0] = push_forward[push_mask]
        delta[push_mask, 1] = torch.clamp(
            0.35 * state["robot_ball_y_body"][push_mask],
            -0.12,
            0.12,
        )
        delta[push_mask, 2] = torch.clamp(
            push_yaw_delta[push_mask],
            -0.35,
            0.35,
        )

        terminal_velocity[push_mask, 0] = torch.clamp(
            0.34 + 0.12 * torch.relu(action[push_mask, 0]),
            0.34,
            0.48,
        )
        terminal_velocity[push_mask, 1] = 0.0
        terminal_velocity[push_mask, 2] = torch.clamp(
            0.18 * action[push_mask, 2],
            -0.18,
            0.18,
        )

        delta[stand_mask] = 0.0
        terminal_velocity[stand_mask] = 0.0

        yaw = state["robot_yaw"]
        cosine = torch.cos(yaw)
        sine = torch.sin(yaw)
        delta_x_world = (
            cosine * delta[:, 0]
            - sine * delta[:, 1]
        )
        delta_y_world = (
            sine * delta[:, 0]
            + cosine * delta[:, 1]
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
            yaw + delta[:, 2]
        )
        self.target_terminal_velocity = terminal_velocity
        self.push_direction_world[:, 0] = (
            state["goal_direction_x"]
        )
        self.push_direction_world[:, 1] = (
            state["goal_direction_y"]
        )
        self.target_elapsed_steps.zero_()
        self.target_duration_steps.fill_(
            self.low_level_steps_per_action
        )

        self.previous_high_action = action
        self.base_done_accum.zero_()
        self.pending_goal.zero_()
        self.pending_own_goal.zero_()
        self.pending_timeout.zero_()

        counts = torch.bincount(
            skill,
            minlength=self.NUM_SKILLS,
        ).float() / float(self.num_envs)

        return {
            "stand_fraction": float(
                counts[self.SKILL_STAND]
            ),
            "locomotion_fraction": float(
                counts[self.SKILL_LOCOMOTION]
            ),
            "push_fraction": float(
                counts[self.SKILL_PUSH]
            ),
        }

    def _applied_torque_penalty(self) -> torch.Tensor:
        torque = getattr(
            self.robot.data,
            "applied_torque",
            None,
        )

        if not torch.is_tensor(torque):
            return torch.zeros(
                self.num_envs,
                device=self.device,
            )

        return torque.square().mean(dim=1)

    def _custom_low_reward(
        self,
        *,
        before_state: dict[str, torch.Tensor],
        before_error: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        after_state: dict[str, torch.Tensor],
        after_error: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        low_actions: torch.Tensor,
        base_done: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        before_xy = torch.sqrt(
            before_error[0].square()
            + before_error[1].square()
            + 1.0e-8
        )
        after_xy = torch.sqrt(
            after_error[0].square()
            + after_error[1].square()
            + 1.0e-8
        )

        pose_track = torch.exp(
            -after_xy.square() / (0.28 ** 2)
        )
        yaw_track = torch.exp(
            -after_error[2].square() / (0.35 ** 2)
        )
        pose_progress = torch.clamp(
            (before_xy - after_xy) / 0.035,
            -1.0,
            1.0,
        )
        yaw_progress = torch.clamp(
            (
                before_error[2].abs()
                - after_error[2].abs()
            ) / 0.08,
            -1.0,
            1.0,
        )

        linear_error = (
            after_state["robot_velocity_body"][:, :2]
            - self.target_terminal_velocity[:, :2]
        )
        angular_error = (
            after_state["robot_angular_velocity_body"][:, 2]
            - self.target_terminal_velocity[:, 2]
        )
        velocity_track = torch.exp(
            -linear_error.square().sum(dim=1) / (0.45 ** 2)
            -angular_error.square() / (0.70 ** 2)
        )

        native = self._native_low_obs
        projected_gravity_xy = native[:, 6:8]
        upright_error = projected_gravity_xy.square().sum(dim=1)
        upright_reward = torch.exp(
            -upright_error / (0.18 ** 2)
        )

        height_error = (
            after_state["robot_position"][:, 2]
            - self.target_height
        )
        height_reward = torch.exp(
            -height_error.square() / (0.08 ** 2)
        )

        action_rate = (
            low_actions - self.previous_low_action
        ).square().mean(dim=1)
        torque_penalty = self._applied_torque_penalty()

        reward = (
            0.70 * pose_track
            + 0.35 * yaw_track
            + 0.55 * pose_progress
            + 0.20 * yaw_progress
            + 0.45 * velocity_track
            + 0.40 * upright_reward
            + 0.18 * height_reward
            - 0.025 * action_rate
            - 0.00002 * torque_penalty
        )

        push_mask = self.skill_id == self.SKILL_PUSH
        ball_delta = (
            after_state["ball_position"][:, :2]
            - before_state["ball_position"][:, :2]
        )
        requested_ball_progress = (
            ball_delta
            * self.push_direction_world
        ).sum(dim=1)
        requested_ball_progress = torch.clamp(
            requested_ball_progress / 0.025,
            -1.0,
            1.0,
        )

        ball_x_error = (
            after_state["robot_ball_x_body"] - 0.34
        )
        ball_y_error = after_state["robot_ball_y_body"]
        ball_control = torch.exp(
            -ball_x_error.square() / (0.28 ** 2)
            -ball_y_error.square() / (0.22 ** 2)
        )
        ball_velocity_requested = (
            after_state["ball_velocity_world"][:, :2]
            * self.push_direction_world
        ).sum(dim=1)
        ball_motion = torch.clamp(
            ball_velocity_requested / 0.40,
            0.0,
            1.0,
        )
        contact_continuity = (
            (after_state["robot_ball_distance"] < 0.72)
            & (after_state["robot_ball_x_body"] > -0.05)
        ).float()

        push_bonus = (
            0.65 * ball_control
            + 0.75 * requested_ball_progress
            + 0.30 * ball_motion
            + 0.20 * contact_continuity
        )
        reward = reward + push_mask.float() * push_bonus

        stand_mask = self.skill_id == self.SKILL_STAND
        stand_bonus = (
            0.35 * upright_reward
            + 0.15 * height_reward
            - 0.04 * low_actions.square().mean(dim=1)
        )
        reward = reward + stand_mask.float() * stand_bonus

        reward = torch.clamp(
            reward,
            -3.0,
            4.0,
        )
        reward[base_done] = -2.0

        robot_speed = torch.linalg.vector_norm(
            after_state["robot_velocity_body"][:, :2],
            dim=1,
        )
        ball_speed = torch.linalg.vector_norm(
            after_state["ball_velocity_world"][:, :2],
            dim=1,
        )
        push_stall = (
            push_mask
            & (robot_speed < 0.035)
            & (ball_speed < 0.035)
            & (after_state["robot_ball_distance"] < 0.65)
        )

        diagnostics = {
            "pose_error_xy": after_xy,
            "pose_error_yaw": after_error[2].abs(),
            "upright_error": upright_error,
            "ball_progress": requested_ball_progress,
            "push_stall": push_stall.float(),
            "robot_speed": robot_speed,
            "ball_speed": ball_speed,
            "action_rate": action_rate,
        }

        return reward, diagnostics

    @torch.inference_mode(False)
    def low_step(
        self,
        low_actions: torch.Tensor,
        *,
        is_last_low_step: bool,
    ) -> tuple[
        TensorDict,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
        dict[str, float],
    ]:
        if low_actions.ndim == 1 and self.num_envs == 1:
            low_actions = low_actions.unsqueeze(0)

        expected = (
            self.num_envs,
            self.LOW_ACTION_DIM,
        )
        if low_actions.shape != expected:
            raise RuntimeError(
                "V4D low action shape mismatch. "
                f"expected={expected} "
                f"received={tuple(low_actions.shape)}"
            )

        low_actions = torch.clamp(
            low_actions.to(self.device),
            -1.0,
            1.0,
        )

        before_state = self._state()
        before_error = self._pose_error()

        reference = self._reference_velocity()
        teacher_command = (
            float(self.teacher_scale) * reference
        )
        teacher_command[self.base_done_accum] = 0.0
        self.command_tensor[:, :3] = teacher_command

        result = self.low_env.step(
            low_actions.detach().clone()
        )
        (
            native_obs,
            native_reward,
            native_done,
            native_extras,
        ) = normalize_step_result(result)

        base_done = normalize_done_tensor(
            native_done,
            num_envs=self.num_envs,
            device=self.device,
        )
        self.base_done_accum |= base_done

        self._native_low_obs = extract_observation(
            native_obs
        )

        after_state = self._state()
        after_error = self._pose_error()

        reward, diagnostic_tensors = (
            self._custom_low_reward(
                before_state=before_state,
                before_error=before_error,
                after_state=after_state,
                after_error=after_error,
                low_actions=low_actions,
                base_done=base_done,
            )
        )

        goal = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        own_goal = torch.zeros_like(goal)
        timeout = torch.zeros_like(goal)

        if is_last_low_step:
            goal = (
                after_state["ball_position"][:, 0]
                >= (
                    after_state["origin_x"]
                    + self.goal_line_x
                )
            ) & (
                torch.abs(
                    after_state["ball_position"][:, 1]
                    - after_state["origin_y"]
                )
                <= self.goal_half_width
            )
            own_goal = (
                after_state["ball_position"][:, 0]
                <= (
                    after_state["origin_x"]
                    - self.goal_line_x
                )
            ) & (
                torch.abs(
                    after_state["ball_position"][:, 1]
                    - after_state["origin_y"]
                )
                <= self.goal_half_width
            )
            timeout = (
                self.episode_length_buf + 1
                >= self.max_episode_length
            )

        self.pending_goal |= goal
        self.pending_own_goal |= own_goal
        self.pending_timeout |= timeout

        done = (
            base_done
            | goal
            | own_goal
            | timeout
        )

        self.target_elapsed_steps += 1
        self.low_episode_length_buf += 1
        self.previous_low_action = low_actions

        self.last_pose_error_xy = (
            diagnostic_tensors["pose_error_xy"]
        )
        self.last_pose_error_yaw = (
            diagnostic_tensors["pose_error_yaw"]
        )
        self.last_upright_error = (
            diagnostic_tensors["upright_error"]
        )
        self.last_push_stall = (
            diagnostic_tensors["push_stall"] > 0.5
        )
        self.last_low_reward = reward

        self.low_obs_buf = self._build_low_observation()
        packed_obs = TensorDict(
            {"policy": self.low_obs_buf},
            batch_size=[self.num_envs],
            device=self.device,
        )

        extras = (
            dict(native_extras)
            if isinstance(native_extras, dict)
            else {}
        )
        extras["time_outs"] = timeout

        diagnostics = {
            "reward": float(reward.mean()),
            "pose_error_xy": float(
                self.last_pose_error_xy.mean()
            ),
            "pose_error_yaw": float(
                self.last_pose_error_yaw.mean()
            ),
            "upright_error": float(
                self.last_upright_error.mean()
            ),
            "push_stall_fraction": float(
                self.last_push_stall.float().mean()
            ),
            "robot_speed": float(
                diagnostic_tensors["robot_speed"].mean()
            ),
            "ball_speed": float(
                diagnostic_tensors["ball_speed"].mean()
            ),
            "ball_progress": float(
                diagnostic_tensors["ball_progress"].mean()
            ),
            "action_rate": float(
                diagnostic_tensors["action_rate"].mean()
            ),
            "native_reward": float(
                torch.as_tensor(
                    native_reward,
                    device=self.device,
                ).float().mean()
            ),
        }

        return packed_obs, reward, done, extras, diagnostics

    def end_high_action(self) -> dict[str, Any]:
        """Finish one high-level decision and expose a strict zero-sum transition."""

        self.episode_length_buf += 1
        self.vector_steps += 1

        state = self._state()
        goal_progress = (
            self.previous_ball_goal_distance
            - state["ball_goal_distance"]
        )

        done = (
            self.base_done_accum
            | self.pending_goal
            | self.pending_own_goal
            | self.pending_timeout
        )

        sparse_reward = (
            self.pending_goal.float()
            - self.pending_own_goal.float()
        )

        valid_support = (
            (sparse_reward == -1.0)
            | (sparse_reward == 0.0)
            | (sparse_reward == 1.0)
        )
        if not bool(valid_support.all()):
            raise RuntimeError(
                "V4D high-level reward escaped {-1, 0, +1}."
            )

        expected_nonzero = (
            self.pending_goal | self.pending_own_goal
        )
        if not torch.equal(
            sparse_reward.ne(0.0),
            expected_nonzero,
        ):
            raise RuntimeError(
                "V4D zero-sum reward/event mismatch."
            )

        reward_tensor = sparse_reward.clone()
        done_tensor = done.clone()
        timeout_tensor = self.pending_timeout.clone()

        completed_count = int(done.sum())

        goal_rate = None
        own_goal_rate = None
        episode_length = None

        if completed_count > 0:
            goal_rate = float(
                self.pending_goal[done].float().mean()
            )
            own_goal_rate = float(
                self.pending_own_goal[done].float().mean()
            )
            episode_length = float(
                self.episode_length_buf[done].float().mean()
            )

        result: dict[str, Any] = {
            "goal_fraction": float(
                self.pending_goal.float().mean()
            ),
            "own_goal_fraction": float(
                self.pending_own_goal.float().mean()
            ),
            "timeout_fraction": float(
                self.pending_timeout.float().mean()
            ),
            "base_done_fraction": float(
                self.base_done_accum.float().mean()
            ),
            "goal_rate": goal_rate,
            "own_goal_rate": own_goal_rate,
            "episode_length": episode_length,
            "ball_goal_progress": float(
                goal_progress.mean()
            ),
            "completed_episodes": float(completed_count),
            "high_sparse_reward": float(
                sparse_reward.mean()
            ),
            "high_nonzero_reward_fraction": float(
                sparse_reward.ne(0.0).float().mean()
            ),
            "reward_tensor": reward_tensor,
            "done_tensor": done_tensor,
            "time_outs_tensor": timeout_tensor,
        }

        self.previous_ball_goal_distance = (
            state["ball_goal_distance"].clone()
        )
        self.previous_robot_ball_distance = (
            state["robot_ball_distance"].clone()
        )

        if done.any():
            reset_ids = torch.nonzero(
                done,
                as_tuple=False,
            ).squeeze(-1)
            self._reset_idx(reset_ids)

        self.high_obs_buf = self._build_high_observation()
        self.low_obs_buf = self._build_low_observation()

        return result


    def get_high_observations(self) -> torch.Tensor:
        self.high_obs_buf = self._build_high_observation()
        return self.high_obs_buf

    def get_low_observations(self) -> TensorDict:
        self.low_obs_buf = self._build_low_observation()
        return TensorDict(
            {"policy": self.low_obs_buf},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def reset(self) -> None:
        env_ids = torch.arange(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        self._reset_idx(env_ids)
        self.high_obs_buf = self._build_high_observation()
        self.low_obs_buf = self._build_low_observation()

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
        robot_y_offset_limit = (
            0.45 + 0.85 * difficulty
        )
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
        yaw_noise_limit = (
            0.55 + 2.20 * difficulty
        )
        robot_yaw = heading_to_ball + (
            2.0 * self._rand((count,)) - 1.0
        ) * yaw_noise_limit

        origins = self.env_origins[env_ids]
        robot_z = self.robot.data.root_pos_w[
            env_ids, 2
        ]
        ball_z = self.ball.data.root_pos_w[
            env_ids, 2
        ]

        robot_pose = torch.zeros(
            count,
            7,
            device=self.device,
        )
        robot_pose[:, 0] = (
            origins[:, 0] + robot_x_local
        )
        robot_pose[:, 1] = (
            origins[:, 1] + robot_y_local
        )
        robot_pose[:, 2] = robot_z
        robot_pose[:, 3] = torch.cos(
            robot_yaw / 2.0
        )
        robot_pose[:, 6] = torch.sin(
            robot_yaw / 2.0
        )

        ball_pose = torch.zeros(
            count,
            7,
            device=self.device,
        )
        ball_pose[:, 0] = (
            origins[:, 0] + ball_x_local
        )
        ball_pose[:, 1] = (
            origins[:, 1] + ball_y_local
        )
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
        self.previous_high_action[env_ids] = 0.0
        self.previous_low_action[env_ids] = 0.0
        self.skill_id[env_ids] = self.SKILL_STAND
        self.target_terminal_velocity[env_ids] = 0.0
        self.target_elapsed_steps[env_ids] = 0
        self.target_duration_steps[env_ids] = (
            self.low_level_steps_per_action
        )
        self.episode_length_buf[env_ids] = 0
        self.low_episode_length_buf[env_ids] = 0

        self.base_done_accum[env_ids] = False
        self.pending_goal[env_ids] = False
        self.pending_own_goal[env_ids] = False
        self.pending_timeout[env_ids] = False

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
        self.target_height[env_ids] = (
            state["robot_position"][env_ids, 2]
        )
        self.push_direction_world[env_ids, 0] = (
            state["goal_direction_x"][env_ids]
        )
        self.push_direction_world[env_ids, 1] = (
            state["goal_direction_y"][env_ids]
        )
        self.previous_ball_goal_distance[env_ids] = (
            state["ball_goal_distance"][env_ids]
        )
        self.previous_robot_ball_distance[env_ids] = (
            state["robot_ball_distance"][env_ids]
        )

        self.total_resets += count
        self._native_low_obs = extract_observation(
            self.low_env.get_observations()
        )

    def close(self) -> None:
        self.low_env.close()


class V4DHighLevelEnvAdapter:
    """RSL-RL adapter for the trainable V4D high-level policy."""

    def __init__(
        self,
        env: HighLevelSoccerVecEnv,
    ) -> None:
        self.env = env
        self.num_envs = env.num_envs
        self.num_actions = env.num_actions
        self.device = env.device
        self.max_episode_length = env.max_episode_length
        self.cfg = env.cfg
        self.episode_length_buf = env.episode_length_buf

    def get_observations(self) -> TensorDict:
        observation = self.env.get_high_observations()
        return TensorDict(
            {"policy": observation},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def reset(self) -> TensorDict:
        self.env.reset()
        return self.get_observations()


class PoseLowLevelEnvAdapter:
    """RSL-RL adapter for the custom 62-dimensional V4D observation."""

    def __init__(
        self,
        env: HighLevelSoccerVecEnv,
    ) -> None:
        self.env = env
        self.num_envs = env.num_envs
        self.num_actions = env.low_num_actions
        self.device = env.device
        self.max_episode_length = (
            env.max_low_episode_length
        )
        self.cfg = env.cfg

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.low_episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(
        self,
        value: torch.Tensor,
    ) -> None:
        self.env.low_episode_length_buf = value

    def get_observations(self) -> TensorDict:
        return self.env.get_low_observations()

    def reset(self) -> TensorDict:
        self.env.reset()
        return self.get_observations()



def create_high_runner_cfg(
    *,
    base_cfg: Any,
    run_name: str,
) -> dict[str, Any]:
    cfg = base_cfg.to_dict()

    cfg["num_steps_per_env"] = int(
        os.environ.get(
            "V4D_HIGH_ROLLOUT_STEPS",
            "24",
        )
    )
    cfg["save_interval"] = 10_000
    cfg["experiment_name"] = (
        "go2_soccer_v4d_joint"
    )
    cfg["run_name"] = run_name

    cfg["obs_groups"] = {
        "policy": ["policy"],
        "critic": ["policy"],
    }

    cfg["policy"]["init_noise_std"] = float(
        os.environ.get(
            "V4D_HIGH_INIT_NOISE_STD",
            "0.05",
        )
    )
    cfg["policy"]["actor_hidden_dims"] = [256, 256, 128]
    cfg["policy"]["critic_hidden_dims"] = [256, 256, 128]
    cfg["policy"]["actor_obs_normalization"] = False
    cfg["policy"]["critic_obs_normalization"] = False

    cfg["algorithm"]["entropy_coef"] = float(
        os.environ.get(
            "V4D_HIGH_ENTROPY_COEF",
            "0.0005",
        )
    )
    cfg["algorithm"]["learning_rate"] = float(
        os.environ.get(
            "V4D_HIGH_LEARNING_RATE",
            "0.00002",
        )
    )
    cfg["algorithm"]["clip_param"] = float(
        os.environ.get(
            "V4D_HIGH_CLIP_PARAM",
            "0.08",
        )
    )
    cfg["algorithm"]["num_learning_epochs"] = int(
        os.environ.get(
            "V4D_HIGH_EPOCHS",
            "3",
        )
    )
    cfg["algorithm"]["num_mini_batches"] = int(
        os.environ.get(
            "V4D_HIGH_MINI_BATCHES",
            "4",
        )
    )
    cfg["algorithm"]["gamma"] = float(
        os.environ.get(
            "V4D_HIGH_GAMMA",
            "0.99",
        )
    )
    cfg["algorithm"]["lam"] = 0.95
    cfg["algorithm"]["desired_kl"] = float(
        os.environ.get(
            "V4D_HIGH_DESIRED_KL",
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
            "V4D_LOW_INIT_NOISE_STD",
            "0.05",
        )
    )

    cfg["algorithm"]["entropy_coef"] = float(
        os.environ.get(
            "V4D_LOW_ENTROPY_COEF",
            "0.0005",
        )
    )
    cfg["algorithm"]["learning_rate"] = float(
        os.environ.get(
            "V4D_LOW_LEARNING_RATE",
            "0.000001",
        )
    )
    cfg["algorithm"]["num_learning_epochs"] = int(
        os.environ.get(
            "V4D_LOW_EPOCHS",
            "2",
        )
    )
    cfg["algorithm"]["num_mini_batches"] = int(
        os.environ.get(
            "V4D_LOW_MINI_BATCHES",
            "4",
        )
    )
    cfg["algorithm"]["desired_kl"] = float(
        os.environ.get(
            "V4D_LOW_DESIRED_KL",
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

    result = torch.nn.Module.load_state_dict(
        module,
        compatible,
        strict=False,
    )

    std_reset = set_action_std(
        module,
        requested_std,
    )

    runner.current_learning_iteration = 0

    print(
        "[V4D_WARM_START] "
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


def warm_start_expanded_low_policy(
    *,
    runner: Any,
    checkpoint: Path,
    requested_std: float,
) -> None:
    """Warm start the 62-D policy from the accepted 48-D locomotion actor.

    Exact-shape tensors are copied directly. For the actor and critic input
    layers, the original 48 observation columns are copied and the new
    pose/push context columns are initialized to zero. At iteration zero,
    the policy therefore reproduces the old velocity-conditioned behavior
    whenever the teacher reference is present, while retaining trainable
    weights for the new direct pose-conditioned inputs.
    """

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
            f"Could not locate low-level state dictionary in {checkpoint}"
        )

    module = policy_module(runner)
    target_state = module.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    expanded_keys: list[str] = []

    def is_noise_parameter(name: str) -> bool:
        return name.rsplit(".", 1)[-1] in {
            "std",
            "log_std",
            "_std",
            "_log_std",
        }

    for key, source_value in source_state.items():
        if key not in target_state or is_noise_parameter(key):
            continue

        target_value = target_state[key]

        if target_value.shape == source_value.shape:
            compatible[key] = source_value
            continue

        if (
            target_value.ndim == 2
            and source_value.ndim == 2
            and target_value.shape[0] == source_value.shape[0]
            and source_value.shape[1]
            == HighLevelSoccerVecEnv.NATIVE_LOW_OBS_DIM
            and target_value.shape[1] >= 62
        ):
            source_on_target = source_value.to(target_value.device)
            expanded = torch.zeros_like(target_value)
            expanded[:, : source_value.shape[1]] = source_on_target

            # Remove the old command pathway from the student network.
            expanded[:, HighLevelSoccerVecEnv.NATIVE_COMMAND_SLICE] = 0.0

            # Seed direct delta-pose context from the accepted velocity-response
            # weights. This is weight initialization, not an execution-time
            # pose-to-velocity controller.
            expanded[:, HighLevelSoccerVecEnv.CONTEXT_ERROR_X] = source_on_target[:, 9]
            expanded[:, HighLevelSoccerVecEnv.CONTEXT_ERROR_Y] = source_on_target[:, 10]
            expanded[:, HighLevelSoccerVecEnv.CONTEXT_SIN_YAW] = source_on_target[:, 11]
            expanded[:, HighLevelSoccerVecEnv.CONTEXT_TERMINAL_VX] = 0.35 * source_on_target[:, 9]
            expanded[:, HighLevelSoccerVecEnv.CONTEXT_TERMINAL_VY] = 0.35 * source_on_target[:, 10]
            expanded[:, HighLevelSoccerVecEnv.CONTEXT_TERMINAL_WZ] = 0.35 * source_on_target[:, 11]

            compatible[key] = expanded
            expanded_keys.append(key)

    if len(compatible) < 8:
        raise RuntimeError(
            "Too few compatible low-level parameters. "
            f"count={len(compatible)}"
        )

    result = torch.nn.Module.load_state_dict(
        module,
        compatible,
        strict=False,
    )
    std_reset = set_action_std(
        module,
        requested_std,
    )
    runner.current_learning_iteration = 0

    print(
        "[V4D_LEGACY_EXPANDED_WARM_START] "
        f"checkpoint={checkpoint} "
        f"compatible_parameters={len(compatible)} "
        f"expanded_input_layers={expanded_keys} "
        "legacy_command_columns_zeroed=true "
        "direct_pose_columns_seeded_from_velocity_response=true "
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
                "v4d_staged_joint": True,
            },
        },
        path,
    )



def teacher_scale_for_iteration(
    iteration: int,
    iterations: int,
) -> float:
    """V4D inherits the verified teacher-free V4BC-R1 interface."""

    del iteration, iterations
    return 0.0


def set_optimizer_learning_rate(
    runner: Any,
    value: float,
) -> None:
    for group in runner.alg.optimizer.param_groups:
        group["lr"] = float(value)


def read_stage_schedule(
    iterations: int,
) -> list[dict[str, Any]]:
    stage1 = int(os.environ.get("V4D_STAGE1_ITERATIONS", "0"))
    stage2 = int(os.environ.get("V4D_STAGE2_ITERATIONS", "0"))
    stage3 = int(os.environ.get("V4D_STAGE3_ITERATIONS", "0"))

    if stage1 + stage2 + stage3 != iterations:
        raise RuntimeError(
            "V4D stage counts must sum to max_iterations. "
            f"stage1={stage1} stage2={stage2} stage3={stage3} "
            f"iterations={iterations}"
        )
    if min(stage1, stage2, stage3) <= 0:
        raise RuntimeError(
            "Every V4D training stage must contain at least one iteration."
        )

    return [
        {
            "name": "high_adapter",
            "start": 0,
            "end": stage1,
            "high_lr": float(os.environ.get("V4D_HIGH_LR_STAGE1", "0.00002")),
            "low_lr": 0.0,
            "low_trainable": False,
        },
        {
            "name": "joint_finetune",
            "start": stage1,
            "end": stage1 + stage2,
            "high_lr": float(os.environ.get("V4D_HIGH_LR_STAGE2", "0.00001")),
            "low_lr": float(os.environ.get("V4D_LOW_LR_STAGE2", "0.000001")),
            "low_trainable": True,
        },
        {
            "name": "stabilization",
            "start": stage1 + stage2,
            "end": iterations,
            "high_lr": float(os.environ.get("V4D_HIGH_LR_STAGE3", "0.000005")),
            "low_lr": 0.0,
            "low_trainable": False,
        },
    ]


def stage_for_iteration(
    schedule: list[dict[str, Any]],
    iteration: int,
) -> dict[str, Any]:
    for stage in schedule:
        if int(stage["start"]) <= iteration < int(stage["end"]):
            return stage
    raise RuntimeError(f"No V4D stage contains iteration {iteration}.")




def run_v4d_training(
    *,
    high_runner: Any,
    low_runner: Any,
    env: HighLevelSoccerVecEnv,
    high_adapter: V4DHighLevelEnvAdapter,
    low_adapter: PoseLowLevelEnvAdapter,
    iterations: int,
    log_dir: Path,
) -> None:
    """Train the zero-sum high level and execution-quality low level separately."""

    schedule = read_stage_schedule(iterations)
    high_before = snapshot_policy(high_runner)
    low_before = snapshot_policy(low_runner)

    high_observations = high_adapter.get_observations().to(
        high_runner.device
    )
    low_observations = low_adapter.get_observations().to(
        low_runner.device
    )

    records: list[dict[str, Any]] = []
    reward_support: set[float] = set()
    stage_deltas: dict[str, dict[str, dict[str, float]]] = {}

    active_stage_name: str | None = None
    stage_high_before: dict[str, torch.Tensor] | None = None
    stage_low_before: dict[str, torch.Tensor] | None = None

    def close_stage(stage_name: str) -> None:
        nonlocal stage_high_before, stage_low_before
        if stage_high_before is None or stage_low_before is None:
            raise RuntimeError("V4D stage snapshot is missing.")
        high_delta = parameter_delta(stage_high_before, high_runner)
        low_delta = parameter_delta(stage_low_before, low_runner)
        stage_deltas[stage_name] = {
            "high": high_delta,
            "low": low_delta,
        }
        print(
            "[V4D_STAGE_DELTA] "
            f"stage={stage_name} "
            f"high_l2={high_delta['l2']:.9f} "
            f"high_max_abs={high_delta['max_abs']:.9f} "
            f"low_l2={low_delta['l2']:.9f} "
            f"low_max_abs={low_delta['max_abs']:.9f}",
            flush=True,
        )

    for iteration in range(iterations):
        stage = stage_for_iteration(schedule, iteration)
        stage_name = str(stage["name"])
        low_trainable = bool(stage["low_trainable"])

        if stage_name != active_stage_name:
            if active_stage_name is not None:
                close_stage(active_stage_name)
            active_stage_name = stage_name
            stage_high_before = snapshot_policy(high_runner)
            stage_low_before = snapshot_policy(low_runner)
            print(
                "[V4D_STAGE_START] "
                f"stage={stage_name} "
                f"iteration={iteration} "
                f"high_lr={float(stage['high_lr']):.9g} "
                f"low_lr={float(stage['low_lr']):.9g} "
                f"high_trainable=true "
                f"low_trainable={str(low_trainable).lower()}",
                flush=True,
            )

        set_optimizer_learning_rate(
            high_runner,
            float(stage["high_lr"]),
        )
        set_optimizer_learning_rate(
            low_runner,
            float(stage["low_lr"]),
        )

        policy_module(high_runner).train()
        if low_trainable:
            policy_module(low_runner).train()
        else:
            policy_module(low_runner).eval()

        env.set_teacher_scale(0.0)

        aggregate: dict[str, list[float]] = {
            "low_reward": [],
            "pose_error_xy": [],
            "pose_error_yaw": [],
            "upright_error": [],
            "push_stall_fraction": [],
            "robot_speed": [],
            "ball_speed": [],
            "ball_progress": [],
            "action_rate": [],
            "native_reward": [],
            "goal_fraction": [],
            "own_goal_fraction": [],
            "timeout_fraction": [],
            "base_done_fraction": [],
            "ball_goal_progress": [],
            "stand_fraction": [],
            "locomotion_fraction": [],
            "push_fraction": [],
            "high_sparse_reward": [],
            "high_nonzero_reward_fraction": [],
        }
        goal_rates: list[float] = []
        own_goal_rates: list[float] = []
        episode_lengths: list[float] = []

        with torch.inference_mode():
            for _ in range(high_runner.cfg["num_steps_per_env"]):
                high_actions = high_runner.alg.act(
                    high_observations
                )

                skill_diag = env.begin_high_action(
                    high_actions.to(env.device)
                )
                for key, value in skill_diag.items():
                    aggregate[key].append(float(value))

                for low_index in range(
                    env.low_level_steps_per_action
                ):
                    if low_trainable:
                        low_actions = low_runner.alg.act(
                            low_observations
                        )
                    else:
                        low_actions = policy_module(
                            low_runner
                        ).act_inference(low_observations)

                    (
                        low_observations,
                        low_rewards,
                        low_dones,
                        low_extras,
                        low_diag,
                    ) = env.low_step(
                        low_actions.to(env.device),
                        is_last_low_step=(
                            low_index
                            == env.low_level_steps_per_action - 1
                        ),
                    )

                    low_observations = low_observations.to(
                        low_runner.device
                    )
                    low_rewards = low_rewards.to(low_runner.device)
                    low_dones = low_dones.to(low_runner.device)

                    if low_trainable:
                        low_runner.alg.process_env_step(
                            low_observations,
                            low_rewards,
                            low_dones,
                            low_extras,
                        )

                    for key, value in low_diag.items():
                        aggregate_key = (
                            "low_reward" if key == "reward" else key
                        )
                        if aggregate_key not in aggregate:
                            raise RuntimeError(
                                "Unexpected V4D low diagnostic key: "
                                f"{key!r}"
                            )
                        aggregate[aggregate_key].append(float(value))

                high_diag = env.end_high_action()
                high_rewards = high_diag.pop("reward_tensor").to(
                    high_runner.device
                )
                high_dones = high_diag.pop("done_tensor").to(
                    high_runner.device
                )
                high_timeouts = high_diag.pop(
                    "time_outs_tensor"
                ).to(high_runner.device)

                reward_support.update(
                    float(value)
                    for value in torch.unique(high_rewards).tolist()
                )
                if not reward_support.issubset({-1.0, 0.0, 1.0}):
                    raise RuntimeError(
                        "V4D high reward support is not zero-sum: "
                        f"{sorted(reward_support)}"
                    )
                if iteration == 0:
                    print(
                        "[V4D_ZERO_SUM_GUARD] "
                        f"support={sorted(reward_support)} "
                        "goal=+1 own_goal=-1 other=0 "
                        "dense_shaping=false reward_mixing=false",
                        flush=True,
                    )

                next_high_observations = (
                    high_adapter.get_observations().to(
                        high_runner.device
                    )
                )
                high_runner.alg.process_env_step(
                    next_high_observations,
                    high_rewards,
                    high_dones,
                    {"time_outs": high_timeouts},
                )
                high_observations = next_high_observations
                low_observations = (
                    low_adapter.get_observations().to(
                        low_runner.device
                    )
                )

                for key in (
                    "goal_fraction",
                    "own_goal_fraction",
                    "timeout_fraction",
                    "base_done_fraction",
                    "ball_goal_progress",
                    "high_sparse_reward",
                    "high_nonzero_reward_fraction",
                ):
                    aggregate[key].append(float(high_diag[key]))

                if high_diag["goal_rate"] is not None:
                    goal_rates.append(float(high_diag["goal_rate"]))
                if high_diag["own_goal_rate"] is not None:
                    own_goal_rates.append(
                        float(high_diag["own_goal_rate"])
                    )
                if high_diag["episode_length"] is not None:
                    episode_lengths.append(
                        float(high_diag["episode_length"])
                    )

            high_runner.alg.compute_returns(high_observations)
            if low_trainable:
                low_runner.alg.compute_returns(low_observations)

        high_losses = high_runner.alg.update()
        low_losses = (
            low_runner.alg.update()
            if low_trainable
            else None
        )
        high_runner.current_learning_iteration = iteration
        if low_trainable:
            low_runner.current_learning_iteration = iteration

        def average(values: list[float]) -> float | None:
            return sum(values) / len(values) if values else None

        record: dict[str, Any] = {
            "iteration": iteration,
            "stage": stage_name,
            "high_learning_rate": float(stage["high_lr"]),
            "low_learning_rate": float(stage["low_lr"]),
            "high_trainable": True,
            "low_trainable": low_trainable,
            "teacher_scale": 0.0,
            "high_sparse_reward": average(
                aggregate["high_sparse_reward"]
            ),
            "high_nonzero_reward_fraction": average(
                aggregate["high_nonzero_reward_fraction"]
            ),
            "goal_rate": average(goal_rates),
            "own_goal_rate": average(own_goal_rates),
            "episode_length": average(episode_lengths),
            "goal_fraction": average(aggregate["goal_fraction"]),
            "own_goal_fraction": average(
                aggregate["own_goal_fraction"]
            ),
            "timeout_fraction": average(
                aggregate["timeout_fraction"]
            ),
            "base_done_fraction": average(
                aggregate["base_done_fraction"]
            ),
            "ball_goal_progress": average(
                aggregate["ball_goal_progress"]
            ),
            "low_reward": average(aggregate["low_reward"]),
            "mean_pose_error": average(
                aggregate["pose_error_xy"]
            ),
            "mean_yaw_error": average(
                aggregate["pose_error_yaw"]
            ),
            "upright_error": average(
                aggregate["upright_error"]
            ),
            "push_stall_fraction": average(
                aggregate["push_stall_fraction"]
            ),
            "robot_speed": average(aggregate["robot_speed"]),
            "ball_speed": average(aggregate["ball_speed"]),
            "ball_progress": average(aggregate["ball_progress"]),
            "action_rate": average(aggregate["action_rate"]),
            "native_reward_diagnostic": average(
                aggregate["native_reward"]
            ),
            "stand_fraction": average(
                aggregate["stand_fraction"]
            ),
            "locomotion_fraction": average(
                aggregate["locomotion_fraction"]
            ),
            "push_fraction": average(
                aggregate["push_fraction"]
            ),
            "high_losses": high_losses,
            "low_losses": low_losses,
        }
        records.append(record)

        print(
            "[V4D_JOINT_ITER] "
            f"iteration={iteration}/{iterations} "
            f"stage={stage_name} "
            f"high_lr={record['high_learning_rate']:.9g} "
            f"low_lr={record['low_learning_rate']:.9g} "
            f"low_trainable={str(low_trainable).lower()} "
            f"high_sparse_reward={record['high_sparse_reward']} "
            f"goal_rate={record['goal_rate']} "
            f"own_goal_rate={record['own_goal_rate']} "
            f"low_reward={record['low_reward']} "
            f"robot_speed={record['robot_speed']} "
            f"push_fraction={record['push_fraction']} "
            f"push_stall_fraction={record['push_stall_fraction']} "
            f"ball_progress={record['ball_progress']} "
            f"upright_error={record['upright_error']} "
            f"action_rate={record['action_rate']} "
            f"high_losses={high_losses} "
            f"low_losses={low_losses}",
            flush=True,
        )

        checkpoint_interval = int(
            os.environ.get("V4D_CHECKPOINT_INTERVAL", "50")
        )
        is_stage_end = iteration + 1 == int(stage["end"])
        if (
            iteration == 0
            or iteration == iterations - 1
            or is_stage_end
            or (
                checkpoint_interval > 0
                and (iteration + 1) % checkpoint_interval == 0
            )
        ):
            suffix = (
                f"{iteration}_{stage_name}"
                if is_stage_end
                else str(iteration)
            )
            save_checkpoint(
                runner=high_runner,
                path=log_dir / f"high_model_{suffix}.pt",
                iteration=iteration,
                role=f"v4d_high_{stage_name}",
            )
            save_checkpoint(
                runner=low_runner,
                path=log_dir / f"low_model_{suffix}.pt",
                iteration=iteration,
                role=f"v4d_low_{stage_name}",
            )

    if active_stage_name is None:
        raise RuntimeError("V4D completed no training stage.")
    close_stage(active_stage_name)

    high_delta = parameter_delta(high_before, high_runner)
    low_delta = parameter_delta(low_before, low_runner)

    high_final = log_dir / "high_model_final.pt"
    low_final = log_dir / "low_model_final.pt"
    save_checkpoint(
        runner=high_runner,
        path=high_final,
        iteration=iterations - 1,
        role="v4d_zero_sum_high_final",
    )
    save_checkpoint(
        runner=low_runner,
        path=low_final,
        iteration=iterations - 1,
        role="v4d_direct_pose_low_final",
    )

    summary = {
        "version": "v4d_staged_zero_sum_joint",
        "high_policy": {
            "source": "V3 zero-sum checkpoint",
            "trainable": True,
            "reward": "goal_plus1_own_goal_minus1_other0",
            "parameter_delta": high_delta,
        },
        "low_policy": {
            "source": "V4D-R1 teacher-free checkpoint",
            "trainable_only_in_stage2": True,
            "reward": "pose_terminal_velocity_posture_smoothness_push_quality",
            "action": "12_joint_position_residuals",
            "parameter_delta": low_delta,
        },
        "reward_separation": {
            "high_receives_low_reward": False,
            "low_receives_goal_reward": False,
            "high_and_low_optimizers": "separate",
        },
        "router": "unchanged_v4bc_r1_stand_locomotion_push",
        "delta_pose": "unchanged_direct_pose_context",
        "kick_included": False,
        "teacher_reference": {
            "actor_observation": "never",
            "ppo_rollout_scale": 0.0,
        },
        "stage_schedule": schedule,
        "stage_parameter_deltas": stage_deltas,
        "high_reward_support": sorted(reward_support),
        "records": records,
    }
    summary_path = log_dir / "v4d_joint_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    contract = {
        "version": "v4d_staged_zero_sum_joint",
        "high_level": {
            "input": "17_dim_soccer_state",
            "action": "3_dim_continuous_intent",
            "reward": {"goal": 1, "own_goal": -1, "other": 0},
            "trainable": True,
        },
        "router": {
            "skills": ["stand", "locomotion", "push"],
            "kick": None,
            "implementation": "unchanged_from_v4bc_r1",
        },
        "low_level": {
            "input": "teacher_free_62_dim_direct_delta_pose_context",
            "action": "12_joint_position_residuals",
            "reward": "execution_quality_only",
            "trainable": "stage2_only",
        },
        "reward_mixing": False,
        "teacher_free_actor_observation": True,
        "stage_schedule": schedule,
    }
    contract_path = log_dir / "v4d_joint_contract.json"
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(
        "[V4D_PARAMETER_DELTA] "
        f"high_l2={high_delta['l2']:.9f} "
        f"high_max_abs={high_delta['max_abs']:.9f} "
        f"low_l2={low_delta['l2']:.9f} "
        f"low_max_abs={low_delta['max_abs']:.9f}",
        flush=True,
    )

    if high_delta["max_abs"] <= 0.0:
        raise RuntimeError("V4D high-level parameters did not update.")
    if low_delta["max_abs"] <= 0.0:
        raise RuntimeError("V4D low-level parameters did not update.")
    if not reward_support.issubset({-1.0, 0.0, 1.0}):
        raise RuntimeError(
            f"V4D zero-sum support guard failed: {sorted(reward_support)}"
        )

    stage1_delta = stage_deltas["high_adapter"]
    stage2_delta = stage_deltas["joint_finetune"]
    stage3_delta = stage_deltas["stabilization"]
    if stage1_delta["low"]["max_abs"] != 0.0:
        raise RuntimeError("V4D stage1 changed the frozen low policy.")
    if stage2_delta["high"]["max_abs"] <= 0.0:
        raise RuntimeError("V4D stage2 did not update the high policy.")
    if stage2_delta["low"]["max_abs"] <= 0.0:
        raise RuntimeError("V4D stage2 did not update the low policy.")
    if stage3_delta["low"]["max_abs"] != 0.0:
        raise RuntimeError("V4D stage3 changed the frozen low policy.")

    final_window = records[-min(20, len(records)):]
    requested_mean = sum(
        float(record["locomotion_fraction"] or 0.0)
        + float(record["push_fraction"] or 0.0)
        for record in final_window
    ) / len(final_window)
    speed_mean = sum(
        float(record["robot_speed"] or 0.0)
        for record in final_window
    ) / len(final_window)
    upright_mean = sum(
        float(record["upright_error"] or 0.0)
        for record in final_window
    ) / len(final_window)
    action_rate_mean = sum(
        float(record["action_rate"] or 0.0)
        for record in final_window
    ) / len(final_window)
    push_fraction_mean = sum(
        float(record["push_fraction"] or 0.0)
        for record in final_window
    ) / len(final_window)
    push_stall_mean = sum(
        float(record["push_stall_fraction"] or 0.0)
        for record in final_window
    ) / len(final_window)
    ball_progress_mean = sum(
        float(record["ball_progress"] or 0.0)
        for record in final_window
    ) / len(final_window)

    final_goal_rates = [
        float(record["goal_rate"])
        for record in final_window
        if record["goal_rate"] is not None
    ]
    final_own_goal_rates = [
        float(record["own_goal_rate"])
        for record in final_window
        if record["own_goal_rate"] is not None
    ]
    goal_rate_mean = (
        sum(final_goal_rates) / len(final_goal_rates)
        if final_goal_rates else None
    )
    own_goal_rate_mean = (
        sum(final_own_goal_rates) / len(final_own_goal_rates)
        if final_own_goal_rates else None
    )

    print(
        "[V4D_FINAL_GUARD] "
        f"requested_fraction={requested_mean:.6f} "
        f"robot_speed={speed_mean:.6f} "
        f"upright_error={upright_mean:.6f} "
        f"action_rate={action_rate_mean:.6f} "
        f"push_fraction={push_fraction_mean:.6f} "
        f"push_stall_fraction={push_stall_mean:.6f} "
        f"ball_progress={ball_progress_mean:.6f} "
        f"goal_rate={goal_rate_mean} "
        f"own_goal_rate={own_goal_rate_mean}",
        flush=True,
    )

    if requested_mean >= 0.25 and speed_mean < float(
        os.environ.get("V4D_MIN_ROBOT_SPEED", "0.05")
    ):
        raise RuntimeError("V4D motion guard failed.")
    if upright_mean > float(
        os.environ.get("V4D_MAX_UPRIGHT_ERROR", "0.03")
    ):
        raise RuntimeError("V4D upright-quality guard failed.")
    if action_rate_mean > float(
        os.environ.get("V4D_MAX_ACTION_RATE", "0.5")
    ):
        raise RuntimeError("V4D action-rate guard failed.")
    if push_stall_mean > float(
        os.environ.get("V4D_MAX_PUSH_STALL", "0.05")
    ):
        raise RuntimeError("V4D push-stall guard failed.")
    if push_fraction_mean >= 0.10 and ball_progress_mean <= 0.01:
        raise RuntimeError("V4D push progress guard failed.")

    minimum_goal_rate = float(
        os.environ.get("V4D_MIN_FINAL_GOAL_RATE", "0.0")
    )
    maximum_own_goal_rate = float(
        os.environ.get("V4D_MAX_FINAL_OWN_GOAL_RATE", "0.02")
    )
    if minimum_goal_rate > 0.0:
        if goal_rate_mean is None or goal_rate_mean < minimum_goal_rate:
            raise RuntimeError(
                "V4D final goal-rate guard failed. "
                f"required={minimum_goal_rate} observed={goal_rate_mean}"
            )
    if (
        own_goal_rate_mean is not None
        and own_goal_rate_mean > maximum_own_goal_rate
    ):
        raise RuntimeError("V4D own-goal guard failed.")

    (log_dir / "V4D_JOINT_OK").touch()
    print(f"V4D_HIGH_CHECKPOINT={high_final}", flush=True)
    print(f"V4D_LOW_CHECKPOINT={low_final}", flush=True)
    print(f"V4D_SUMMARY={summary_path}", flush=True)
    print(f"V4D_CONTRACT={contract_path}", flush=True)
    print("V4D_JOINT_TRAINING_COMPLETE", flush=True)




def run_v4d_evaluation(
    *,
    high_runner: Any,
    low_runner: Any,
    env: HighLevelSoccerVecEnv,
    video_path: Path,
    trajectory_path: Path,
    summary_path: Path,
    steps: int,
    fps: int,
) -> None:
    import csv
    import imageio.v2 as imageio

    if env.num_envs != 1:
        raise RuntimeError(
            "V4D local evaluation requires --num_envs 1."
        )

    policy_module(high_runner).eval()
    policy_module(low_runner).eval()
    env.set_teacher_scale(0.0)
    env.reset()

    high_adapter = V4DHighLevelEnvAdapter(env)
    low_adapter = PoseLowLevelEnvAdapter(env)

    high_obs = high_adapter.get_observations().to(
        high_runner.device
    )
    low_obs = low_adapter.get_observations().to(
        low_runner.device
    )

    render_env: Any = env.low_env
    chain = [type(render_env).__name__]
    seen: set[int] = set()

    while not callable(getattr(render_env, "render", None)):
        object_id = id(render_env)
        if object_id in seen:
            raise RuntimeError(
                f"Render unwrap cycle: {chain}"
            )
        seen.add(object_id)

        next_env = getattr(render_env, "env", None)
        if next_env is None or next_env is render_env:
            next_env = getattr(
                render_env,
                "unwrapped",
                None,
            )
        if next_env is None or next_env is render_env:
            raise RuntimeError(
                f"No render-capable environment: {chain}"
            )
        render_env = next_env
        chain.append(type(render_env).__name__)

    print(
        "[V4D_EVAL] "
        f"render_chain={' -> '.join(chain)}",
        flush=True,
    )

    video_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    trajectory_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "high_step",
        "sim_time_s",
        "skill",
        "goal",
        "own_goal",
        "done",
        "robot_x",
        "robot_y",
        "robot_yaw",
        "ball_x",
        "ball_y",
        "ball_vx",
        "ball_vy",
        "robot_ball_distance",
        "pose_error_xy",
        "pose_error_yaw",
        "upright_error",
        "push_stall",
        "reference_vx",
        "reference_vy",
        "reference_wz",
        "high_action_0",
        "high_action_1",
        "high_action_2",
    ]

    rows: list[dict[str, Any]] = []
    goal_count = 0
    own_goal_count = 0
    done_count = 0
    push_stall_count = 0
    skill_counts = {
        name: 0
        for name in env.SKILL_NAMES
    }

    writer = imageio.get_writer(
        video_path,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
    )

    try:
        with torch.no_grad():
            for high_step in range(steps):
                high_action = policy_module(
                    high_runner
                ).act_inference(high_obs)

                env.begin_high_action(
                    high_action.to(env.device)
                )

                for low_index in range(
                    env.low_level_steps_per_action
                ):
                    low_action = policy_module(
                        low_runner
                    ).act_inference(low_obs)

                    (
                        low_obs,
                        _,
                        _,
                        _,
                        _,
                    ) = env.low_step(
                        low_action.to(env.device),
                        is_last_low_step=(
                            low_index
                            == env.low_level_steps_per_action - 1
                        ),
                    )
                    low_obs = low_obs.to(
                        low_runner.device
                    )

                high_diag = env.end_high_action()
                high_obs = (
                    high_adapter.get_observations().to(
                        high_runner.device
                    )
                )
                low_obs = (
                    low_adapter.get_observations().to(
                        low_runner.device
                    )
                )

                frame = render_env.render()
                if isinstance(frame, list):
                    frame = frame[0]
                if frame is None:
                    raise RuntimeError(
                        "render() returned no frame."
                    )
                writer.append_data(np.asarray(frame))

                state = env._state()
                skill_name = env.SKILL_NAMES[
                    int(env.skill_id[0])
                ]
                skill_counts[skill_name] += 1

                goal = (
                    high_diag["goal_fraction"] > 0.5
                )
                own_goal = (
                    high_diag["own_goal_fraction"] > 0.5
                )
                done = (
                    goal
                    or own_goal
                    or high_diag["timeout_fraction"] > 0.5
                    or high_diag["base_done_fraction"] > 0.5
                )

                goal_count += int(goal)
                own_goal_count += int(own_goal)
                done_count += int(done)
                push_stall_count += int(
                    bool(env.last_push_stall[0])
                )

                rows.append(
                    {
                        "high_step": high_step,
                        "sim_time_s": (
                            high_step
                            * env.low_level_steps_per_action
                            * 0.02
                        ),
                        "skill": skill_name,
                        "goal": int(goal),
                        "own_goal": int(own_goal),
                        "done": int(done),
                        "robot_x": float(
                            state["robot_position"][0, 0]
                        ),
                        "robot_y": float(
                            state["robot_position"][0, 1]
                        ),
                        "robot_yaw": float(
                            state["robot_yaw"][0]
                        ),
                        "ball_x": float(
                            state["ball_position"][0, 0]
                        ),
                        "ball_y": float(
                            state["ball_position"][0, 1]
                        ),
                        "ball_vx": float(
                            state["ball_velocity_world"][0, 0]
                        ),
                        "ball_vy": float(
                            state["ball_velocity_world"][0, 1]
                        ),
                        "robot_ball_distance": float(
                            state["robot_ball_distance"][0]
                        ),
                        "pose_error_xy": float(
                            env.last_pose_error_xy[0]
                        ),
                        "pose_error_yaw": float(
                            env.last_pose_error_yaw[0]
                        ),
                        "upright_error": float(
                            env.last_upright_error[0]
                        ),
                        "push_stall": int(
                            bool(env.last_push_stall[0])
                        ),
                        "reference_vx": float(
                            env.last_reference_command[0, 0]
                        ),
                        "reference_vy": float(
                            env.last_reference_command[0, 1]
                        ),
                        "reference_wz": float(
                            env.last_reference_command[0, 2]
                        ),
                        "high_action_0": float(
                            high_action[0, 0]
                        ),
                        "high_action_1": float(
                            high_action[0, 1]
                        ),
                        "high_action_2": float(
                            high_action[0, 2]
                        ),
                    }
                )

                if (
                    high_step == 0
                    or (high_step + 1) % 100 == 0
                    or high_step + 1 == steps
                ):
                    print(
                        "[V4D_EVAL] "
                        f"step={high_step + 1}/{steps} "
                        f"skill={skill_name} "
                        f"goal={int(goal)} "
                        f"stall={int(env.last_push_stall[0])}",
                        flush=True,
                    )
    finally:
        writer.close()

    with trajectory_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        csv_writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )
        csv_writer.writeheader()
        csv_writer.writerows(rows)

    summary = {
        "version": "v4d_joint",
        "teacher_scale": 0.0,
        "steps": steps,
        "fps": fps,
        "simulated_seconds": (
            steps
            * env.low_level_steps_per_action
            * 0.02
        ),
        "goal_count": goal_count,
        "own_goal_count": own_goal_count,
        "done_count": done_count,
        "push_stall_count": push_stall_count,
        "skill_counts": skill_counts,
        "video": str(video_path),
        "trajectory": str(trajectory_path),
    }
    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(
        f"V4D_EVAL_VIDEO={video_path}",
        flush=True,
    )
    print(
        f"V4D_EVAL_TRAJECTORY={trajectory_path}",
        flush=True,
    )
    print(
        f"V4D_EVAL_SUMMARY={summary_path}",
        flush=True,
    )
    print("V4D_EVAL_VIDEO_OK", flush=True)


# V4K1_ISOLATED_DELTA_POSE_KICK
class IsolatedDeltaPoseKickEnv(HighLevelSoccerVecEnv):
    """Isolated low-level kick skill trained from the accepted V4D policy.

    The skill keeps the direct planar base Delta-Pose interface.  A short
    phase-conditioned kick command augments the V4D observation, while the
    policy still outputs exactly 12 joint-position residuals.  No high-level
    soccer policy and no goal/own-goal reward is used in this isolated stage.
    """

    SKILL_KICK = 3
    KICK_NUM_SKILLS = 4
    KICK_OBS_DIM = 70

    KICK_CONTEXT_ERROR_X = 48
    KICK_CONTEXT_ERROR_Y = 49
    KICK_CONTEXT_SIN_YAW = 50
    KICK_CONTEXT_COS_YAW = 51
    KICK_CONTEXT_TERMINAL_VX = 52
    KICK_CONTEXT_TERMINAL_VY = 53
    KICK_CONTEXT_TERMINAL_WZ = 54
    KICK_CONTEXT_REMAINING = 55
    KICK_CONTEXT_SKILL_START = 56
    KICK_CONTEXT_BALL_X = 60
    KICK_CONTEXT_BALL_Y = 61
    KICK_CONTEXT_TEACHER_FLAG = 62
    KICK_CONTEXT_PHASE_SIN = 63
    KICK_CONTEXT_PHASE_COS = 64
    KICK_CONTEXT_LEG_START = 65
    KICK_CONTEXT_DIRECTION_X = 67
    KICK_CONTEXT_DIRECTION_Y = 68
    KICK_CONTEXT_STRENGTH = 69

    def __init__(
        self,
        *,
        low_env: RslRlVecEnvWrapper,
        kick_episode_steps: int,
        seed: int,
        fixed_leg: str,
    ) -> None:
        self._kick_initialized = False
        # Preserve the inherited 62-D initialization path until the base class
        # has allocated simulator-facing tensors.
        self.NUM_SKILLS = 3
        self.SKILL_NAMES = ("stand", "locomotion", "push")
        super().__init__(
            low_env=low_env,
            low_level_steps_per_action=1,
            episode_steps=max(1, int(kick_episode_steps)),
            goal_line_x=4.02,
            goal_center_x=4.60,
            goal_half_width=0.78,
            seed=seed,
        )

        self.NUM_SKILLS = self.KICK_NUM_SKILLS
        self.SKILL_NAMES = ("stand", "locomotion", "push", "kick")
        self.kick_episode_steps = int(kick_episode_steps)
        self.max_episode_length = self.kick_episode_steps
        self.max_low_episode_length = self.kick_episode_steps
        self.fixed_leg = fixed_leg
        self.curriculum_stage = "contact_discovery"

        self.kick_leg = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        self.kick_phase = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.strike_strength = torch.ones(
            self.num_envs,
            device=self.device,
        )
        self.strike_direction_world = torch.zeros(
            self.num_envs,
            2,
            device=self.device,
        )
        self.strike_direction_world[:, 0] = 1.0
        self.kick_ball_start_xy = torch.zeros(
            self.num_envs,
            2,
            device=self.device,
        )
        self.kick_stance_xy = torch.zeros(
            self.num_envs,
            2,
            device=self.device,
        )
        self.kick_target_yaw = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.previous_ball_direction_speed = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.peak_ball_direction_speed = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.contact_seen = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self.selected_foot_distance = torch.full(
            (self.num_envs,),
            9.0,
            device=self.device,
        )
        self.other_foot_distance = torch.full(
            (self.num_envs,),
            9.0,
            device=self.device,
        )
        self.selected_foot_height = torch.zeros(
            self.num_envs,
            device=self.device,
        )

        self.front_foot_ids = self._resolve_front_foot_ids()
        self._kick_initialized = True
        self.reset()
        self.low_obs_buf = self._build_low_observation()
        self.low_num_obs = int(self.low_obs_buf.shape[1])

        if self.low_num_obs != self.KICK_OBS_DIM:
            raise RuntimeError(
                "V4K1 kick observation dimension mismatch. "
                f"expected={self.KICK_OBS_DIM} observed={self.low_num_obs}"
            )

        print(
            "[V4K1_BOOT] "
            f"low_obs={self.low_num_obs} "
            f"low_actions={self.low_num_actions} "
            f"episode_steps={self.kick_episode_steps} "
            f"front_foot_ids={self.front_foot_ids}",
            flush=True,
        )

    def _resolve_front_foot_ids(self) -> tuple[int, int]:
        resolved: list[int] = []
        for patterns in (
            ("FL_foot", ".*FL.*foot.*", ".*FL.*FOOT.*"),
            ("FR_foot", ".*FR.*foot.*", ".*FR.*FOOT.*"),
        ):
            body_id = -1
            for pattern in patterns:
                try:
                    ids, _ = self.robot.find_bodies(pattern)
                except Exception:
                    ids = []
                if ids:
                    body_id = int(ids[0])
                    break
            resolved.append(body_id)
        return int(resolved[0]), int(resolved[1])

    def set_curriculum_stage(self, name: str) -> None:
        if name not in {
            "contact_discovery",
            "directional_kick",
            "recovery_generalization",
        }:
            raise ValueError(f"Unknown V4K1 curriculum stage: {name}")
        self.curriculum_stage = name

    def _phase(self) -> torch.Tensor:
        return torch.clamp(
            self.low_episode_length_buf.float()
            / max(1.0, float(self.kick_episode_steps - 1)),
            0.0,
            1.0,
        )

    def _selected_front_foot_state(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        body_pos = getattr(self.robot.data, "body_pos_w", None)
        if not torch.is_tensor(body_pos) or min(self.front_foot_ids) < 0:
            large = torch.full(
                (self.num_envs,),
                9.0,
                device=self.device,
            )
            zero = torch.zeros(
                self.num_envs,
                device=self.device,
            )
            return large, large, zero

        left = body_pos[:, self.front_foot_ids[0], :3]
        right = body_pos[:, self.front_foot_ids[1], :3]
        ball = self.ball.data.root_pos_w[:, :3]
        left_distance = torch.linalg.vector_norm(left - ball, dim=1)
        right_distance = torch.linalg.vector_norm(right - ball, dim=1)
        selected = torch.where(
            self.kick_leg == 0,
            left_distance,
            right_distance,
        )
        other = torch.where(
            self.kick_leg == 0,
            right_distance,
            left_distance,
        )
        selected_z = torch.where(
            self.kick_leg == 0,
            left[:, 2],
            right[:, 2],
        )
        ground_z = self.env_origins[:, 2]
        height = torch.clamp(selected_z - ground_z, min=0.0)
        return selected, other, height

    def _update_kick_target(self) -> None:
        self.kick_phase = self._phase()
        phase = self.kick_phase

        # The base remains anchored behind the ball during wind-up, advances
        # only slightly during impact, and returns to a stable recovery pose.
        strike_ramp = torch.clamp((phase - 0.42) / 0.16, 0.0, 1.0)
        recovery_ramp = torch.clamp((phase - 0.68) / 0.22, 0.0, 1.0)
        forward_shift = (
            0.075 * strike_ramp * (1.0 - recovery_ramp)
            + 0.025 * recovery_ramp
        )
        forward_shift = forward_shift * self.strike_strength

        self.target_x_world = (
            self.kick_stance_xy[:, 0]
            + forward_shift * self.strike_direction_world[:, 0]
        )
        self.target_y_world = (
            self.kick_stance_xy[:, 1]
            + forward_shift * self.strike_direction_world[:, 1]
        )
        self.target_yaw_world = self.kick_target_yaw
        self.target_terminal_velocity.zero_()
        strike_window = (phase >= 0.42) & (phase <= 0.68)
        self.target_terminal_velocity[strike_window, 0] = (
            0.10 * self.strike_strength[strike_window]
        )
        self.target_elapsed_steps = self.low_episode_length_buf.clone()
        self.target_duration_steps.fill_(self.kick_episode_steps)

    def _build_low_observation(self) -> torch.Tensor:
        if not getattr(self, "_kick_initialized", False):
            return HighLevelSoccerVecEnv._build_low_observation(self)

        native = extract_observation(self.low_env.get_observations())
        if native.shape != (self.num_envs, self.NATIVE_LOW_OBS_DIM):
            raise RuntimeError(
                "Native V4K1 observation shape changed. "
                f"received={tuple(native.shape)}"
            )
        native = native.clone()
        native[:, self.NATIVE_COMMAND_SLICE] = 0.0
        self._native_low_obs = native

        self._update_kick_target()
        error_x, error_y, error_yaw = self._pose_error()
        state = self._state()
        one_hot_skill = torch.nn.functional.one_hot(
            self.skill_id,
            num_classes=self.KICK_NUM_SKILLS,
        ).float()
        one_hot_leg = torch.nn.functional.one_hot(
            self.kick_leg,
            num_classes=2,
        ).float()

        direction_body_x, direction_body_y = world_to_body(
            self.strike_direction_world[:, 0],
            self.strike_direction_world[:, 1],
            state["robot_yaw"],
        )
        phase_angle = 2.0 * math.pi * self.kick_phase
        remaining = 1.0 - self.kick_phase

        context = torch.cat(
            (
                (error_x / 0.60).unsqueeze(1),
                (error_y / 0.40).unsqueeze(1),
                torch.sin(error_yaw).unsqueeze(1),
                torch.cos(error_yaw).unsqueeze(1),
                self.target_terminal_velocity[:, 0:1] / 0.70,
                self.target_terminal_velocity[:, 1:2] / 0.40,
                self.target_terminal_velocity[:, 2:3] / 1.00,
                remaining.unsqueeze(1),
                one_hot_skill,
                state["robot_ball_x_body"].unsqueeze(1) / 1.0,
                state["robot_ball_y_body"].unsqueeze(1) / 0.6,
                torch.zeros((self.num_envs, 1), device=self.device),
                torch.sin(phase_angle).unsqueeze(1),
                torch.cos(phase_angle).unsqueeze(1),
                one_hot_leg,
                direction_body_x.unsqueeze(1),
                direction_body_y.unsqueeze(1),
                self.strike_strength.unsqueeze(1),
            ),
            dim=1,
        )
        observation = torch.cat((native, context), dim=1)
        if observation.shape[1] != self.KICK_OBS_DIM:
            raise RuntimeError(
                "V4K1 low observation assembly failed. "
                f"shape={tuple(observation.shape)}"
            )
        if not torch.isfinite(observation).all():
            raise RuntimeError("V4K1 observations contain non-finite values.")
        return observation

    def _kick_reward(
        self,
        *,
        before_state: dict[str, torch.Tensor],
        after_state: dict[str, torch.Tensor],
        low_actions: torch.Tensor,
        base_done: torch.Tensor,
        terminal: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._update_kick_target()
        error_x, error_y, error_yaw = self._pose_error()
        error_xy = torch.sqrt(error_x.square() + error_y.square() + 1.0e-8)

        pose_track = torch.exp(-error_xy.square() / (0.24 ** 2))
        yaw_track = torch.exp(-error_yaw.square() / (0.28 ** 2))
        projected_gravity_xy = self._native_low_obs[:, 6:8]
        upright_error = projected_gravity_xy.square().sum(dim=1)
        upright_reward = torch.exp(-upright_error / (0.18 ** 2))
        action_rate = (
            low_actions - self.previous_low_action
        ).square().mean(dim=1)
        torque_penalty = self._applied_torque_penalty()

        ball_velocity = after_state["ball_velocity_world"][:, :2]
        direction = self.strike_direction_world
        direction_speed = (ball_velocity * direction).sum(dim=1)
        lateral = torch.stack((-direction[:, 1], direction[:, 0]), dim=1)
        lateral_speed = torch.abs((ball_velocity * lateral).sum(dim=1))
        displacement_vec = (
            after_state["ball_position"][:, :2] - self.kick_ball_start_xy
        )
        directional_displacement = (displacement_vec * direction).sum(dim=1)
        lateral_displacement = torch.abs((displacement_vec * lateral).sum(dim=1))
        positive_impulse = torch.relu(
            direction_speed - self.previous_ball_direction_speed
        )
        self.peak_ball_direction_speed = torch.maximum(
            self.peak_ball_direction_speed,
            direction_speed,
        )

        selected_distance, other_distance, selected_height = (
            self._selected_front_foot_state()
        )
        self.selected_foot_distance = selected_distance
        self.other_foot_distance = other_distance
        self.selected_foot_height = selected_height

        phase = self.kick_phase
        windup = ((phase >= 0.18) & (phase < 0.43)).float()
        strike = ((phase >= 0.38) & (phase <= 0.72)).float()
        recovery = (phase > 0.72).float()
        early = (phase < 0.30).float()

        foot_proximity = torch.exp(
            -selected_distance.square() / (0.16 ** 2)
        )
        wrong_foot_closer = (other_distance + 0.03 < selected_distance).float()
        foot_lift = torch.exp(
            -(selected_height - 0.10).square() / (0.09 ** 2)
        )
        contact_now = (
            (selected_distance < 0.22)
            & (direction_speed > 0.08)
            & (phase >= 0.30)
        )
        self.contact_seen |= contact_now

        body_crowding = torch.relu(
            0.31 - after_state["robot_ball_distance"]
        ) / 0.31
        early_ball_motion = early * torch.relu(direction_speed - 0.05)

        reward = (
            0.30 * pose_track
            + 0.18 * yaw_track
            + 0.48 * upright_reward
            - 0.020 * action_rate
            - 0.00002 * torque_penalty
            + 0.18 * windup * foot_lift
            + 0.34 * strike * foot_proximity
            + 0.95 * torch.clamp(direction_speed / 1.20, 0.0, 1.5)
            + 1.20 * torch.clamp(positive_impulse / 0.45, 0.0, 1.5)
            + 0.42 * torch.clamp(directional_displacement / 0.80, 0.0, 1.0)
            - 0.28 * torch.clamp(lateral_speed / 0.50, 0.0, 1.0)
            - 0.20 * torch.clamp(lateral_displacement / 0.45, 0.0, 1.0)
            - 0.16 * strike * wrong_foot_closer
            - 0.55 * body_crowding
            - 0.75 * early_ball_motion
            + 0.24 * recovery * upright_reward
        )

        success = (
            terminal
            & self.contact_seen
            & (self.peak_ball_direction_speed > 0.45)
            & (directional_displacement > 0.45)
            & (upright_error < 0.03)
        )
        terminal_failure = terminal & ~success
        reward = reward + 4.0 * success.float() - 0.75 * terminal_failure.float()
        reward[base_done] = -2.5
        reward = torch.clamp(reward, -4.0, 6.0)

        diagnostics = {
            "pose_error_xy": error_xy,
            "pose_error_yaw": error_yaw.abs(),
            "upright_error": upright_error,
            "action_rate": action_rate,
            "direction_speed": direction_speed,
            "peak_direction_speed": self.peak_ball_direction_speed,
            "directional_displacement": directional_displacement,
            "lateral_displacement": lateral_displacement,
            "selected_foot_distance": selected_distance,
            "other_foot_distance": other_distance,
            "selected_foot_height": selected_height,
            "contact_seen": self.contact_seen.float(),
            "success": success.float(),
        }
        return reward, diagnostics

    @torch.inference_mode(False)
    def kick_step(
        self,
        low_actions: torch.Tensor,
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, Any], dict[str, float]]:
        if low_actions.ndim == 1 and self.num_envs == 1:
            low_actions = low_actions.unsqueeze(0)
        expected = (self.num_envs, self.LOW_ACTION_DIM)
        if low_actions.shape != expected:
            raise RuntimeError(
                "V4K1 action shape mismatch. "
                f"expected={expected} received={tuple(low_actions.shape)}"
            )
        low_actions = torch.clamp(low_actions.to(self.device), -1.0, 1.0)
        self.command_tensor[:, :3] = 0.0
        self._update_kick_target()

        before_state = self._state()
        result = self.low_env.step(low_actions.detach().clone())
        native_obs, native_reward, native_done, native_extras = normalize_step_result(result)
        base_done = normalize_done_tensor(
            native_done,
            num_envs=self.num_envs,
            device=self.device,
        )
        self._native_low_obs = extract_observation(native_obs)
        self._native_low_obs[:, self.NATIVE_COMMAND_SLICE] = 0.0

        self.low_episode_length_buf += 1
        terminal = self.low_episode_length_buf >= self.kick_episode_steps
        after_state = self._state()
        reward, tensors = self._kick_reward(
            before_state=before_state,
            after_state=after_state,
            low_actions=low_actions,
            base_done=base_done,
            terminal=terminal,
        )
        done = base_done | terminal
        timeouts = terminal & ~base_done

        completed = int(done.sum())
        successful = int((tensors["success"] > 0.5).sum())
        contact_successes = int((done & self.contact_seen).sum())
        completed_peak_sum = float(
            tensors["peak_direction_speed"][done].sum()
        ) if completed else 0.0
        completed_displacement_sum = float(
            tensors["directional_displacement"][done].sum()
        ) if completed else 0.0

        self.previous_ball_direction_speed = tensors["direction_speed"].clone()
        self.previous_low_action = low_actions
        self.last_pose_error_xy = tensors["pose_error_xy"]
        self.last_pose_error_yaw = tensors["pose_error_yaw"]
        self.last_upright_error = tensors["upright_error"]
        self.last_low_reward = reward

        extras = dict(native_extras) if isinstance(native_extras, dict) else {}
        extras["time_outs"] = timeouts

        diagnostics = {
            "reward": float(reward.mean()),
            "pose_error_xy": float(tensors["pose_error_xy"].mean()),
            "pose_error_yaw": float(tensors["pose_error_yaw"].mean()),
            "upright_error": float(tensors["upright_error"].mean()),
            "action_rate": float(tensors["action_rate"].mean()),
            "direction_speed": float(tensors["direction_speed"].mean()),
            "peak_direction_speed": float(tensors["peak_direction_speed"].mean()),
            "directional_displacement": float(tensors["directional_displacement"].mean()),
            "lateral_displacement": float(tensors["lateral_displacement"].mean()),
            "selected_foot_distance": float(tensors["selected_foot_distance"].mean()),
            "selected_foot_height": float(tensors["selected_foot_height"].mean()),
            "contact_fraction": float(tensors["contact_seen"].mean()),
            "completed_episodes": float(completed),
            "successful_episodes": float(successful),
            "contact_successes": float(contact_successes),
            "completed_peak_speed_sum": completed_peak_sum,
            "completed_displacement_sum": completed_displacement_sum,
            "base_done_fraction": float(base_done.float().mean()),
            "native_reward": float(torch.as_tensor(native_reward, device=self.device).float().mean()),
        }

        if done.any():
            reset_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
            self._reset_idx(reset_ids)

        self.low_obs_buf = self._build_low_observation()
        packed = TensorDict(
            {"policy": self.low_obs_buf},
            batch_size=[self.num_envs],
            device=self.device,
        )
        return packed, reward, done, extras, diagnostics

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        if not getattr(self, "_kick_initialized", False):
            HighLevelSoccerVecEnv._reset_idx(self, env_ids)
            return
        if env_ids.numel() == 0:
            return
        if not hasattr(self.base_env, "_reset_idx"):
            raise RuntimeError("ManagerBased environment has no _reset_idx method.")
        self.base_env._reset_idx(env_ids)

        count = int(env_ids.numel())
        origins = self.env_origins[env_ids]
        if self.curriculum_stage == "contact_discovery":
            direction_limit = 0.05
            ball_y_limit = 0.025
            distance_min, distance_max = 0.40, 0.44
            position_noise = 0.015
            yaw_noise = 0.04
            strength_min, strength_max = 0.70, 0.85
        elif self.curriculum_stage == "directional_kick":
            direction_limit = 0.32
            ball_y_limit = 0.08
            distance_min, distance_max = 0.38, 0.47
            position_noise = 0.035
            yaw_noise = 0.10
            strength_min, strength_max = 0.65, 1.00
        else:
            direction_limit = 0.48
            ball_y_limit = 0.14
            distance_min, distance_max = 0.36, 0.50
            position_noise = 0.060
            yaw_noise = 0.18
            strength_min, strength_max = 0.55, 1.00

        target_angle = (
            2.0 * self._rand((count,)) - 1.0
        ) * direction_limit
        direction = torch.stack(
            (torch.cos(target_angle), torch.sin(target_angle)),
            dim=1,
        )
        ball_local_x = 0.65 + 0.15 * self._rand((count,))
        ball_local_y = (
            2.0 * self._rand((count,)) - 1.0
        ) * ball_y_limit
        ball_xy = torch.stack(
            (origins[:, 0] + ball_local_x, origins[:, 1] + ball_local_y),
            dim=1,
        )
        stance_distance = distance_min + (
            distance_max - distance_min
        ) * self._rand((count,))
        lateral = torch.stack((-direction[:, 1], direction[:, 0]), dim=1)
        stance_xy = ball_xy - stance_distance.unsqueeze(1) * direction
        stance_xy = stance_xy + (
            2.0 * self._rand((count, 1)) - 1.0
        ) * position_noise * lateral
        robot_xy = stance_xy + (
            2.0 * self._rand((count, 2)) - 1.0
        ) * position_noise
        robot_yaw = target_angle + (
            2.0 * self._rand((count,)) - 1.0
        ) * yaw_noise

        robot_z = self.robot.data.root_pos_w[env_ids, 2]
        ball_z = self.ball.data.root_pos_w[env_ids, 2]
        robot_pose = torch.zeros(count, 7, device=self.device)
        robot_pose[:, 0:2] = robot_xy
        robot_pose[:, 2] = robot_z
        robot_pose[:, 3] = torch.cos(robot_yaw / 2.0)
        robot_pose[:, 6] = torch.sin(robot_yaw / 2.0)
        ball_pose = torch.zeros(count, 7, device=self.device)
        ball_pose[:, 0:2] = ball_xy
        ball_pose[:, 2] = ball_z
        ball_pose[:, 3] = 1.0
        zero_velocity = torch.zeros(count, 6, device=self.device)
        self.robot.write_root_pose_to_sim(robot_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)
        self.ball.write_root_pose_to_sim(ball_pose, env_ids=env_ids)
        self.ball.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)

        if self.fixed_leg == "left":
            leg = torch.zeros(count, dtype=torch.long, device=self.device)
        elif self.fixed_leg == "right":
            leg = torch.ones(count, dtype=torch.long, device=self.device)
        else:
            leg = torch.randint(
                0,
                2,
                (count,),
                generator=self.generator,
                device=self.device,
            )
        strength = strength_min + (
            strength_max - strength_min
        ) * self._rand((count,))

        self.kick_leg[env_ids] = leg
        self.strike_strength[env_ids] = strength
        self.strike_direction_world[env_ids] = direction
        self.kick_ball_start_xy[env_ids] = ball_xy
        self.kick_stance_xy[env_ids] = stance_xy
        self.kick_target_yaw[env_ids] = target_angle
        self.command_tensor[env_ids, :3] = 0.0
        self.previous_low_action[env_ids] = 0.0
        self.skill_id[env_ids] = self.SKILL_KICK
        self.target_terminal_velocity[env_ids] = 0.0
        self.low_episode_length_buf[env_ids] = 0
        self.episode_length_buf[env_ids] = 0
        self.previous_ball_direction_speed[env_ids] = 0.0
        self.peak_ball_direction_speed[env_ids] = 0.0
        self.contact_seen[env_ids] = False
        self.selected_foot_distance[env_ids] = 9.0
        self.other_foot_distance[env_ids] = 9.0
        self.selected_foot_height[env_ids] = 0.0

        state = self._state()
        self.target_height[env_ids] = state["robot_position"][env_ids, 2]
        self._update_kick_target()
        self.total_resets += count
        self._native_low_obs = extract_observation(self.low_env.get_observations())

    def reset(self) -> None:
        env_ids = torch.arange(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        self._reset_idx(env_ids)
        self.low_obs_buf = self._build_low_observation()


class IsolatedKickEnvAdapter:
    def __init__(self, env: IsolatedDeltaPoseKickEnv) -> None:
        self.env = env
        self.num_envs = env.num_envs
        self.num_actions = env.low_num_actions
        self.device = env.device
        self.max_episode_length = env.max_low_episode_length
        self.cfg = env.cfg

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.low_episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.env.low_episode_length_buf = value

    def get_observations(self) -> TensorDict:
        return self.env.get_low_observations()

    def reset(self) -> TensorDict:
        self.env.reset()
        return self.get_observations()


def create_v4k1_runner_cfg(*, base_cfg: Any, run_name: str) -> dict[str, Any]:
    cfg = base_cfg.to_dict()
    cfg["num_steps_per_env"] = int(os.environ.get("V4K1_ROLLOUT_STEPS", "60"))
    cfg["save_interval"] = 10_000
    cfg["experiment_name"] = "go2_soccer_v4k1_isolated_kick"
    cfg["run_name"] = run_name
    cfg["obs_groups"] = {"policy": ["policy"], "critic": ["policy"]}
    cfg["policy"]["init_noise_std"] = float(os.environ.get("V4K1_INIT_NOISE_STD", "0.10"))
    cfg["policy"]["actor_hidden_dims"] = [256, 256, 128]
    cfg["policy"]["critic_hidden_dims"] = [256, 256, 128]
    cfg["policy"]["actor_obs_normalization"] = False
    cfg["policy"]["critic_obs_normalization"] = False
    cfg["algorithm"]["entropy_coef"] = float(os.environ.get("V4K1_ENTROPY_COEF", "0.001"))
    cfg["algorithm"]["learning_rate"] = float(os.environ.get("V4K1_LEARNING_RATE", "0.000003"))
    cfg["algorithm"]["clip_param"] = float(os.environ.get("V4K1_CLIP_PARAM", "0.10"))
    cfg["algorithm"]["num_learning_epochs"] = int(os.environ.get("V4K1_EPOCHS", "3"))
    cfg["algorithm"]["num_mini_batches"] = int(os.environ.get("V4K1_MINI_BATCHES", "4"))
    cfg["algorithm"]["gamma"] = float(os.environ.get("V4K1_GAMMA", "0.995"))
    cfg["algorithm"]["lam"] = 0.95
    cfg["algorithm"]["desired_kl"] = float(os.environ.get("V4K1_DESIRED_KL", "0.004"))
    cfg["algorithm"]["schedule"] = "fixed"
    return cfg


def load_state_dictionary(checkpoint: Path) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(checkpoint, map_location=args_cli.device, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location=args_cli.device)
    if isinstance(payload, dict):
        for key in ("model_state_dict", "policy_state_dict", "actor_critic_state_dict", "state_dict"):
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                return candidate
        if payload and all(torch.is_tensor(value) for value in payload.values()):
            return payload
    raise RuntimeError(f"Could not locate model state in {checkpoint}")


def warm_start_v4k1_policy(
    *,
    runner: Any,
    checkpoint: Path,
    requested_std: float,
) -> None:
    source = load_state_dictionary(checkpoint)
    module = policy_module(runner)
    target = module.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    expanded: list[str] = []

    def noise_parameter(name: str) -> bool:
        return name.rsplit(".", 1)[-1] in {"std", "log_std", "_std", "_log_std"}

    for key, source_value in source.items():
        if key not in target or noise_parameter(key):
            continue
        target_value = target[key]
        if target_value.shape == source_value.shape:
            compatible[key] = source_value
            continue
        if (
            target_value.ndim == 2
            and source_value.ndim == 2
            and target_value.shape[0] == source_value.shape[0]
            and source_value.shape[1] == 62
            and target_value.shape[1] == IsolatedDeltaPoseKickEnv.KICK_OBS_DIM
        ):
            source_on_target = source_value.to(target_value.device)
            mapped = torch.zeros_like(target_value)
            # Native proprioception and direct pose/terminal/remaining context.
            mapped[:, :56] = source_on_target[:, :56]
            # Existing stand/locomotion/push one-hot columns remain unchanged.
            mapped[:, 56:59] = source_on_target[:, 56:59]
            # Column 59 is the new kick one-hot and starts at zero.
            # Move V4D ball-relative columns from 59:61 to 60:62.
            mapped[:, 60:62] = source_on_target[:, 59:61]
            # Teacher flag and all new phase/leg/direction/strength columns are zero.
            mapped[:, IsolatedDeltaPoseKickEnv.NATIVE_COMMAND_SLICE] = 0.0
            compatible[key] = mapped
            expanded.append(key)

    if len(compatible) < 8 or not expanded:
        raise RuntimeError(
            "V4K1 warm start did not find the expected 62->70 input layers. "
            f"compatible={len(compatible)} expanded={expanded}"
        )
    result = torch.nn.Module.load_state_dict(module, compatible, strict=False)
    std_reset = set_action_std(module, requested_std)
    runner.current_learning_iteration = 0
    print(
        "[V4K1_WARM_START] "
        f"checkpoint={checkpoint} "
        f"compatible_parameters={len(compatible)} "
        f"expanded_input_layers={expanded} "
        "source_obs=62 target_obs=70 "
        "legacy_command_columns_zeroed=true "
        "new_kick_context_columns_zero_initialized=true "
        f"missing_keys={len(result.missing_keys)} "
        f"unexpected_keys={len(result.unexpected_keys)} "
        f"optimizer_loaded=false action_std_reset={std_reset} "
        f"requested_std={requested_std}",
        flush=True,
    )


def read_v4k1_schedule(iterations: int) -> list[dict[str, Any]]:
    s1 = int(os.environ.get("V4K1_STAGE1_ITERATIONS", "0"))
    s2 = int(os.environ.get("V4K1_STAGE2_ITERATIONS", "0"))
    s3 = int(os.environ.get("V4K1_STAGE3_ITERATIONS", "0"))
    if s1 + s2 + s3 != iterations or min(s1, s2, s3) <= 0:
        raise RuntimeError(
            "V4K1 stage counts must be positive and sum to max_iterations. "
            f"stages={s1},{s2},{s3} iterations={iterations}"
        )
    return [
        {"name": "contact_discovery", "start": 0, "end": s1, "lr": float(os.environ.get("V4K1_LR_STAGE1", "0.000005"))},
        {"name": "directional_kick", "start": s1, "end": s1 + s2, "lr": float(os.environ.get("V4K1_LR_STAGE2", "0.000003"))},
        {"name": "recovery_generalization", "start": s1 + s2, "end": iterations, "lr": float(os.environ.get("V4K1_LR_STAGE3", "0.000001"))},
    ]


def v4k1_stage_for_iteration(schedule: list[dict[str, Any]], iteration: int) -> dict[str, Any]:
    for stage in schedule:
        if int(stage["start"]) <= iteration < int(stage["end"]):
            return stage
    raise RuntimeError(f"No V4K1 stage contains iteration {iteration}")


def run_v4k1_training(
    *,
    runner: Any,
    env: IsolatedDeltaPoseKickEnv,
    adapter: IsolatedKickEnvAdapter,
    iterations: int,
    log_dir: Path,
) -> None:
    schedule = read_v4k1_schedule(iterations)
    before = snapshot_policy(runner)
    observations = adapter.get_observations().to(runner.device)
    records: list[dict[str, Any]] = []
    active_stage: str | None = None
    stage_before: dict[str, torch.Tensor] | None = None
    stage_deltas: dict[str, dict[str, float]] = {}

    for iteration in range(iterations):
        stage = v4k1_stage_for_iteration(schedule, iteration)
        stage_name = str(stage["name"])
        if stage_name != active_stage:
            if active_stage is not None and stage_before is not None:
                delta = parameter_delta(stage_before, runner)
                stage_deltas[active_stage] = delta
                print(
                    "[V4K1_STAGE_DELTA] "
                    f"stage={active_stage} l2={delta['l2']:.9f} "
                    f"max_abs={delta['max_abs']:.9f}",
                    flush=True,
                )
            active_stage = stage_name
            stage_before = snapshot_policy(runner)
            env.set_curriculum_stage(stage_name)
            env.reset()
            observations = adapter.get_observations().to(runner.device)
            print(
                "[V4K1_STAGE_START] "
                f"stage={stage_name} iteration={iteration} "
                f"learning_rate={float(stage['lr']):.9g}",
                flush=True,
            )

        set_optimizer_learning_rate(runner, float(stage["lr"]))
        policy_module(runner).train()
        totals = {
            "reward": 0.0,
            "pose_error_xy": 0.0,
            "pose_error_yaw": 0.0,
            "upright_error": 0.0,
            "action_rate": 0.0,
            "direction_speed": 0.0,
            "peak_direction_speed": 0.0,
            "directional_displacement": 0.0,
            "lateral_displacement": 0.0,
            "selected_foot_distance": 0.0,
            "selected_foot_height": 0.0,
            "contact_fraction": 0.0,
            "base_done_fraction": 0.0,
            "native_reward": 0.0,
        }
        completed = successful = contact_successes = 0.0
        peak_sum = displacement_sum = 0.0
        steps = int(runner.cfg["num_steps_per_env"])

        with torch.inference_mode():
            for _ in range(steps):
                actions = runner.alg.act(observations)
                next_obs, rewards, dones, extras, diag = env.kick_step(actions.to(env.device))
                next_obs = next_obs.to(runner.device)
                runner.alg.process_env_step(
                    next_obs,
                    rewards.to(runner.device),
                    dones.to(runner.device),
                    extras,
                )
                observations = next_obs
                for key in totals:
                    totals[key] += float(diag[key])
                completed += float(diag["completed_episodes"])
                successful += float(diag["successful_episodes"])
                contact_successes += float(diag["contact_successes"])
                peak_sum += float(diag["completed_peak_speed_sum"])
                displacement_sum += float(diag["completed_displacement_sum"])
            runner.alg.compute_returns(observations)

        losses = runner.alg.update()
        runner.current_learning_iteration = iteration
        success_rate = successful / completed if completed > 0 else None
        contact_rate = contact_successes / completed if completed > 0 else None
        episode_peak = peak_sum / completed if completed > 0 else None
        episode_displacement = displacement_sum / completed if completed > 0 else None
        record = {
            "iteration": iteration,
            "stage": stage_name,
            "learning_rate": float(stage["lr"]),
            **{key: value / steps for key, value in totals.items()},
            "completed_episodes": completed,
            "kick_success_rate": success_rate,
            "episode_contact_rate": contact_rate,
            "episode_peak_direction_speed": episode_peak,
            "episode_directional_displacement": episode_displacement,
            "losses": losses,
        }
        records.append(record)
        print(
            "[V4K1_ITER] "
            f"iteration={iteration}/{iterations} stage={stage_name} "
            f"lr={record['learning_rate']:.9g} reward={record['reward']:.6f} "
            f"kick_success_rate={success_rate} contact_rate={contact_rate} "
            f"episode_peak_speed={episode_peak} "
            f"episode_displacement={episode_displacement} "
            f"upright_error={record['upright_error']:.6f} "
            f"action_rate={record['action_rate']:.6f} losses={losses}",
            flush=True,
        )

        interval = int(os.environ.get("V4K1_CHECKPOINT_INTERVAL", "50"))
        if interval > 0 and (iteration + 1) % interval == 0:
            save_checkpoint(
                runner=runner,
                path=log_dir / f"kick_model_{iteration}.pt",
                iteration=iteration,
                role="v4k1_isolated_delta_pose_kick",
            )

    if active_stage is not None and stage_before is not None:
        delta = parameter_delta(stage_before, runner)
        stage_deltas[active_stage] = delta
        print(
            "[V4K1_STAGE_DELTA] "
            f"stage={active_stage} l2={delta['l2']:.9f} "
            f"max_abs={delta['max_abs']:.9f}",
            flush=True,
        )

    overall_delta = parameter_delta(before, runner)
    final_path = log_dir / "kick_model_final.pt"
    save_checkpoint(
        runner=runner,
        path=final_path,
        iteration=iterations - 1,
        role="v4k1_isolated_delta_pose_kick_final",
    )
    final_window = records[-min(20, len(records)):]
    def mean_available(key: str) -> float | None:
        values = [float(r[key]) for r in final_window if isinstance(r.get(key), (int, float))]
        return sum(values) / len(values) if values else None

    final_success = mean_available("kick_success_rate")
    final_contact = mean_available("episode_contact_rate")
    final_peak = mean_available("episode_peak_direction_speed")
    final_displacement = mean_available("episode_directional_displacement")
    final_upright = mean_available("upright_error")
    final_action_rate = mean_available("action_rate")

    summary = {
        "version": "v4k1_isolated_delta_pose_kick",
        "source_low_policy": "V4D final 62-D direct Delta-Pose low policy",
        "high_level_policy_used": False,
        "soccer_goal_reward_used": False,
        "skill": "kick_only",
        "delta_pose": "planar_base_delta_pose_plus_kick_phase_leg_direction_strength",
        "action": "12_joint_position_residuals",
        "teacher_velocity_actor_observation": False,
        "curriculum": schedule,
        "stage_parameter_deltas": stage_deltas,
        "parameter_delta": overall_delta,
        "records": records,
    }
    summary_path = log_dir / "v4k1_kick_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    contract = {
        "version": "v4k1_isolated_delta_pose_kick",
        "training_scope": "isolated_low_level_kick_only",
        "high_level": None,
        "soccer_zero_sum_reward": None,
        "input": {
            "native_proprioception": 48,
            "legacy_velocity_command_scrubbed": True,
            "planar_delta_pose": ["delta_x", "delta_y", "delta_yaw"],
            "kick_context": ["phase", "leg", "direction", "strength"],
            "total_observation_dimension": 70,
        },
        "output": "12_joint_position_residuals",
        "reward": "contact_ball_speed_direction_base_pose_posture_recovery_only",
        "later_integration": "add kick to V4D router only after isolated acceptance",
    }
    contract_path = log_dir / "v4k1_kick_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")

    print(
        "[V4K1_PARAMETER_DELTA] "
        f"l2={overall_delta['l2']:.9f} max_abs={overall_delta['max_abs']:.9f}",
        flush=True,
    )
    print(
        "[V4K1_FINAL_GUARD] "
        f"kick_success_rate={final_success} contact_rate={final_contact} "
        f"peak_direction_speed={final_peak} "
        f"directional_displacement={final_displacement} "
        f"upright_error={final_upright} action_rate={final_action_rate}",
        flush=True,
    )
    if overall_delta["max_abs"] <= 0.0:
        raise RuntimeError("V4K1 kick policy parameters did not update.")
    if final_upright is not None and final_upright > float(os.environ.get("V4K1_MAX_UPRIGHT_ERROR", "0.04")):
        raise RuntimeError("V4K1 upright guard failed.")
    if final_action_rate is not None and final_action_rate > float(os.environ.get("V4K1_MAX_ACTION_RATE", "0.8")):
        raise RuntimeError("V4K1 action-rate guard failed.")
    minimum_success = float(os.environ.get("V4K1_MIN_SUCCESS_RATE", "0.0"))
    if minimum_success > 0.0 and (final_success is None or final_success < minimum_success):
        raise RuntimeError(
            "V4K1 kick-success guard failed. "
            f"required={minimum_success} observed={final_success}"
        )
    minimum_contact = float(os.environ.get("V4K1_MIN_CONTACT_RATE", "0.0"))
    if minimum_contact > 0.0 and (final_contact is None or final_contact < minimum_contact):
        raise RuntimeError(
            "V4K1 contact guard failed. "
            f"required={minimum_contact} observed={final_contact}"
        )

    (log_dir / "V4K1_ISOLATED_KICK_OK").touch()
    print(f"V4K1_KICK_CHECKPOINT={final_path}", flush=True)
    print(f"V4K1_SUMMARY={summary_path}", flush=True)
    print(f"V4K1_CONTRACT={contract_path}", flush=True)
    print("V4K1_ISOLATED_KICK_TRAINING_COMPLETE", flush=True)


def run_v4k1_evaluation(
    *,
    runner: Any,
    env: IsolatedDeltaPoseKickEnv,
    video_path: Path,
    trajectory_path: Path,
    summary_path: Path,
    episodes: int,
    fps: int,
) -> None:
    import csv
    import imageio.v2 as imageio
    if env.num_envs != 1:
        raise RuntimeError("V4K1 evaluation requires --num_envs 1.")
    policy_module(runner).eval()
    env.set_curriculum_stage("recovery_generalization")
    env.reset()
    adapter = IsolatedKickEnvAdapter(env)
    obs = adapter.get_observations().to(runner.device)

    render_env: Any = env.low_env
    seen: set[int] = set()
    while not callable(getattr(render_env, "render", None)):
        if id(render_env) in seen:
            raise RuntimeError("V4K1 render unwrap cycle.")
        seen.add(id(render_env))
        next_env = getattr(render_env, "env", None)
        if next_env is None or next_env is render_env:
            next_env = getattr(render_env, "unwrapped", None)
        if next_env is None or next_env is render_env:
            raise RuntimeError("No render-capable V4K1 environment.")
        render_env = next_env

    video_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        video_path,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
    )
    fields = [
        "step", "episode", "phase", "kick_leg", "robot_x", "robot_y",
        "robot_yaw", "ball_x", "ball_y", "ball_vx", "ball_vy",
        "direction_speed", "peak_direction_speed", "directional_displacement",
        "selected_foot_distance", "selected_foot_height", "contact_seen",
        "completed", "success",
    ]
    rows: list[dict[str, Any]] = []
    completed_total = success_total = 0
    step_index = 0
    try:
        with torch.no_grad():
            while completed_total < episodes:
                action = policy_module(runner).act_inference(obs)
                obs, _, _, _, diag = env.kick_step(action.to(env.device))
                obs = obs.to(runner.device)
                frame = render_env.render()
                if isinstance(frame, list):
                    frame = frame[0]
                if frame is None:
                    raise RuntimeError("V4K1 render() returned no frame.")
                writer.append_data(np.asarray(frame))
                state = env._state()
                completed_now = int(diag["completed_episodes"])
                success_now = int(diag["successful_episodes"])
                rows.append({
                    "step": step_index,
                    "episode": completed_total,
                    "phase": float(env.kick_phase[0]),
                    "kick_leg": int(env.kick_leg[0]),
                    "robot_x": float(state["robot_position"][0, 0]),
                    "robot_y": float(state["robot_position"][0, 1]),
                    "robot_yaw": float(state["robot_yaw"][0]),
                    "ball_x": float(state["ball_position"][0, 0]),
                    "ball_y": float(state["ball_position"][0, 1]),
                    "ball_vx": float(state["ball_velocity_world"][0, 0]),
                    "ball_vy": float(state["ball_velocity_world"][0, 1]),
                    "direction_speed": diag["direction_speed"],
                    "peak_direction_speed": diag["peak_direction_speed"],
                    "directional_displacement": diag["directional_displacement"],
                    "selected_foot_distance": diag["selected_foot_distance"],
                    "selected_foot_height": diag["selected_foot_height"],
                    "contact_seen": diag["contact_fraction"],
                    "completed": completed_now,
                    "success": success_now,
                })
                completed_total += completed_now
                success_total += success_now
                step_index += 1
                if step_index > episodes * env.kick_episode_steps * 3:
                    raise RuntimeError("V4K1 evaluation exceeded safety step limit.")
    finally:
        writer.close()

    with trajectory_path.open("w", newline="", encoding="utf-8") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=fields)
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    payload = {
        "episodes": completed_total,
        "successes": success_total,
        "success_rate": success_total / completed_total if completed_total else None,
        "steps": step_index,
        "video": str(video_path),
        "trajectory": str(trajectory_path),
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"V4K1_EVAL_VIDEO={video_path}", flush=True)
    print(f"V4K1_EVAL_SUMMARY={summary_path}", flush=True)
    print("V4K1_EVAL_VIDEO_OK", flush=True)


def main() -> None:
    if args_cli.num_envs <= 0 or args_cli.max_iterations <= 0:
        raise ValueError("V4K1 num_envs and max_iterations must be positive.")
    source_checkpoint = Path(args_cli.low_level_checkpoint).resolve()
    if not source_checkpoint.is_file():
        raise FileNotFoundError(f"V4K1 source checkpoint not found: {source_checkpoint}")
    log_dir = Path(args_cli.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args_cli.seed)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    env_cfg.episode_length_s = max(4.0, args_cli.kick_episode_steps * 0.02 + 1.0)
    try:
        env_cfg.observations.policy.enable_corruption = False
    except AttributeError:
        pass
    try:
        env_cfg.events.push_robot = None
    except AttributeError:
        pass
    try:
        env_cfg.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
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
        render_mode="rgb_array" if args_cli.eval_video else None,
    )
    low_env = RslRlVecEnvWrapper(gym_env, clip_actions=low_agent_cfg.clip_actions)
    env = IsolatedDeltaPoseKickEnv(
        low_env=low_env,
        kick_episode_steps=args_cli.kick_episode_steps,
        seed=args_cli.seed,
        fixed_leg=args_cli.kick_fixed_leg,
    )
    adapter = IsolatedKickEnvAdapter(env)
    runner_cfg = create_v4k1_runner_cfg(base_cfg=low_agent_cfg, run_name=args_cli.run_name)

    print(
        "[V4K1_CONTRACT] "
        "training_scope=isolated_low_level_kick_only "
        "high_policy=not_used "
        "soccer_goal_reward=not_used "
        "source_low_policy=v4d_teacher_free_delta_pose "
        "delta_pose=planar_base_delta_pose_direct_context "
        "kick_context=phase_leg_direction_strength "
        "skill=kick_only "
        "low_action=12_joint_position_residuals "
        "teacher_velocity_actor_observation=never "
        "later_router_integration=deferred_until_isolated_acceptance",
        flush=True,
    )

    runner = OnPolicyRunner(adapter, runner_cfg, log_dir=None, device=args_cli.device)
    if args_cli.eval_video:
        trained_raw = args_cli.trained_low_checkpoint.strip()
        if not trained_raw:
            raise RuntimeError("--trained_low_checkpoint is required for V4K1 evaluation.")
        trained = Path(trained_raw).resolve()
        if not trained.is_file():
            raise FileNotFoundError(f"V4K1 trained checkpoint missing: {trained}")
        runner.load(str(trained), load_optimizer=False)
        video_path = Path(args_cli.eval_video).resolve()
        trajectory_path = Path(
            args_cli.trajectory_csv or str(video_path.with_suffix("")) + "_trajectory.csv"
        ).resolve()
        summary_path = Path(
            args_cli.eval_summary or str(video_path.with_suffix("")) + "_summary.json"
        ).resolve()
        run_v4k1_evaluation(
            runner=runner,
            env=env,
            video_path=video_path,
            trajectory_path=trajectory_path,
            summary_path=summary_path,
            episodes=args_cli.eval_episodes,
            fps=args_cli.eval_fps,
        )
    else:
        warm_start_v4k1_policy(
            runner=runner,
            checkpoint=source_checkpoint,
            requested_std=float(runner_cfg["policy"]["init_noise_std"]),
        )
        env.reset()
        preflight_obs = adapter.get_observations()["policy"]
        command_abs = float(preflight_obs[:, env.NATIVE_COMMAND_SLICE].abs().max())
        teacher_abs = float(preflight_obs[:, env.KICK_CONTEXT_TEACHER_FLAG].abs().max())
        kick_onehot = float(preflight_obs[:, 59].mean())
        print(
            "[V4K1_INPUT_GUARD] "
            f"legacy_command_abs={command_abs:.9f} "
            f"teacher_flag_abs={teacher_abs:.9f} "
            f"kick_onehot_mean={kick_onehot:.9f} "
            f"obs_shape={tuple(preflight_obs.shape)}",
            flush=True,
        )
        if command_abs > 1.0e-8 or teacher_abs > 1.0e-8:
            raise RuntimeError("Teacher leakage detected in V4K1 actor observation.")
        if abs(kick_onehot - 1.0) > 1.0e-6:
            raise RuntimeError("V4K1 kick skill one-hot is not active.")
        with torch.no_grad():
            action = policy_module(runner).act_inference(adapter.get_observations().to(runner.device))
            next_obs, reward, done, _, diag = env.kick_step(action.to(env.device))
        if not torch.isfinite(next_obs["policy"]).all() or not torch.isfinite(reward).all():
            raise RuntimeError("V4K1 preflight produced non-finite tensors.")
        print(
            "[V4K1_PREFLIGHT] "
            f"reward_mean={float(reward.mean()):.6f} "
            f"done_count={int(done.sum())} diagnostics={diag}",
            flush=True,
        )
        env.reset()
        run_v4k1_training(
            runner=runner,
            env=env,
            adapter=adapter,
            iterations=args_cli.max_iterations,
            log_dir=log_dir,
        )
    env.close()


try:
    main()
except BaseException as error:
    print(f"[V4K1_FATAL] {type(error).__name__}: {error}", flush=True)
    raise
finally:
    print("[V4K1_BOOT] closing_simulation_app", flush=True)
    simulation_app.close()
