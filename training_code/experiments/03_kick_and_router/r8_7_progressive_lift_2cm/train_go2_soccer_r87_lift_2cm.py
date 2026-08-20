# R8_7_PROGRESSIVE_LOAD_PRESERVING_LIFT_2CM
# R8_6_PROGRESSIVE_LOAD_PRESERVING_LIFT
# R8_5_2_SIMULTANEOUS_BRIDGE_CONSOLIDATION
# R8_1_GATED_UNLOAD_CURRICULUM_PPO
# R8_4_1_DIRECT_LIFT_FROM_R83
# R8_4_CAPABILITY_SPRINT
# R8_4_LOAD_PRESERVING_PROGRESSIVE_LIFT_PPO
# R8_3_REAL_LOAD_TRANSFER_PPO
# R8_SUPPORT_TRANSFER_SKILL_PPO
# V4K1_R5_3_CAUSAL_DISCOVERY_REPAIR
# standing_positive_baseline_removed_in_stage1=true
# V4D_STAGED_ZERO_SUM_JOINT_INTEGRATION
"""R8 isolated support-transfer, single-leg lift, swing, and recovery PPO."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
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
        description="R8 isolated low-level support-transfer skill training."
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
    parser.add_argument("--kick_fixed_leg", type=str, default="left", choices=("alternate", "left", "right"))
    parser.add_argument(
        "--r8_profile",
        type=str,
        default="pilot",
        choices=("pilot", "full"),
    )

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
    # Preserve the exact V4D low-policy MLP widths from the registered
    # Round-2 configuration. V4D intentionally inherited these widths.
    # Changing them here prevents the 62->70 first-layer transplant.
    actor_hidden_dims = list(cfg["policy"]["actor_hidden_dims"])
    critic_hidden_dims = list(cfg["policy"]["critic_hidden_dims"])
    cfg["policy"]["actor_hidden_dims"] = actor_hidden_dims
    cfg["policy"]["critic_hidden_dims"] = critic_hidden_dims
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
        f"source_std_preserved={source_std_preserved} "
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




def stable_stance_mask(
    *,
    vertical_speed: torch.Tensor,
    horizontal_speed: torch.Tensor,
    angular_xy_speed: torch.Tensor,
    upright_error: torch.Tensor,
    base_height: torch.Tensor,
    support_count: torch.Tensor,
) -> torch.Tensor:
    """Return environments that are genuinely settled on the ground."""
    return (
        (vertical_speed < 0.05)
        & (horizontal_speed < 0.12)
        & (angular_xy_speed < 0.18)
        & (upright_error < 0.012)
        & (base_height > 0.22)
        & (base_height < 0.45)
        & (support_count >= 3)
    )


def precontact_ball_motion_mask(
    *,
    active: torch.Tensor,
    contact_seen: torch.Tensor,
    before_ball_speed: torch.Tensor,
    before_ball_displacement: torch.Tensor,
) -> torch.Tensor:
    """Reject ball motion that existed before the current control action."""
    speed_limit = float(os.environ.get("V4K1_PRECONTACT_SPEED", "0.030"))
    displacement_limit = float(
        os.environ.get("V4K1_PRECONTACT_DISPLACEMENT", "0.020")
    )
    return (
        active
        & ~contact_seen
        & (
            (before_ball_speed > speed_limit)
            | (before_ball_displacement > displacement_limit)
        )
    )


def valid_kick_contact_mask(
    *,
    active: torch.Tensor,
    geometric_contact: torch.Tensor,
    stable_at_contact: torch.Tensor,
    foot_lift_seen: torch.Tensor,
    foot_swing_seen: torch.Tensor,
    deliberate_approach: torch.Tensor,
    wrong_foot_contact: torch.Tensor,
    body_ball_contact: torch.Tensor,
    precontact_ball_motion: torch.Tensor,
) -> torch.Tensor:
    """Accept only a deliberate selected-foot contact from a stable stance."""
    return (
        active
        & geometric_contact
        & stable_at_contact
        & foot_lift_seen
        & foot_swing_seen
        & deliberate_approach
        & ~wrong_foot_contact
        & ~body_ball_contact
        & ~precontact_ball_motion
    )


def invalid_kick_contact_mask(
    *,
    geometric_contact: torch.Tensor,
    valid_contact: torch.Tensor,
    ball_response_now: torch.Tensor,
    wrong_foot_contact: torch.Tensor,
    body_ball_contact: torch.Tensor,
    precontact_ball_motion: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Separate harmless near-contact from unsafe or passive ball motion."""
    passive_selected_response = (
        geometric_contact
        & ~valid_contact
        & ball_response_now
        & ~wrong_foot_contact
        & ~body_ball_contact
        & ~precontact_ball_motion
    )
    invalid_contact = (
        wrong_foot_contact
        | body_ball_contact
        | precontact_ball_motion
        | passive_selected_response
    )
    return invalid_contact, passive_selected_response


def next_valid_contact_duration(
    *,
    previous_duration: torch.Tensor,
    contact_seen_after_update: torch.Tensor,
    geometric_contact: torch.Tensor,
) -> torch.Tensor:
    """Count contact duration only from the first valid selected-foot contact."""
    return previous_duration + (
        contact_seen_after_update & geometric_contact
    ).long()



def stable_kick_success_mask(
    *,
    valid_contact_seen: torch.Tensor,
    impact_seen: torch.Tensor,
    peak_direction_speed: torch.Tensor,
    directional_displacement: torch.Tensor,
    contact_duration_steps: torch.Tensor,
    violation_seen: torch.Tensor,
    base_displacement: torch.Tensor,
    recovery_seen: torch.Tensor,
) -> torch.Tensor:
    """Require a short foot strike, ball motion, anchored base, and recovery."""
    short_contact = (
        (contact_duration_steps >= 1)
        & (contact_duration_steps <= 12)
    )
    return (
        valid_contact_seen
        & impact_seen
        & (peak_direction_speed > 0.30)
        & (directional_displacement > 0.20)
        & short_contact
        & ~violation_seen
        & (base_displacement < 0.10)
        & recovery_seen
    )

# V4K1_R5_STABLE_STANCE_DELTA_POSE_KICK
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
        self.settle_min_steps = int(os.environ.get("V4K1_SETTLE_MIN_STEPS", "25"))
        self.settle_max_steps = int(os.environ.get("V4K1_SETTLE_MAX_STEPS", "150"))
        self.settle_required_steps = int(os.environ.get("V4K1_SETTLE_REQUIRED_STEPS", "10"))
        self.ball_arm_steps = int(os.environ.get("V4K1_BALL_ARM_STEPS", "6"))
        if not (
            1 <= self.settle_min_steps < self.settle_max_steps
            and 1 <= self.settle_required_steps <= self.settle_max_steps
            and self.ball_arm_steps >= 1
        ):
            raise ValueError(
                "Invalid V4K1-R5 settle configuration: "
                f"min={self.settle_min_steps} max={self.settle_max_steps} "
                f"required={self.settle_required_steps} arm={self.ball_arm_steps}"
            )
        self.total_episode_steps = (
            self.settle_max_steps
            + self.ball_arm_steps
            + self.kick_episode_steps
        )
        self.max_episode_length = self.total_episode_steps
        self.max_low_episode_length = self.total_episode_steps
        # R5 intentionally trains one reliable left-front-foot kick first.
        self.fixed_leg = "left"
        self.curriculum_stage = "contact_discovery"

        self.kick_leg = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.kick_phase = torch.zeros(self.num_envs, device=self.device)
        self.strike_strength = torch.ones(self.num_envs, device=self.device)
        self.strike_direction_world = torch.zeros(self.num_envs, 2, device=self.device)
        self.strike_direction_world[:, 0] = 1.0
        self.strike_angle_offset = torch.zeros(self.num_envs, device=self.device)
        self.kick_ball_start_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.kick_stance_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.kick_stance_yaw = torch.zeros(self.num_envs, device=self.device)
        self.kick_target_yaw = torch.zeros(self.num_envs, device=self.device)

        self.settle_step_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.stable_consecutive_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.ball_arm_step_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.active_kick_step_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.ball_placed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.kick_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.stable_activation_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.settle_timeout_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.previous_ball_direction_speed = torch.zeros(self.num_envs, device=self.device)
        self.peak_ball_direction_speed = torch.zeros(self.num_envs, device=self.device)
        self.contact_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.impact_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.geometric_contact_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.motion_ready_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.passive_response_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.violation_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.precontact_motion_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.invalid_contact_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.body_contact_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.wrong_foot_contact_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.foot_lift_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.foot_swing_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.recovery_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.recovery_stable_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.contact_duration_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.previous_selected_foot_distance = torch.full((self.num_envs,), 0.24, device=self.device)
        self.minimum_selected_foot_distance = torch.full((self.num_envs,), 9.0, device=self.device)
        self.selected_foot_distance = torch.full((self.num_envs,), 9.0, device=self.device)
        self.other_foot_distance = torch.full((self.num_envs,), 9.0, device=self.device)
        self.selected_foot_height = torch.zeros(self.num_envs, device=self.device)
        self.selected_foot_start_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.previous_selected_foot_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.max_selected_foot_lift = torch.zeros(self.num_envs, device=self.device)
        self.max_selected_foot_travel = torch.zeros(self.num_envs, device=self.device)
        self.max_foot_toward_ball_speed = torch.zeros(self.num_envs, device=self.device)

        self.front_foot_ids = self._resolve_front_foot_ids()
        self.all_foot_ids = self._resolve_all_foot_ids()
        if min(self.front_foot_ids) < 0 or min(self.all_foot_ids) < 0:
            raise RuntimeError(
                "V4K1-R5 could not resolve all Go2 foot bodies. "
                f"front={self.front_foot_ids} all={self.all_foot_ids}"
            )

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
            f"low_obs={self.low_num_obs} low_actions={self.low_num_actions} "
            f"active_kick_steps={self.kick_episode_steps} "
            f"settle_min={self.settle_min_steps} settle_max={self.settle_max_steps} "
            f"ball_arm_steps={self.ball_arm_steps} "
            f"front_foot_ids={self.front_foot_ids} all_foot_ids={self.all_foot_ids}",
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


    def _resolve_all_foot_ids(self) -> tuple[int, int, int, int]:
        resolved: list[int] = []
        for patterns in (
            ("FL_foot", ".*FL.*foot.*", ".*FL.*FOOT.*"),
            ("FR_foot", ".*FR.*foot.*", ".*FR.*FOOT.*"),
            ("RL_foot", ".*RL.*foot.*", ".*RL.*FOOT.*"),
            ("RR_foot", ".*RR.*foot.*", ".*RR.*FOOT.*"),
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
        return tuple(int(value) for value in resolved)  # type: ignore[return-value]

    def _front_foot_positions(self) -> tuple[torch.Tensor, torch.Tensor]:
        body_pos = getattr(self.robot.data, "body_pos_w", None)
        if not torch.is_tensor(body_pos):
            raise RuntimeError("V4K1-R5 robot body positions are unavailable.")
        left = body_pos[:, self.front_foot_ids[0], :3]
        right = body_pos[:, self.front_foot_ids[1], :3]
        selected = torch.where((self.kick_leg == 0).unsqueeze(1), left, right)
        other = torch.where((self.kick_leg == 0).unsqueeze(1), right, left)
        return selected, other

    def _support_foot_count(self) -> torch.Tensor:
        body_pos = getattr(self.robot.data, "body_pos_w", None)
        if not torch.is_tensor(body_pos):
            return torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        ids = torch.as_tensor(self.all_foot_ids, dtype=torch.long, device=self.device)
        foot_height = body_pos[:, ids, 2] - self.env_origins[:, 2:3]
        return (foot_height < 0.10).sum(dim=1)

    def _stability_state(self) -> dict[str, torch.Tensor]:
        root_lin_vel_w = self.robot.data.root_lin_vel_w
        root_ang_vel_b = self.robot.data.root_ang_vel_b
        vertical_speed = root_lin_vel_w[:, 2].abs()
        horizontal_speed = torch.linalg.vector_norm(root_lin_vel_w[:, :2], dim=1)
        angular_xy_speed = torch.linalg.vector_norm(root_ang_vel_b[:, :2], dim=1)
        upright_error = self._native_low_obs[:, 6:8].square().sum(dim=1)
        base_height = self.robot.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        support_count = self._support_foot_count()
        stable = stable_stance_mask(
            vertical_speed=vertical_speed,
            horizontal_speed=horizontal_speed,
            angular_xy_speed=angular_xy_speed,
            upright_error=upright_error,
            base_height=base_height,
            support_count=support_count,
        )
        return {
            "stable": stable,
            "vertical_speed": vertical_speed,
            "horizontal_speed": horizontal_speed,
            "angular_xy_speed": angular_xy_speed,
            "upright_error": upright_error,
            "base_height": base_height,
            "support_count": support_count,
        }

    def _place_ball_for_kick(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        selected, _ = self._front_foot_positions()
        state = self._state()
        count = int(env_ids.numel())
        yaw = state["robot_yaw"][env_ids]
        angle = yaw + self.strike_angle_offset[env_ids]
        direction = torch.stack((torch.cos(angle), torch.sin(angle)), dim=1)
        lateral = torch.stack((-direction[:, 1], direction[:, 0]), dim=1)
        if self.curriculum_stage == "contact_discovery":
            distance = 0.155 + 0.010 * self._rand((count,))
            lateral_noise = (2.0 * self._rand((count,)) - 1.0) * 0.004
        elif self.curriculum_stage == "directional_kick":
            distance = 0.170 + 0.020 * self._rand((count,))
            lateral_noise = (2.0 * self._rand((count,)) - 1.0) * 0.012
        else:
            distance = 0.175 + 0.030 * self._rand((count,))
            lateral_noise = (2.0 * self._rand((count,)) - 1.0) * 0.020
        ball_xy = (
            selected[env_ids, :2]
            + distance.unsqueeze(1) * direction
            + lateral_noise.unsqueeze(1) * lateral
        )
        parked_z = self.ball.data.root_pos_w[env_ids, 2]
        ground_z = self.env_origins[env_ids, 2]
        ball_z = torch.clamp(parked_z, min=ground_z + 0.035, max=ground_z + 0.10)
        ball_pose = torch.zeros(count, 7, device=self.device)
        ball_pose[:, :2] = ball_xy
        ball_pose[:, 2] = ball_z
        ball_pose[:, 3] = 1.0
        zero_velocity = torch.zeros(count, 6, device=self.device)
        self.ball.write_root_pose_to_sim(ball_pose, env_ids=env_ids)
        self.ball.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)

        self.strike_direction_world[env_ids] = direction
        self.kick_ball_start_xy[env_ids] = ball_xy
        self.kick_stance_xy[env_ids] = state["robot_position"][env_ids, :2]
        self.kick_stance_yaw[env_ids] = yaw
        self.kick_target_yaw[env_ids] = yaw
        self.target_height[env_ids] = state["robot_position"][env_ids, 2]
        self.ball_placed[env_ids] = True
        self.ball_arm_step_buf[env_ids] = 0
        self.stable_consecutive_buf[env_ids] = 0
        self.previous_ball_direction_speed[env_ids] = 0.0
        self.peak_ball_direction_speed[env_ids] = 0.0
        self.contact_seen[env_ids] = False
        self.impact_seen[env_ids] = False
        self.geometric_contact_seen[env_ids] = False
        self.motion_ready_seen[env_ids] = False
        self.passive_response_seen[env_ids] = False
        self.violation_seen[env_ids] = False
        self.precontact_motion_seen[env_ids] = False
        self.invalid_contact_seen[env_ids] = False
        self.body_contact_seen[env_ids] = False
        self.wrong_foot_contact_seen[env_ids] = False
        self.foot_lift_seen[env_ids] = False
        self.foot_swing_seen[env_ids] = False
        self.recovery_seen[env_ids] = False
        self.recovery_stable_steps[env_ids] = 0
        self.contact_duration_steps[env_ids] = 0
        self.selected_foot_start_pos[env_ids] = selected[env_ids]
        self.previous_selected_foot_pos[env_ids] = selected[env_ids]
        self.max_selected_foot_lift[env_ids] = 0.0
        self.max_selected_foot_travel[env_ids] = 0.0
        self.max_foot_toward_ball_speed[env_ids] = 0.0

    def _advance_pre_kick_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        stability = self._stability_state()
        stable = stability["stable"]
        not_placed = ~self.ball_placed
        settling = not_placed & ~self.kick_active
        self.settle_step_buf[settling] += 1
        self.stable_consecutive_buf[settling] = torch.where(
            stable[settling],
            self.stable_consecutive_buf[settling] + 1,
            torch.zeros_like(self.stable_consecutive_buf[settling]),
        )
        ready = (
            settling
            & (self.settle_step_buf >= self.settle_min_steps)
            & (self.stable_consecutive_buf >= self.settle_required_steps)
        )
        if ready.any():
            self._place_ball_for_kick(torch.nonzero(ready, as_tuple=False).squeeze(-1))
        settle_timeout = (
            settling
            & (self.settle_step_buf >= self.settle_max_steps)
            & ~ready
        )
        self.settle_timeout_seen |= settle_timeout

        arming = self.ball_placed & ~self.kick_active
        ball_speed = torch.linalg.vector_norm(self.ball.data.root_lin_vel_w[:, :2], dim=1)
        ball_displacement = torch.linalg.vector_norm(
            self.ball.data.root_pos_w[:, :2] - self.kick_ball_start_xy,
            dim=1,
        )
        arming_violation = arming & (
            (ball_speed > 0.04)
            | (ball_displacement > 0.025)
            | ~stable
        )
        self.ball_arm_step_buf[arming] += 1
        self.stable_consecutive_buf[arming] = torch.where(
            (stable & (ball_speed < 0.025))[arming],
            self.stable_consecutive_buf[arming] + 1,
            torch.zeros_like(self.stable_consecutive_buf[arming]),
        )
        arm_ready = (
            arming
            & (self.ball_arm_step_buf >= self.ball_arm_steps)
            & (self.stable_consecutive_buf >= self.ball_arm_steps)
            & ~arming_violation
        )
        if arm_ready.any():
            ids = torch.nonzero(arm_ready, as_tuple=False).squeeze(-1)
            self.kick_active[ids] = True
            self.stable_activation_seen[ids] = True
            self.active_kick_step_buf[ids] = 0
            self.low_episode_length_buf[ids] = 0
            self.kick_ball_start_xy[ids] = self.ball.data.root_pos_w[ids, :2]
            selected, _ = self._front_foot_positions()
            self.selected_foot_start_pos[ids] = selected[ids]
            self.previous_selected_foot_pos[ids] = selected[ids]
            state = self._state()
            self.kick_stance_xy[ids] = state["robot_position"][ids, :2]
            self.kick_stance_yaw[ids] = state["robot_yaw"][ids]
            self.kick_target_yaw[ids] = state["robot_yaw"][ids]
            self.target_height[ids] = state["robot_position"][ids, 2]
        return settle_timeout, arming_violation


    def set_curriculum_stage(self, name: str) -> None:
        if name not in {
            "contact_discovery",
            "directional_kick",
            "recovery_generalization",
        }:
            raise ValueError(f"Unknown V4K1 curriculum stage: {name}")
        self.curriculum_stage = name

    def _phase(self) -> torch.Tensor:
        return torch.where(
            self.kick_active,
            torch.clamp(
                self.active_kick_step_buf.float()
                / max(1.0, float(self.kick_episode_steps - 1)),
                0.0,
                1.0,
            ),
            torch.zeros(self.num_envs, device=self.device),
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
        strike_ramp = torch.clamp((phase - 0.42) / 0.16, 0.0, 1.0)
        recovery_ramp = torch.clamp((phase - 0.68) / 0.22, 0.0, 1.0)
        forward_shift = (
            0.045 * strike_ramp * (1.0 - recovery_ramp)
            + 0.010 * recovery_ramp
        ) * self.strike_strength
        self.target_x_world = self.kick_stance_xy[:, 0] + forward_shift * self.strike_direction_world[:, 0]
        self.target_y_world = self.kick_stance_xy[:, 1] + forward_shift * self.strike_direction_world[:, 1]
        self.target_yaw_world = self.kick_target_yaw
        self.target_terminal_velocity.zero_()
        self.target_elapsed_steps = self.active_kick_step_buf.clone()
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
        active = self.kick_active
        error_x, error_y, error_yaw = self._pose_error()
        error_xy = torch.sqrt(error_x.square() + error_y.square() + 1.0e-8)
        pose_track = torch.exp(-error_xy.square() / (0.18 ** 2))
        yaw_track = torch.exp(-error_yaw.square() / (0.22 ** 2))
        stability = self._stability_state()
        upright_error = stability["upright_error"]
        action_rate = (low_actions - self.previous_low_action).square().mean(dim=1)
        torque_penalty = self._applied_torque_penalty()

        direction = self.strike_direction_world
        lateral = torch.stack((-direction[:, 1], direction[:, 0]), dim=1)
        before_ball_velocity = before_state["ball_velocity_world"][:, :2]
        after_ball_velocity = after_state["ball_velocity_world"][:, :2]
        before_ball_speed = torch.linalg.vector_norm(before_ball_velocity, dim=1)
        after_ball_speed = torch.linalg.vector_norm(after_ball_velocity, dim=1)
        before_displacement_vec = (
            before_state["ball_position"][:, :2] - self.kick_ball_start_xy
        )
        before_ball_displacement = torch.linalg.vector_norm(
            before_displacement_vec, dim=1
        )
        displacement_vec = (
            after_state["ball_position"][:, :2] - self.kick_ball_start_xy
        )
        direction_speed = (after_ball_velocity * direction).sum(dim=1)
        lateral_speed = torch.abs((after_ball_velocity * lateral).sum(dim=1))
        directional_displacement = (displacement_vec * direction).sum(dim=1)
        lateral_displacement = torch.abs((displacement_vec * lateral).sum(dim=1))
        positive_impulse = torch.relu(
            direction_speed - self.previous_ball_direction_speed
        )
        ball_speed_gain = torch.relu(after_ball_speed - before_ball_speed)
        self.peak_ball_direction_speed = torch.where(
            active,
            torch.maximum(self.peak_ball_direction_speed, direction_speed),
            self.peak_ball_direction_speed,
        )

        selected_pos, other_pos = self._front_foot_positions()
        ball_pos = self.ball.data.root_pos_w[:, :3]
        selected_distance = torch.linalg.vector_norm(selected_pos - ball_pos, dim=1)
        other_distance = torch.linalg.vector_norm(other_pos - ball_pos, dim=1)
        selected_height = torch.clamp(
            selected_pos[:, 2] - self.env_origins[:, 2], min=0.0
        )
        foot_lift = selected_pos[:, 2] - self.selected_foot_start_pos[:, 2]
        foot_travel = torch.linalg.vector_norm(
            selected_pos - self.selected_foot_start_pos, dim=1
        )
        foot_forward = (
            (selected_pos[:, :2] - self.selected_foot_start_pos[:, :2])
            * direction
        ).sum(dim=1)
        step_dt = float(getattr(self.base_env, "step_dt", 0.02))
        if not math.isfinite(step_dt) or step_dt <= 0.0:
            step_dt = 0.02
        foot_velocity = (
            selected_pos - self.previous_selected_foot_pos
        ) / step_dt
        foot_to_ball_before = (
            before_state["ball_position"][:, :2]
            - self.previous_selected_foot_pos[:, :2]
        )
        foot_to_ball_norm = torch.linalg.vector_norm(
            foot_to_ball_before, dim=1
        ).clamp_min(1.0e-6)
        foot_to_ball_unit = foot_to_ball_before / foot_to_ball_norm.unsqueeze(1)
        foot_toward_ball_speed = (
            foot_velocity[:, :2] * foot_to_ball_unit
        ).sum(dim=1)
        self.max_foot_toward_ball_speed = torch.where(
            active,
            torch.maximum(
                self.max_foot_toward_ball_speed,
                foot_toward_ball_speed,
            ),
            self.max_foot_toward_ball_speed,
        )

        previous_lift_seen = self.foot_lift_seen.clone()
        previous_swing_seen = self.foot_swing_seen.clone()
        previous_contact_seen = self.contact_seen.clone()
        lift_threshold = float(os.environ.get("V4K1_LIFT_HEIGHT", "0.012"))
        swing_forward_threshold = float(
            os.environ.get("V4K1_SWING_FORWARD", "0.025")
        )
        swing_travel_threshold = float(
            os.environ.get("V4K1_SWING_TRAVEL", "0.035")
        )
        toward_speed_threshold = float(
            os.environ.get("V4K1_TOWARD_BALL_SPEED", "0.060")
        )
        self.max_selected_foot_lift = torch.where(
            active,
            torch.maximum(self.max_selected_foot_lift, foot_lift),
            self.max_selected_foot_lift,
        )
        self.max_selected_foot_travel = torch.where(
            active,
            torch.maximum(self.max_selected_foot_travel, foot_travel),
            self.max_selected_foot_travel,
        )
        self.foot_lift_seen |= active & (
            self.max_selected_foot_lift > lift_threshold
        )
        self.foot_swing_seen |= (
            active
            & (foot_forward > swing_forward_threshold)
            & (self.max_selected_foot_travel > swing_travel_threshold)
        )
        motion_ready = self.foot_lift_seen & self.foot_swing_seen
        deliberate_approach = motion_ready & (
            foot_toward_ball_speed > toward_speed_threshold
        )
        self.motion_ready_seen |= active & motion_ready
        new_lift = self.foot_lift_seen & ~previous_lift_seen
        new_swing = self.foot_swing_seen & ~previous_swing_seen

        self.selected_foot_distance = selected_distance
        self.other_foot_distance = other_distance
        self.selected_foot_height = selected_height
        self.minimum_selected_foot_distance = torch.where(
            active,
            torch.minimum(self.minimum_selected_foot_distance, selected_distance),
            self.minimum_selected_foot_distance,
        )
        foot_progress = torch.clamp(
            (self.previous_selected_foot_distance - selected_distance) / 0.035,
            -1.0,
            1.0,
        )

        phase = self.kick_phase
        windup = ((phase >= 0.10) & (phase < 0.42)).float()
        strike = ((phase >= 0.30) & (phase <= 0.76)).float()
        recovery = (phase > 0.74).float()
        lift_amount = torch.clamp(
            self.max_selected_foot_lift / max(lift_threshold * 2.5, 1.0e-4),
            0.0,
            1.5,
        )
        swing_amount = torch.clamp(
            torch.relu(foot_forward) / max(swing_forward_threshold * 2.0, 1.0e-4),
            0.0,
            1.5,
        )
        approach_amount = torch.clamp(
            torch.relu(foot_toward_ball_speed) / 0.25,
            0.0,
            1.5,
        )
        foot_contact_shell = torch.exp(
            -(selected_distance - 0.145).square() / (0.035 ** 2)
        )

        contact_distance = float(
            os.environ.get("V4K1_CONTACT_DISTANCE", "0.155")
        )
        impact_speed = float(os.environ.get("V4K1_IMPACT_SPEED", "0.060"))
        geometric_contact = (
            active & (selected_distance < contact_distance) & (phase >= 0.18)
        )
        self.geometric_contact_seen |= geometric_contact
        ball_response_now = (
            (positive_impulse > 0.025)
            | (ball_speed_gain > 0.035)
        )
        wrong_foot_proximity = active & (
            other_distance < contact_distance
        )
        body_ball_proximity = active & (
            after_state["robot_ball_distance"] < 0.27
        )
        # A geometric proximity proxy alone is not treated as a collision.
        # Wrong-foot/body violations require a simultaneous causal ball
        # response, preventing false permanent failures from coarse distances.
        wrong_foot_contact = wrong_foot_proximity & ball_response_now
        body_ball_contact = body_ball_proximity & ball_response_now
        precontact_ball_motion = precontact_ball_motion_mask(
            active=active,
            contact_seen=previous_contact_seen,
            before_ball_speed=before_ball_speed,
            before_ball_displacement=before_ball_displacement,
        )
        stable_at_contact = (
            (stability["vertical_speed"] < 0.08)
            & (stability["upright_error"] < 0.020)
            & (stability["support_count"] >= 3)
        )
        valid_contact = valid_kick_contact_mask(
            active=active,
            geometric_contact=geometric_contact,
            stable_at_contact=stable_at_contact,
            foot_lift_seen=self.foot_lift_seen,
            foot_swing_seen=self.foot_swing_seen,
            deliberate_approach=deliberate_approach,
            wrong_foot_contact=wrong_foot_contact,
            body_ball_contact=body_ball_contact,
            precontact_ball_motion=precontact_ball_motion,
        )
        invalid_contact, passive_selected_response = invalid_kick_contact_mask(
            geometric_contact=geometric_contact,
            valid_contact=valid_contact,
            ball_response_now=ball_response_now,
            wrong_foot_contact=wrong_foot_contact,
            body_ball_contact=body_ball_contact,
            precontact_ball_motion=precontact_ball_motion,
        )
        new_contact = valid_contact & ~previous_contact_seen
        contact_seen_after = previous_contact_seen | valid_contact
        self.contact_seen = contact_seen_after
        self.contact_duration_steps = next_valid_contact_duration(
            previous_duration=self.contact_duration_steps,
            contact_seen_after_update=contact_seen_after,
            geometric_contact=geometric_contact,
        )
        self.precontact_motion_seen |= precontact_ball_motion
        self.invalid_contact_seen |= invalid_contact
        self.body_contact_seen |= body_ball_contact
        self.wrong_foot_contact_seen |= wrong_foot_contact
        self.passive_response_seen |= passive_selected_response
        self.violation_seen |= invalid_contact

        impact_now = (
            self.contact_seen
            & (
                (direction_speed > impact_speed)
                | (positive_impulse > 0.035)
                | (ball_speed_gain > 0.050)
            )
            & (phase >= 0.24)
            & ~self.violation_seen
        )
        new_impact = impact_now & ~self.impact_seen
        self.impact_seen |= impact_now

        base_displacement = torch.linalg.vector_norm(
            after_state["robot_position"][:, :2] - self.kick_stance_xy,
            dim=1,
        )
        foot_returned = torch.abs(
            selected_pos[:, 2] - self.selected_foot_start_pos[:, 2]
        ) < 0.035
        recovery_stable = (
            active
            & self.impact_seen
            & (phase > 0.72)
            & stability["stable"]
            & foot_returned
        )
        self.recovery_stable_steps = torch.where(
            recovery_stable,
            self.recovery_stable_steps + 1,
            torch.where(
                active & (phase > 0.72),
                torch.zeros_like(self.recovery_stable_steps),
                self.recovery_stable_steps,
            ),
        )
        self.recovery_seen |= self.recovery_stable_steps >= 5

        kick_success = stable_kick_success_mask(
            valid_contact_seen=self.contact_seen,
            impact_seen=self.impact_seen,
            peak_direction_speed=self.peak_ball_direction_speed,
            directional_displacement=directional_displacement,
            contact_duration_steps=self.contact_duration_steps,
            violation_seen=self.violation_seen,
            base_displacement=base_displacement,
            recovery_seen=self.recovery_seen,
        )
        contact_stage_success = self.contact_seen & ~self.violation_seen
        directional_stage_success = (
            self.impact_seen
            & (self.peak_ball_direction_speed > 0.22)
            & (directional_displacement > 0.12)
            & ~self.violation_seen
        )

        unsafe_penalty = (
            1.80 * precontact_ball_motion.float()
            + 1.50 * passive_selected_response.float()
            + 1.40 * body_ball_contact.float()
            + 1.00 * wrong_foot_contact.float()
        )
        if self.curriculum_stage == "contact_discovery":
            # No positive standing baseline: progress must come from deliberate
            # single-leg lift, forward swing, approach speed, and valid contact.
            reward = (
                0.20 * windup * lift_amount
                + 0.30 * strike * swing_amount
                + 0.35 * strike * approach_amount
                + 0.30 * strike * torch.relu(foot_progress)
                + 0.35 * new_lift.float()
                + 0.55 * new_swing.float()
                + 0.20 * strike * foot_contact_shell * motion_ready.float()
                + 2.50 * new_contact.float()
                + 1.00 * new_impact.float()
                - unsafe_penalty
                - 0.25 * (~stability["stable"]).float()
                - 0.40 * torch.clamp(base_displacement / 0.10, 0.0, 2.0)
                - 0.030 * action_rate
                - 0.0004 * torque_penalty
            )
            stage_success = contact_stage_success
            terminal_bonus = 3.0 * stage_success.float()
            terminal_penalty = -0.8 * (terminal & ~stage_success).float()
        else:
            reward = (
                0.10 * pose_track
                + 0.08 * yaw_track
                + 0.06 * stability["stable"].float()
                + 0.16 * windup * lift_amount
                + 0.20 * strike * swing_amount
                + 0.24 * strike * approach_amount
                + 0.18 * strike * torch.relu(foot_progress)
                + 0.12 * strike * foot_contact_shell * motion_ready.float()
                + 1.50 * new_contact.float()
                + 1.80 * new_impact.float()
                + 0.85 * self.contact_seen.float()
                * torch.clamp(direction_speed / 0.45, 0.0, 1.5)
                + 0.60 * self.contact_seen.float()
                * torch.clamp(directional_displacement / 0.35, 0.0, 1.5)
                + 0.20 * recovery * self.recovery_seen.float()
                - unsafe_penalty
                - 0.55 * torch.clamp(base_displacement / 0.10, 0.0, 2.0)
                - 0.20 * torch.clamp(lateral_speed / 0.25, 0.0, 2.0)
                - 0.15 * torch.clamp(lateral_displacement / 0.20, 0.0, 2.0)
                - 0.020 * action_rate
                - 0.0004 * torque_penalty
            )
            if self.curriculum_stage == "directional_kick":
                stage_success = directional_stage_success
                terminal_bonus = 3.5 * stage_success.float()
                terminal_penalty = -1.2 * (terminal & ~stage_success).float()
            else:
                stage_success = kick_success
                terminal_bonus = 4.5 * kick_success.float()
                terminal_penalty = -1.5 * (terminal & ~kick_success).float()

        reward = reward + terminal_bonus + terminal_penalty
        reward = torch.where(active, reward, torch.zeros_like(reward))
        reward[base_done] = -3.0
        reward = torch.clamp(reward, -5.0, 7.0)

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
            "minimum_selected_foot_distance": self.minimum_selected_foot_distance,
            "selected_foot_height": selected_height,
            "foot_progress": foot_progress,
            "foot_toward_ball_speed": foot_toward_ball_speed,
            "max_foot_toward_ball_speed": self.max_foot_toward_ball_speed,
            "base_displacement": base_displacement,
            "geometric_contact": geometric_contact.float(),
            "valid_contact": valid_contact.float(),
            "ball_response_now": ball_response_now.float(),
            "wrong_foot_proximity": wrong_foot_proximity.float(),
            "body_ball_proximity": body_ball_proximity.float(),
            "passive_selected_response": passive_selected_response.float(),
            "contact_seen": self.contact_seen.float(),
            "impact_seen": self.impact_seen.float(),
            "violation_seen": self.violation_seen.float(),
            "precontact_motion_seen": self.precontact_motion_seen.float(),
            "invalid_contact_seen": self.invalid_contact_seen.float(),
            "body_contact_seen": self.body_contact_seen.float(),
            "wrong_foot_contact_seen": self.wrong_foot_contact_seen.float(),
            "geometric_contact_seen": self.geometric_contact_seen.float(),
            "foot_lift_seen": self.foot_lift_seen.float(),
            "foot_swing_seen": self.foot_swing_seen.float(),
            "motion_ready_seen": self.motion_ready_seen.float(),
            "passive_response_seen": self.passive_response_seen.float(),
            "recovery_seen": self.recovery_seen.float(),
            "stable_activation_seen": self.stable_activation_seen.float(),
            "kick_active": active.float(),
            "support_count": stability["support_count"].float(),
            "vertical_speed": stability["vertical_speed"],
            "stage_success": stage_success.float(),
            "success": kick_success.float(),
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
        requested_actions = torch.clamp(
            low_actions.to(self.device), -1.0, 1.0
        )
        applied_actions = torch.where(
            self.kick_active.unsqueeze(1),
            requested_actions,
            torch.zeros_like(requested_actions),
        )
        self.command_tensor[:, :3] = 0.0
        self._update_kick_target()
        before_state = self._state()
        result = self.low_env.step(applied_actions.detach().clone())
        native_obs, native_reward, native_done, native_extras = normalize_step_result(
            result
        )
        base_done = normalize_done_tensor(
            native_done, num_envs=self.num_envs, device=self.device
        )
        self._native_low_obs = extract_observation(native_obs)
        self._native_low_obs[:, self.NATIVE_COMMAND_SLICE] = 0.0

        active_before = self.kick_active.clone()
        self.active_kick_step_buf[active_before] += 1
        self.low_episode_length_buf[active_before] = (
            self.active_kick_step_buf[active_before]
        )
        settle_timeout, arming_violation = self._advance_pre_kick_state()
        terminal = self.kick_active & (
            self.active_kick_step_buf >= self.kick_episode_steps
        )
        after_state = self._state()
        reward, tensors = self._kick_reward(
            before_state=before_state,
            after_state=after_state,
            low_actions=applied_actions,
            base_done=base_done,
            terminal=terminal,
        )
        done = base_done | terminal | settle_timeout | arming_violation
        timeouts = terminal & ~base_done

        completed = int(done.sum())
        successful = int((done & (tensors["success"] > 0.5)).sum())
        stage_successes = int(
            (done & (tensors["stage_success"] > 0.5)).sum()
        )
        contact_successes = int((done & self.contact_seen).sum())
        impact_successes = int((done & self.impact_seen).sum())
        lift_successes = int((done & self.foot_lift_seen).sum())
        swing_successes = int((done & self.foot_swing_seen).sum())
        motion_ready_successes = int((done & self.motion_ready_seen).sum())
        geometric_contact_episodes = int(
            (done & self.geometric_contact_seen).sum()
        )
        passive_response_episodes = int(
            (done & self.passive_response_seen).sum()
        )
        precontact_motion_episodes = int(
            (done & self.precontact_motion_seen).sum()
        )
        invalid_contact_episodes = int(
            (done & self.invalid_contact_seen).sum()
        )
        body_contact_episodes = int(
            (done & self.body_contact_seen).sum()
        )
        wrong_foot_contact_episodes = int(
            (done & self.wrong_foot_contact_seen).sum()
        )
        violation_episodes = int((done & self.violation_seen).sum())
        stable_activation_episodes = int(
            (done & self.stable_activation_seen).sum()
        )
        completed_peak_sum = (
            float(tensors["peak_direction_speed"][done].sum())
            if completed else 0.0
        )
        completed_displacement_sum = (
            float(tensors["directional_displacement"][done].sum())
            if completed else 0.0
        )
        completed_min_distance_sum = (
            float(tensors["minimum_selected_foot_distance"][done].sum())
            if completed else 0.0
        )
        completed_max_toward_speed_sum = (
            float(tensors["max_foot_toward_ball_speed"][done].sum())
            if completed else 0.0
        )

        self.previous_ball_direction_speed = tensors["direction_speed"].clone()
        self.previous_selected_foot_distance = tensors[
            "selected_foot_distance"
        ].clone()
        selected_pos, _ = self._front_foot_positions()
        self.previous_selected_foot_pos = selected_pos.clone()
        self.previous_low_action = applied_actions
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
            "minimum_selected_foot_distance": float(tensors["minimum_selected_foot_distance"].mean()),
            "selected_foot_height": float(tensors["selected_foot_height"].mean()),
            "foot_progress": float(tensors["foot_progress"].mean()),
            "foot_toward_ball_speed": float(tensors["foot_toward_ball_speed"].mean()),
            "max_foot_toward_ball_speed": float(tensors["max_foot_toward_ball_speed"].mean()),
            "base_displacement": float(tensors["base_displacement"].mean()),
            "geometric_contact_fraction": float(tensors["geometric_contact"].mean()),
            "valid_contact_fraction": float(tensors["valid_contact"].mean()),
            "ball_response_fraction": float(tensors["ball_response_now"].mean()),
            "wrong_foot_proximity_fraction": float(tensors["wrong_foot_proximity"].mean()),
            "body_ball_proximity_fraction": float(tensors["body_ball_proximity"].mean()),
            "passive_response_fraction": float(tensors["passive_selected_response"].mean()),
            "precontact_motion_fraction": float(tensors["precontact_motion_seen"].mean()),
            "invalid_contact_fraction": float(tensors["invalid_contact_seen"].mean()),
            "body_contact_fraction": float(tensors["body_contact_seen"].mean()),
            "wrong_foot_contact_fraction": float(tensors["wrong_foot_contact_seen"].mean()),
            "contact_fraction": float(tensors["contact_seen"].mean()),
            "impact_fraction": float(tensors["impact_seen"].mean()),
            "foot_lift_fraction": float(tensors["foot_lift_seen"].mean()),
            "foot_swing_fraction": float(tensors["foot_swing_seen"].mean()),
            "motion_ready_fraction": float(tensors["motion_ready_seen"].mean()),
            "violation_fraction": float(tensors["violation_seen"].mean()),
            "stable_activation_fraction": float(tensors["stable_activation_seen"].mean()),
            "kick_active_fraction": float(tensors["kick_active"].mean()),
            "support_count": float(tensors["support_count"].mean()),
            "vertical_speed": float(tensors["vertical_speed"].mean()),
            "completed_episodes": float(completed),
            "successful_episodes": float(successful),
            "stage_successful_episodes": float(stage_successes),
            "contact_successes": float(contact_successes),
            "impact_successes": float(impact_successes),
            "lift_successes": float(lift_successes),
            "swing_successes": float(swing_successes),
            "motion_ready_successes": float(motion_ready_successes),
            "geometric_contact_episodes": float(geometric_contact_episodes),
            "passive_response_episodes": float(passive_response_episodes),
            "precontact_motion_episodes": float(precontact_motion_episodes),
            "invalid_contact_episodes": float(invalid_contact_episodes),
            "body_contact_episodes": float(body_contact_episodes),
            "wrong_foot_contact_episodes": float(wrong_foot_contact_episodes),
            "violation_episodes": float(violation_episodes),
            "stable_activation_episodes": float(stable_activation_episodes),
            "completed_peak_speed_sum": completed_peak_sum,
            "completed_displacement_sum": completed_displacement_sum,
            "completed_min_distance_sum": completed_min_distance_sum,
            "completed_max_toward_speed_sum": completed_max_toward_speed_sum,
            "settle_timeout_fraction": float(settle_timeout.float().mean()),
            "arming_violation_fraction": float(arming_violation.float().mean()),
            "base_done_fraction": float(base_done.float().mean()),
            "native_reward": float(
                torch.as_tensor(native_reward, device=self.device).float().mean()
            ),
        }
        for key in (
            "selected_load_ratio",
            "selected_fz",
            "support3_fz",
            "total4_fz",
            "load_ready_seen",
            "min_selected_load_ratio",
            "max_unload_hold_steps",
            "prelift_unload_seen",
            "max_prelift_unload_hold_steps",
            "bridge_now_r852",
            "bridge_seen_r852",
            "max_bridge_hold_steps_r852",
        ):
            if key in tensors:
                diagnostics[key] = float(
                    tensors[key].mean()
                )

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
            direction_limit = 0.025
            strength_min, strength_max = 0.78, 0.90
        elif self.curriculum_stage == "directional_kick":
            direction_limit = 0.20
            strength_min, strength_max = 0.75, 1.00
        else:
            direction_limit = 0.32
            strength_min, strength_max = 0.65, 1.00
        angle_offset = (2.0 * self._rand((count,)) - 1.0) * direction_limit
        strength = strength_min + (strength_max - strength_min) * self._rand((count,))

        robot_z = self.robot.data.root_pos_w[env_ids, 2]
        robot_pose = torch.zeros(count, 7, device=self.device)
        robot_pose[:, :2] = origins[:, :2]
        robot_pose[:, 2] = robot_z
        robot_pose[:, 3] = 1.0
        zero_velocity = torch.zeros(count, 6, device=self.device)
        self.robot.write_root_pose_to_sim(robot_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)

        # Keep the ball far away during the entire robot-settling window.
        ball_z = self.ball.data.root_pos_w[env_ids, 2]
        ball_pose = torch.zeros(count, 7, device=self.device)
        ball_pose[:, 0] = origins[:, 0] + 2.20
        ball_pose[:, 1] = origins[:, 1] + 1.80
        ball_pose[:, 2] = ball_z
        ball_pose[:, 3] = 1.0
        self.ball.write_root_pose_to_sim(ball_pose, env_ids=env_ids)
        self.ball.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)

        self.kick_leg[env_ids] = 0
        self.strike_strength[env_ids] = strength
        self.strike_angle_offset[env_ids] = angle_offset
        self.strike_direction_world[env_ids] = torch.tensor([1.0, 0.0], device=self.device)
        self.kick_ball_start_xy[env_ids] = ball_pose[:, :2]
        self.kick_stance_xy[env_ids] = origins[:, :2]
        self.kick_stance_yaw[env_ids] = 0.0
        self.kick_target_yaw[env_ids] = 0.0
        self.command_tensor[env_ids, :3] = 0.0
        self.previous_low_action[env_ids] = 0.0
        self.skill_id[env_ids] = self.SKILL_KICK
        self.target_terminal_velocity[env_ids] = 0.0
        self.low_episode_length_buf[env_ids] = 0
        self.episode_length_buf[env_ids] = 0
        self.settle_step_buf[env_ids] = 0
        self.stable_consecutive_buf[env_ids] = 0
        self.ball_arm_step_buf[env_ids] = 0
        self.active_kick_step_buf[env_ids] = 0
        self.ball_placed[env_ids] = False
        self.kick_active[env_ids] = False
        self.stable_activation_seen[env_ids] = False
        self.settle_timeout_seen[env_ids] = False
        self.previous_ball_direction_speed[env_ids] = 0.0
        self.peak_ball_direction_speed[env_ids] = 0.0
        self.contact_seen[env_ids] = False
        self.impact_seen[env_ids] = False
        self.geometric_contact_seen[env_ids] = False
        self.motion_ready_seen[env_ids] = False
        self.passive_response_seen[env_ids] = False
        self.violation_seen[env_ids] = False
        self.precontact_motion_seen[env_ids] = False
        self.invalid_contact_seen[env_ids] = False
        self.body_contact_seen[env_ids] = False
        self.wrong_foot_contact_seen[env_ids] = False
        self.foot_lift_seen[env_ids] = False
        self.foot_swing_seen[env_ids] = False
        self.recovery_seen[env_ids] = False
        self.recovery_stable_steps[env_ids] = 0
        self.contact_duration_steps[env_ids] = 0
        self.selected_foot_distance[env_ids] = 9.0
        self.other_foot_distance[env_ids] = 9.0
        self.selected_foot_height[env_ids] = 0.0
        self.previous_selected_foot_distance[env_ids] = 0.24
        self.minimum_selected_foot_distance[env_ids] = 9.0
        self.selected_foot_start_pos[env_ids] = 0.0
        self.previous_selected_foot_pos[env_ids] = 0.0
        self.max_selected_foot_lift[env_ids] = 0.0
        self.max_selected_foot_travel[env_ids] = 0.0
        self.max_foot_toward_ball_speed[env_ids] = 0.0

        self._native_low_obs = extract_observation(self.low_env.get_observations())
        self._native_low_obs[:, self.NATIVE_COMMAND_SLICE] = 0.0
        state = self._state()
        self.target_height[env_ids] = state["robot_position"][env_ids, 2]
        self._update_kick_target()
        self.total_resets += count

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


def create_v4k1_runner_cfg(
    *,
    base_cfg: Any,
    run_name: str,
    actor_hidden_dims: list[int],
    critic_hidden_dims: list[int],
) -> dict[str, Any]:
    cfg = base_cfg.to_dict()
    cfg["num_steps_per_env"] = int(os.environ.get("V4K1_ROLLOUT_STEPS", "60"))
    cfg["save_interval"] = 10_000
    cfg["experiment_name"] = "go2_soccer_v4k1_r5_stable_kick"
    cfg["run_name"] = run_name
    cfg["obs_groups"] = {"policy": ["policy"], "critic": ["policy"]}
    cfg["policy"]["init_noise_std"] = float(os.environ.get("V4K1_INIT_NOISE_STD", "0.35"))
    cfg["policy"]["actor_hidden_dims"] = list(actor_hidden_dims)
    cfg["policy"]["critic_hidden_dims"] = list(critic_hidden_dims)
    cfg["policy"]["actor_obs_normalization"] = False
    cfg["policy"]["critic_obs_normalization"] = False
    cfg["algorithm"]["entropy_coef"] = float(os.environ.get("V4K1_ENTROPY_COEF", "0.010"))
    cfg["algorithm"]["learning_rate"] = float(os.environ.get("V4K1_LEARNING_RATE", "0.000030"))
    cfg["algorithm"]["clip_param"] = float(os.environ.get("V4K1_CLIP_PARAM", "0.20"))
    cfg["algorithm"]["num_learning_epochs"] = int(os.environ.get("V4K1_EPOCHS", "3"))
    cfg["algorithm"]["num_mini_batches"] = int(os.environ.get("V4K1_MINI_BATCHES", "4"))
    cfg["algorithm"]["gamma"] = float(os.environ.get("V4K1_GAMMA", "0.995"))
    cfg["algorithm"]["lam"] = 0.95
    cfg["algorithm"]["desired_kl"] = float(os.environ.get("V4K1_DESIRED_KL", "0.010"))
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




def infer_v4k1_source_architecture(
    checkpoint: Path,
) -> dict[str, Any]:
    """Infer the exact actor/critic MLP widths from the source checkpoint.

    Supported sources:
      * 62-D V4D checkpoint for the initial isolated-kick bootstrap.
      * 70-D isolated-kick checkpoint for Pilot -> Full continuation.
    """

    state = load_state_dictionary(checkpoint)

    def ordered_linear_weights(
        prefix: str,
    ) -> list[tuple[int, str, torch.Tensor]]:
        rows: list[tuple[int, str, torch.Tensor]] = []
        pattern = re.compile(
            rf"^{re.escape(prefix)}\.(\d+)\.weight$"
        )
        for key, value in state.items():
            if not torch.is_tensor(value) or value.ndim != 2:
                continue
            match = pattern.fullmatch(key)
            if match is not None:
                rows.append((int(match.group(1)), key, value))
        rows.sort(key=lambda item: item[0])
        return rows

    actor_rows = ordered_linear_weights("actor")
    critic_rows = ordered_linear_weights("critic")

    if len(actor_rows) < 2 or len(critic_rows) < 2:
        raise RuntimeError(
            "V4K1-R5 could not infer the source MLP architecture. "
            f"actor_layers={[key for _, key, _ in actor_rows]} "
            f"critic_layers={[key for _, key, _ in critic_rows]}"
        )

    actor_input = int(actor_rows[0][2].shape[1])
    critic_input = int(critic_rows[0][2].shape[1])
    actor_output = int(actor_rows[-1][2].shape[0])
    critic_output = int(critic_rows[-1][2].shape[0])
    actor_hidden = [
        int(value.shape[0])
        for _, _, value in actor_rows[:-1]
    ]
    critic_hidden = [
        int(value.shape[0])
        for _, _, value in critic_rows[:-1]
    ]

    allowed_inputs = {
        62,
        IsolatedDeltaPoseKickEnv.KICK_OBS_DIM,
    }
    if (
        actor_input not in allowed_inputs
        or critic_input != actor_input
    ):
        raise RuntimeError(
            "V4K1-R5 source observation dimension is incompatible. "
            "Expected a 62-D V4D bootstrap checkpoint or a 70-D "
            "isolated-kick continuation checkpoint. "
            f"actor_input={actor_input} critic_input={critic_input}"
        )
    if actor_output != 12 or critic_output != 1:
        raise RuntimeError(
            "V4K1-R5 source output dimensions are incompatible. "
            f"actor_output={actor_output} critic_output={critic_output}"
        )
    if actor_hidden != critic_hidden:
        raise RuntimeError(
            "V4K1-R5 requires matching actor/critic hidden widths. "
            f"actor_hidden={actor_hidden} "
            f"critic_hidden={critic_hidden}"
        )

    return {
        "actor_input": actor_input,
        "critic_input": critic_input,
        "actor_output": actor_output,
        "critic_output": critic_output,
        "actor_hidden_dims": actor_hidden,
        "critic_hidden_dims": critic_hidden,
        "actor_weight_shapes": {
            key: tuple(value.shape)
            for _, key, value in actor_rows
        },
        "critic_weight_shapes": {
            key: tuple(value.shape)
            for _, key, value in critic_rows
        },
    }




def warm_start_v4k1_policy(
    *,
    runner: Any,
    checkpoint: Path,
    requested_std: float,
) -> None:
    """Bootstrap from V4D (62-D) or resume from Kick Pilot (70-D)."""

    source = load_state_dictionary(checkpoint)
    module = policy_module(runner)
    target = module.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    expanded: list[str] = []

    source_actor_first = source.get("actor.0.weight")
    source_critic_first = source.get("critic.0.weight")
    if (
        not torch.is_tensor(source_actor_first)
        or not torch.is_tensor(source_critic_first)
        or source_actor_first.ndim != 2
        or source_critic_first.ndim != 2
    ):
        raise RuntimeError(
            "V4K1-R5 source checkpoint lacks actor/critic input layers."
        )

    source_obs = int(source_actor_first.shape[1])
    critic_source_obs = int(source_critic_first.shape[1])
    allowed_source_obs = {
        62,
        IsolatedDeltaPoseKickEnv.KICK_OBS_DIM,
    }
    if (
        source_obs not in allowed_source_obs
        or critic_source_obs != source_obs
    ):
        raise RuntimeError(
            "V4K1-R5 source observation dimension is incompatible. "
            f"actor_input={source_obs} "
            f"critic_input={critic_source_obs}"
        )

    def noise_parameter(name: str) -> bool:
        return name.rsplit(".", 1)[-1] in {
            "std",
            "log_std",
            "_std",
            "_log_std",
        }

    for key, source_value in source.items():
        if key not in target:
            continue
        if noise_parameter(key) and source_obs == 62:
            continue
        target_value = target[key]

        # Pilot -> Full: exact 70-D continuation.
        if target_value.shape == source_value.shape:
            compatible[key] = source_value
            continue

        # V4D -> isolated Kick: expand only the first Actor/Critic inputs.
        if (
            target_value.ndim == 2
            and source_value.ndim == 2
            and target_value.shape[0] == source_value.shape[0]
            and source_value.shape[1] == 62
            and target_value.shape[1]
            == IsolatedDeltaPoseKickEnv.KICK_OBS_DIM
        ):
            source_on_target = source_value.to(target_value.device)
            mapped = torch.zeros_like(target_value)

            # Native proprioception and direct pose/terminal context.
            mapped[:, :56] = source_on_target[:, :56]

            # Existing stand/locomotion/push one-hot columns.
            mapped[:, 56:59] = source_on_target[:, 56:59]

            # Column 59 is the new Kick one-hot and starts at zero.
            # Move V4D ball-relative columns from 59:61 to 60:62.
            mapped[:, 60:62] = source_on_target[:, 59:61]

            # Legacy command slots, teacher flag, and new Kick context
            # remain zero-initialized.
            mapped[
                :,
                IsolatedDeltaPoseKickEnv.NATIVE_COMMAND_SLICE,
            ] = 0.0

            compatible[key] = mapped
            expanded.append(key)

    required_input_layers = {
        "actor.0.weight",
        "critic.0.weight",
    }
    bootstrap_62_to_70 = (
        source_obs == 62
        and required_input_layers.issubset(set(expanded))
    )
    resume_70_to_70 = (
        source_obs == IsolatedDeltaPoseKickEnv.KICK_OBS_DIM
        and required_input_layers.issubset(set(compatible))
    )

    if (
        len(compatible) < 8
        or not (bootstrap_62_to_70 or resume_70_to_70)
    ):
        source_2d = {
            key: tuple(value.shape)
            for key, value in source.items()
            if torch.is_tensor(value) and value.ndim == 2
        }
        target_2d = {
            key: tuple(value.shape)
            for key, value in target.items()
            if torch.is_tensor(value) and value.ndim == 2
        }
        raise RuntimeError(
            "V4K1-R5 warm start could not bootstrap or resume. "
            f"source_obs={source_obs} "
            f"compatible={len(compatible)} "
            f"expanded={expanded} "
            f"source_2d={source_2d} "
            f"target_2d={target_2d}"
        )

    result = torch.nn.Module.load_state_dict(
        module,
        compatible,
        strict=False,
    )
    if resume_70_to_70:
        std_reset = False
        source_std_preserved = True
    else:
        std_reset = set_action_std(
            module, requested_std
        )
        source_std_preserved = False
    runner.current_learning_iteration = 0

    warm_start_mode = (
        "bootstrap_62_to_70"
        if bootstrap_62_to_70
        else "resume_70_to_70"
    )
    new_context_zero_initialized = bootstrap_62_to_70

    print(
        "[V4K1_WARM_START] "
        f"checkpoint={checkpoint} "
        f"compatible_parameters={len(compatible)} "
        f"expanded_input_layers={expanded} "
        f"warm_start_mode={warm_start_mode} "
        f"source_obs={source_obs} "
        f"target_obs={IsolatedDeltaPoseKickEnv.KICK_OBS_DIM} "
        "legacy_command_columns_zeroed=true "
        "new_kick_context_columns_zero_initialized="
        f"{str(new_context_zero_initialized).lower()} "
        f"missing_keys={len(result.missing_keys)} "
        f"unexpected_keys={len(result.unexpected_keys)} "
        "optimizer_loaded=false "
        f"action_std_reset={std_reset} "
        f"requested_std={requested_std}",
        flush=True,
    )



def read_v4k1_schedule(
    iterations: int,
) -> list[dict[str, Any]]:
    s1 = int(os.environ.get("V4K1_STAGE1_ITERATIONS", "0"))
    s2 = int(os.environ.get("V4K1_STAGE2_ITERATIONS", "0"))
    s3 = int(os.environ.get("V4K1_STAGE3_ITERATIONS", "0"))
    if s1 + s2 + s3 != iterations or min(s1, s2, s3) <= 0:
        raise RuntimeError(
            "V4K1 stage counts must be positive and sum to max_iterations. "
            f"stages={s1},{s2},{s3} iterations={iterations}"
        )
    return [
        {
            "name": "contact_discovery",
            "start": 0,
            "end": s1,
            "lr": float(
                os.environ.get("V4K1_LR_STAGE1", "0.000030")
            ),
            "action_std": float(
                os.environ.get("V4K1_STD_STAGE1", "0.35")
            ),
        },
        {
            "name": "directional_kick",
            "start": s1,
            "end": s1 + s2,
            "lr": float(
                os.environ.get("V4K1_LR_STAGE2", "0.000015")
            ),
            "action_std": float(
                os.environ.get("V4K1_STD_STAGE2", "0.25")
            ),
        },
        {
            "name": "recovery_generalization",
            "start": s1 + s2,
            "end": iterations,
            "lr": float(
                os.environ.get("V4K1_LR_STAGE3", "0.000005")
            ),
            "action_std": float(
                os.environ.get("V4K1_STD_STAGE3", "0.15")
            ),
        },
    ]


def v4k1_stage_for_iteration(schedule: list[dict[str, Any]], iteration: int) -> dict[str, Any]:
    for stage in schedule:
        if int(stage["start"]) <= iteration < int(stage["end"]):
            return stage
    raise RuntimeError(f"No V4K1 stage contains iteration {iteration}")




def run_v4k1_stable_reset_guard(env: IsolatedDeltaPoseKickEnv) -> None:
    """Physically verify stable ground support before any kick is activated."""
    env.reset()
    zero = torch.zeros(env.num_envs, env.low_num_actions, device=env.device)
    activated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    maximum_steps = env.settle_max_steps + env.ball_arm_steps + 12
    minimum_activation_rate = float(
        os.environ.get("V4K1_MIN_STABLE_ACTIVATION_RATE", "0.95")
    )
    if not (0.0 < minimum_activation_rate <= 1.0):
        raise ValueError(
            "V4K1_MIN_STABLE_ACTIVATION_RATE must be in (0, 1]. "
            f"observed={minimum_activation_rate}"
        )

    max_preactive_ball_speed = 0.0
    max_preactive_vertical_speed = 0.0
    max_settle_timeout_rate = 0.0
    max_arming_violation_rate = 0.0
    max_violation_rate = 0.0
    max_base_done_rate = 0.0

    with torch.no_grad():
        for _ in range(maximum_steps):
            preactive = ~env.kick_active.clone()
            if bool(preactive.any()):
                ball_speed = torch.linalg.vector_norm(
                    env.ball.data.root_lin_vel_w[:, :2], dim=1
                )
                vertical_speed = env.robot.data.root_lin_vel_w[:, 2].abs()
                max_preactive_ball_speed = max(
                    max_preactive_ball_speed,
                    float(ball_speed[preactive].max()),
                )
                max_preactive_vertical_speed = max(
                    max_preactive_vertical_speed,
                    float(vertical_speed[preactive].max()),
                )

            _, _, _, _, diag = env.kick_step(zero)
            activated |= env.kick_active
            max_settle_timeout_rate = max(
                max_settle_timeout_rate,
                float(diag["settle_timeout_fraction"]),
            )
            max_arming_violation_rate = max(
                max_arming_violation_rate,
                float(diag["arming_violation_fraction"]),
            )
            max_violation_rate = max(
                max_violation_rate,
                float(diag["violation_fraction"]),
            )
            max_base_done_rate = max(
                max_base_done_rate,
                float(diag["base_done_fraction"]),
            )
            if bool(activated.all()):
                break

    activation_rate = float(activated.float().mean())
    never_activated_envs = (
        torch.nonzero(~activated, as_tuple=False)
        .squeeze(-1)
        .detach()
        .cpu()
        .tolist()
    )
    print(
        "[V4K1_R5_STABLE_RESET_GUARD] "
        f"activation_rate={activation_rate:.9f} "
        f"required_activation_rate={minimum_activation_rate:.9f} "
        f"settle_max_steps={env.settle_max_steps} "
        f"guard_steps={maximum_steps} "
        f"preactive_ball_speed_max={max_preactive_ball_speed:.9f} "
        f"preactive_vertical_speed_max={max_preactive_vertical_speed:.9f} "
        f"settle_timeout_rate_max={max_settle_timeout_rate:.9f} "
        f"arming_violation_rate_max={max_arming_violation_rate:.9f} "
        f"violation_rate_max={max_violation_rate:.9f} "
        f"base_done_rate_max={max_base_done_rate:.9f} "
        f"never_activated_envs={never_activated_envs}",
        flush=True,
    )
    if activation_rate < minimum_activation_rate:
        raise RuntimeError(
            "V4K1-R5 stable-reset activation guard failed. "
            f"required={minimum_activation_rate} observed={activation_rate} "
            f"never_activated_envs={never_activated_envs}"
        )
    if max_settle_timeout_rate > 0.0:
        raise RuntimeError(
            "V4K1-R5 stable-reset settle timeout occurred. "
            f"rate={max_settle_timeout_rate}"
        )
    if max_arming_violation_rate > 0.0:
        raise RuntimeError(
            "V4K1-R5 ball-arming violation occurred. "
            f"rate={max_arming_violation_rate}"
        )
    if max_preactive_ball_speed > 0.05 or max_violation_rate > 0.0:
        raise RuntimeError(
            "V4K1-R5 pre-kick motion guard failed. "
            f"ball_speed={max_preactive_ball_speed} "
            f"violation={max_violation_rate}"
        )
    if max_base_done_rate > 0.0:
        raise RuntimeError(
            "V4K1-R5 robot terminated during stable-reset guard. "
            f"base_done_rate={max_base_done_rate}"
        )
    env.reset()


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

    def stage_mean(
        stage_name: str,
        key: str,
        *,
        window: int = 12,
    ) -> float | None:
        values = [
            float(record[key])
            for record in records
            if (
                record.get("stage") == stage_name
                and isinstance(record.get(key), (int, float))
            )
        ]
        if not values:
            return None
        values = values[-min(window, len(values)):]
        return sum(values) / len(values)

    def close_stage(stage_name: str) -> None:
        nonlocal stage_before
        if stage_before is None:
            raise RuntimeError(
                f"V4K1 stage snapshot missing for {stage_name}"
            )
        delta = parameter_delta(stage_before, runner)
        stage_deltas[stage_name] = delta
        print(
            "[V4K1_STAGE_DELTA] "
            f"stage={stage_name} l2={delta['l2']:.9f} "
            f"max_abs={delta['max_abs']:.9f}",
            flush=True,
        )

        if stage_name == "contact_discovery":
            observed_lift = stage_mean(stage_name, "episode_foot_lift_rate")
            observed_swing = stage_mean(stage_name, "episode_foot_swing_rate")
            observed_contact = stage_mean(stage_name, "episode_contact_rate")
            required_lift = float(os.environ.get("V4K1_STAGE1_MIN_LIFT_RATE", "0.0"))
            required_swing = float(os.environ.get("V4K1_STAGE1_MIN_SWING_RATE", "0.0"))
            required_contact = float(os.environ.get("V4K1_STAGE1_MIN_CONTACT_RATE", "0.0"))
            for metric, required, observed in (
                ("foot_lift_rate", required_lift, observed_lift),
                ("foot_swing_rate", required_swing, observed_swing),
                ("contact_rate", required_contact, observed_contact),
            ):
                print(
                    "[V4K1_STAGE_GATE] "
                    f"stage={stage_name} metric={metric} "
                    f"required={required} observed={observed}",
                    flush=True,
                )
                if required > 0.0 and (observed is None or observed < required):
                    raise RuntimeError(
                        "V4K1 contact-discovery subskill gate failed. "
                        f"metric={metric} required={required} observed={observed}"
                    )

        if stage_name == "directional_kick":
            observed_impact = stage_mean(
                stage_name,
                "episode_impact_rate",
            )
            observed_kick = stage_mean(
                stage_name,
                "kick_success_rate",
            )
            required_impact = float(
                os.environ.get(
                    "V4K1_STAGE2_MIN_IMPACT_RATE",
                    "0.0",
                )
            )
            required_kick = float(
                os.environ.get(
                    "V4K1_STAGE2_MIN_KICK_RATE",
                    "0.0",
                )
            )
            print(
                "[V4K1_STAGE_GATE] "
                f"stage={stage_name} metric=impact_rate "
                f"required={required_impact} "
                f"observed={observed_impact}",
                flush=True,
            )
            print(
                "[V4K1_STAGE_GATE] "
                f"stage={stage_name} metric=kick_success_rate "
                f"required={required_kick} "
                f"observed={observed_kick}",
                flush=True,
            )
            if required_impact > 0.0 and (
                observed_impact is None
                or observed_impact < required_impact
            ):
                raise RuntimeError(
                    "V4K1 directional-impact stage gate failed. "
                    f"required={required_impact} "
                    f"observed={observed_impact}"
                )
            if required_kick > 0.0 and (
                observed_kick is None
                or observed_kick < required_kick
            ):
                raise RuntimeError(
                    "V4K1 directional-kick stage gate failed. "
                    f"required={required_kick} "
                    f"observed={observed_kick}"
                )

    for iteration in range(iterations):
        stage = v4k1_stage_for_iteration(
            schedule,
            iteration,
        )
        stage_name = str(stage["name"])
        if stage_name != active_stage:
            if active_stage is not None:
                close_stage(active_stage)
            active_stage = stage_name
            stage_before = snapshot_policy(runner)
            env.set_curriculum_stage(stage_name)
            env.reset()
            observations = adapter.get_observations().to(
                runner.device
            )
            std_reset = set_action_std(
                policy_module(runner),
                float(stage["action_std"]),
            )
            print(
                "[V4K1_STAGE_START] "
                f"stage={stage_name} iteration={iteration} "
                f"learning_rate={float(stage['lr']):.9g} "
                f"action_std={float(stage['action_std']):.6f} "
                f"action_std_reset={std_reset}",
                flush=True,
            )

        set_optimizer_learning_rate(
            runner,
            float(stage["lr"]),
        )
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
            "minimum_selected_foot_distance": 0.0,
            "selected_foot_height": 0.0,
            "foot_progress": 0.0,
            "foot_toward_ball_speed": 0.0,
            "max_foot_toward_ball_speed": 0.0,
            "base_displacement": 0.0,
            "geometric_contact_fraction": 0.0,
            "valid_contact_fraction": 0.0,
            "ball_response_fraction": 0.0,
            "wrong_foot_proximity_fraction": 0.0,
            "body_ball_proximity_fraction": 0.0,
            "passive_response_fraction": 0.0,
            "precontact_motion_fraction": 0.0,
            "invalid_contact_fraction": 0.0,
            "body_contact_fraction": 0.0,
            "wrong_foot_contact_fraction": 0.0,
            "foot_lift_fraction": 0.0,
            "foot_swing_fraction": 0.0,
            "motion_ready_fraction": 0.0,
            "contact_fraction": 0.0,
            "impact_fraction": 0.0,
            "violation_fraction": 0.0,
            "stable_activation_fraction": 0.0,
            "kick_active_fraction": 0.0,
            "support_count": 0.0,
            "vertical_speed": 0.0,
            "settle_timeout_fraction": 0.0,
            "arming_violation_fraction": 0.0,
            "base_done_fraction": 0.0,
            "native_reward": 0.0,
        }
        completed = 0.0
        successful = 0.0
        stage_successful = 0.0
        contact_successes = 0.0
        impact_successes = 0.0
        lift_successes = 0.0
        swing_successes = 0.0
        motion_ready_successes = 0.0
        geometric_contact_episodes = 0.0
        passive_response_episodes = 0.0
        precontact_motion_episodes = 0.0
        invalid_contact_episodes = 0.0
        body_contact_episodes = 0.0
        wrong_foot_contact_episodes = 0.0
        violation_episodes = 0.0
        stable_activation_episodes = 0.0
        peak_sum = 0.0
        displacement_sum = 0.0
        min_distance_sum = 0.0
        steps = int(runner.cfg["num_steps_per_env"])

        completed_env_seen = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        with torch.inference_mode():
            for _ in range(steps):
                actions = runner.alg.act(observations)
                (
                    next_obs,
                    rewards,
                    dones,
                    extras,
                    diag,
                ) = env.kick_step(actions.to(env.device))
                next_obs = next_obs.to(runner.device)
                runner.alg.process_env_step(
                    next_obs,
                    rewards.to(runner.device),
                    dones.to(runner.device),
                    extras,
                )
                observations = next_obs
                completed_env_seen |= dones.to(env.device).bool()
                for key in totals:
                    totals[key] += float(diag[key])
                completed += float(
                    diag["completed_episodes"]
                )
                successful += float(
                    diag["successful_episodes"]
                )
                stage_successful += float(
                    diag["stage_successful_episodes"]
                )
                contact_successes += float(
                    diag["contact_successes"]
                )
                impact_successes += float(diag["impact_successes"])
                lift_successes += float(diag["lift_successes"])
                swing_successes += float(diag["swing_successes"])
                motion_ready_successes += float(diag["motion_ready_successes"])
                geometric_contact_episodes += float(
                    diag["geometric_contact_episodes"]
                )
                passive_response_episodes += float(
                    diag["passive_response_episodes"]
                )
                precontact_motion_episodes += float(
                    diag["precontact_motion_episodes"]
                )
                invalid_contact_episodes += float(
                    diag["invalid_contact_episodes"]
                )
                body_contact_episodes += float(
                    diag["body_contact_episodes"]
                )
                wrong_foot_contact_episodes += float(
                    diag["wrong_foot_contact_episodes"]
                )
                violation_episodes += float(diag["violation_episodes"])
                stable_activation_episodes += float(diag["stable_activation_episodes"])
                peak_sum += float(
                    diag["completed_peak_speed_sum"]
                )
                displacement_sum += float(
                    diag["completed_displacement_sum"]
                )
                min_distance_sum += float(
                    diag["completed_min_distance_sum"]
                )
            runner.alg.compute_returns(observations)

        losses = runner.alg.update()
        runner.current_learning_iteration = iteration
        success_rate = (
            successful / completed
            if completed > 0
            else None
        )
        stage_success_rate = (
            stage_successful / completed
            if completed > 0
            else None
        )
        contact_rate = (
            contact_successes / completed
            if completed > 0
            else None
        )
        impact_rate = (impact_successes / completed if completed > 0 else None)
        foot_lift_rate = (lift_successes / completed if completed > 0 else None)
        foot_swing_rate = (swing_successes / completed if completed > 0 else None)
        motion_ready_rate = (motion_ready_successes / completed if completed > 0 else None)
        geometric_contact_rate = (geometric_contact_episodes / completed if completed > 0 else None)
        passive_response_rate = (
            passive_response_episodes / completed
            if completed > 0 else None
        )
        precontact_motion_rate = (
            precontact_motion_episodes / completed
            if completed > 0 else None
        )
        invalid_contact_rate = (
            invalid_contact_episodes / completed
            if completed > 0 else None
        )
        body_contact_rate = (
            body_contact_episodes / completed
            if completed > 0 else None
        )
        wrong_foot_contact_rate = (
            wrong_foot_contact_episodes / completed
            if completed > 0 else None
        )
        violation_rate = (
            violation_episodes / completed
            if completed > 0
            else None
        )
        stable_activation_rate = (
            stable_activation_episodes / completed
            if completed > 0
            else None
        )
        episode_peak = (
            peak_sum / completed
            if completed > 0
            else None
        )
        episode_displacement = (
            displacement_sum / completed
            if completed > 0
            else None
        )
        episode_min_distance = (
            min_distance_sum / completed
            if completed > 0
            else None
        )

        record = {
            "iteration": iteration,
            "stage": stage_name,
            "learning_rate": float(stage["lr"]),
            "action_std": float(stage["action_std"]),
            **{
                key: value / steps
                for key, value in totals.items()
            },
            "completed_episodes": completed,
            "completed_env_fraction": float(
                completed_env_seen.float().mean()
            ),
            "kick_success_rate": success_rate,
            "stage_success_rate": stage_success_rate,
            "episode_contact_rate": contact_rate,
            "episode_impact_rate": impact_rate,
            "episode_foot_lift_rate": foot_lift_rate,
            "episode_foot_swing_rate": foot_swing_rate,
            "episode_motion_ready_rate": motion_ready_rate,
            "episode_geometric_contact_rate": geometric_contact_rate,
            "episode_passive_response_rate": passive_response_rate,
            "episode_precontact_motion_rate": precontact_motion_rate,
            "episode_invalid_contact_rate": invalid_contact_rate,
            "episode_body_contact_rate": body_contact_rate,
            "episode_wrong_foot_contact_rate": wrong_foot_contact_rate,
            "episode_violation_rate": violation_rate,
            "episode_stable_activation_rate": stable_activation_rate,
            "episode_peak_direction_speed": episode_peak,
            "episode_directional_displacement": (
                episode_displacement
            ),
            "episode_minimum_foot_distance": (
                episode_min_distance
            ),
            "losses": losses,
        }
        records.append(record)
        print(
            "[V4K1_ITER] "
            f"iteration={iteration}/{iterations} "
            f"stage={stage_name} "
            f"lr={record['learning_rate']:.9g} "
            f"std={record['action_std']:.4f} "
            f"reward={record['reward']:.6f} "
            f"stage_success_rate={stage_success_rate} "
            f"kick_success_rate={success_rate} "
            f"contact_rate={contact_rate} "
            f"impact_rate={impact_rate} "
            f"foot_lift_rate={foot_lift_rate} "
            f"foot_swing_rate={foot_swing_rate} "
            f"motion_ready_rate={motion_ready_rate} "
            f"geometric_contact_rate={geometric_contact_rate} "
            f"passive_response_rate={passive_response_rate} "
            f"precontact_motion_rate={precontact_motion_rate} "
            f"invalid_contact_rate={invalid_contact_rate} "
            f"body_contact_rate={body_contact_rate} "
            f"wrong_foot_contact_rate={wrong_foot_contact_rate} "
            f"violation_rate={violation_rate} "
            f"stable_activation_rate={stable_activation_rate} "
            f"episode_peak_speed={episode_peak} "
            f"episode_displacement={episode_displacement} "
            f"episode_min_foot_distance={episode_min_distance} "
            f"upright_error={record['upright_error']:.6f} "
            f"action_rate={record['action_rate']:.6f} "
            f"losses={losses}",
            flush=True,
        )

        interval = int(
            os.environ.get(
                "V4K1_CHECKPOINT_INTERVAL",
                "50",
            )
        )
        if interval > 0 and (iteration + 1) % interval == 0:
            save_checkpoint(
                runner=runner,
                path=log_dir
                / f"kick_model_{iteration}.pt",
                iteration=iteration,
                role="v4k1_r5_3_stable_stance_delta_pose_kick",
            )

    if active_stage is not None:
        close_stage(active_stage)

    overall_delta = parameter_delta(before, runner)
    final_path = log_dir / "kick_model_final.pt"
    save_checkpoint(
        runner=runner,
        path=final_path,
        iteration=iterations - 1,
        role="v4k1_r5_3_stable_stance_delta_pose_kick_final",
    )
    final_window = records[-min(20, len(records)):]

    def mean_available(key: str) -> float | None:
        values = [
            float(record[key])
            for record in final_window
            if isinstance(record.get(key), (int, float))
        ]
        return sum(values) / len(values) if values else None

    final_success = mean_available("kick_success_rate")
    final_contact = mean_available("episode_contact_rate")
    final_impact = mean_available("episode_impact_rate")
    final_foot_lift = mean_available("episode_foot_lift_rate")
    final_foot_swing = mean_available("episode_foot_swing_rate")
    final_motion_ready = mean_available("episode_motion_ready_rate")
    final_geometric_contact = mean_available("episode_geometric_contact_rate")
    final_passive_response = mean_available(
        "episode_passive_response_rate"
    )
    final_precontact_motion = mean_available(
        "episode_precontact_motion_rate"
    )
    final_invalid_contact = mean_available(
        "episode_invalid_contact_rate"
    )
    final_body_contact = mean_available(
        "episode_body_contact_rate"
    )
    final_wrong_foot_contact = mean_available(
        "episode_wrong_foot_contact_rate"
    )
    final_violation = mean_available("episode_violation_rate")
    final_stable_activation = mean_available("episode_stable_activation_rate")
    final_peak = mean_available(
        "episode_peak_direction_speed"
    )
    final_displacement = mean_available(
        "episode_directional_displacement"
    )
    final_min_distance = mean_available(
        "episode_minimum_foot_distance"
    )
    final_upright = mean_available("upright_error")
    final_action_rate = mean_available("action_rate")
    final_completed_env_fraction = mean_available(
        "completed_env_fraction"
    )

    summary = {
        "version": "v4k1_r5_3_stable_stance_delta_pose_kick",
        "source_low_policy": (
            "V4D final 62-D direct Delta-Pose low policy"
        ),
        "high_level_policy_used": False,
        "soccer_goal_reward_used": False,
        "skill": "kick_only",
        "delta_pose": (
            "planar_base_delta_pose_plus_"
            "kick_phase_leg_direction_strength"
        ),
        "action": "12_joint_position_residuals",
        "teacher_velocity_actor_observation": False,
        "contact_definition": (
            "stable_stance_selected_foot_short_contact_after_lift_and_swing"
        ),
        "impact_definition": (
            "valid_selected_foot_contact_then_directional_ball_speed_or_impulse"
        ),
        "curriculum": schedule,
        "stage_parameter_deltas": stage_deltas,
        "parameter_delta": overall_delta,
        "records": records,
    }
    summary_path = log_dir / "v4k1_kick_summary.json"
    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    contract = {
        "version": "v4k1_r5_3_stable_stance_delta_pose_kick",
        "training_scope": "isolated_low_level_kick_only",
        "high_level": None,
        "soccer_zero_sum_reward": None,
        "input": {
            "native_proprioception": 48,
            "legacy_velocity_command_scrubbed": True,
            "planar_delta_pose": [
                "delta_x",
                "delta_y",
                "delta_yaw",
            ],
            "kick_context": [
                "phase",
                "leg",
                "direction",
                "strength",
            ],
            "total_observation_dimension": 70,
        },
        "output": "12_joint_position_residuals",
        "reward": (
            "stage_specific_foot_approach_contact_"
            "impact_direction_recovery"
        ),
        "contact_impact_success_separated": True,
        "later_integration": (
            "add kick to V4D router only after "
            "isolated acceptance"
        ),
    }
    contract_path = log_dir / "v4k1_kick_contract.json"
    contract_path.write_text(
        json.dumps(
            contract,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(
        "[V4K1_PARAMETER_DELTA] "
        f"l2={overall_delta['l2']:.9f} "
        f"max_abs={overall_delta['max_abs']:.9f}",
        flush=True,
    )
    print(
        "[V4K1_FINAL_GUARD] "
        f"kick_success_rate={final_success} "
        f"contact_rate={final_contact} "
        f"impact_rate={final_impact} "
        f"foot_lift_rate={final_foot_lift} "
        f"foot_swing_rate={final_foot_swing} "
        f"motion_ready_rate={final_motion_ready} "
        f"geometric_contact_rate={final_geometric_contact} "
        f"passive_response_rate={final_passive_response} "
        f"precontact_motion_rate={final_precontact_motion} "
        f"invalid_contact_rate={final_invalid_contact} "
        f"body_contact_rate={final_body_contact} "
        f"wrong_foot_contact_rate={final_wrong_foot_contact} "
        f"violation_rate={final_violation} "
        f"stable_activation_rate={final_stable_activation} "
        f"peak_direction_speed={final_peak} "
        f"directional_displacement={final_displacement} "
        f"minimum_foot_distance={final_min_distance} "
        f"upright_error={final_upright} "
        f"action_rate={final_action_rate} "
        f"completed_env_fraction={final_completed_env_fraction}",
        flush=True,
    )
    if overall_delta["max_abs"] <= 0.0:
        raise RuntimeError(
            "V4K1 kick policy parameters did not update."
        )
    if (
        final_upright is not None
        and final_upright
        > float(
            os.environ.get(
                "V4K1_MAX_UPRIGHT_ERROR",
                "0.04",
            )
        )
    ):
        raise RuntimeError("V4K1 upright guard failed.")
    if (
        final_action_rate is not None
        and final_action_rate
        > float(
            os.environ.get(
                "V4K1_MAX_ACTION_RATE",
                "0.9",
            )
        )
    ):
        raise RuntimeError("V4K1 action-rate guard failed.")

    minimum_completed_env_fraction = float(
        os.environ.get("V4K1_MIN_COMPLETED_ENV_FRACTION", "0.0")
    )
    if minimum_completed_env_fraction > 0.0 and (
        final_completed_env_fraction is None
        or final_completed_env_fraction < minimum_completed_env_fraction
    ):
        raise RuntimeError(
            "V4K1-R5.3 rollout-completion guard failed. "
            f"required={minimum_completed_env_fraction} "
            f"observed={final_completed_env_fraction}"
        )

    maximum_violation = float(os.environ.get("V4K1_MAX_VIOLATION_RATE", "0.02"))
    if final_violation is None or final_violation > maximum_violation:
        raise RuntimeError(
            "V4K1-R5 invalid-contact/exploit guard failed. "
            f"required_max={maximum_violation} observed={final_violation}"
        )
    minimum_stable_activation = float(
        os.environ.get("V4K1_MIN_STABLE_ACTIVATION_RATE", "0.95")
    )
    if (
        final_stable_activation is None
        or final_stable_activation < minimum_stable_activation
    ):
        raise RuntimeError(
            "V4K1-R5 stable-activation guard failed. "
            f"required={minimum_stable_activation} "
            f"observed={final_stable_activation}"
        )

    minimum_success = float(
        os.environ.get("V4K1_MIN_SUCCESS_RATE", "0.0")
    )
    if minimum_success > 0.0 and (
        final_success is None
        or final_success < minimum_success
    ):
        raise RuntimeError(
            "V4K1 kick-success guard failed. "
            f"required={minimum_success} "
            f"observed={final_success}"
        )

    minimum_contact = float(
        os.environ.get("V4K1_MIN_CONTACT_RATE", "0.0")
    )
    if minimum_contact > 0.0 and (
        final_contact is None
        or final_contact < minimum_contact
    ):
        raise RuntimeError(
            "V4K1 contact guard failed. "
            f"required={minimum_contact} "
            f"observed={final_contact}"
        )

    minimum_impact = float(
        os.environ.get("V4K1_MIN_IMPACT_RATE", "0.0")
    )
    if minimum_impact > 0.0 and (
        final_impact is None
        or final_impact < minimum_impact
    ):
        raise RuntimeError(
            "V4K1 impact guard failed. "
            f"required={minimum_impact} "
            f"observed={final_impact}"
        )

    (log_dir / "V4K1_ISOLATED_KICK_OK").touch()
    print(f"V4K1_KICK_CHECKPOINT={final_path}", flush=True)
    print(f"V4K1_SUMMARY={summary_path}", flush=True)
    print(f"V4K1_CONTRACT={contract_path}", flush=True)
    print(
        "V4K1_R5_STABLE_STANCE_CONTACT_CAUSALITY=true",
        flush=True,
    )
    print(
        "V4K1_ISOLATED_KICK_TRAINING_COMPLETE",
        flush=True,
    )


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
    env.set_curriculum_stage(os.environ.get("V4K1_EVAL_STAGE", "recovery_generalization"))
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
        "kick_active", "stable_activation", "violation", "base_displacement",
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
                    "kick_active": diag["kick_active_fraction"],
                    "stable_activation": diag["stable_activation_fraction"],
                    "violation": diag["violation_fraction"],
                    "base_displacement": diag["base_displacement"],
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



# R8_SUPPORT_TRANSFER_SKILL_PPO
class IsolatedSupportTransferSkillEnv(IsolatedDeltaPoseKickEnv):
    """Teacher-free 70-D low-level skill for coordinated single-leg motion.

    The policy controls all 12 joint-position residuals.  No high-level policy,
    ball reward, goal reward, teacher action, or Cartesian target is exposed to
    the Actor.  The only skill context beyond proprioception is the existing
    kick phase, selected leg, direction, and strength context.
    """

    R8_STAGES = (
        "lift_5mm",
        "lift_1cm",
        "lift_2cm",
        "lift_3cm",
        "lift_4cm",
        "swing_3cm",
        "swing_6cm",
        "recovery",
    )

    R8_STAGE_SPECS: dict[str, dict[str, float | bool]] = {
        "unload_10pct": {
            "lift": 0.000, "forward": 0.000,
            "success_lift": 0.000, "success_forward": 0.000,
            "load_target": 0.070, "load_success": 0.100,
            "load_hold_steps": 6.0,
            "prelift_hold_steps": 0.0,
            "max_base": 0.068, "max_height": 0.038,
            "max_upright": 0.028, "recovery": False,
        },
        "lift_5mm": {
            "lift": 0.005, "forward": 0.000,
            "success_lift": 0.004, "success_forward": 0.000,
            "load_target": 0.080, "load_success": 0.120,
            "load_hold_steps": 6.0,
            "prelift_hold_steps": 4.0,
            "max_base": 0.070, "max_height": 0.040,
            "max_upright": 0.030, "recovery": False,
        },
        "lift_1cm": {
            "lift": 0.010, "forward": 0.000,
            "success_lift": 0.008, "success_forward": 0.000,
            "load_target": 0.080, "load_success": 0.120,
            "load_hold_steps": 6.0,
            "prelift_hold_steps": 4.0,
            "max_base": 0.070, "max_height": 0.040,
            "max_upright": 0.030, "recovery": False,
        },
        "lift_2cm": {
            "lift": 0.020, "forward": 0.000,
            "success_lift": 0.016, "success_forward": 0.000,
            "load_target": 0.070, "load_success": 0.110,
            "load_hold_steps": 6.0,
            "prelift_hold_steps": 4.0,
            "max_base": 0.068, "max_height": 0.038,
            "max_upright": 0.028, "recovery": False,
        },
        "lift_3cm": {
            "lift": 0.030, "forward": 0.000,
            "success_lift": 0.026, "success_forward": 0.000,
            "load_target": 0.060, "load_success": 0.100,
            "load_hold_steps": 6.0,
            "prelift_hold_steps": 4.0,
            "max_base": 0.066, "max_height": 0.036,
            "max_upright": 0.026, "recovery": False,
        },
        "lift_4cm": {
            "lift": 0.040, "forward": 0.000,
            "success_lift": 0.035, "success_forward": 0.000,
            "load_target": 0.055, "load_success": 0.095,
            "load_hold_steps": 6.0,
            "prelift_hold_steps": 4.0,
            "max_base": 0.065, "max_height": 0.035,
            "max_upright": 0.025, "recovery": False,
        },
        "swing_3cm": {
            "lift": 0.040, "forward": 0.030,
            "success_lift": 0.035, "success_forward": 0.025,
            "load_target": 0.050, "load_success": 0.100,
            "load_hold_steps": 6.0,
            "prelift_hold_steps": 4.0,
            "max_base": 0.065, "max_height": 0.035,
            "max_upright": 0.025, "recovery": False,
        },
        "swing_6cm": {
            "lift": 0.045, "forward": 0.060,
            "success_lift": 0.038, "success_forward": 0.050,
            "load_target": 0.050, "load_success": 0.100,
            "load_hold_steps": 6.0,
            "prelift_hold_steps": 4.0,
            "max_base": 0.065, "max_height": 0.035,
            "max_upright": 0.025, "recovery": False,
        },
        "recovery": {
            "lift": 0.045, "forward": 0.065,
            "success_lift": 0.040, "success_forward": 0.055,
            "load_target": 0.050, "load_success": 0.100,
            "load_hold_steps": 6.0,
            "prelift_hold_steps": 4.0,
            "max_base": 0.060, "max_height": 0.030,
            "max_upright": 0.020, "recovery": True,
        },
    }

    def __init__(
        self,
        *,
        low_env: RslRlVecEnvWrapper,
        kick_episode_steps: int,
        seed: int,
        fixed_leg: str,
    ) -> None:
        self._r8_initialized = False
        super().__init__(
            low_env=low_env,
            kick_episode_steps=kick_episode_steps,
            seed=seed,
            fixed_leg="left",
        )
        self.curriculum_stage = self.R8_STAGES[0]
        self.max_selected_foot_forward = torch.zeros(
            self.num_envs, device=self.device
        )
        self.max_airborne_foot_forward = torch.zeros(
            self.num_envs, device=self.device
        )
        self.max_base_displacement_r8 = torch.zeros(
            self.num_envs, device=self.device
        )
        self.max_base_height_error_r8 = torch.zeros(
            self.num_envs, device=self.device
        )
        self.max_upright_error_r8 = torch.zeros(
            self.num_envs, device=self.device
        )
        self.final_foot_error_r8 = torch.full(
            (self.num_envs,), 9.0, device=self.device
        )
        self.min_selected_load_ratio_r83 = torch.ones(
            self.num_envs, device=self.device
        )
        self.unload_consecutive_steps_r83 = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.max_unload_consecutive_steps_r83 = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.load_ready_seen_r83 = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.prelift_unload_consecutive_r84 = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.max_prelift_unload_consecutive_r84 = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.prelift_unload_seen_r84 = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.bridge_consecutive_steps_r852 = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.max_bridge_consecutive_steps_r852 = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.bridge_seen_r852 = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._init_r83_contact_force_source()
        self._r8_initialized = True
        self.reset()
        print(
            "[R8_4_BOOT] "
            f"obs={self.low_num_obs} actions={self.low_num_actions} "
            f"episode_steps={self.kick_episode_steps} "
            "curriculum=gated_load_preserving_progressive_lift "
            "unload_signal=real_contact_force_ratio_prelift_ordered "
            "ball_reward=false high_level_reward=false",
            flush=True,
        )

    @staticmethod
    def _smoothstep(value: torch.Tensor) -> torch.Tensor:
        value = torch.clamp(value, 0.0, 1.0)
        return value.square() * (3.0 - 2.0 * value)

    def _init_r83_contact_force_source(self) -> None:
        try:
            sensor = self.base_env.scene["contact_forces"]
        except Exception as exc:
            raise RuntimeError(
                "R8.3 requires scene['contact_forces']; R8.2 verified it."
            ) from exc

        forces = getattr(
            getattr(sensor, "data", None), "net_forces_w", None
        )
        if (
            not torch.is_tensor(forces)
            or forces.ndim != 3
            or int(forces.shape[0]) != self.num_envs
            or int(forces.shape[-1]) != 3
        ):
            raise RuntimeError(
                "R8.3 invalid net_forces_w. "
                f"shape={getattr(forces, 'shape', None)}"
            )

        names = [str(x) for x in getattr(sensor, "body_names", [])]
        required = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
        mapping: dict[str, int] = {}
        for target in required:
            ids = [
                i for i, name in enumerate(names)
                if name == target
            ]
            if len(ids) != 1:
                raise RuntimeError(
                    "R8.3 foot mapping failed. "
                    f"target={target} names={names}"
                )
            mapping[target] = ids[0]

        self._r83_contact_sensor = sensor
        self._r83_foot_force_ids = torch.as_tensor(
            [mapping[name] for name in required],
            dtype=torch.long,
            device=self.device,
        )
        print(
            "[R8_4_CONTACT_SOURCE] "
            f"shape={tuple(forces.shape)} "
            f"FL={mapping['FL_foot']} "
            f"FR={mapping['FR_foot']} "
            f"RL={mapping['RL_foot']} "
            f"RR={mapping['RR_foot']}",
            flush=True,
        )

    def _r83_contact_metrics(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        forces = self._r83_contact_sensor.data.net_forces_w
        if not torch.is_tensor(forces) or forces.ndim != 3:
            raise RuntimeError(
                "R8.3 contact tensor invalid at runtime."
            )
        if not torch.isfinite(forces).all():
            raise RuntimeError(
                "R8.3 contact tensor contains non-finite values."
            )

        foot = forces.index_select(
            1, self._r83_foot_force_ids
        )
        fz = foot[:, :, 2].abs()

        selected_fz = torch.where(
            self.kick_leg == 0,
            fz[:, 0],
            fz[:, 1],
        )
        support3_fz = torch.where(
            self.kick_leg == 0,
            fz[:, 1:].sum(dim=1),
            fz[:, [0, 2, 3]].sum(dim=1),
        )
        total4_fz = fz.sum(dim=1)
        ratio = selected_fz / torch.clamp(
            total4_fz, min=20.0
        )
        return selected_fz, support3_fz, total4_fz, ratio

    def _stage_spec(self) -> dict[str, float | bool]:
        try:
            return self.R8_STAGE_SPECS[self.curriculum_stage]
        except KeyError as exc:
            raise RuntimeError(
                f"Unknown R8.1 stage: {self.curriculum_stage}"
            ) from exc

    def _stage_amplitudes(self) -> tuple[float, float]:
        spec = self._stage_spec()
        return float(spec["lift"]), float(spec["forward"])

    def _target_foot_offset(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        phase = self._phase()
        lift_amplitude, forward_amplitude = self._stage_amplitudes()
        strength = torch.clamp(self.strike_strength, 0.90, 1.08)

        lift_up = self._smoothstep((phase - 0.40) / 0.14)
        swing_out = self._smoothstep((phase - 0.58) / 0.12)
        recovery = self._smoothstep((phase - 0.78) / 0.16)

        lift = (
            float(lift_amplitude)
            * strength
            * lift_up
            * (1.0 - recovery)
        )
        forward = (
            float(forward_amplitude)
            * strength
            * swing_out
            * (1.0 - recovery)
        )
        return lift, forward

    def _build_low_observation(self) -> torch.Tensor:
        observation = super()._build_low_observation()
        if getattr(self, "_r8_initialized", False):
            observation = observation.clone()
            # No physical ball participates in this stage.  Keep the deployed
            # ball slots present but neutral rather than exposing a fake target.
            observation[:, self.KICK_CONTEXT_BALL_X] = 0.0
            observation[:, self.KICK_CONTEXT_BALL_Y] = 0.0
            observation[:, self.KICK_CONTEXT_TEACHER_FLAG] = 0.0
        return observation

    def _update_kick_target(self) -> None:
        self.kick_phase = self._phase()
        self.target_x_world = self.kick_stance_xy[:, 0]
        self.target_y_world = self.kick_stance_xy[:, 1]
        self.target_yaw_world = self.kick_target_yaw
        self.target_terminal_velocity.zero_()
        self.target_elapsed_steps = self.active_kick_step_buf.clone()
        self.target_duration_steps.fill_(self.kick_episode_steps)

    def _place_ball_for_kick(self, env_ids: torch.Tensor) -> None:
        """Park the soccer ball far away while preserving reset semantics."""
        if env_ids.numel() == 0:
            return
        selected, _ = self._front_foot_positions()
        state = self._state()
        count = int(env_ids.numel())
        yaw = state["robot_yaw"][env_ids]
        direction = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=1)

        ball_pose = torch.zeros(count, 7, device=self.device)
        ball_pose[:, 0] = self.env_origins[env_ids, 0] + 2.50
        ball_pose[:, 1] = self.env_origins[env_ids, 1] + 2.00
        ground_z = self.env_origins[env_ids, 2]
        ball_pose[:, 2] = ground_z + 0.055
        ball_pose[:, 3] = 1.0
        zero_velocity = torch.zeros(count, 6, device=self.device)
        self.ball.write_root_pose_to_sim(ball_pose, env_ids=env_ids)
        self.ball.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)

        self.strike_direction_world[env_ids] = direction
        self.kick_ball_start_xy[env_ids] = ball_pose[:, :2]
        self.kick_stance_xy[env_ids] = state["robot_position"][env_ids, :2]
        self.kick_stance_yaw[env_ids] = yaw
        self.kick_target_yaw[env_ids] = yaw
        self.target_height[env_ids] = state["robot_position"][env_ids, 2]
        self.ball_placed[env_ids] = True
        self.ball_arm_step_buf[env_ids] = 0
        self.stable_consecutive_buf[env_ids] = 0
        self.previous_ball_direction_speed[env_ids] = 0.0
        self.peak_ball_direction_speed[env_ids] = 0.0
        self.contact_seen[env_ids] = False
        self.impact_seen[env_ids] = False
        self.geometric_contact_seen[env_ids] = False
        self.motion_ready_seen[env_ids] = False
        self.passive_response_seen[env_ids] = False
        self.violation_seen[env_ids] = False
        self.precontact_motion_seen[env_ids] = False
        self.invalid_contact_seen[env_ids] = False
        self.body_contact_seen[env_ids] = False
        self.wrong_foot_contact_seen[env_ids] = False
        self.foot_lift_seen[env_ids] = False
        self.foot_swing_seen[env_ids] = False
        self.recovery_seen[env_ids] = False
        self.recovery_stable_steps[env_ids] = 0
        self.contact_duration_steps[env_ids] = 0
        self.selected_foot_start_pos[env_ids] = selected[env_ids]
        self.previous_selected_foot_pos[env_ids] = selected[env_ids]
        self.max_selected_foot_lift[env_ids] = 0.0
        self.max_selected_foot_travel[env_ids] = 0.0
        self.max_foot_toward_ball_speed[env_ids] = 0.0
        if getattr(self, "_r8_initialized", False):
            self.max_selected_foot_forward[env_ids] = 0.0
            self.max_airborne_foot_forward[env_ids] = 0.0
            self.max_base_displacement_r8[env_ids] = 0.0
            self.max_base_height_error_r8[env_ids] = 0.0
            self.max_upright_error_r8[env_ids] = 0.0
            self.final_foot_error_r8[env_ids] = 9.0
            self.min_selected_load_ratio_r83[env_ids] = 1.0
            self.unload_consecutive_steps_r83[env_ids] = 0
            self.max_unload_consecutive_steps_r83[env_ids] = 0
            self.load_ready_seen_r83[env_ids] = False
            self.prelift_unload_consecutive_r84[env_ids] = 0
            self.max_prelift_unload_consecutive_r84[env_ids] = 0
            self.prelift_unload_seen_r84[env_ids] = False
            self.bridge_consecutive_steps_r852[env_ids] = 0
            self.max_bridge_consecutive_steps_r852[env_ids] = 0
            self.bridge_seen_r852[env_ids] = False

    def set_curriculum_stage(self, name: str) -> None:
        if name not in self.R8_STAGES:
            raise ValueError(f"Unknown R8 curriculum stage: {name}")
        self.curriculum_stage = name

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)
        if not getattr(self, "_r8_initialized", False):
            return
        if env_ids.numel() == 0:
            return
        count = int(env_ids.numel())
        stage_index = self.R8_STAGES.index(self.curriculum_stage)
        if stage_index <= 1:
            low, high = 0.94, 1.00
        elif stage_index <= 3:
            low, high = 0.92, 1.04
        else:
            low, high = 0.90, 1.08
        self.strike_strength[env_ids] = (
            low + (high - low) * self._rand((count,))
        )
        self.strike_angle_offset[env_ids] = 0.0
        self.strike_direction_world[env_ids] = torch.tensor(
            [1.0, 0.0], device=self.device
        )
        self.max_selected_foot_forward[env_ids] = 0.0
        self.max_airborne_foot_forward[env_ids] = 0.0
        self.max_base_displacement_r8[env_ids] = 0.0
        self.max_base_height_error_r8[env_ids] = 0.0
        self.max_upright_error_r8[env_ids] = 0.0
        self.final_foot_error_r8[env_ids] = 9.0
        self.min_selected_load_ratio_r83[env_ids] = 1.0
        self.unload_consecutive_steps_r83[env_ids] = 0
        self.max_unload_consecutive_steps_r83[env_ids] = 0
        self.load_ready_seen_r83[env_ids] = False
        self.prelift_unload_consecutive_r84[env_ids] = 0
        self.max_prelift_unload_consecutive_r84[env_ids] = 0
        self.prelift_unload_seen_r84[env_ids] = False
        self.bridge_consecutive_steps_r852[env_ids] = 0
        self.max_bridge_consecutive_steps_r852[env_ids] = 0
        self.bridge_seen_r852[env_ids] = False

    def _other_support_count_r8(self) -> torch.Tensor:
        body_pos = getattr(self.robot.data, "body_pos_w", None)
        if not torch.is_tensor(body_pos):
            return torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
        ids = torch.as_tensor(
            self.all_foot_ids, dtype=torch.long, device=self.device
        )
        heights = body_pos[:, ids, 2] - self.env_origins[:, 2:3]
        left_other = (heights[:, [1, 2, 3]] < 0.085).sum(dim=1)
        right_other = (heights[:, [0, 2, 3]] < 0.085).sum(dim=1)
        return torch.where(self.kick_leg == 0, left_other, right_other)

    def _kick_reward(
        self,
        *,
        before_state: dict[str, torch.Tensor],
        after_state: dict[str, torch.Tensor],
        low_actions: torch.Tensor,
        base_done: torch.Tensor,
        terminal: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del before_state
        self._update_kick_target()
        active = self.kick_active
        phase = self._phase()
        selected_pos, _ = self._front_foot_positions()
        direction = self.strike_direction_world
        lateral_direction = torch.stack(
            (-direction[:, 1], direction[:, 0]), dim=1
        )
        foot_delta = selected_pos - self.selected_foot_start_pos
        foot_lift = foot_delta[:, 2]
        foot_forward = (foot_delta[:, :2] * direction).sum(dim=1)
        foot_lateral = (foot_delta[:, :2] * lateral_direction).sum(dim=1)
        target_lift, target_forward = self._target_foot_offset()
        spec = self._stage_spec()
        lift_amplitude = float(spec["lift"])
        forward_amplitude = float(spec["forward"])
        load_target = float(spec["load_target"])
        load_success = float(spec["load_success"])
        load_hold_steps = int(float(spec["load_hold_steps"]))
        prelift_hold_steps = int(float(spec["prelift_hold_steps"]))

        lift_sigma = max(0.0045, 0.30 * max(lift_amplitude, 0.010))
        forward_sigma = max(
            0.010, 0.30 * max(forward_amplitude, 0.020)
        )
        target_error = torch.sqrt(
            (foot_lift - target_lift).square()
            + (foot_forward - target_forward).square()
            + 0.50 * foot_lateral.square()
            + 1.0e-8
        )
        foot_track = torch.exp(
            -(
                (foot_lift - target_lift).square() / (lift_sigma ** 2)
                + (foot_forward - target_forward).square()
                / (forward_sigma ** 2)
                + foot_lateral.square() / (0.025 ** 2)
            )
        )
        lift_progress = torch.clamp(
            foot_lift / max(lift_amplitude, 0.008), 0.0, 1.25
        )
        airborne_threshold = min(
            0.020, max(0.006, 0.55 * lift_amplitude)
        )
        airborne_now = active & (foot_lift >= airborne_threshold)
        airborne_gate = torch.clamp(
            (foot_lift - 0.004)
            / max(0.008, 0.45 * lift_amplitude),
            0.0,
            1.0,
        )
        if forward_amplitude > 0.0:
            raw_forward_progress = torch.clamp(
                foot_forward / forward_amplitude, 0.0, 1.25
            )
            forward_progress = raw_forward_progress * airborne_gate
        else:
            forward_progress = torch.zeros_like(foot_forward)

        stability = self._stability_state()
        upright_error = stability["upright_error"]
        base_height_error = torch.abs(
            stability["base_height"] - self.target_height
        )
        base_displacement = torch.linalg.vector_norm(
            after_state["robot_position"][:, :2] - self.kick_stance_xy,
            dim=1,
        )
        support_count = stability["support_count"]
        other_support_count = self._other_support_count_r8()

        selected_fz, support3_fz, total4_fz, load_ratio = (
            self._r83_contact_metrics()
        )
        load_window = (
            active
            & (phase >= 0.20)
            & (phase < 0.72)
        )
        load_ready_now = (
            load_window
            & (load_ratio <= load_success)
            & (support3_fz >= 80.0)
            & (
                base_displacement
                <= float(spec["max_base"])
            )
            & (
                base_height_error
                <= float(spec["max_height"])
            )
            & (
                upright_error
                <= float(spec["max_upright"])
            )
            & (other_support_count >= 3)
        )
        self.unload_consecutive_steps_r83 = torch.where(
            load_ready_now,
            self.unload_consecutive_steps_r83 + 1,
            torch.zeros_like(
                self.unload_consecutive_steps_r83
            ),
        )
        self.max_unload_consecutive_steps_r83 = torch.maximum(
            self.max_unload_consecutive_steps_r83,
            self.unload_consecutive_steps_r83,
        )
        self.load_ready_seen_r83 |= (
            self.max_unload_consecutive_steps_r83
            >= load_hold_steps
        )

        prelift_window_r84 = (
            active
            & (phase >= 0.18)
            & (phase < 0.38)
            & (lift_amplitude > 0.0)
        )
        prelift_ready_now_r84 = (
            prelift_window_r84
            & (load_ratio <= load_success)
            & (support3_fz >= 80.0)
            & (
                base_displacement
                <= float(spec["max_base"])
            )
            & (
                base_height_error
                <= float(spec["max_height"])
            )
            & (
                upright_error
                <= float(spec["max_upright"])
            )
            & (other_support_count >= 3)
        )
        self.prelift_unload_consecutive_r84 = torch.where(
            prelift_ready_now_r84,
            self.prelift_unload_consecutive_r84 + 1,
            torch.zeros_like(
                self.prelift_unload_consecutive_r84
            ),
        )
        self.max_prelift_unload_consecutive_r84 = torch.maximum(
            self.max_prelift_unload_consecutive_r84,
            self.prelift_unload_consecutive_r84,
        )
        if prelift_hold_steps > 0:
            self.prelift_unload_seen_r84 |= (
                self.max_prelift_unload_consecutive_r84
                >= prelift_hold_steps
            )
        else:
            self.prelift_unload_seen_r84[:] = True

        bridge_hold_steps_r852 = int(
            os.environ.get(
                "R852_BRIDGE_HOLD_STEPS",
                "3",
            )
        )

        bridge_now_r852 = (
            active
            & self.prelift_unload_seen_r84
            & (phase >= 0.30)
            & (phase < 0.82)
            & (
                foot_lift
                >= float(spec["success_lift"])
            )
            & (load_ratio <= load_success)
            & (support3_fz >= 80.0)
            & (
                base_displacement
                <= float(spec["max_base"])
            )
            & (
                base_height_error
                <= float(spec["max_height"])
            )
            & (
                upright_error
                <= float(spec["max_upright"])
            )
            & (other_support_count >= 3)
        )

        self.bridge_consecutive_steps_r852 = torch.where(
            bridge_now_r852,
            self.bridge_consecutive_steps_r852 + 1,
            torch.zeros_like(
                self.bridge_consecutive_steps_r852
            ),
        )

        self.max_bridge_consecutive_steps_r852 = (
            torch.maximum(
                self.max_bridge_consecutive_steps_r852,
                self.bridge_consecutive_steps_r852,
            )
        )

        self.bridge_seen_r852 |= (
            self.max_bridge_consecutive_steps_r852
            >= bridge_hold_steps_r852
        )

        self.min_selected_load_ratio_r83 = torch.where(
            load_window,
            torch.minimum(
                self.min_selected_load_ratio_r83,
                load_ratio,
            ),
            self.min_selected_load_ratio_r83,
        )

        measured_baseline_ratio = 0.30
        load_progress_r83 = torch.clamp(
            (
                measured_baseline_ratio
                - load_ratio
            )
            / max(
                0.05,
                measured_baseline_ratio - load_target,
            ),
            -1.0,
            1.25,
        )
        load_excess_r83 = torch.clamp(
            (load_ratio - load_success)
            / max(
                0.08,
                0.46 - load_success,
            ),
            0.0,
            1.5,
        )
        load_target_reward_r83 = torch.exp(
            -torch.relu(
                load_ratio - load_target
            ).square()
            / (0.070 ** 2)
        )
        support3_reward_r83 = torch.clamp(
            (support3_fz - 80.0) / 60.0,
            0.0,
            1.25,
        )

        upright_reward = torch.exp(-upright_error / 0.025)
        height_reward = torch.exp(
            -base_height_error.square() / (0.040 ** 2)
        )
        base_hold_reward = torch.exp(
            -base_displacement.square() / (0.070 ** 2)
        )
        other_support_reward = torch.clamp(
            other_support_count.float() - 2.0, 0.0, 1.0
        )
        low_speed_reward = torch.exp(
            -stability["vertical_speed"].square() / (0.25 ** 2)
            - stability["angular_xy_speed"].square() / (0.75 ** 2)
        )

        action_rate = (
            low_actions - self.previous_low_action
        ).square().mean(dim=1)
        torque_penalty = self._applied_torque_penalty()

        self.max_selected_foot_lift = torch.where(
            active,
            torch.maximum(self.max_selected_foot_lift, foot_lift),
            self.max_selected_foot_lift,
        )
        self.max_selected_foot_forward = torch.where(
            active,
            torch.maximum(self.max_selected_foot_forward, foot_forward),
            self.max_selected_foot_forward,
        )
        self.max_airborne_foot_forward = torch.where(
            airborne_now,
            torch.maximum(self.max_airborne_foot_forward, foot_forward),
            self.max_airborne_foot_forward,
        )
        self.max_selected_foot_travel = torch.where(
            active,
            torch.maximum(
                self.max_selected_foot_travel,
                torch.linalg.vector_norm(foot_delta, dim=1),
            ),
            self.max_selected_foot_travel,
        )
        self.max_base_displacement_r8 = torch.where(
            active,
            torch.maximum(
                self.max_base_displacement_r8, base_displacement
            ),
            self.max_base_displacement_r8,
        )
        self.max_base_height_error_r8 = torch.where(
            active,
            torch.maximum(
                self.max_base_height_error_r8, base_height_error
            ),
            self.max_base_height_error_r8,
        )
        self.max_upright_error_r8 = torch.where(
            active,
            torch.maximum(self.max_upright_error_r8, upright_error),
            self.max_upright_error_r8,
        )
        final_foot_error = torch.linalg.vector_norm(foot_delta, dim=1)
        self.final_foot_error_r8 = torch.where(
            active, final_foot_error, self.final_foot_error_r8
        )

        self.foot_lift_seen |= active & (
            self.max_selected_foot_lift >= 0.040
        )
        self.foot_swing_seen |= (
            active
            & (self.max_selected_foot_lift >= 0.030)
            & (self.max_airborne_foot_forward >= 0.050)
        )
        self.motion_ready_seen |= (
            self.foot_lift_seen & self.foot_swing_seen
        )
        recovery_window = active & (phase >= 0.88)
        recovery_now = (
            recovery_window
            & (final_foot_error <= 0.040)
            & (base_displacement <= 0.070)
            & (base_height_error <= 0.040)
            & (upright_error <= 0.040)
            & (support_count >= 3)
        )
        self.recovery_seen |= recovery_now

        severe_violation = (
            base_done
            | (base_displacement > 0.18)
            | (base_height_error > 0.14)
            | (upright_error > 0.35)
        )
        self.violation_seen |= active & severe_violation

        stage_success = (
            self.load_ready_seen_r83
            & (
                self.prelift_unload_seen_r84
                if lift_amplitude > 0.0
                else torch.ones_like(
                    self.load_ready_seen_r83
                )
            )
            & self.bridge_seen_r852
            & (
                self.max_base_displacement_r8
                <= float(spec["max_base"])
            )
            & (
                self.max_base_height_error_r8
                <= float(spec["max_height"])
            )
            & (
                self.max_upright_error_r8
                <= float(spec["max_upright"])
            )
            & (support_count >= 3)
            & ~self.violation_seen
        )
        if float(spec["success_forward"]) > 0.0:
            stage_success &= (
                self.max_airborne_foot_forward
                >= float(spec["success_forward"])
            )
        if bool(spec["recovery"]):
            stage_success &= self.recovery_seen

        strict_success = (
            (self.max_selected_foot_lift >= 0.040)
            & (self.max_airborne_foot_forward >= 0.060)
            & self.recovery_seen
            & (self.max_base_displacement_r8 <= 0.060)
            & (self.max_base_height_error_r8 <= 0.030)
            & (self.max_upright_error_r8 <= 0.020)
            & (support_count >= 3)
            & ~self.violation_seen
        )

        motion_window = (target_lift >= 0.003).float()
        lift_deficit = torch.clamp(
            (target_lift - foot_lift)
            / max(lift_amplitude, 0.008),
            0.0,
            1.0,
        )
        planted = motion_window * (foot_lift < 0.004).float()
        if forward_amplitude > 0.0:
            swing_window = (target_forward >= 0.004).float()
            forward_deficit = torch.clamp(
                (target_forward - foot_forward) / forward_amplitude,
                0.0,
                1.0,
            )
        else:
            swing_window = torch.zeros_like(foot_forward)
            forward_deficit = torch.zeros_like(foot_forward)

        foot_hold_r83 = torch.exp(
            -foot_lift.square() / (0.012 ** 2)
            -foot_forward.square() / (0.020 ** 2)
            -foot_lateral.square() / (0.020 ** 2)
        )
        load_window_f_r83 = load_window.float()

        if lift_amplitude <= 0.0:
            reward = (
                load_window_f_r83
                * (
                    4.00 * load_progress_r83
                    + 1.50 * load_target_reward_r83
                    + 0.90 * support3_reward_r83
                    - 2.50 * load_excess_r83
                )
                + 0.35 * foot_hold_r83
                + 0.35 * upright_reward
                + 0.30 * height_reward
                + 0.25 * base_hold_reward
                + 0.15 * other_support_reward
                + 0.08 * low_speed_reward
                - 0.020 * action_rate
                - 0.00015 * torque_penalty
            )
        else:
            soft_unload_gate_r84 = torch.clamp(
                (0.22 - load_ratio)
                / max(0.08, 0.22 - load_success),
                0.0,
                1.0,
            )
            ordered_lift_gate_r84 = torch.maximum(
                0.35 * soft_unload_gate_r84,
                self.prelift_unload_seen_r84.float(),
            )
            if forward_amplitude > 0.0:
                horizontal_hold_r84 = torch.exp(
                    -foot_lateral.square() / (0.018 ** 2)
                )
                movement_reward_r84 = (
                    2.60 * lift_progress
                    + 2.40 * forward_progress
                    + 1.20 * foot_track
                )
                swing_penalty_r84 = (
                    2.00 * swing_window * forward_deficit
                )
            else:
                horizontal_hold_r84 = torch.exp(
                    -foot_forward.square() / (0.018 ** 2)
                    -foot_lateral.square() / (0.018 ** 2)
                )
                movement_reward_r84 = (
                    2.80 * lift_progress
                    + 1.60 * foot_track
                )
                swing_penalty_r84 = torch.zeros_like(
                    foot_forward
                )
            lift_deficit_r84 = torch.clamp(
                (target_lift - foot_lift)
                / max(lift_amplitude, 0.010),
                0.0,
                1.0,
            )
            loaded_lift_penalty_r84 = torch.clamp(
                (load_ratio - load_success)
                / max(0.08, 0.30 - load_success),
                0.0,
                1.5,
            )
            reward = (
                load_window_f_r83
                * (
                    3.20 * load_progress_r83
                    + 1.20 * load_target_reward_r83
                    + 0.75 * support3_reward_r83
                    - 2.40 * load_excess_r83
                )
                + ordered_lift_gate_r84
                * movement_reward_r84
                + 1.25 * bridge_now_r852.float()
                + 0.45 * horizontal_hold_r84
                + 0.40 * upright_reward
                + 0.32 * height_reward
                + 0.28 * base_hold_reward
                + 0.18 * other_support_reward
                + 0.08 * low_speed_reward
                - 2.00
                * (1.0 - ordered_lift_gate_r84)
                * lift_deficit_r84
                - 2.60 * loaded_lift_penalty_r84
                - swing_penalty_r84
                - 0.020 * action_rate
                - 0.00015 * torque_penalty
            )
        terminal_bonus = torch.where(
            terminal,
            10.0 * stage_success.float()
            - 5.0 * (~stage_success).float(),
            torch.zeros_like(reward),
        )
        reward = reward + terminal_bonus
        reward = torch.where(active, reward, torch.zeros_like(reward))
        reward[base_done] = -6.0
        reward = torch.clamp(reward, -6.0, 10.0)

        step_dt = float(getattr(self.base_env, "step_dt", 0.02))
        if not math.isfinite(step_dt) or step_dt <= 0.0:
            step_dt = 0.02
        foot_velocity = (
            selected_pos - self.previous_selected_foot_pos
        ) / step_dt
        foot_forward_speed = (
            foot_velocity[:, :2] * direction
        ).sum(dim=1)
        self.peak_ball_direction_speed = torch.where(
            active,
            torch.maximum(
                self.peak_ball_direction_speed, foot_forward_speed
            ),
            self.peak_ball_direction_speed,
        )
        self.minimum_selected_foot_distance = torch.where(
            active,
            torch.minimum(
                self.minimum_selected_foot_distance, target_error
            ),
            self.minimum_selected_foot_distance,
        )

        zeros = torch.zeros_like(reward)
        false = torch.zeros_like(active)
        diagnostics = {
            "pose_error_xy": base_displacement,
            "pose_error_yaw": torch.abs(
                wrap_to_pi(
                    after_state["robot_yaw"] - self.kick_stance_yaw
                )
            ),
            "upright_error": upright_error,
            "action_rate": action_rate,
            "direction_speed": foot_forward_speed,
            "peak_direction_speed": self.peak_ball_direction_speed,
            "directional_displacement": foot_forward,
            "lateral_displacement": torch.abs(foot_lateral),
            "selected_foot_distance": target_error,
            "minimum_selected_foot_distance": self.minimum_selected_foot_distance,
            "selected_foot_height": foot_lift,
            "foot_progress": lift_progress + forward_progress,
            "foot_toward_ball_speed": foot_forward_speed,
            "max_foot_toward_ball_speed": self.peak_ball_direction_speed,
            "base_displacement": base_displacement,
            "geometric_contact": false.float(),
            "valid_contact": false.float(),
            "ball_response_now": false.float(),
            "wrong_foot_proximity": false.float(),
            "body_ball_proximity": false.float(),
            "passive_selected_response": false.float(),
            "precontact_motion_seen": false.float(),
            "invalid_contact_seen": false.float(),
            "body_contact_seen": false.float(),
            "wrong_foot_contact_seen": false.float(),
            "contact_seen": false.float(),
            "impact_seen": false.float(),
            "foot_lift_seen": self.foot_lift_seen.float(),
            "foot_swing_seen": self.foot_swing_seen.float(),
            "motion_ready_seen": self.motion_ready_seen.float(),
            "violation_seen": self.violation_seen.float(),
            "stable_activation_seen": self.stable_activation_seen.float(),
            "kick_active": active.float(),
            "support_count": support_count.float(),
            "vertical_speed": stability["vertical_speed"],
            "direction_speed_unused": zeros,
            "success": strict_success.float(),
            "stage_success": stage_success.float(),
            "selected_load_ratio": load_ratio,
            "selected_fz": selected_fz,
            "support3_fz": support3_fz,
            "total4_fz": total4_fz,
            "load_ready_seen": (
                self.load_ready_seen_r83.float()
            ),
            "min_selected_load_ratio": (
                self.min_selected_load_ratio_r83
            ),
            "max_unload_hold_steps": (
                self.max_unload_consecutive_steps_r83.float()
            ),
            "prelift_unload_seen": (
                self.prelift_unload_seen_r84.float()
            ),
            "max_prelift_unload_hold_steps": (
                self.max_prelift_unload_consecutive_r84.float()
            ),
            "bridge_now_r852": (
                bridge_now_r852.float()
            ),
            "bridge_seen_r852": (
                self.bridge_seen_r852.float()
            ),
            "max_bridge_hold_steps_r852": (
                self.max_bridge_consecutive_steps_r852.float()
            ),
        }
        return reward, diagnostics


def create_r8_runner_cfg(
    *,
    base_cfg: Any,
    run_name: str,
    actor_hidden_dims: list[int],
    critic_hidden_dims: list[int],
) -> dict[str, Any]:
    cfg = create_v4k1_runner_cfg(
        base_cfg=base_cfg,
        run_name=run_name,
        actor_hidden_dims=actor_hidden_dims,
        critic_hidden_dims=critic_hidden_dims,
    )
    cfg["experiment_name"] = "go2_soccer_r8_support_skill"
    cfg["num_steps_per_env"] = int(
        os.environ.get("R8_ROLLOUT_STEPS", "48")
    )
    cfg["policy"]["init_noise_std"] = float(
        os.environ.get("R8_INIT_NOISE_STD", "0.25")
    )
    cfg["algorithm"]["entropy_coef"] = float(
        os.environ.get("R8_ENTROPY_COEF", "0.006")
    )
    cfg["algorithm"]["learning_rate"] = float(
        os.environ.get("R8_LEARNING_RATE", "0.000030")
    )
    cfg["algorithm"]["num_learning_epochs"] = int(
        os.environ.get("R8_EPOCHS", "3")
    )
    cfg["algorithm"]["num_mini_batches"] = int(
        os.environ.get("R8_MINI_BATCHES", "4")
    )
    cfg["algorithm"]["gamma"] = float(
        os.environ.get("R8_GAMMA", "0.995")
    )
    cfg["algorithm"]["desired_kl"] = float(
        os.environ.get("R8_DESIRED_KL", "0.012")
    )
    return cfg


def read_r8_schedule(
    iterations: int,
) -> list[dict[str, Any]]:
    """
    R8.7:
      1 cm reconsolidation
      -> 2 cm simultaneous load-preserving lift.

    Progression gates are capability-training gates,
    not final physical acceptance thresholds.
    """

    if iterations < 25:
        raise RuntimeError(
            "R8.7 requires at least 25 iterations."
        )

    return [
        {
            "name": "lift_1cm",

            "min_iterations": 5,
            "max_iterations": 10,

            "lr": float(
                os.environ.get(
                    "R87_LR_1CM",
                    "0.000004",
                )
            ),

            "action_std": float(
                os.environ.get(
                    "R87_NOMINAL_STD_1CM",
                    "0.26",
                )
            ),

            "gate_success": float(
                os.environ.get(
                    "R87_GATE_1CM",
                    "0.030",
                )
            ),
        },

        {
            "name": "lift_2cm",

            "min_iterations": 15,
            "max_iterations": 45,

            "lr": float(
                os.environ.get(
                    "R87_LR_2CM",
                    "0.000004",
                )
            ),

            "action_std": float(
                os.environ.get(
                    "R87_NOMINAL_STD_2CM",
                    "0.26",
                )
            ),

            # Capability-discovery/progression gate only.
            # Final Kick acceptance will be much stricter.
            "gate_success": float(
                os.environ.get(
                    "R87_GATE_2CM",
                    "0.015",
                )
            ),
        },
    ]


def run_r8_training(
    *,
    runner: Any,
    env: IsolatedSupportTransferSkillEnv,
    adapter: IsolatedKickEnvAdapter,
    iterations: int,
    log_dir: Path,
    profile: str,
) -> None:
    schedule = read_r8_schedule(iterations)
    before = snapshot_policy(runner)
    observations = adapter.get_observations().to(runner.device)
    records: list[dict[str, Any]] = []
    active_stage: str | None = None
    stage_before: dict[str, torch.Tensor] | None = None
    stage_deltas: dict[str, dict[str, float]] = {}
    stage_gate_history: list[dict[str, Any]] = []
    stage_index = 0
    stage_iteration = 0
    highest_passed_stage: str | None = None
    evaluation_stage = str(schedule[0]["name"])
    curriculum_complete = False
    stop_reason = "iteration_budget_exhausted"
    stage_closed = False

    gate_window = int(os.environ.get("R84_GATE_WINDOW", "20"))
    gate_min_episodes = int(
        os.environ.get("R84_GATE_MIN_EPISODES", "256")
    )
    if gate_window <= 0 or gate_min_episodes <= 0:
        raise RuntimeError(
            "R8.1 gate window and minimum episode count must be positive."
        )

    def recent_gate_stats(
        stage_name: str,
    ) -> tuple[float | None, float]:
        rows = [
            record
            for record in records
            if record.get("stage") == stage_name
            and float(record.get("completed_episodes", 0.0)) > 0.0
        ]
        rows = rows[-min(gate_window, len(rows)):]
        completed = sum(
            float(row.get("completed_episodes", 0.0)) for row in rows
        )
        successful = sum(
            float(row.get("stage_successful_episodes", 0.0))
            for row in rows
        )
        if completed <= 0.0:
            return None, 0.0
        return successful / completed, completed

    def close_stage(stage_name: str, status: str) -> None:
        nonlocal stage_before, stage_closed
        if stage_before is None:
            raise RuntimeError(f"R8.1 stage snapshot missing: {stage_name}")
        delta = parameter_delta(stage_before, runner)
        stage_deltas[stage_name] = delta
        gate_rate, gate_episodes = recent_gate_stats(stage_name)
        stage_gate_history.append(
            {
                "stage": stage_name,
                "status": status,
                "stage_iterations": stage_iteration,
                "recent_success": gate_rate,
                "recent_completed_episodes": gate_episodes,
                "parameter_delta": delta,
            }
        )
        print(
            "[R8_4_STAGE_CLOSE] "
            f"stage={stage_name} status={status} "
            f"stage_iterations={stage_iteration} "
            f"recent_stage_success={gate_rate} "
            f"recent_completed_episodes={gate_episodes:.1f} "
            f"parameter_l2={delta['l2']:.9f} "
            f"parameter_max_abs={delta['max_abs']:.9f}",
            flush=True,
        )
        if delta["max_abs"] <= 0.0:
            raise RuntimeError(
                f"R8.1 stage did not update policy parameters: {stage_name}"
            )
        stage_before = None
        stage_closed = True

    for iteration in range(iterations):
        stage = schedule[stage_index]
        stage_name = str(stage["name"])
        if stage_name != active_stage:
            active_stage = stage_name
            evaluation_stage = stage_name
            stage_iteration = 0
            stage_closed = False
            stage_before = snapshot_policy(runner)
            env.set_curriculum_stage(stage_name)
            env.reset()
            observations = adapter.get_observations().to(runner.device)
            std_reset = False
            print(
                "[R8_4_STAGE_START] "
                f"stage={stage_name} global_iteration={iteration + 1} "
                f"min_iters={int(stage['min_iterations'])} "
                f"max_iters={int(stage['max_iterations'])} "
                f"gate_success={float(stage['gate_success']):.3f} "
                f"lr={float(stage['lr']):.9g} "
                f"std={float(stage['action_std']):.6f} "
                f"std_reset={std_reset} "
                "source_std_preserved=true",
                flush=True,
            )

        set_optimizer_learning_rate(runner, float(stage["lr"]))
        policy_module(runner).train()
        totals = {
            "reward": 0.0,
            "pose_error_xy": 0.0,
            "upright_error": 0.0,
            "action_rate": 0.0,
            "directional_displacement": 0.0,
            "selected_foot_height": 0.0,
            "base_displacement": 0.0,
            "foot_lift_fraction": 0.0,
            "foot_swing_fraction": 0.0,
            "motion_ready_fraction": 0.0,
            "violation_fraction": 0.0,
            "stable_activation_fraction": 0.0,
            "support_count": 0.0,
            "vertical_speed": 0.0,
            "base_done_fraction": 0.0,
            "selected_load_ratio": 0.0,
            "selected_fz": 0.0,
            "support3_fz": 0.0,
            "total4_fz": 0.0,
            "load_ready_seen": 0.0,
            "min_selected_load_ratio": 0.0,
            "max_unload_hold_steps": 0.0,
            "prelift_unload_seen": 0.0,
            "max_prelift_unload_hold_steps": 0.0,
            "bridge_now_r852": 0.0,
            "bridge_seen_r852": 0.0,
            "max_bridge_hold_steps_r852": 0.0,
        }
        completed = 0.0
        successful = 0.0
        stage_successful = 0.0
        steps = int(runner.cfg["num_steps_per_env"])
        completed_env_seen = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

        with torch.inference_mode():
            for _ in range(steps):
                actions = runner.alg.act(observations)
                next_obs, rewards, dones, extras, diag = env.kick_step(
                    actions.to(env.device)
                )
                next_obs = next_obs.to(runner.device)
                runner.alg.process_env_step(
                    next_obs,
                    rewards.to(runner.device),
                    dones.to(runner.device),
                    extras,
                )
                observations = next_obs
                completed_env_seen |= dones.to(env.device).bool()
                for key in totals:
                    totals[key] += float(diag[key])
                completed += float(diag["completed_episodes"])
                successful += float(diag["successful_episodes"])
                stage_successful += float(
                    diag["stage_successful_episodes"]
                )
            runner.alg.compute_returns(observations)

        losses = runner.alg.update()
        runner.current_learning_iteration = iteration
        stage_iteration += 1
        success_rate = successful / completed if completed > 0 else None
        stage_success_rate = (
            stage_successful / completed if completed > 0 else None
        )
        record = {
            "iteration": iteration,
            "stage_iteration": stage_iteration,
            "stage": stage_name,
            "learning_rate": float(stage["lr"]),
            "action_std": float(stage["action_std"]),
            **{key: value / steps for key, value in totals.items()},
            "completed_episodes": completed,
            "stage_successful_episodes": stage_successful,
            "completed_env_fraction": float(
                completed_env_seen.float().mean()
            ),
            "strict_success_rate": success_rate,
            "stage_success_rate": stage_success_rate,
            "losses": losses,
        }
        records.append(record)
        print(
            "[R8_4_ITER] "
            f"iteration={iteration + 1}/{iterations} "
            f"stage={stage_name} stage_iter={stage_iteration} "
            f"reward={record['reward']:.6f} "
            f"stage_success={stage_success_rate} "
            f"strict_success={success_rate} "
            f"lift_fraction={record['foot_lift_fraction']:.6f} "
            f"swing_fraction={record['foot_swing_fraction']:.6f} "
            f"foot_z={record['selected_foot_height']:.6f} "
            f"load_ratio={record['selected_load_ratio']:.6f} "
            f"fl_fz={record['selected_fz']:.3f} "
            f"support3_fz={record['support3_fz']:.3f} "
            f"load_ready={record['load_ready_seen']:.6f} "
            f"prelift={record['prelift_unload_seen']:.6f} "
            f"bridge_now={record['bridge_now_r852']:.6f} "
            f"bridge_seen={record['bridge_seen_r852']:.6f} "
            f"bridge_hold={record['max_bridge_hold_steps_r852']:.3f} "
            f"base={record['base_displacement']:.6f} "
            f"upright={record['upright_error']:.6f} "
            f"violation={record['violation_fraction']:.6f} "
            f"losses={losses}",
            flush=True,
        )

        interval = int(os.environ.get("R8_CHECKPOINT_INTERVAL", "50"))
        if interval > 0 and (iteration + 1) % interval == 0:
            save_checkpoint(
                runner=runner,
                path=log_dir / f"support_model_{iteration}.pt",
                iteration=iteration,
                role="r8_3_gated_support_transfer_skill",
            )

        gate_rate, gate_episodes = recent_gate_stats(stage_name)
        minimum_reached = stage_iteration >= int(stage["min_iterations"])
        gate_passed = (
            minimum_reached
            and gate_rate is not None
            and gate_episodes >= gate_min_episodes
            and gate_rate >= float(stage["gate_success"])
        )
        gate_timeout = stage_iteration >= int(stage["max_iterations"])
        if minimum_reached and (
            gate_passed
            or gate_timeout
            or stage_iteration % 10 == 0
        ):
            print(
                "[R8_4_STAGE_GATE] "
                f"stage={stage_name} stage_iter={stage_iteration} "
                f"recent_success={gate_rate} "
                f"recent_completed_episodes={gate_episodes:.1f} "
                f"required_success={float(stage['gate_success']):.3f} "
                f"required_completed={gate_min_episodes} "
                f"passed={int(gate_passed)} timeout={int(gate_timeout)}",
                flush=True,
            )

        if gate_passed:
            close_stage(stage_name, "passed")
            highest_passed_stage = stage_name
            if stage_index == len(schedule) - 1:
                curriculum_complete = True
                stop_reason = "curriculum_complete"
                print(
                    "[R8_4_CURRICULUM_COMPLETE] "
                    f"iteration={iteration + 1} stage={stage_name}",
                    flush=True,
                )
                break
            stage_index += 1
            active_stage = None
            continue

        if gate_timeout:
            close_stage(stage_name, "timeout")
            stop_reason = f"stage_gate_timeout:{stage_name}"
            print(
                "[R8_4_STAGE_TIMEOUT] "
                f"stage={stage_name} stage_iter={stage_iteration} "
                f"recent_success={gate_rate} "
                "later_stages_not_started=true",
                flush=True,
            )
            break

    if active_stage is not None and not stage_closed:
        close_stage(active_stage, "iteration_budget_exhausted")

    overall_delta = parameter_delta(before, runner)
    final_path = log_dir / "support_model_final.pt"
    save_checkpoint(
        runner=runner,
        path=final_path,
        iteration=len(records) - 1,
        role="r8_3_gated_support_transfer_skill_final",
    )
    final_window = records[-min(25, len(records)):]

    def mean_available(key: str) -> float | None:
        values = [
            float(record[key])
            for record in final_window
            if isinstance(record.get(key), (int, float))
        ]
        return sum(values) / len(values) if values else None

    final_strict = mean_available("strict_success_rate")
    final_stage = mean_available("stage_success_rate")
    final_lift = mean_available("foot_lift_fraction")
    final_swing = mean_available("foot_swing_fraction")
    final_violation = mean_available("violation_fraction")
    final_base = mean_available("base_displacement")
    final_upright = mean_available("upright_error")
    final_completed = mean_available(
        "completed_env_fraction"
    )
    final_load_ratio = mean_available(
        "selected_load_ratio"
    )
    final_selected_fz = mean_available(
        "selected_fz"
    )
    final_support3_fz = mean_available(
        "support3_fz"
    )
    final_load_ready = mean_available(
        "load_ready_seen"
    )
    final_min_load_ratio = mean_available(
        "min_selected_load_ratio"
    )

    summary = {
        "version": "r8_3_gated_unload_support_skill_ppo",
        "profile": profile,
        "source_low_policy": "accepted_v4d_low_62D_warm_start",
        "high_level_policy_used": False,
        "soccer_goal_reward_used": False,
        "ball_reward_used": False,
        "actor_observation_dimension": 70,
        "actor_observation_teacher_free": True,
        "action_dimension": 12,
        "curriculum": schedule,
        "curriculum_complete": curriculum_complete,
        "highest_passed_stage": highest_passed_stage,
        "evaluation_stage": evaluation_stage,
        "stop_reason": stop_reason,
        "stage_gate_history": stage_gate_history,
        "stage_parameter_deltas": stage_deltas,
        "parameter_delta": overall_delta,
        "final_metrics": {
            "strict_success_rate": final_strict,
            "stage_success_rate": final_stage,
            "foot_lift_fraction": final_lift,
            "foot_swing_fraction": final_swing,
            "violation_fraction": final_violation,
            "base_displacement": final_base,
            "upright_error": final_upright,
            "completed_env_fraction_informational_only": final_completed,
            "selected_load_ratio": final_load_ratio,
            "selected_fz_N": final_selected_fz,
            "support3_fz_N": final_support3_fz,
            "load_ready_fraction": final_load_ready,
            "min_selected_load_ratio": final_min_load_ratio,
        },
        "records": records,
    }
    (log_dir / "r8_support_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    contract = {
        "version": "r8_3_gated_unload_support_skill_ppo",
        "training_scope": "isolated_low_level_no_ball_support_skill",
        "input": {
            "dimension": 70,
            "teacher_free": True,
            "legacy_velocity_command_scrubbed": True,
            "phase_leg_direction_strength_context": True,
        },
        "output": "12_joint_position_residuals",
        "high_level_router_interface": "unchanged",
        "high_level_sparse_reward": "unchanged_and_not_used_here",
        "curriculum": "success_gated_not_fixed_time",
        "swing_metric": "airborne_forward_only",
        "next_step": "ball_contact_training_only_after_video_review",
    }
    (log_dir / "r8_support_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(
        "[R8_4_FINAL_GUARD] "
        f"profile={profile} stop_reason={stop_reason} "
        f"curriculum_complete={int(curriculum_complete)} "
        f"evaluation_stage={evaluation_stage} "
        f"strict_success={final_strict} stage_success={final_stage} "
        f"lift_fraction={final_lift} swing_fraction={final_swing} "
        f"violation={final_violation} base={final_base} "
        f"upright={final_upright} "
        f"load_ratio={final_load_ratio} "
        f"fl_fz={final_selected_fz} "
        f"support3_fz={final_support3_fz} "
        f"load_ready={final_load_ready} "
        f"completed_env_fraction_info={final_completed} "
        f"parameter_max_abs={overall_delta['max_abs']:.9f}",
        flush=True,
    )
    if overall_delta["max_abs"] <= 0.0:
        raise RuntimeError("R8.1 policy parameters did not update.")

    if profile == "full":
        if not curriculum_complete:
            raise RuntimeError(
                "R8.1 full curriculum did not complete. "
                f"stop_reason={stop_reason}"
            )
        if final_violation is None or final_violation > 0.05:
            raise RuntimeError(
                "R8.1 full violation guard failed. "
                f"observed={final_violation}"
            )
        minimum = float(os.environ.get("R8_FULL_MIN_STRICT_SUCCESS", "0.80"))
        if final_strict is None or final_strict < minimum:
            raise RuntimeError(
                "R8.1 full strict-success guard failed. "
                f"required={minimum} observed={final_strict}"
            )
        (log_dir / "R8_4_SUPPORT_SKILL_ACCEPTED").touch()
        (log_dir / "R8_SUPPORT_SKILL_ACCEPTED").touch()
        print("R8_4_SUPPORT_SKILL_ACCEPTED", flush=True)
    else:
        (log_dir / "R8_4_SUPPORT_PILOT_COMPLETE").touch()
        (log_dir / "R8_SUPPORT_PILOT_COMPLETE").touch()
        print(
            "[R8_4_PILOT_OUTCOME] "
            f"stop_reason={stop_reason} "
            f"highest_passed_stage={highest_passed_stage} "
            f"evaluation_stage={evaluation_stage} "
            "later_stage_training_blocked_by_gate=true",
            flush=True,
        )
        print("R8_4_SUPPORT_PILOT_COMPLETE", flush=True)


def run_r8_evaluation(
    *,
    runner: Any,
    env: IsolatedSupportTransferSkillEnv,
    video_path: Path,
    trajectory_path: Path,
    summary_path: Path,
    episodes: int,
    fps: int,
    stage_name: str,
    profile: str,
) -> None:
    import csv
    import imageio.v2 as imageio

    if env.num_envs != 1:
        raise RuntimeError("R8.4 evaluation requires --num_envs 1.")
    env.set_curriculum_stage(stage_name)
    env.reset()
    print(
        f"[R8_4_EVAL_STAGE] stage={stage_name}",
        flush=True,
    )
    policy_module(runner).eval()
    video_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(video_path, fps=fps)
    rows: list[dict[str, Any]] = []
    completed = 0
    successes = 0
    stage_successes = 0
    step = 0
    try:
        with torch.no_grad():
            while completed < episodes:
                obs = TensorDict(
                    {"policy": env.low_obs_buf},
                    batch_size=[env.num_envs],
                    device=env.device,
                )
                action = policy_module(runner).act_inference(
                    obs.to(runner.device)
                )
                _, _, done, _, diag = env.kick_step(action.to(env.device))
                frame = env.low_env.unwrapped.render()
                if isinstance(frame, list):
                    frame = frame[0]
                if frame is None:
                    raise RuntimeError("R8 render() returned no frame.")
                writer.append_data(np.asarray(frame))
                selected, _ = env._front_foot_positions()
                delta = selected - env.selected_foot_start_pos
                direction = env.strike_direction_world
                forward = (delta[:, :2] * direction).sum(dim=1)
                rows.append(
                    {
                        "step": step,
                        "episode": completed,
                        "phase": float(env.kick_phase[0]),
                        "foot_lift": float(delta[0, 2]),
                        "foot_forward": float(forward[0]),
                        "base_displacement": float(diag["base_displacement"]),
                        "upright_error": float(diag["upright_error"]),
                        "support_count": float(diag["support_count"]),
                        "selected_load_ratio": float(
                            diag.get("selected_load_ratio", 0.0)
                        ),
                        "selected_fz_N": float(
                            diag.get("selected_fz", 0.0)
                        ),
                        "support3_fz_N": float(
                            diag.get("support3_fz", 0.0)
                        ),
                        "load_ready": float(
                            diag.get("load_ready_seen", 0.0)
                        ),
                        "prelift_unload_seen": float(
                            diag.get("prelift_unload_seen", 0.0)
                        ),
                        "active": float(diag["kick_active_fraction"]),
                        "completed": int(diag["completed_episodes"]),
                        "success": int(diag["successful_episodes"]),
                        "stage_success": int(
                            diag["stage_successful_episodes"]
                        ),
                    }
                )
                completed += int(diag["completed_episodes"])
                successes += int(diag["successful_episodes"])
                stage_successes += int(
                    diag["stage_successful_episodes"]
                )
                step += 1
                if step > episodes * (env.total_episode_steps + 60):
                    raise RuntimeError("R8 evaluation exceeded safety limit.")
    finally:
        writer.close()

    fields = sorted({key for row in rows for key in row})
    with trajectory_path.open("w", newline="", encoding="utf-8") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=fields)
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    success_rate = successes / completed if completed else None
    stage_success_rate = (
        stage_successes / completed if completed else None
    )
    payload = {
        "profile": profile,
        "evaluation_stage": stage_name,
        "episodes": completed,
        "successes": successes,
        "success_rate": success_rate,
        "stage_successes": stage_successes,
        "stage_success_rate": stage_success_rate,
        "video": str(video_path),
        "trajectory": str(trajectory_path),
        "actor_observation_dimension": 70,
        "action_dimension": 12,
        "high_level_router_interface": "unchanged",
    }
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    if completed != episodes:
        raise RuntimeError(f"R8 evaluation incomplete: {payload}")
    if profile == "full" and (success_rate is None or success_rate < 0.75):
        raise RuntimeError(f"R8.1 full video guard failed: {payload}")
    video_path.with_suffix(video_path.suffix + ".ready").touch()
    print(f"R8_EVAL_VIDEO={video_path}", flush=True)
    print(f"R8_EVAL_SUMMARY={summary_path}", flush=True)
    print("R8_EVAL_VIDEO_READY", flush=True)


def main() -> None:
    if args_cli.num_envs <= 0 or args_cli.max_iterations <= 0:
        raise ValueError("R8 num_envs and max_iterations must be positive.")
    source_checkpoint = Path(args_cli.low_level_checkpoint).resolve()
    if not source_checkpoint.is_file():
        raise FileNotFoundError(f"R8 source checkpoint missing: {source_checkpoint}")
    log_dir = Path(args_cli.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    settle_budget = int(os.environ.get("V4K1_SETTLE_MAX_STEPS", "150"))
    arm_budget = int(os.environ.get("V4K1_BALL_ARM_STEPS", "4"))
    env_cfg.episode_length_s = max(
        7.0,
        (args_cli.kick_episode_steps + settle_budget + arm_budget) * 0.02 + 1.0,
    )
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
    low_env = RslRlVecEnvWrapper(
        gym_env, clip_actions=low_agent_cfg.clip_actions
    )
    env = IsolatedSupportTransferSkillEnv(
        low_env=low_env,
        kick_episode_steps=args_cli.kick_episode_steps,
        seed=args_cli.seed,
        fixed_leg="left",
    )
    adapter = IsolatedKickEnvAdapter(env)
    source_architecture = infer_v4k1_source_architecture(source_checkpoint)
    runner_cfg = create_r8_runner_cfg(
        base_cfg=low_agent_cfg,
        run_name=args_cli.run_name,
        actor_hidden_dims=source_architecture["actor_hidden_dims"],
        critic_hidden_dims=source_architecture["critic_hidden_dims"],
    )
    runner = OnPolicyRunner(
        adapter, runner_cfg, log_dir=None, device=args_cli.device
    )

    print(
        "[R8_4_CONTRACT] "
        f"profile={args_cli.r8_profile} "
        "training_scope=isolated_low_level_support_transfer_no_ball "
        "actor_observation=70D_teacher_free "
        "low_action=12D_joint_position_residuals "
        "high_level_policy=not_used "
        "high_level_router_interface=unchanged "
        "soccer_goal_reward=not_used "
        "dense_reward=low_level_skill_only "
        "ball_contact_training=deferred_until_video_review",
        flush=True,
    )

    if args_cli.eval_video:
        trained_raw = args_cli.trained_low_checkpoint.strip()
        if not trained_raw:
            raise RuntimeError(
                "--trained_low_checkpoint is required for R8 evaluation."
            )
        trained = Path(trained_raw).resolve()
        if not trained.is_file():
            raise FileNotFoundError(f"R8 trained checkpoint missing: {trained}")
        evaluation_stage = os.environ.get("R8_EVAL_STAGE", "").strip()
        training_summary = trained.parent / "r8_support_summary.json"
        if not evaluation_stage and training_summary.is_file():
            try:
                training_payload = json.loads(
                    training_summary.read_text(encoding="utf-8")
                )
                candidate = str(
                    training_payload.get("evaluation_stage", "")
                ).strip()
                if candidate in env.R8_STAGES:
                    evaluation_stage = candidate
            except Exception as exc:
                print(
                    "[R8_4_EVAL_STAGE_WARNING] "
                    f"summary_read_failed={type(exc).__name__}:{exc}",
                    flush=True,
                )
        if not evaluation_stage:
            evaluation_stage = env.R8_STAGES[-1]
        if evaluation_stage not in env.R8_STAGES:
            raise RuntimeError(
                f"Invalid R8.4 evaluation stage: {evaluation_stage}"
            )
        runner.load(str(trained), load_optimizer=False)
        video_path = Path(args_cli.eval_video).resolve()
        trajectory_path = Path(
            args_cli.trajectory_csv
            or str(video_path.with_suffix("")) + "_trajectory.csv"
        ).resolve()
        summary_path = Path(
            args_cli.eval_summary
            or str(video_path.with_suffix("")) + "_summary.json"
        ).resolve()
        run_r8_evaluation(
            runner=runner,
            env=env,
            video_path=video_path,
            trajectory_path=trajectory_path,
            summary_path=summary_path,
            episodes=args_cli.eval_episodes,
            fps=args_cli.eval_fps,
            stage_name=evaluation_stage,
            profile=args_cli.r8_profile,
        )
    else:
        warm_start_v4k1_policy(
            runner=runner,
            checkpoint=source_checkpoint,
            requested_std=float(runner_cfg["policy"]["init_noise_std"]),
        )
        env.reset()
        preflight_obs = adapter.get_observations()["policy"]
        command_abs = float(
            preflight_obs[:, env.NATIVE_COMMAND_SLICE].abs().max()
        )
        teacher_abs = float(
            preflight_obs[:, env.KICK_CONTEXT_TEACHER_FLAG].abs().max()
        )
        kick_onehot = float(preflight_obs[:, 59].mean())
        print(
            "[R8_INPUT_GUARD] "
            f"legacy_command_abs={command_abs:.9f} "
            f"teacher_flag_abs={teacher_abs:.9f} "
            f"kick_onehot_mean={kick_onehot:.9f} "
            f"obs_shape={tuple(preflight_obs.shape)}",
            flush=True,
        )
        if command_abs > 1.0e-8 or teacher_abs > 1.0e-8:
            raise RuntimeError("R8 Actor observation contains teacher leakage.")
        if abs(kick_onehot - 1.0) > 1.0e-6:
            raise RuntimeError("R8 kick skill one-hot is not active.")
        run_v4k1_stable_reset_guard(env)
        with torch.no_grad():
            (
                selected_fz,
                support3_fz,
                total4_fz,
                load_ratio,
            ) = env._r83_contact_metrics()
        print(
            "[R8_4_LOAD_BASELINE] "
            f"fl_fz={float(selected_fz.mean()):.6f} "
            f"support3_fz={float(support3_fz.mean()):.6f} "
            f"total4_fz={float(total4_fz.mean()):.6f} "
            f"fl_ratio={float(load_ratio.mean()):.6f} "
            "r82_reference_ratio=0.293567",
            flush=True,
        )
        if float(total4_fz.mean()) < 100.0:
            raise RuntimeError(
                "R8.3 implausible contact baseline."
            )
        with torch.no_grad():
            action = policy_module(runner).act_inference(
                adapter.get_observations().to(runner.device)
            )
            next_obs, reward, done, _, diag = env.kick_step(
                action.to(env.device)
            )
        if not torch.isfinite(next_obs["policy"]).all() or not torch.isfinite(reward).all():
            raise RuntimeError("R8 preflight produced non-finite tensors.")
        print(
            "[R8_PREFLIGHT] "
            f"reward_mean={float(reward.mean()):.6f} "
            f"done_count={int(done.sum())} diagnostics={diag}",
            flush=True,
        )
        env.reset()
        run_r8_training(
            runner=runner,
            env=env,
            adapter=adapter,
            iterations=args_cli.max_iterations,
            log_dir=log_dir,
            profile=args_cli.r8_profile,
        )
    env.close()


try:
    main()
except BaseException as error:
    print(f"[R8_4_FATAL] {type(error).__name__}: {error}", flush=True)
    print(f"[R8_FATAL] {type(error).__name__}: {error}", flush=True)
    print("[R8_4_BOOT] fatal_fast_exit=true", flush=True)
    os._exit(1)
else:
    print("[R8_4_BOOT] clean_fast_exit=true", flush=True)
    os._exit(0)
