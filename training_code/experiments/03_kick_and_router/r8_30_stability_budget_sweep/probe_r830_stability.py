# R8_30_STABILITY_BUDGET_SWEEP
# R8_29_EXACT_RECOVERY_TIMING_SWEEP
# R8_28_RECOVERY_STABILIZATION_SWEEP
# R8_27_RECOVERY_MODE_CORE
# R8_26_RETRACTION_REACHABILITY_SWEEP
# R8_25_RECOVERY_COMPOSITION_SWEEP
# R8_24_LIFT_SWING_COMPOSITION_SWEEP
# R8_23_MODE_CORE_RESIDUAL_AB
# R8_20_PREBRIDGE_LIFT_RESIDUAL
# R8_17_SUCCESS_ACTION_ATTRIBUTION
# R8_14_ROOT_CAUSE_AUDIT
# R8_11_DIRECT_COORDINATE_RECOVERY
# R8_10_RECOVERY_REINTEGRATION
# R8_9_THREE_LEG_SUPPORT_SINGLE_LEG_AIR_SWING
# R8_8_AIRBORNE_LOAD_PRESERVING_LIFT_4CM
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
            "swing_now_r89",
            "swing_seen_r89",
            "max_swing_hold_steps_r89",
            "recovery_now_r810",
            "recovery_seen_r810",
            "recovery_hold_steps_r810",
            "recovery_final_foot_error_r810",
            "recovery_fl_contact_r810",
            "recovery_support4_r810",

            "r814_rec_window",
            "r814_rec_candidate",
            "r814_rec_forward_ok",
            "r814_rec_lift_ok",
            "r814_rec_return_ok",
            "r814_rec_contact_ok",
            "r814_rec_support4_ok",
            "r814_rec_base_ok",
            "r814_rec_height_ok",
            "r814_rec_upright_ok",
            "r814_rec_body_ok",
            "r814_rec_touchdown_ok",
            "r814_rec_all_now",
            "r814_rec_hold1",
            "r814_rec_hold2",
            "r814_rec_hold3",
            "r814_rec_dx_num",
            "r814_rec_dz_num",
        ):
            if key in tensors:
                diagnostics[key] = float(
                    tensors[key].mean()
                )

        # ====================================================
        # R8.14 completed-episode capability histogram.
        # All values are measured before reset.
        # ====================================================

        r814_done = done.bool()

        diagnostics["r814_done_count"] = float(
            r814_done.sum()
        )

        diagnostics["r814_done_max_forward_sum"] = float(
            torch.where(
                r814_done,
                self.max_airborne_foot_forward,
                torch.zeros_like(
                    self.max_airborne_foot_forward
                ),
            ).sum()
        )

        diagnostics["r814_done_max_lift_sum"] = float(
            torch.where(
                r814_done,
                self.max_selected_foot_lift,
                torch.zeros_like(
                    self.max_selected_foot_lift
                ),
            ).sum()
        )

        diagnostics["r814_done_bridge_seen"] = float(
            (
                r814_done
                & self.bridge_seen_r852
            ).sum()
        )

        diagnostics["r814_done_swing_seen"] = float(
            (
                r814_done
                & self.swing_seen_r89
            ).sum()
        )

        diagnostics["r814_done_recovery_seen"] = float(
            (
                r814_done
                & self.recovery_seen
            ).sum()
        )

        for r814_cm in range(1, 11):

            r814_threshold = (
                float(r814_cm)
                / 100.0
            )

            diagnostics[
                f"r814_done_fwd_ge_{r814_cm}cm"
            ] = float(
                (
                    r814_done
                    & (
                        self.max_airborne_foot_forward
                        >= r814_threshold
                    )
                ).sum()
            )

        for r814_cm in range(1, 7):

            r814_threshold = (
                float(r814_cm)
                / 100.0
            )

            diagnostics[
                f"r814_done_lift_ge_{r814_cm}cm"
            ] = float(
                (
                    r814_done
                    & (
                        self.max_selected_foot_lift
                        >= r814_threshold
                    )
                ).sum()
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

            # Capability-training bounds.
            # Final acceptance remains separate.
            "max_base": 0.065,
            "max_height": 0.035,
            "max_upright": 0.025,

            "recovery": True,
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
        self.swing_consecutive_steps_r89 = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        self.max_swing_consecutive_steps_r89 = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        self.swing_seen_r89 = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
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
            self.swing_consecutive_steps_r89[env_ids] = 0
            self.max_swing_consecutive_steps_r89[env_ids] = 0
            self.swing_seen_r89[env_ids] = False

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
        self.swing_consecutive_steps_r89[env_ids] = 0
        self.max_swing_consecutive_steps_r89[env_ids] = 0
        self.swing_seen_r89[env_ids] = False

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

        swing_hold_steps_r89 = int(
            os.environ.get(
                "R89_SWING_HOLD_STEPS",
                "2",
            )
        )

        if forward_amplitude > 0.0:

            swing_now_r89 = (
                active
                & self.prelift_unload_seen_r84
                & (phase >= 0.58)
                & (phase < 0.82)
                & (
                    foot_lift
                    >= float(spec["success_lift"])
                )
                & (
                    foot_forward
                    >= float(spec["success_forward"])
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

        else:
            swing_now_r89 = torch.zeros_like(
                active
            )

        self.swing_consecutive_steps_r89 = torch.where(
            swing_now_r89,
            self.swing_consecutive_steps_r89 + 1,
            torch.zeros_like(
                self.swing_consecutive_steps_r89
            ),
        )

        self.max_swing_consecutive_steps_r89 = (
            torch.maximum(
                self.max_swing_consecutive_steps_r89,
                self.swing_consecutive_steps_r89,
            )
        )

        self.swing_seen_r89 |= (
            self.max_swing_consecutive_steps_r89
            >= swing_hold_steps_r89
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
        recovery_window = (
            active
            & (phase >= 0.88)
        )

        recovery_foot_contact_r810 = (
            selected_fz >= 8.0
        )

        # ----------------------------------------------------
        # R8.11 direct-coordinate recovery.
        #
        # These are the same relative coordinates already used
        # successfully by the airborne lift / swing curriculum.
        # ----------------------------------------------------

        recovery_forward_error_r811 = torch.abs(
            foot_forward
        )

        recovery_lift_error_r811 = torch.abs(
            foot_lift
        )

        recovery_return_position_r811 = (
            (recovery_forward_error_r811 <= 0.045)
            & (recovery_lift_error_r811 <= 0.015)
        )

        recovery_touchdown_r811 = (
            recovery_window
            & self.swing_seen_r89
            & recovery_foot_contact_r810
            & (support_count >= 4)
            & (base_displacement <= 0.065)
            & (base_height_error <= 0.035)
            & (upright_error <= 0.025)
        )

        recovery_now = (
            recovery_touchdown_r811
            & recovery_return_position_r811
        )

        self.recovery_stable_steps = torch.where(
            recovery_now,
            self.recovery_stable_steps + 1,
            torch.zeros_like(
                self.recovery_stable_steps
            ),
        )

        # Curriculum capability threshold.
        # Final visual/physics acceptance remains separate.
        self.recovery_seen |= (
            self.recovery_stable_steps >= 3
        )

        recovery_gate_r810 = self._smoothstep(
            (phase - 0.78) / 0.16
        )

        # Continuous return rewards.
        # Do NOT depend on the previous broken final_foot_error.
        recovery_forward_reward_r811 = torch.exp(
            -(
                recovery_forward_error_r811
                / 0.050
            ).square()
        )

        recovery_lift_reward_r811 = torch.exp(
            -(
                recovery_lift_error_r811
                / 0.020
            ).square()
        )

        recovery_pose_reward_r810 = (
            recovery_forward_reward_r811
            * recovery_lift_reward_r811
        )

        recovery_support4_reward_r810 = (
            support_count >= 4
        ).float()

        recovery_contact_reward_r810 = torch.clamp(
            (selected_fz - 5.0)
            / 20.0,
            0.0,
            1.0,
        )

        recovery_body_reward_r810 = (
            upright_reward
            * height_reward
            * base_hold_reward
        )

        # ====================================================
        # R8.14 mask-aware recovery decomposition.
        # ====================================================

        r814_recovery_window = recovery_window

        r814_recovery_candidate = (
            recovery_window
            & self.swing_seen_r89
        )

        r814_forward_return_ok = (
            recovery_forward_error_r811 <= 0.045
        )

        r814_lift_return_ok = (
            recovery_lift_error_r811 <= 0.015
        )

        r814_return_ok = (
            r814_forward_return_ok
            & r814_lift_return_ok
        )

        r814_contact_ok = (
            recovery_foot_contact_r810
        )

        r814_support4_ok = (
            support_count >= 4
        )

        r814_base_ok = (
            base_displacement <= 0.065
        )

        r814_height_ok = (
            base_height_error <= 0.035
        )

        r814_upright_ok = (
            upright_error <= 0.025
        )

        r814_body_ok = (
            r814_base_ok
            & r814_height_ok
            & r814_upright_ok
        )

        r814_touchdown_ok = (
            r814_contact_ok
            & r814_support4_ok
            & r814_body_ok
        )

        r814_recovery_all_now = (
            r814_recovery_candidate
            & r814_return_ok
            & r814_touchdown_ok
        )

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
            stage_success &= self.swing_seen_r89
        if bool(spec["recovery"]):
            stage_success &= self.recovery_seen

        strict_success = (
            self.swing_seen_r89
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
                + (
                    1.0
                    - recovery_gate_r810
                )
                * ordered_lift_gate_r84
                * movement_reward_r84

                + (
                    1.0
                    - recovery_gate_r810
                )
                * 1.25
                * bridge_now_r852.float()

                + (
                    1.0
                    - recovery_gate_r810
                )
                * 1.00
                * swing_now_r89.float()

                # During recovery the objective changes:
                # return FL to stance and restore 4-foot support.
                + recovery_gate_r810
                * (
                    3.00
                    * recovery_pose_reward_r810

                    + 1.20
                    * recovery_support4_reward_r810

                    + 0.80
                    * recovery_contact_reward_r810

                    + 0.60
                    * recovery_body_reward_r810
                )

                + 0.45 * horizontal_hold_r84
                + 0.40 * upright_reward
                + 0.32 * height_reward
                + 0.28 * base_hold_reward
                + 0.18 * other_support_reward
                + 0.08 * low_speed_reward
                - (
                    1.0
                    - recovery_gate_r810
                )
                * 2.00
                * (1.0 - ordered_lift_gate_r84)
                * lift_deficit_r84

                # Reloading FL is CORRECT during recovery,
                # therefore loaded-lift penalty must turn off.
                - (
                    1.0
                    - recovery_gate_r810
                )
                * 2.60
                * loaded_lift_penalty_r84

                - (
                    1.0
                    - recovery_gate_r810
                )
                * swing_penalty_r84
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
            "swing_now_r89": (
                swing_now_r89.float()
            ),
            "swing_seen_r89": (
                self.swing_seen_r89.float()
            ),
            "max_swing_hold_steps_r89": (
                self.max_swing_consecutive_steps_r89.float()
            ),
            "recovery_now_r810": (
                recovery_now.float()
            ),
            "recovery_seen_r810": (
                self.recovery_seen.float()
            ),
            "recovery_hold_steps_r810": (
                self.recovery_stable_steps.float()
            ),
            "recovery_final_foot_error_r810": (
                torch.maximum(
                    recovery_forward_error_r811,
                    recovery_lift_error_r811,
                )
            ),
            "recovery_fl_contact_r810": (
                recovery_foot_contact_r810.float()
            ),
            "recovery_support4_r810": (
                (support_count >= 4).float()
            ),

            "r814_rec_window": (
                r814_recovery_window.float()
            ),

            "r814_rec_candidate": (
                r814_recovery_candidate.float()
            ),

            "r814_rec_forward_ok": (
                (
                    r814_recovery_candidate
                    & r814_forward_return_ok
                ).float()
            ),

            "r814_rec_lift_ok": (
                (
                    r814_recovery_candidate
                    & r814_lift_return_ok
                ).float()
            ),

            "r814_rec_return_ok": (
                (
                    r814_recovery_candidate
                    & r814_return_ok
                ).float()
            ),

            "r814_rec_contact_ok": (
                (
                    r814_recovery_candidate
                    & r814_contact_ok
                ).float()
            ),

            "r814_rec_support4_ok": (
                (
                    r814_recovery_candidate
                    & r814_support4_ok
                ).float()
            ),

            "r814_rec_base_ok": (
                (
                    r814_recovery_candidate
                    & r814_base_ok
                ).float()
            ),

            "r814_rec_height_ok": (
                (
                    r814_recovery_candidate
                    & r814_height_ok
                ).float()
            ),

            "r814_rec_upright_ok": (
                (
                    r814_recovery_candidate
                    & r814_upright_ok
                ).float()
            ),

            "r814_rec_body_ok": (
                (
                    r814_recovery_candidate
                    & r814_body_ok
                ).float()
            ),

            "r814_rec_touchdown_ok": (
                (
                    r814_recovery_candidate
                    & r814_touchdown_ok
                ).float()
            ),

            "r814_rec_all_now": (
                r814_recovery_all_now.float()
            ),

            "r814_rec_hold1": (
                (
                    r814_recovery_candidate
                    & (
                        self.recovery_stable_steps >= 1
                    )
                ).float()
            ),

            "r814_rec_hold2": (
                (
                    r814_recovery_candidate
                    & (
                        self.recovery_stable_steps >= 2
                    )
                ).float()
            ),

            "r814_rec_hold3": (
                (
                    r814_recovery_candidate
                    & (
                        self.recovery_stable_steps >= 3
                    )
                ).float()
            ),

            "r814_rec_dx_num": torch.where(
                r814_recovery_candidate,
                recovery_forward_error_r811,
                torch.zeros_like(
                    recovery_forward_error_r811
                ),
            ),

            "r814_rec_dz_num": torch.where(
                r814_recovery_candidate,
                recovery_lift_error_r811,
                torch.zeros_like(
                    recovery_lift_error_r811
                ),
            ),
        }
        # ====================================================
        # R8.29 exact per-environment recovery decomposition.
        # ====================================================

        self.r829_phase = (
            phase.detach().clone()
        )

        self.r829_rec_window = (
            r814_recovery_window.detach().clone()
        )

        self.r829_rec_candidate = (
            r814_recovery_candidate.detach().clone()
        )

        self.r829_forward_ok = (
            (
                r814_recovery_candidate
                & r814_forward_return_ok
            ).detach().clone()
        )

        self.r829_lift_ok = (
            (
                r814_recovery_candidate
                & r814_lift_return_ok
            ).detach().clone()
        )

        self.r829_return_ok = (
            (
                r814_recovery_candidate
                & r814_return_ok
            ).detach().clone()
        )

        self.r829_contact_ok = (
            (
                r814_recovery_candidate
                & r814_contact_ok
            ).detach().clone()
        )

        self.r829_support4_ok = (
            (
                r814_recovery_candidate
                & r814_support4_ok
            ).detach().clone()
        )

        self.r829_base_ok = (
            (
                r814_recovery_candidate
                & r814_base_ok
            ).detach().clone()
        )

        self.r829_height_ok = (
            (
                r814_recovery_candidate
                & r814_height_ok
            ).detach().clone()
        )

        self.r829_upright_ok = (
            (
                r814_recovery_candidate
                & r814_upright_ok
            ).detach().clone()
        )

        self.r829_body_ok = (
            (
                r814_recovery_candidate
                & r814_body_ok
            ).detach().clone()
        )

        self.r829_touchdown_ok = (
            (
                r814_recovery_candidate
                & r814_touchdown_ok
            ).detach().clone()
        )

        self.r829_all_now = (
            r814_recovery_all_now.detach().clone()
        )

        self.r829_recovery_hold = (
            self.recovery_stable_steps
            .detach().clone()
        )

        self.r829_recovery_seen = (
            self.recovery_seen
            .detach().clone()
        )

        self.r829_support_count = (
            support_count.detach().clone()
        )

        self.r829_base_displacement = (
            base_displacement.detach().clone()
        )

        self.r829_height_error = (
            base_height_error.detach().clone()
        )

        self.r829_upright_error = (
            upright_error.detach().clone()
        )

        self.r829_stage_load_ok = (
            self.load_ready_seen_r83
            .detach().clone()
        )

        self.r829_stage_prelift_ok = (
            self.prelift_unload_seen_r84
            .detach().clone()
        )

        self.r829_stage_bridge_ok = (
            self.bridge_seen_r852
            .detach().clone()
        )

        self.r829_stage_swing_ok = (
            self.swing_seen_r89
            .detach().clone()
        )

        self.r829_stage_base_ok = (
            (
                self.max_base_displacement_r8
                <= float(spec["max_base"])
            ).detach().clone()
        )

        self.r829_stage_height_ok = (
            (
                self.max_base_height_error_r8
                <= float(spec["max_height"])
            ).detach().clone()
        )

        self.r829_stage_upright_ok = (
            (
                self.max_upright_error_r8
                <= float(spec["max_upright"])
            ).detach().clone()
        )

        self.r829_stage_support3_ok = (
            (support_count >= 3)
            .detach().clone()
        )

        self.r829_stage_no_violation = (
            (~self.violation_seen)
            .detach().clone()
        )

        self.r829_stage_success = (
            stage_success.detach().clone()
        )

        # ====================================================
        # R8.17 per-environment attribution state.
        # ====================================================

        self.r817_bridge_now = (
            bridge_now_r852.detach().clone()
        )

        self.r817_swing_now = (
            swing_now_r89.detach().clone()
        )

        self.r817_foot_lift = (
            foot_lift.detach().clone()
        )

        self.r817_foot_forward = (
            foot_forward.detach().clone()
        )

        self.r817_load_ratio = (
            load_ratio.detach().clone()
        )

        self.r817_selected_fz = (
            selected_fz.detach().clone()
        )

        self.r817_support3_fz = (
            support3_fz.detach().clone()
        )

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

    if iterations < 25:
        raise RuntimeError(
            "R8.11 requires >=25 iterations."
        )

    return [
        {
            "name": "swing_6cm",
            "min_iterations": 5,
            "max_iterations": 10,
            "lr": 0.0000025,
            "action_std": 0.26,
            "gate_success": 0.0015,
        },

        {
            "name": "recovery",
            "min_iterations": 15,
            "max_iterations": 45,
            "lr": 0.0000020,
            "action_std": 0.26,

            # Discovery/progression gate only.
            "gate_success": 0.0010,
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
            "swing_now_r89": 0.0,
            "swing_seen_r89": 0.0,
            "max_swing_hold_steps_r89": 0.0,
            "recovery_now_r810": 0.0,
            "recovery_seen_r810": 0.0,
            "recovery_hold_steps_r810": 0.0,
            "recovery_final_foot_error_r810": 0.0,
            "recovery_fl_contact_r810": 0.0,
            "recovery_support4_r810": 0.0,
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
            f"swing_now={record['swing_now_r89']:.6f} "
            f"swing_seen={record['swing_seen_r89']:.6f} "
            f"swing_hold={record['max_swing_hold_steps_r89']:.3f} "
            f"recover_now={record['recovery_now_r810']:.6f} "
            f"recover_seen={record['recovery_seen_r810']:.6f} "
            f"recover_hold={record['recovery_hold_steps_r810']:.3f} "
            f"recover_err={record['recovery_final_foot_error_r810']:.5f} "
            f"recover_contact={record['recovery_fl_contact_r810']:.6f} "
            f"support4={record['recovery_support4_r810']:.6f} "
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



def _r814_tensor_list(
    value: Any,
) -> list[float] | None:

    if isinstance(
        value,
        torch.nn.Parameter,
    ):
        value = value.detach()

    if not torch.is_tensor(value):
        return None

    flat = (
        value.detach()
        .float()
        .reshape(-1)
        .cpu()
    )

    if flat.numel() > 128:
        return None

    return [
        float(x)
        for x in flat.tolist()
    ]


def _r814_module_norm(
    module: Any,
) -> float | None:

    if module is None:
        return None

    total = 0.0
    found = False

    try:
        params = module.parameters()
    except Exception:
        return None

    for param in params:
        found = True
        total += float(
            param.detach()
            .float()
            .square()
            .sum()
        )

    if not found:
        return None

    return float(
        math.sqrt(total)
    )


def _r814_policy_snapshot(
    runner: Any,
) -> dict[str, Any]:

    policy = policy_module(runner)

    result: dict[str, Any] = {
        "class":
            policy.__class__.__name__,
    }

    for name in (
        "std",
        "log_std",
        "action_std",
    ):
        try:
            value = getattr(
                policy,
                name,
            )
        except Exception:
            continue

        arr = _r814_tensor_list(
            value
        )

        if arr is not None:
            result[name] = arr

    result["actor_l2"] = (
        _r814_module_norm(
            getattr(
                policy,
                "actor",
                None,
            )
        )
    )

    result["critic_l2"] = (
        _r814_module_norm(
            getattr(
                policy,
                "critic",
                None,
            )
        )
    )

    state_std = {}

    for key, value in (
        policy.state_dict().items()
    ):
        if "std" not in key.lower():
            continue

        arr = _r814_tensor_list(
            value
        )

        if arr is not None:
            state_std[key] = arr

    result[
        "state_dict_std_entries"
    ] = state_std

    return result


def _r814_checkpoint_snapshot(
    checkpoint: Path,
) -> dict[str, Any]:

    try:
        payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(
            checkpoint,
            map_location="cpu",
        )

    result: dict[str, Any] = {
        "checkpoint":
            str(checkpoint),
    }

    if isinstance(payload, dict):

        result["top_level_keys"] = sorted(
            str(k)
            for k in payload.keys()
        )

        result["optimizer_present"] = any(
            "optim" in str(k).lower()
            for k in payload.keys()
        )

        for key in (
            "iter",
            "iteration",
            "current_learning_iteration",
        ):
            value = payload.get(key)

            if isinstance(
                value,
                (int, float),
            ):
                result[key] = value

        state = None

        for key in (
            "model_state_dict",
            "model",
            "state_dict",
        ):
            candidate = payload.get(key)

            if isinstance(
                candidate,
                dict,
            ):
                state = candidate
                break

        if state is not None:

            std_entries = {}

            for key, value in (
                state.items()
            ):
                if (
                    "std"
                    not in str(key).lower()
                ):
                    continue

                arr = _r814_tensor_list(
                    value
                )

                if arr is not None:
                    std_entries[
                        str(key)
                    ] = arr

            result[
                "checkpoint_std_entries"
            ] = std_entries

    del payload

    return result


def _r814_stats_init(
    dim: int,
) -> dict[str, Any]:

    return {
        "n": 0,
        "sum":
            torch.zeros(
                dim,
                dtype=torch.float64,
            ),
        "sq":
            torch.zeros(
                dim,
                dtype=torch.float64,
            ),
        "abs":
            torch.zeros(
                dim,
                dtype=torch.float64,
            ),
        "max_abs":
            torch.zeros(
                dim,
                dtype=torch.float64,
            ),
    }


def _r814_stats_add(
    acc: dict[str, Any],
    x: torch.Tensor,
) -> None:

    if x.numel() == 0:
        return

    y = (
        x.detach()
        .float()
        .reshape(-1, x.shape[-1])
        .cpu()
        .double()
    )

    acc["n"] += int(
        y.shape[0]
    )

    acc["sum"] += y.sum(dim=0)
    acc["sq"] += y.square().sum(dim=0)
    acc["abs"] += y.abs().sum(dim=0)

    acc["max_abs"] = torch.maximum(
        acc["max_abs"],
        y.abs().amax(dim=0),
    )


def _r814_stats_finish(
    acc: dict[str, Any],
) -> dict[str, Any]:

    n = int(acc["n"])

    if n <= 0:
        return {
            "n": 0,
        }

    mean = (
        acc["sum"]
        / n
    )

    var = torch.clamp(
        (
            acc["sq"]
            / n
        )
        - mean.square(),
        min=0.0,
    )

    return {
        "n":
            n,

        "mean":
            [
                float(x)
                for x in mean.tolist()
            ],

        "std":
            [
                float(x)
                for x in torch.sqrt(
                    var
                ).tolist()
            ],

        "abs_mean":
            [
                float(x)
                for x in (
                    acc["abs"]
                    / n
                ).tolist()
            ],

        "max_abs":
            [
                float(x)
                for x in (
                    acc["max_abs"]
                ).tolist()
            ],
    }


def run_r814_case(
    *,
    runner: Any,
    env: IsolatedSupportTransferSkillEnv,
    stage: str,
    mode: str,
    episodes: int,
    seed: int,
) -> dict[str, Any]:

    if mode not in (
        "inference",
        "policy_sample",
        "alg_sample",
    ):
        raise ValueError(
            f"invalid R814 mode: {mode}"
        )

    torch.manual_seed(seed)
    np.random.seed(seed)

    generator = getattr(
        env,
        "generator",
        None,
    )

    if isinstance(
        generator,
        torch.Generator,
    ):
        generator.manual_seed(
            seed
        )

    env.set_curriculum_stage(
        stage
    )

    env.reset()

    policy = policy_module(
        runner
    )

    policy.eval()

    action_dim = 12

    active_action = _r814_stats_init(
        action_dim
    )

    swing_action = _r814_stats_init(
        action_dim
    )

    active_noise = _r814_stats_init(
        action_dim
    )

    swing_noise = _r814_stats_init(
        action_dim
    )

    completed = 0
    stage_success = 0
    strict_success = 0

    episode = {
        "max_forward_sum": 0.0,
        "max_lift_sum": 0.0,
        "bridge_seen": 0.0,
        "swing_seen": 0.0,
        "recovery_seen": 0.0,
    }

    for cm in range(1, 11):
        episode[
            f"forward_ge_{cm}cm"
        ] = 0.0

    for cm in range(1, 7):
        episode[
            f"lift_ge_{cm}cm"
        ] = 0.0

    rec_keys = (
        "r814_rec_window",
        "r814_rec_candidate",
        "r814_rec_forward_ok",
        "r814_rec_lift_ok",
        "r814_rec_return_ok",
        "r814_rec_contact_ok",
        "r814_rec_support4_ok",
        "r814_rec_base_ok",
        "r814_rec_height_ok",
        "r814_rec_upright_ok",
        "r814_rec_body_ok",
        "r814_rec_touchdown_ok",
        "r814_rec_all_now",
        "r814_rec_hold1",
        "r814_rec_hold2",
        "r814_rec_hold3",
        "r814_rec_dx_num",
        "r814_rec_dz_num",
    )

    recovery_totals = {
        key: 0.0
        for key in rec_keys
    }

    reward_n = 0
    reward_sum = 0.0
    reward_sq = 0.0

    value_n = 0
    value_sum = 0.0
    value_sq = 0.0

    steps = 0

    safety_limit = max(
        2500,
        episodes
        * (
            env.total_episode_steps
            + 80
        ),
    )

    first_std_snapshot = None

    with torch.no_grad():

        while completed < episodes:

            obs = TensorDict(
                {
                    "policy":
                        env.low_obs_buf
                },
                batch_size=[
                    env.num_envs
                ],
                device=env.device,
            )

            obs_runner = obs.to(
                runner.device
            )

            active_before = (
                env.kick_active
                .detach()
                .bool()
                .clone()
            )

            phase_before = (
                env._phase()
                .detach()
                .clone()
            )

            swing_mask = (
                active_before
                & (
                    phase_before
                    >= 0.58
                )
                & (
                    phase_before
                    < 0.82
                )
            )

            mean_action = (
                policy
                .act_inference(
                    obs_runner
                )
            )

            if mode == "inference":

                action = mean_action

            elif mode == "policy_sample":

                action = policy.act(
                    obs_runner
                )

            else:

                # Exact sampling path used by PPO training.
                # We intentionally do NOT call update().
                action = runner.alg.act(
                    obs_runner
                )

            if first_std_snapshot is None:
                first_std_snapshot = (
                    _r814_policy_snapshot(
                        runner
                    )
                )

            noise = (
                action
                - mean_action
            )

            if active_before.any():

                _r814_stats_add(
                    active_action,
                    action[
                        active_before
                    ],
                )

                _r814_stats_add(
                    active_noise,
                    noise[
                        active_before
                    ],
                )

            if swing_mask.any():

                _r814_stats_add(
                    swing_action,
                    action[
                        swing_mask
                    ],
                )

                _r814_stats_add(
                    swing_noise,
                    noise[
                        swing_mask
                    ],
                )

            # Critic scale audit.
            try:
                values = policy.evaluate(
                    obs_runner
                )

                values = (
                    values.detach()
                    .float()
                    .reshape(
                        env.num_envs,
                        -1,
                    )
                    .mean(dim=1)
                )

                if active_before.any():

                    selected_values = (
                        values[
                            active_before
                        ]
                    )

                    value_n += int(
                        selected_values.numel()
                    )

                    value_sum += float(
                        selected_values.sum()
                    )

                    value_sq += float(
                        selected_values.square().sum()
                    )

            except Exception:
                pass

            (
                _,
                rewards,
                _,
                _,
                diag,
            ) = env.kick_step(
                action.to(
                    env.device
                )
            )

            rewards_flat = (
                rewards.detach()
                .float()
                .reshape(-1)
            )

            if active_before.any():

                active_reward = (
                    rewards_flat[
                        active_before
                    ]
                )

                reward_n += int(
                    active_reward.numel()
                )

                reward_sum += float(
                    active_reward.sum()
                )

                reward_sq += float(
                    active_reward.square().sum()
                )

            completed += int(
                diag[
                    "completed_episodes"
                ]
            )

            stage_success += int(
                diag[
                    "stage_successful_episodes"
                ]
            )

            strict_success += int(
                diag[
                    "successful_episodes"
                ]
            )

            episode[
                "max_forward_sum"
            ] += float(
                diag.get(
                    "r814_done_max_forward_sum",
                    0.0,
                )
            )

            episode[
                "max_lift_sum"
            ] += float(
                diag.get(
                    "r814_done_max_lift_sum",
                    0.0,
                )
            )

            episode[
                "bridge_seen"
            ] += float(
                diag.get(
                    "r814_done_bridge_seen",
                    0.0,
                )
            )

            episode[
                "swing_seen"
            ] += float(
                diag.get(
                    "r814_done_swing_seen",
                    0.0,
                )
            )

            episode[
                "recovery_seen"
            ] += float(
                diag.get(
                    "r814_done_recovery_seen",
                    0.0,
                )
            )

            for cm in range(
                1,
                11,
            ):

                episode[
                    f"forward_ge_{cm}cm"
                ] += float(
                    diag.get(
                        f"r814_done_fwd_ge_{cm}cm",
                        0.0,
                    )
                )

            for cm in range(
                1,
                7,
            ):

                episode[
                    f"lift_ge_{cm}cm"
                ] += float(
                    diag.get(
                        f"r814_done_lift_ge_{cm}cm",
                        0.0,
                    )
                )

            for key in rec_keys:
                recovery_totals[
                    key
                ] += float(
                    diag.get(
                        key,
                        0.0,
                    )
                )

            steps += 1

            if steps > safety_limit:
                raise RuntimeError(
                    "R8.14 case exceeded safety limit."
                )

    def scalar_stats(
        n,
        total,
        total_sq,
    ):

        if n <= 0:
            return None

        mean = total / n

        variance = max(
            0.0,
            total_sq / n
            - mean * mean,
        )

        return {
            "n": n,
            "mean": mean,
            "std": math.sqrt(
                variance
            ),
        }

    result = {
        "stage":
            stage,

        "mode":
            mode,

        "seed":
            seed,

        "episodes":
            completed,

        "steps":
            steps,

        "stage_success_rate": (
            stage_success
            / completed
            if completed
            else None
        ),

        "strict_success_rate": (
            strict_success
            / completed
            if completed
            else None
        ),

        "episode":
            episode,

        "active_action":
            _r814_stats_finish(
                active_action
            ),

        "swing_phase_action":
            _r814_stats_finish(
                swing_action
            ),

        "active_noise":
            _r814_stats_finish(
                active_noise
            ),

        "swing_phase_noise":
            _r814_stats_finish(
                swing_noise
            ),

        "reward":
            scalar_stats(
                reward_n,
                reward_sum,
                reward_sq,
            ),

        "value":
            scalar_stats(
                value_n,
                value_sum,
                value_sq,
            ),

        "recovery_totals":
            recovery_totals,

        "policy_snapshot":
            first_std_snapshot,
    }

    return result


def run_r814_root_audit(
    *,
    runner: Any,
    env: IsolatedSupportTransferSkillEnv,
    checkpoint: Path,
    checkpoint_name: str,
    summary_path: Path,
    episodes: int,
    seed: int,
) -> None:

    runner.load(
        str(checkpoint),
        load_optimizer=False,
    )

    policy_module(
        runner
    ).eval()

    # Seven cases per checkpoint.
    #
    # lift:
    #   mean vs exact PPO stochastic path
    #
    # swing:
    #   mean vs direct actor sample vs PPO sample
    #
    # recovery:
    #   mean vs PPO stochastic path
    plan = (
        (
            "lift_4cm",
            "inference",
        ),
        (
            "lift_4cm",
            "alg_sample",
        ),
        (
            "swing_6cm",
            "inference",
        ),
        (
            "swing_6cm",
            "policy_sample",
        ),
        (
            "swing_6cm",
            "alg_sample",
        ),
        (
            "recovery",
            "inference",
        ),
        (
            "recovery",
            "alg_sample",
        ),
    )

    results = []

    stage_seed_offset = {
        "lift_4cm": 1000,
        "swing_6cm": 2000,
        "recovery": 3000,
    }

    for stage, mode in plan:

        case_seed = (
            seed
            + stage_seed_offset[
                stage
            ]
        )

        print(
            "[R814_CASE_START] "
            f"checkpoint={checkpoint_name} "
            f"stage={stage} "
            f"mode={mode}",
            flush=True,
        )

        result = run_r814_case(
            runner=runner,
            env=env,
            stage=stage,
            mode=mode,
            episodes=episodes,
            seed=case_seed,
        )

        results.append(
            result
        )

        ep = result["episode"]

        fwd6 = (
            ep["forward_ge_6cm"]
            / result["episodes"]
        )

        lift4 = (
            ep["lift_ge_4cm"]
            / result["episodes"]
        )

        print(
            "[R814_CASE_RESULT] "
            f"checkpoint={checkpoint_name} "
            f"stage={stage} "
            f"mode={mode} "
            f"success={result['stage_success_rate']} "
            f"lift4={lift4:.6f} "
            f"fwd6={fwd6:.6f} "
            f"swing_seen="
            f"{ep['swing_seen']/result['episodes']:.6f}",
            flush=True,
        )

    payload = {
        "version":
            "r8_14_root_cause_audit",

        "training":
            False,

        "ppo_update":
            False,

        "video":
            False,

        "ball":
            False,

        "checkpoint_name":
            checkpoint_name,

        "checkpoint":
            str(checkpoint),

        "episodes_per_case":
            episodes,

        "checkpoint_snapshot":
            _r814_checkpoint_snapshot(
                checkpoint
            ),

        "policy_snapshot":
            _r814_policy_snapshot(
                runner
            ),

        "cases":
            results,
    }

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(
        f"R814_SUMMARY={summary_path}",
        flush=True,
    )

    print(
        "R814_AUDIT_OK",
        flush=True,
    )



def _r817_stats(
    dim: int = 12,
) -> dict[str, Any]:

    return {
        "count": 0,

        "mean_sum":
            torch.zeros(
                dim,
                dtype=torch.float64,
            ),

        "sample_sum":
            torch.zeros(
                dim,
                dtype=torch.float64,
            ),

        "noise_sum":
            torch.zeros(
                dim,
                dtype=torch.float64,
            ),

        "noise_sq":
            torch.zeros(
                dim,
                dtype=torch.float64,
            ),

        "noise_abs":
            torch.zeros(
                dim,
                dtype=torch.float64,
            ),

        "lift_sum": 0.0,
        "forward_sum": 0.0,
        "load_sum": 0.0,
        "fz_sum": 0.0,
        "support3_sum": 0.0,
    }


def _r817_add(
    acc: dict[str, Any],
    mask: torch.Tensor,
    mean_action: torch.Tensor,
    sample_action: torch.Tensor,
    *,
    foot_lift: torch.Tensor,
    foot_forward: torch.Tensor,
    load_ratio: torch.Tensor,
    selected_fz: torch.Tensor,
    support3_fz: torch.Tensor,
) -> None:

    mask = mask.bool()

    n = int(
        mask.sum()
    )

    if n <= 0:
        return

    mean = (
        mean_action[
            mask
        ]
        .detach()
        .float()
        .cpu()
        .double()
    )

    sample = (
        sample_action[
            mask
        ]
        .detach()
        .float()
        .cpu()
        .double()
    )

    noise = (
        sample - mean
    )

    acc["count"] += n

    acc["mean_sum"] += (
        mean.sum(dim=0)
    )

    acc["sample_sum"] += (
        sample.sum(dim=0)
    )

    acc["noise_sum"] += (
        noise.sum(dim=0)
    )

    acc["noise_sq"] += (
        noise.square().sum(dim=0)
    )

    acc["noise_abs"] += (
        noise.abs().sum(dim=0)
    )

    acc["lift_sum"] += float(
        foot_lift[
            mask
        ].sum()
    )

    acc["forward_sum"] += float(
        foot_forward[
            mask
        ].sum()
    )

    acc["load_sum"] += float(
        load_ratio[
            mask
        ].sum()
    )

    acc["fz_sum"] += float(
        selected_fz[
            mask
        ].sum()
    )

    acc["support3_sum"] += float(
        support3_fz[
            mask
        ].sum()
    )


def _r817_finish(
    acc: dict[str, Any],
) -> dict[str, Any]:

    n = int(
        acc["count"]
    )

    if n <= 0:
        return {
            "count": 0,
        }

    mean_action = (
        acc["mean_sum"]
        / n
    )

    sample_action = (
        acc["sample_sum"]
        / n
    )

    noise_mean = (
        acc["noise_sum"]
        / n
    )

    noise_var = torch.clamp(
        (
            acc["noise_sq"]
            / n
        )
        - noise_mean.square(),
        min=0.0,
    )

    return {
        "count":
            n,

        "mean_action":
            [
                float(x)
                for x in mean_action.tolist()
            ],

        "sample_action":
            [
                float(x)
                for x in sample_action.tolist()
            ],

        "noise_mean":
            [
                float(x)
                for x in noise_mean.tolist()
            ],

        "noise_std":
            [
                float(x)
                for x in torch.sqrt(
                    noise_var
                ).tolist()
            ],

        "noise_abs_mean":
            [
                float(x)
                for x in (
                    acc["noise_abs"]
                    / n
                ).tolist()
            ],

        "mean_lift_m":
            acc["lift_sum"] / n,

        "mean_forward_m":
            acc["forward_sum"] / n,

        "mean_load_ratio":
            acc["load_sum"] / n,

        "mean_selected_fz_N":
            acc["fz_sum"] / n,

        "mean_support3_fz_N":
            acc["support3_sum"] / n,
    }


def _r817_action_joint_map(
    env: Any,
) -> dict[str, Any]:

    robot_names = list(
        getattr(
            env.robot.data,
            "joint_names",
            [],
        )
    )

    result = {
        "robot_joint_names":
            robot_names,

        "action_joint_names":
            None,

        "action_joint_ids":
            None,
    }

    try:

        manager = (
            env.low_env
            .unwrapped
            .action_manager
        )

        terms = getattr(
            manager,
            "_terms",
            None,
        )

        candidates = []

        if isinstance(
            terms,
            dict,
        ):
            candidates = list(
                terms.values()
            )

        elif isinstance(
            terms,
            (list, tuple),
        ):
            candidates = list(
                terms
            )

        for term in candidates:

            ids = getattr(
                term,
                "_joint_ids",
                None,
            )

            if ids is None:
                ids = getattr(
                    term,
                    "joint_ids",
                    None,
                )

            if ids is None:
                continue

            if torch.is_tensor(ids):
                ids = [
                    int(x)
                    for x in ids
                    .detach()
                    .cpu()
                    .tolist()
                ]

            elif isinstance(
                ids,
                range,
            ):
                ids = list(ids)

            elif isinstance(
                ids,
                tuple,
            ):
                ids = list(ids)

            if not isinstance(
                ids,
                list,
            ):
                continue

            if len(ids) != 12:
                continue

            names = []

            valid = True

            for idx in ids:

                idx = int(idx)

                if (
                    idx < 0
                    or idx >= len(
                        robot_names
                    )
                ):
                    valid = False
                    break

                names.append(
                    robot_names[idx]
                )

            if valid:

                result[
                    "action_joint_ids"
                ] = ids

                result[
                    "action_joint_names"
                ] = names

                break

    except Exception as exc:

        result[
            "mapping_error"
        ] = (
            f"{type(exc).__name__}:"
            f"{exc}"
        )

    return result


def run_r817_action_attribution(
    *,
    runner: Any,
    env: IsolatedSupportTransferSkillEnv,
    checkpoint: Path,
    summary_path: Path,
    episodes: int,
    seed: int,
) -> None:

    if episodes <= 0:
        raise ValueError(
            "R8.17 episodes must be positive."
        )

    runner.load(
        str(checkpoint),
        load_optimizer=False,
    )

    policy = policy_module(
        runner
    )

    policy.eval()

    torch.manual_seed(
        seed
    )

    np.random.seed(
        seed
    )

    generator = getattr(
        env,
        "generator",
        None,
    )

    if isinstance(
        generator,
        torch.Generator,
    ):
        generator.manual_seed(
            seed
        )

    env.set_curriculum_stage(
        "swing_6cm"
    )

    env.reset()

    spec = env._stage_spec()

    success_lift = float(
        spec["success_lift"]
    )

    success_forward = float(
        spec["success_forward"]
    )

    load_success = float(
        spec["load_success"]
    )

    print(
        "[R817_CONTRACT] "
        f"success_lift={success_lift:.6f} "
        f"success_forward={success_forward:.6f} "
        f"load_success={load_success:.6f}",
        flush=True,
    )

    std_tensor = getattr(
        policy,
        "std",
        None,
    )

    if torch.is_tensor(
        std_tensor
    ):

        std_values = [
            float(x)
            for x in (
                std_tensor.detach()
                .float()
                .reshape(-1)
                .cpu()
                .tolist()
            )
        ]

    else:
        std_values = None

    categories = {
        "swing_phase":
            _r817_stats(),

        "forward_low_lift":
            _r817_stats(),

        "lift_but_loaded":
            _r817_stats(),

        "bridge_now":
            _r817_stats(),

        "swing_now":
            _r817_stats(),
    }

    completed = 0
    stage_successes = 0
    steps = 0

    max_steps = max(
        3000,
        episodes
        * (
            env.total_episode_steps
            + 80
        ),
    )

    with torch.no_grad():

        while completed < episodes:

            obs = TensorDict(
                {
                    "policy":
                        env.low_obs_buf
                },
                batch_size=[
                    env.num_envs
                ],
                device=env.device,
            )

            obs_runner = obs.to(
                runner.device
            )

            phase_before = (
                env._phase()
                .detach()
                .clone()
            )

            active_before = (
                env.kick_active
                .detach()
                .bool()
                .clone()
            )

            mean_action = (
                policy
                .act_inference(
                    obs_runner
                )
            )

            # Exact stochastic actor path used in PPO rollout.
            sample_action = (
                runner.alg.act(
                    obs_runner
                )
            )

            (
                _,
                _,
                _,
                _,
                diag,
            ) = env.kick_step(
                sample_action.to(
                    env.device
                )
            )

            foot_lift = (
                env.r817_foot_lift
            )

            foot_forward = (
                env.r817_foot_forward
            )

            load_ratio = (
                env.r817_load_ratio
            )

            selected_fz = (
                env.r817_selected_fz
            )

            support3_fz = (
                env.r817_support3_fz
            )

            bridge_now = (
                env.r817_bridge_now
                .bool()
            )

            swing_now = (
                env.r817_swing_now
                .bool()
            )

            swing_phase = (
                active_before
                & (
                    phase_before
                    >= 0.58
                )
                & (
                    phase_before
                    < 0.82
                )
            )

            # The exact local optimum we want to understand:
            # enough forward reach, but still below the
            # required airborne lift.
            forward_low_lift = (
                swing_phase
                & (
                    foot_forward
                    >= success_forward
                )
                & (
                    foot_lift
                    < success_lift
                )
            )

            # It reaches required lift but has not unloaded FL.
            lift_but_loaded = (
                swing_phase
                & (
                    foot_lift
                    >= success_lift
                )
                & (
                    load_ratio
                    > load_success
                )
            )

            masks = {
                "swing_phase":
                    swing_phase,

                "forward_low_lift":
                    forward_low_lift,

                "lift_but_loaded":
                    lift_but_loaded,

                "bridge_now":
                    bridge_now,

                "swing_now":
                    swing_now,
            }

            for name, mask in (
                masks.items()
            ):

                _r817_add(
                    categories[
                        name
                    ],
                    mask,
                    mean_action,
                    sample_action,
                    foot_lift=foot_lift,
                    foot_forward=foot_forward,
                    load_ratio=load_ratio,
                    selected_fz=selected_fz,
                    support3_fz=support3_fz,
                )

            completed += int(
                diag[
                    "completed_episodes"
                ]
            )

            stage_successes += int(
                diag[
                    "stage_successful_episodes"
                ]
            )

            steps += 1

            if steps > max_steps:
                raise RuntimeError(
                    "R8.17 exceeded safety limit."
                )

    result = {
        "version":
            "r8_17_success_action_attribution",

        "training":
            False,

        "video":
            False,

        "checkpoint":
            str(checkpoint),

        "stage":
            "swing_6cm",

        "episodes":
            completed,

        "steps":
            steps,

        "stage_success_rate": (
            stage_successes
            / completed
            if completed
            else None
        ),

        "actual_std":
            std_values,

        "contract": {
            "success_lift_m":
                success_lift,

            "success_forward_m":
                success_forward,

            "load_success":
                load_success,
        },

        "joint_mapping":
            _r817_action_joint_map(
                env
            ),

        "categories": {
            key:
                _r817_finish(value)

            for key, value in (
                categories.items()
            )
        },
    }

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(
        "[R817_RESULT] "
        f"episodes={completed} "
        f"success={result['stage_success_rate']} "
        f"forward_low_lift="
        f"{result['categories']['forward_low_lift']['count']} "
        f"bridge_now="
        f"{result['categories']['bridge_now']['count']} "
        f"swing_now="
        f"{result['categories']['swing_now']['count']}",
        flush=True,
    )

    print(
        "R817_ATTRIBUTION_OK",
        flush=True,
    )



class R820LiftResidualHead(torch.nn.Module):
    """Small phase-specific residual head for pre-bridge lift."""

    def __init__(
        self,
        obs_dim: int = 70,
        hidden: int = 64,
        action_dim: int = 12,
        scale: float = 0.30,
    ) -> None:
        super().__init__()

        self.scale = float(scale)

        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden),
            torch.nn.ELU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ELU(),
            torch.nn.Linear(hidden, action_dim),
        )

        # Start exactly at the frozen base policy.
        torch.nn.init.zeros_(
            self.net[-1].weight
        )
        torch.nn.init.zeros_(
            self.net[-1].bias
        )

    def forward(
        self,
        normalized_obs: torch.Tensor,
    ) -> torch.Tensor:

        return (
            self.scale
            * torch.tanh(
                self.net(
                    normalized_obs
                )
            )
        )


def _r820_actor_input(
    policy: Any,
    obs: TensorDict,
) -> torch.Tensor:

    raw = policy.get_actor_obs(obs)

    return policy.actor_obs_normalizer(
        raw
    )


def _r820_residual_stats(
    x: torch.Tensor,
) -> dict[str, Any]:

    y = (
        x.detach()
        .float()
        .reshape(-1, x.shape[-1])
        .cpu()
    )

    return {
        "n":
            int(y.shape[0]),

        "mean_abs":
            [
                float(v)
                for v in y.abs().mean(
                    dim=0
                ).tolist()
            ],

        "rms":
            [
                float(v)
                for v in torch.sqrt(
                    y.square().mean(
                        dim=0
                    )
                ).tolist()
            ],

        "max_abs":
            [
                float(v)
                for v in y.abs().amax(
                    dim=0
                ).tolist()
            ],
    }


def run_r820_lift_residual(
    *,
    runner: Any,
    env: IsolatedSupportTransferSkillEnv,
    checkpoint: Path,
    bundle_path: Path,
    summary_path: Path,
    episodes: int,
    history_steps: int,
    optimization_steps: int,
    evaluation_episodes: int,
    seed: int,
) -> None:

    # ========================================================
    # Load/freeze base actor.
    # ========================================================

    runner.load(
        str(checkpoint),
        load_optimizer=False,
    )

    policy = policy_module(runner)

    policy.eval()

    for parameter in policy.parameters():
        parameter.requires_grad_(False)

    torch.manual_seed(seed)
    np.random.seed(seed)

    generator = getattr(
        env,
        "generator",
        None,
    )

    if isinstance(
        generator,
        torch.Generator,
    ):
        generator.manual_seed(seed)

    env.set_curriculum_stage(
        "swing_6cm"
    )

    env.reset()

    num_envs = env.num_envs

    K = int(history_steps)

    if K < 2:
        raise RuntimeError(
            "R8.20 history_steps must be >=2."
        )

    # ========================================================
    # Collect histories LEADING TO FIRST bridge event.
    #
    # Target:
    # stochastic_action - frozen_base_mean
    #
    # We do NOT imitate post-bridge forward motion.
    # ========================================================

    ring_obs = torch.zeros(
        K,
        num_envs,
        70,
        dtype=torch.float32,
        device=env.device,
    )

    ring_target = torch.zeros(
        K,
        num_envs,
        12,
        dtype=torch.float32,
        device=env.device,
    )

    valid = torch.zeros(
        num_envs,
        dtype=torch.long,
        device=env.device,
    )

    previous_bridge = torch.zeros(
        num_envs,
        dtype=torch.bool,
        device=env.device,
    )

    elite_obs = []
    elite_target = []
    elite_weight = []

    anchor_obs = []

    completed = 0
    steps = 0
    bridge_events = 0
    swing_events = 0

    max_elite = int(
        os.environ.get(
            "R820_MAX_ELITE",
            "12000",
        )
    )

    max_anchor = int(
        os.environ.get(
            "R820_MAX_ANCHOR",
            "5000",
        )
    )

    safety_limit = max(
        5000,
        episodes
        * (
            env.total_episode_steps
            + 80
        ),
    )

    with torch.no_grad():

        while (
            completed < episodes
            and len(elite_obs) < max_elite
        ):

            raw_obs = (
                env.low_obs_buf
                .detach()
                .clone()
            )

            obs = TensorDict(
                {
                    "policy":
                        raw_obs
                },
                batch_size=[
                    num_envs
                ],
                device=env.device,
            )

            obs_runner = obs.to(
                runner.device
            )

            normalized_obs = (
                _r820_actor_input(
                    policy,
                    obs_runner,
                )
            )

            base_mean = (
                policy.act_inference(
                    obs_runner
                )
            )

            sampled_action = (
                policy.act(
                    obs_runner
                )
            )

            target_residual = (
                sampled_action
                - base_mean
            )

            pre_bridge = (
                ~env.bridge_seen_r852
            )

            phase = (
                env._phase()
                .detach()
                .clone()
            )

            # Keep neutral examples so the head does not
            # modify every pre-bridge state aggressively.
            anchor_mask = (
                pre_bridge
                & env.kick_active
                & (phase >= 0.30)
                & (phase < 0.82)
            )

            if (
                len(anchor_obs) < max_anchor
                and anchor_mask.any()
            ):

                ids = torch.nonzero(
                    anchor_mask,
                    as_tuple=False,
                ).squeeze(-1)

                take = min(
                    6,
                    int(ids.numel()),
                    max_anchor
                    - len(anchor_obs),
                )

                for idx in ids[:take].tolist():

                    anchor_obs.append(
                        normalized_obs[idx]
                        .detach()
                        .cpu()
                    )

            slot = steps % K

            ring_obs[slot].copy_(
                normalized_obs.to(
                    env.device
                )
            )

            ring_target[slot].copy_(
                target_residual.to(
                    env.device
                )
            )

            valid = torch.where(
                pre_bridge,
                torch.clamp(
                    valid + 1,
                    max=K,
                ),
                torch.zeros_like(valid),
            )

            (
                _,
                _,
                done,
                _,
                diag,
            ) = env.kick_step(
                sampled_action.to(
                    env.device
                )
            )

            bridge_now = (
                env.r817_bridge_now
                .detach()
                .bool()
            )

            swing_now = (
                env.r817_swing_now
                .detach()
                .bool()
            )

            bridge_rise = (
                bridge_now
                & (~previous_bridge)
            )

            swing_events += int(
                swing_now.sum()
            )

            ids = torch.nonzero(
                bridge_rise,
                as_tuple=False,
            ).squeeze(-1)

            bridge_events += int(
                ids.numel()
            )

            for idx in ids.tolist():

                length = int(
                    valid[idx]
                )

                if length <= 0:
                    continue

                for j in range(length):

                    history_slot = (
                        slot
                        - length
                        + 1
                        + j
                    ) % K

                    # Strongly emphasize the final actions
                    # closest to physical lift-off.
                    recency = (
                        float(j + 1)
                        / float(length)
                    )

                    weight = (
                        0.20
                        + 1.80
                        * recency
                        * recency
                    )

                    elite_obs.append(
                        ring_obs[
                            history_slot,
                            idx,
                        ]
                        .detach()
                        .cpu()
                    )

                    elite_target.append(
                        ring_target[
                            history_slot,
                            idx,
                        ]
                        .detach()
                        .cpu()
                    )

                    elite_weight.append(
                        weight
                    )

                    if len(elite_obs) >= max_elite:
                        break

                if len(elite_obs) >= max_elite:
                    break

            completed += int(
                diag[
                    "completed_episodes"
                ]
            )

            done_mask = (
                done.to(
                    env.device
                )
                .bool()
            )

            previous_bridge = (
                bridge_now.clone()
            )

            if done_mask.any():

                previous_bridge[
                    done_mask
                ] = False

                valid[
                    done_mask
                ] = 0

            steps += 1

            if steps > safety_limit:
                raise RuntimeError(
                    "R8.20 collection safety limit."
                )

    if bridge_events < 100:
        raise RuntimeError(
            "R8.20 insufficient bridge events: "
            f"{bridge_events}"
        )

    if len(elite_obs) < 1000:
        raise RuntimeError(
            "R8.20 insufficient elite samples: "
            f"{len(elite_obs)}"
        )

    if len(anchor_obs) < 500:
        raise RuntimeError(
            "R8.20 insufficient anchors."
        )

    elite_obs_t = torch.stack(
        elite_obs
    )

    elite_target_t = torch.stack(
        elite_target
    )

    elite_weight_t = torch.tensor(
        elite_weight,
        dtype=torch.float32,
    )

    anchor_obs_t = torch.stack(
        anchor_obs
    )

    # ========================================================
    # Train ONLY small residual head.
    # ========================================================

    residual_scale = float(
        os.environ.get(
            "R820_RESIDUAL_SCALE",
            "0.30",
        )
    )

    residual = R820LiftResidualHead(
        obs_dim=70,
        hidden=64,
        action_dim=12,
        scale=residual_scale,
    ).to(
        runner.device
    )

    optimizer = torch.optim.Adam(
        residual.parameters(),
        lr=float(
            os.environ.get(
                "R820_LR",
                "0.0003",
            )
        ),
    )

    batch_size = int(
        os.environ.get(
            "R820_BATCH",
            "512",
        )
    )

    anchor_weight = float(
        os.environ.get(
            "R820_ANCHOR_WEIGHT",
            "0.05",
        )
    )

    n_elite = int(
        elite_obs_t.shape[0]
    )

    n_anchor = int(
        anchor_obs_t.shape[0]
    )

    initial_loss = None
    final_loss = None

    for update in range(
        int(optimization_steps)
    ):

        e_idx = torch.randint(
            0,
            n_elite,
            (
                min(
                    batch_size,
                    n_elite,
                ),
            ),
        )

        a_idx = torch.randint(
            0,
            n_anchor,
            (
                min(
                    batch_size,
                    n_anchor,
                ),
            ),
        )

        e_obs = elite_obs_t[
            e_idx
        ].to(
            runner.device
        )

        e_target = elite_target_t[
            e_idx
        ].to(
            runner.device
        )

        e_weight = elite_weight_t[
            e_idx
        ].to(
            runner.device
        )

        a_obs = anchor_obs_t[
            a_idx
        ].to(
            runner.device
        )

        pred = residual(
            e_obs
        )

        per_sample = (
            (
                pred - e_target
            )
            .square()
            .mean(dim=1)
        )

        elite_loss = (
            per_sample
            * e_weight
        ).sum() / (
            e_weight.sum()
            + 1.0e-8
        )

        # Neutral-state regularization.
        anchor_pred = residual(
            a_obs
        )

        anchor_loss = (
            anchor_pred
            .square()
            .mean()
        )

        loss = (
            elite_loss
            + anchor_weight
            * anchor_loss
        )

        if initial_loss is None:
            initial_loss = float(
                loss.detach()
            )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            residual.parameters(),
            1.0,
        )

        optimizer.step()

        final_loss = float(
            loss.detach()
        )

        if (
            update == 0
            or (update + 1) % 50 == 0
            or update + 1 == optimization_steps
        ):

            print(
                "[R820_RESIDUAL_TRAIN] "
                f"update={update+1}/"
                f"{optimization_steps} "
                f"loss={float(loss):.8f} "
                f"elite={float(elite_loss):.8f} "
                f"anchor={float(anchor_loss):.8f}",
                flush=True,
            )

    residual.eval()

    # Residual magnitude on elite states.
    with torch.no_grad():

        probe_count = min(
            4096,
            n_elite,
        )

        probe_residual = residual(
            elite_obs_t[
                :probe_count
            ].to(
                runner.device
            )
        )

    residual_stats = (
        _r820_residual_stats(
            probe_residual
        )
    )

    # ========================================================
    # Deterministic composite evaluation.
    #
    # BEFORE bridge:
    #   base_mean + lift_residual
    #
    # AFTER bridge:
    #   frozen base_mean only
    # ========================================================

    torch.manual_seed(
        seed + 10000
    )

    np.random.seed(
        seed + 10000
    )

    if isinstance(
        generator,
        torch.Generator,
    ):
        generator.manual_seed(
            seed + 10000
        )

    env.set_curriculum_stage(
        "swing_6cm"
    )

    env.reset()

    eval_completed = 0
    eval_steps = 0

    done_lift4 = 0.0
    done_bridge = 0.0
    done_swing = 0.0
    done_recovery = 0.0

    max_lift_sum = 0.0
    max_forward_sum = 0.0

    residual_active_steps = 0
    residual_abs_sum = 0.0

    eval_safety = max(
        3000,
        evaluation_episodes
        * (
            env.total_episode_steps
            + 80
        ),
    )

    with torch.no_grad():

        while (
            eval_completed
            < evaluation_episodes
        ):

            obs = TensorDict(
                {
                    "policy":
                        env.low_obs_buf
                },
                batch_size=[
                    num_envs
                ],
                device=env.device,
            )

            obs_runner = obs.to(
                runner.device
            )

            base_mean = (
                policy.act_inference(
                    obs_runner
                )
            )

            normalized_obs = (
                _r820_actor_input(
                    policy,
                    obs_runner,
                )
            )

            correction = residual(
                normalized_obs
            )

            pre_bridge = (
                ~env.bridge_seen_r852
            ).to(
                runner.device
            )

            applied = torch.where(
                pre_bridge.unsqueeze(1),
                correction,
                torch.zeros_like(
                    correction
                ),
            )

            action = (
                base_mean
                + applied
            )

            residual_active_steps += int(
                pre_bridge.sum()
            )

            residual_abs_sum += float(
                applied.abs().sum()
            )

            (
                _,
                _,
                _,
                _,
                diag,
            ) = env.kick_step(
                action.to(
                    env.device
                )
            )

            eval_completed += int(
                diag[
                    "completed_episodes"
                ]
            )

            done_lift4 += float(
                diag.get(
                    "r814_done_lift_ge_4cm",
                    0.0,
                )
            )

            done_bridge += float(
                diag.get(
                    "r814_done_bridge_seen",
                    0.0,
                )
            )

            done_swing += float(
                diag.get(
                    "r814_done_swing_seen",
                    0.0,
                )
            )

            done_recovery += float(
                diag.get(
                    "r814_done_recovery_seen",
                    0.0,
                )
            )

            max_lift_sum += float(
                diag.get(
                    "r814_done_max_lift_sum",
                    0.0,
                )
            )

            max_forward_sum += float(
                diag.get(
                    "r814_done_max_forward_sum",
                    0.0,
                )
            )

            eval_steps += 1

            if eval_steps > eval_safety:
                raise RuntimeError(
                    "R8.20 evaluation safety limit."
                )

    payload = {
        "version":
            "r8_20_prebridge_lift_residual",

        "training_type":
            "self_generated_residual_warm_start",

        "external_teacher":
            False,

        "base_actor_updated":
            False,

        "critic_updated":
            False,

        "std_updated":
            False,

        "base_checkpoint":
            str(checkpoint),

        "collection": {
            "episodes":
                completed,

            "steps":
                steps,

            "bridge_events":
                bridge_events,

            "swing_step_events":
                swing_events,

            "elite_samples":
                int(
                    elite_obs_t.shape[0]
                ),

            "anchor_samples":
                int(
                    anchor_obs_t.shape[0]
                ),

            "history_steps":
                K,
        },

        "optimization": {
            "steps":
                int(
                    optimization_steps
                ),

            "initial_loss":
                initial_loss,

            "final_loss":
                final_loss,

            "residual_scale":
                residual_scale,

            "residual_stats":
                residual_stats,
        },

        "deterministic_composite": {
            "episodes":
                eval_completed,

            "lift4_rate":
                done_lift4
                / eval_completed,

            "bridge_rate":
                done_bridge
                / eval_completed,

            "swing_rate":
                done_swing
                / eval_completed,

            "recovery_rate":
                done_recovery
                / eval_completed,

            "mean_max_lift_m":
                max_lift_sum
                / eval_completed,

            "mean_max_forward_m":
                max_forward_sum
                / eval_completed,

            "mean_abs_residual_per_action":
                (
                    residual_abs_sum
                    / (
                        residual_active_steps
                        * 12
                    )
                    if residual_active_steps
                    else 0.0
                ),
        },
    }

    bundle_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "version":
                "r8_20_prebridge_lift_residual",

            "base_checkpoint":
                str(checkpoint),

            "obs_dim":
                70,

            "action_dim":
                12,

            "hidden_dim":
                64,

            "residual_scale":
                residual_scale,

            "residual_state_dict":
                residual.state_dict(),
        },
        bundle_path,
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = payload[
        "deterministic_composite"
    ]

    print(
        "[R820_RESULT] "
        f"episodes={eval_completed} "
        f"lift4={result['lift4_rate']:.6f} "
        f"bridge={result['bridge_rate']:.6f} "
        f"swing={result['swing_rate']:.6f} "
        f"mean_lift="
        f"{result['mean_max_lift_m']:.6f} "
        f"mean_forward="
        f"{result['mean_max_forward_m']:.6f}",
        flush=True,
    )

    print(
        "R820_RESIDUAL_OK",
        flush=True,
    )



def _r823_eval_case(
    *,
    runner: Any,
    env: IsolatedSupportTransferSkillEnv,
    policy: Any,
    residual: Any,
    episodes: int,
    seed: int,
) -> dict[str, Any]:

    torch.manual_seed(seed)
    np.random.seed(seed)

    generator = getattr(
        env,
        "generator",
        None,
    )

    if isinstance(generator, torch.Generator):
        generator.manual_seed(seed)

    env.set_curriculum_stage(
        "lift_4cm"
    )

    env.reset()

    completed = 0
    steps = 0

    lift4 = 0.0
    bridge = 0.0
    swing = 0.0

    max_lift = 0.0
    max_forward = 0.0

    residual_abs = 0.0
    residual_count = 0

    safety = max(
        3000,
        episodes
        * (
            env.total_episode_steps
            + 80
        ),
    )

    with torch.no_grad():

        while completed < episodes:

            obs = TensorDict(
                {
                    "policy":
                        env.low_obs_buf
                },
                batch_size=[
                    env.num_envs
                ],
                device=env.device,
            )

            obs_runner = obs.to(
                runner.device
            )

            base = policy.act_inference(
                obs_runner
            )

            if residual is None:

                action = base

            else:

                normalized = _r820_actor_input(
                    policy,
                    obs_runner,
                )

                correction = residual(
                    normalized
                )

                pre_bridge = (
                    ~env.bridge_seen_r852
                ).to(
                    runner.device
                )

                correction = torch.where(
                    pre_bridge.unsqueeze(1),
                    correction,
                    torch.zeros_like(
                        correction
                    ),
                )

                action = base + correction

                residual_abs += float(
                    correction.abs().sum()
                )

                residual_count += int(
                    pre_bridge.sum()
                ) * 12

            (
                _,
                _,
                _,
                _,
                diag,
            ) = env.kick_step(
                action.to(
                    env.device
                )
            )

            completed += int(
                diag["completed_episodes"]
            )

            lift4 += float(
                diag.get(
                    "r814_done_lift_ge_4cm",
                    0.0,
                )
            )

            bridge += float(
                diag.get(
                    "r814_done_bridge_seen",
                    0.0,
                )
            )

            swing += float(
                diag.get(
                    "r814_done_swing_seen",
                    0.0,
                )
            )

            max_lift += float(
                diag.get(
                    "r814_done_max_lift_sum",
                    0.0,
                )
            )

            max_forward += float(
                diag.get(
                    "r814_done_max_forward_sum",
                    0.0,
                )
            )

            steps += 1

            if steps > safety:
                raise RuntimeError(
                    "R8.23 eval safety limit."
                )

    return {
        "episodes":
            completed,

        "lift4_rate":
            lift4 / completed,

        "bridge_rate":
            bridge / completed,

        "swing_rate":
            swing / completed,

        "mean_max_lift_m":
            max_lift / completed,

        "mean_max_forward_m":
            max_forward / completed,

        "mean_abs_residual":
            (
                residual_abs
                / residual_count
                if residual_count
                else 0.0
            ),
    }


def _r823_train_mode(
    *,
    policy: Any,
    runner: Any,
    obs_history: np.ndarray,
    residual_history: np.ndarray,
    labels: np.ndarray,
    mode: int,
    core_fraction: float,
    optimization_steps: int,
) -> tuple[Any, dict[str, Any]]:

    ids = np.flatnonzero(
        labels == mode
    )

    if len(ids) < 50:
        raise RuntimeError(
            f"R8.23 mode {mode} too small"
        )

    trajectories = (
        residual_history[
            ids
        ]
        .reshape(
            len(ids),
            -1,
        )
        .astype(
            np.float64
        )
    )

    norms = np.linalg.norm(
        trajectories,
        axis=1,
    )

    unit = trajectories / np.maximum(
        norms[:, None],
        1.0e-12,
    )

    centroid = unit.mean(
        axis=0
    )

    centroid /= max(
        np.linalg.norm(centroid),
        1.0e-12,
    )

    cosine = unit @ centroid

    keep_n = max(
        32,
        int(
            round(
                len(ids)
                * core_fraction
            )
        ),
    )

    order = np.argsort(
        -cosine
    )

    core_local = order[
        :keep_n
    ]

    core_ids = ids[
        core_local
    ]

    core_cos = cosine[
        core_local
    ]

    raw_obs = (
        obs_history[
            core_ids
        ]
        .reshape(
            -1,
            70,
        )
        .astype(
            np.float32
        )
    )

    target = (
        residual_history[
            core_ids
        ]
        .reshape(
            -1,
            12,
        )
        .astype(
            np.float32
        )
    )

    # Stronger weight for steps nearest bridge:
    # t-7 ... t-0
    time_weights = np.linspace(
        0.25,
        1.50,
        residual_history.shape[1],
        dtype=np.float32,
    )

    weight = np.tile(
        time_weights,
        len(core_ids),
    )

    # Also prefer the most centroid-consistent trajectories.
    trajectory_weight = (
        0.5
        + 0.5
        * (
            (
                core_cos
                - core_cos.min()
            )
            / max(
                float(
                    core_cos.max()
                    - core_cos.min()
                ),
                1.0e-6,
            )
        )
    ).astype(
        np.float32
    )

    weight *= np.repeat(
        trajectory_weight,
        residual_history.shape[1],
    )

    raw_obs_t = torch.from_numpy(
        raw_obs
    ).to(
        runner.device
    )

    target_t = torch.from_numpy(
        target
    ).to(
        runner.device
    )

    weight_t = torch.from_numpy(
        weight
    ).to(
        runner.device
    )

    with torch.no_grad():

        td = TensorDict(
            {
                "policy":
                    raw_obs_t
            },
            batch_size=[
                raw_obs_t.shape[0]
            ],
            device=runner.device,
        )

        normalized = _r820_actor_input(
            policy,
            td,
        ).detach()

    head = R820LiftResidualHead(
        obs_dim=70,
        hidden=64,
        action_dim=12,
        scale=float(
            os.environ.get(
                "R823_RESIDUAL_SCALE",
                "0.35",
            )
        ),
    ).to(
        runner.device
    )

    optimizer = torch.optim.Adam(
        head.parameters(),
        lr=float(
            os.environ.get(
                "R823_LR",
                "0.0005",
            )
        ),
        weight_decay=1.0e-5,
    )

    batch_size = int(
        os.environ.get(
            "R823_BATCH",
            "512",
        )
    )

    n = int(
        normalized.shape[0]
    )

    initial = None
    final = None

    for update in range(
        optimization_steps
    ):

        idx = torch.randint(
            0,
            n,
            (
                min(
                    batch_size,
                    n,
                ),
            ),
            device=runner.device,
        )

        pred = head(
            normalized[idx]
        )

        per_sample = (
            (
                pred
                - target_t[idx]
            )
            .square()
            .mean(dim=1)
        )

        imitation = (
            per_sample
            * weight_t[idx]
        ).sum() / (
            weight_t[idx].sum()
            + 1.0e-8
        )

        # Mild magnitude regularizer only.
        reg = (
            pred.square().mean()
        )

        loss = (
            imitation
            + 0.01 * reg
        )

        if initial is None:
            initial = float(
                loss.detach()
            )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            head.parameters(),
            1.0,
        )

        optimizer.step()

        final = float(
            loss.detach()
        )

        if (
            update == 0
            or (update + 1) % 100 == 0
            or update + 1 == optimization_steps
        ):

            print(
                "[R823_TRAIN] "
                f"mode={mode} "
                f"update={update+1}/"
                f"{optimization_steps} "
                f"loss={float(loss):.8f}",
                flush=True,
            )

    head.eval()

    with torch.no_grad():

        all_pred = head(
            normalized
        )

    stats = {
        "mode":
            int(mode),

        "all_mode_trajectories":
            int(len(ids)),

        "core_trajectories":
            int(len(core_ids)),

        "core_fraction":
            float(
                len(core_ids)
                / len(ids)
            ),

        "core_cosine_mean":
            float(
                core_cos.mean()
            ),

        "core_cosine_min":
            float(
                core_cos.min()
            ),

        "core_cosine_max":
            float(
                core_cos.max()
            ),

        "training_samples":
            int(n),

        "initial_loss":
            initial,

        "final_loss":
            final,

        "target_abs_mean":
            float(
                target_t.abs().mean()
            ),

        "pred_abs_mean":
            float(
                all_pred.abs().mean()
            ),
    }

    return head, stats


def run_r823_mode_core_ab(
    *,
    runner: Any,
    env: IsolatedSupportTransferSkillEnv,
    checkpoint: Path,
    dataset_path: Path,
    output_dir: Path,
    evaluation_episodes: int,
    seed: int,
) -> None:

    runner.load(
        str(checkpoint),
        load_optimizer=False,
    )

    policy = policy_module(
        runner
    )

    policy.eval()

    for parameter in policy.parameters():
        parameter.requires_grad_(False)

    data = np.load(
        dataset_path
    )

    obs_history = data[
        "obs_history"
    ]

    residual_history = data[
        "residual_history"
    ]

    labels = data[
        "labels_k2"
    ]

    if (
        obs_history.ndim != 3
        or obs_history.shape[1:] != (8, 70)
    ):
        raise RuntimeError(
            f"bad obs shape: {obs_history.shape}"
        )

    if (
        residual_history.ndim != 3
        or residual_history.shape[1:] != (8, 12)
    ):
        raise RuntimeError(
            "bad residual shape: "
            f"{residual_history.shape}"
        )

    core_fraction = float(
        os.environ.get(
            "R823_CORE_FRACTION",
            "0.50",
        )
    )

    optimization_steps = int(
        os.environ.get(
            "R823_OPT_STEPS",
            "400",
        )
    )

    results = {}

    # --------------------------------------------------------
    # Fresh deterministic base evaluation.
    # --------------------------------------------------------

    results["base"] = _r823_eval_case(
        runner=runner,
        env=env,
        policy=policy,
        residual=None,
        episodes=evaluation_episodes,
        seed=seed + 1000,
    )

    heads = {}

    for mode in (0, 1):

        head, training = _r823_train_mode(
            policy=policy,
            runner=runner,
            obs_history=obs_history,
            residual_history=residual_history,
            labels=labels,
            mode=mode,
            core_fraction=core_fraction,
            optimization_steps=optimization_steps,
        )

        heads[mode] = head

        result = _r823_eval_case(
            runner=runner,
            env=env,
            policy=policy,
            residual=head,
            episodes=evaluation_episodes,
            seed=seed + 1000,
        )

        results[
            f"mode{mode}"
        ] = {
            "training":
                training,

            "evaluation":
                result,
        }

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for mode in (0, 1):

        torch.save(
            {
                "version":
                    "r8_23_mode_core_residual",

                "mode":
                    mode,

                "base_checkpoint":
                    str(checkpoint),

                "state_dict":
                    heads[
                        mode
                    ].state_dict(),

                "residual_scale":
                    float(
                        os.environ.get(
                            "R823_RESIDUAL_SCALE",
                            "0.35",
                        )
                    ),
            },
            output_dir
            / f"mode{mode}_head.pt",
        )

    base_lift = results[
        "base"
    ][
        "mean_max_lift_m"
    ]

    ranking = []

    for mode in (0, 1):

        e = results[
            f"mode{mode}"
        ][
            "evaluation"
        ]

        ranking.append(
            (
                e["bridge_rate"] > 0,
                e["lift4_rate"] > 0,
                e["mean_max_lift_m"],
                -abs(
                    e["mean_max_forward_m"]
                    - results["base"][
                        "mean_max_forward_m"
                    ]
                ),
                mode,
            )
        )

    ranking.sort(
        reverse=True
    )

    winner = int(
        ranking[0][-1]
    )

    winner_eval = results[
        f"mode{winner}"
    ][
        "evaluation"
    ]

    gain = (
        winner_eval[
            "mean_max_lift_m"
        ]
        - base_lift
    )

    if winner_eval[
        "bridge_rate"
    ] > 0:

        verdict = (
            "MODE_CORE_DET_BRIDGE_ESTABLISHED"
        )

    elif winner_eval[
        "lift4_rate"
    ] > 0:

        verdict = (
            "MODE_CORE_DET_LIFT4_ESTABLISHED"
        )

    elif gain >= 0.005:

        verdict = (
            "MODE_CORE_DIRECTION_STRONG"
        )

    elif gain >= 0.0025:

        verdict = (
            "MODE_CORE_DIRECTION_POSITIVE"
        )

    else:

        verdict = (
            "MODE_SPECIFIC_BC_EXHAUSTED"
        )

    summary = {
        "version":
            "r8_23_mode_core_residual_ab",

        "training":
            "offline_mode_specific_residual_only",

        "ppo_update":
            False,

        "reward_change":
            False,

        "base_checkpoint":
            str(checkpoint),

        "dataset":
            str(dataset_path),

        "core_fraction":
            core_fraction,

        "evaluation_episodes":
            evaluation_episodes,

        "results":
            results,

        "winner_mode":
            winner,

        "winner_lift_gain_m":
            gain,

        "verdict":
            verdict,
    }

    (
        output_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(
        "[R823_RESULT] "
        f"base_lift={base_lift:.6f} "
        f"mode0_lift="
        f"{results['mode0']['evaluation']['mean_max_lift_m']:.6f} "
        f"mode1_lift="
        f"{results['mode1']['evaluation']['mean_max_lift_m']:.6f} "
        f"winner={winner} "
        f"verdict={verdict}",
        flush=True,
    )

    print(
        "R823_MODE_CORE_OK",
        flush=True,
    )



def _r824_load_policy_copy(
    runner: Any,
    checkpoint: Path,
) -> Any:

    import copy as _copy

    runner.load(
        str(checkpoint),
        load_optimizer=False,
    )

    source = policy_module(
        runner
    )

    result = _copy.deepcopy(
        source
    ).to(
        runner.device
    )

    result.eval()

    result.distribution = None

    for parameter in result.parameters():
        parameter.requires_grad_(False)

    return result


def _r824_eval_composition(
    *,
    runner: Any,
    env: IsolatedSupportTransferSkillEnv,
    lift_policy: Any,
    lift_head: Any,
    post_policy: Any,
    switch_mode: str,
    episodes: int,
    seed: int,
) -> dict[str, Any]:

    torch.manual_seed(seed)
    np.random.seed(seed)

    generator = getattr(
        env,
        "generator",
        None,
    )

    if isinstance(
        generator,
        torch.Generator,
    ):
        generator.manual_seed(seed)

    env.set_curriculum_stage(
        "swing_6cm"
    )

    env.reset()

    num_envs = env.num_envs

    bridge_age = torch.zeros(
        num_envs,
        dtype=torch.long,
        device=env.device,
    )

    completed = 0
    steps = 0

    lift4 = 0.0
    bridge = 0.0
    forward6 = 0.0
    swing = 0.0

    max_lift = 0.0
    max_forward = 0.0

    post_steps = 0
    post_lift_sum = 0.0
    post_forward_sum = 0.0
    post_load_sum = 0.0
    post_fz_sum = 0.0
    post_support_sum = 0.0

    pre_residual_abs = 0.0
    pre_residual_count = 0

    safety = max(
        4000,
        episodes
        * (
            env.total_episode_steps
            + 80
        ),
    )

    with torch.no_grad():

        while completed < episodes:

            obs = TensorDict(
                {
                    "policy":
                        env.low_obs_buf
                },
                batch_size=[
                    num_envs
                ],
                device=env.device,
            )

            obs_runner = obs.to(
                runner.device
            )

            bridge_seen = (
                env.bridge_seen_r852
                .detach()
                .bool()
                .clone()
            )

            bridge_age = torch.where(
                bridge_seen,
                bridge_age + 1,
                torch.zeros_like(
                    bridge_age
                ),
            )

            # -----------------------------------------------
            # Pre-bridge deterministic Lift primitive.
            # -----------------------------------------------

            lift_base_action = (
                lift_policy.act_inference(
                    obs_runner
                )
            )

            lift_normalized = (
                _r820_actor_input(
                    lift_policy,
                    obs_runner,
                )
            )

            correction = lift_head(
                lift_normalized
            )

            lift_action = (
                lift_base_action
                + correction
            )

            # -----------------------------------------------
            # Candidate deterministic post-bridge actor.
            # -----------------------------------------------

            post_action = (
                post_policy.act_inference(
                    obs_runner
                )
            )

            if switch_mode == "immediate":

                alpha = bridge_seen.float()

            elif switch_mode == "hold2":

                alpha = (
                    bridge_seen
                    & (bridge_age >= 3)
                ).float()

            elif switch_mode == "blend4":

                alpha = torch.where(
                    bridge_seen,
                    torch.clamp(
                        (
                            bridge_age.float()
                            - 1.0
                        ) / 4.0,
                        0.0,
                        1.0,
                    ),
                    torch.zeros_like(
                        bridge_age,
                        dtype=torch.float32,
                    ),
                )

            else:
                raise RuntimeError(
                    f"Unknown R8.24 switch: "
                    f"{switch_mode}"
                )

            action = (
                (
                    1.0
                    - alpha.unsqueeze(1)
                )
                * lift_action

                + alpha.unsqueeze(1)
                * post_action
            )

            pre_mask = (
                ~bridge_seen
            )

            if pre_mask.any():

                pre_residual_abs += float(
                    correction[
                        pre_mask.to(
                            runner.device
                        )
                    ]
                    .abs()
                    .sum()
                )

                pre_residual_count += (
                    int(
                        pre_mask.sum()
                    )
                    * 12
                )

            (
                _,
                _,
                done,
                _,
                diag,
            ) = env.kick_step(
                action.to(
                    env.device
                )
            )

            # Metrics after action, using the exact
            # per-environment physics exported by R8.17.
            post_mask = (
                bridge_seen
            )

            if post_mask.any():

                n = int(
                    post_mask.sum()
                )

                post_steps += n

                post_lift_sum += float(
                    env.r817_foot_lift[
                        post_mask
                    ].sum()
                )

                post_forward_sum += float(
                    env.r817_foot_forward[
                        post_mask
                    ].sum()
                )

                post_load_sum += float(
                    env.r817_load_ratio[
                        post_mask
                    ].sum()
                )

                post_fz_sum += float(
                    env.r817_selected_fz[
                        post_mask
                    ].sum()
                )

                post_support_sum += float(
                    env.r817_support3_fz[
                        post_mask
                    ].sum()
                )

            completed += int(
                diag[
                    "completed_episodes"
                ]
            )

            lift4 += float(
                diag.get(
                    "r814_done_lift_ge_4cm",
                    0.0,
                )
            )

            bridge += float(
                diag.get(
                    "r814_done_bridge_seen",
                    0.0,
                )
            )

            forward6 += float(
                diag.get(
                    "r814_done_forward_ge_6cm",
                    0.0,
                )
            )

            swing += float(
                diag.get(
                    "r814_done_swing_seen",
                    0.0,
                )
            )

            max_lift += float(
                diag.get(
                    "r814_done_max_lift_sum",
                    0.0,
                )
            )

            max_forward += float(
                diag.get(
                    "r814_done_max_forward_sum",
                    0.0,
                )
            )

            done_mask = (
                done.to(
                    env.device
                )
                .bool()
            )

            if done_mask.any():

                bridge_age[
                    done_mask
                ] = 0

            steps += 1

            if steps > safety:
                raise RuntimeError(
                    "R8.24 evaluation safety limit."
                )

    return {
        "episodes":
            completed,

        "lift4_rate":
            lift4 / completed,

        "bridge_rate":
            bridge / completed,

        "forward6_rate":
            forward6 / completed,

        "swing_rate":
            swing / completed,

        "mean_max_lift_m":
            max_lift / completed,

        "mean_max_forward_m":
            max_forward / completed,

        "pre_mean_abs_residual":
            (
                pre_residual_abs
                / pre_residual_count
                if pre_residual_count
                else 0.0
            ),

        "post_samples":
            post_steps,

        "post_mean_lift_m":
            (
                post_lift_sum
                / post_steps
                if post_steps
                else 0.0
            ),

        "post_mean_forward_m":
            (
                post_forward_sum
                / post_steps
                if post_steps
                else 0.0
            ),

        "post_mean_load_ratio":
            (
                post_load_sum
                / post_steps
                if post_steps
                else 0.0
            ),

        "post_mean_selected_fz_N":
            (
                post_fz_sum
                / post_steps
                if post_steps
                else 0.0
            ),

        "post_mean_support3_fz_N":
            (
                post_support_sum
                / post_steps
                if post_steps
                else 0.0
            ),
    }


def run_r824_composition_sweep(
    *,
    runner: Any,
    env: IsolatedSupportTransferSkillEnv,
    lift_checkpoint: Path,
    lift_head_path: Path,
    post_checkpoints: dict[str, Path],
    summary_path: Path,
    episodes: int,
    seed: int,
) -> None:

    # ========================================================
    # Load all policies once.
    # ========================================================

    lift_policy = _r824_load_policy_copy(
        runner,
        lift_checkpoint,
    )

    head_data = torch.load(
        lift_head_path,
        map_location=runner.device,
        weights_only=False,
    )

    lift_head = R820LiftResidualHead(
        obs_dim=70,
        hidden=64,
        action_dim=12,
        scale=float(
            head_data[
                "residual_scale"
            ]
        ),
    ).to(
        runner.device
    )

    lift_head.load_state_dict(
        head_data[
            "state_dict"
        ]
    )

    lift_head.eval()

    for parameter in lift_head.parameters():
        parameter.requires_grad_(False)

    post_policies = {}

    for name, checkpoint in (
        post_checkpoints.items()
    ):

        post_policies[name] = (
            _r824_load_policy_copy(
                runner,
                checkpoint,
            )
        )

        print(
            "[R824_POLICY_LOADED] "
            f"name={name} "
            f"checkpoint={checkpoint}",
            flush=True,
        )

    cases = []

    switch_modes = (
        "immediate",
        "hold2",
        "blend4",
    )

    # Same seed per case on purpose:
    # deterministic comparison gets the same reset stream.
    for actor_name, actor in (
        post_policies.items()
    ):

        for switch_mode in switch_modes:

            label = (
                f"{actor_name}__"
                f"{switch_mode}"
            )

            print(
                "[R824_CASE_START] "
                f"{label}",
                flush=True,
            )

            result = (
                _r824_eval_composition(
                    runner=runner,
                    env=env,
                    lift_policy=lift_policy,
                    lift_head=lift_head,
                    post_policy=actor,
                    switch_mode=switch_mode,
                    episodes=episodes,
                    seed=seed + 9000,
                )
            )

            result[
                "post_actor"
            ] = actor_name

            result[
                "switch_mode"
            ] = switch_mode

            result[
                "label"
            ] = label

            cases.append(
                result
            )

            print(
                "[R824_CASE_RESULT] "
                f"label={label} "
                f"lift4="
                f"{result['lift4_rate']:.6f} "
                f"bridge="
                f"{result['bridge_rate']:.6f} "
                f"fwd6="
                f"{result['forward6_rate']:.6f} "
                f"swing="
                f"{result['swing_rate']:.6f} "
                f"lift="
                f"{result['mean_max_lift_m']:.6f} "
                f"forward="
                f"{result['mean_max_forward_m']:.6f} "
                f"post_load="
                f"{result['post_mean_load_ratio']:.6f}",
                flush=True,
            )

    # ========================================================
    # Rank:
    #
    # 1 swing
    # 2 bridge
    # 3 forward6
    # 4 retain airborne lift
    # 5 low post FL load
    # ========================================================

    ranked = sorted(
        cases,
        key=lambda x: (
            x["swing_rate"],
            x["bridge_rate"],
            x["forward6_rate"],
            x["mean_max_lift_m"],
            -x["post_mean_load_ratio"],
            x["post_mean_support3_fz_N"],
        ),
        reverse=True,
    )

    winner = ranked[0]

    if winner[
        "swing_rate"
    ] > 0.50:

        verdict = (
            "COMPOSED_DETERMINISTIC_SWING_STRONG"
        )

    elif winner[
        "swing_rate"
    ] > 0.0:

        verdict = (
            "COMPOSED_DETERMINISTIC_SWING_EMERGED"
        )

    elif (
        winner[
            "bridge_rate"
        ] > 0.90
        and winner[
            "forward6_rate"
        ] > 0.90
    ):

        verdict = (
            "BRIDGE_AND_FORWARD_EXIST_"
            "BUT_NOT_SIMULTANEOUS"
        )

    elif winner[
        "bridge_rate"
    ] < 0.50:

        verdict = (
            "SWITCH_DESTROYS_LIFT_PRIMITIVE"
        )

    else:

        verdict = (
            "POST_BRIDGE_SWING_ACTOR_INSUFFICIENT"
        )

    summary = {
        "version":
            "r8_24_lift_swing_composition_sweep",

        "training":
            False,

        "reward_change":
            False,

        "video":
            False,

        "lift_checkpoint":
            str(lift_checkpoint),

        "lift_head":
            str(lift_head_path),

        "evaluation_episodes_per_case":
            episodes,

        "post_checkpoints": {
            key:
                str(value)

            for key, value in (
                post_checkpoints.items()
            )
        },

        "cases":
            cases,

        "ranking":
            [
                x["label"]
                for x in ranked
            ],

        "winner":
            winner["label"],

        "verdict":
            verdict,
    }

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(
        "[R824_RESULT] "
        f"winner={winner['label']} "
        f"swing={winner['swing_rate']:.6f} "
        f"bridge={winner['bridge_rate']:.6f} "
        f"fwd6={winner['forward6_rate']:.6f} "
        f"verdict={verdict}",
        flush=True,
    )

    print(
        "R824_COMPOSITION_OK",
        flush=True,
    )



def _r825_eval_chain(
    *,
    runner: Any,
    env: IsolatedSupportTransferSkillEnv,
    lift_policy: Any,
    lift_head: Any,
    swing_policy: Any,
    recovery_policy: Any | None,
    stage: str,
    recovery_switch: str,
    episodes: int,
    seed: int,
) -> dict[str, Any]:

    torch.manual_seed(seed)
    np.random.seed(seed)

    generator = getattr(
        env,
        "generator",
        None,
    )

    if isinstance(generator, torch.Generator):
        generator.manual_seed(seed)

    env.set_curriculum_stage(stage)
    env.reset()

    N = env.num_envs

    bridge_age = torch.zeros(
        N,
        dtype=torch.long,
        device=env.device,
    )

    swing_age = torch.zeros_like(
        bridge_age
    )

    bridge_seen_ep = torch.zeros(
        N,
        dtype=torch.bool,
        device=env.device,
    )

    swing_seen_ep = torch.zeros_like(
        bridge_seen_ep
    )

    candidate_seen_ep = torch.zeros_like(
        bridge_seen_ep
    )

    max_lift_ep = torch.zeros(
        N,
        dtype=torch.float32,
        device=env.device,
    )

    max_forward_ep = torch.zeros_like(
        max_lift_ep
    )

    completed = 0
    steps = 0

    done_lift4 = 0
    done_bridge = 0
    done_fwd6 = 0
    done_swing = 0
    done_candidate = 0

    official_stage_success = 0

    done_max_lift_sum = 0.0
    done_max_forward_sum = 0.0

    done_abs_forward_sum = 0.0
    done_abs_lift_sum = 0.0
    done_fz_sum = 0.0
    done_support3_sum = 0.0

    post_swing_samples = 0
    post_swing_abs_forward = 0.0
    post_swing_abs_lift = 0.0
    post_swing_fz = 0.0
    post_swing_support3 = 0.0

    safety = max(
        4000,
        episodes
        * (
            env.total_episode_steps
            + 100
        ),
    )

    with torch.no_grad():

        while completed < episodes:

            obs = TensorDict(
                {
                    "policy":
                        env.low_obs_buf
                },
                batch_size=[N],
                device=env.device,
            )

            obs_runner = obs.to(
                runner.device
            )

            # =================================================
            # 1. Deterministic Lift primitive.
            # =================================================

            lift_base = (
                lift_policy.act_inference(
                    obs_runner
                )
            )

            lift_normalized = (
                _r820_actor_input(
                    lift_policy,
                    obs_runner,
                )
            )

            lift_correction = (
                lift_head(
                    lift_normalized
                )
            )

            lift_action = (
                lift_base
                + lift_correction
            )

            # =================================================
            # 2. R8.24 winner:
            #    bridge + hold2 -> R8.15 swing.
            # =================================================

            bridge_state = (
                env.bridge_seen_r852
                .detach()
                .bool()
                .clone()
            )

            bridge_age = torch.where(
                bridge_state,
                bridge_age + 1,
                torch.zeros_like(
                    bridge_age
                ),
            )

            swing_action = (
                swing_policy.act_inference(
                    obs_runner
                )
            )

            swing_alpha = (
                bridge_state
                & (bridge_age >= 3)
            ).float()

            chain_action = (
                (
                    1.0
                    - swing_alpha.unsqueeze(1)
                )
                * lift_action
                + swing_alpha.unsqueeze(1)
                * swing_action
            )

            # =================================================
            # 3. Optional Recovery policy after swing event.
            # =================================================

            if recovery_policy is None:

                action = chain_action

            else:

                recovery_action = (
                    recovery_policy.act_inference(
                        obs_runner
                    )
                )

                # swing_seen_ep is our own persistent latch.
                swing_age = torch.where(
                    swing_seen_ep,
                    swing_age + 1,
                    torch.zeros_like(
                        swing_age
                    ),
                )

                if recovery_switch == "immediate":

                    beta = (
                        swing_seen_ep.float()
                    )

                elif recovery_switch == "hold2":

                    beta = (
                        swing_seen_ep
                        & (swing_age >= 3)
                    ).float()

                elif recovery_switch == "blend4":

                    beta = torch.where(
                        swing_seen_ep,
                        torch.clamp(
                            (
                                swing_age.float()
                                - 1.0
                            ) / 4.0,
                            0.0,
                            1.0,
                        ),
                        torch.zeros(
                            N,
                            dtype=torch.float32,
                            device=env.device,
                        ),
                    )

                else:
                    raise RuntimeError(
                        "Unknown recovery switch: "
                        f"{recovery_switch}"
                    )

                action = (
                    (
                        1.0
                        - beta.unsqueeze(1)
                    )
                    * chain_action
                    + beta.unsqueeze(1)
                    * recovery_action
                )

            (
                _,
                _,
                done,
                _,
                diag,
            ) = env.kick_step(
                action.to(
                    env.device
                )
            )

            # =================================================
            # Physics from THIS step, before our own latches
            # are reset.
            # =================================================

            foot_lift = (
                env.r817_foot_lift
                .detach()
            )

            foot_forward = (
                env.r817_foot_forward
                .detach()
            )

            selected_fz = (
                env.r817_selected_fz
                .detach()
            )

            support3 = (
                env.r817_support3_fz
                .detach()
            )

            bridge_now = (
                env.r817_bridge_now
                .detach()
                .bool()
            )

            swing_now = (
                env.r817_swing_now
                .detach()
                .bool()
            )

            bridge_seen_ep |= bridge_now
            swing_seen_ep |= swing_now

            max_lift_ep = torch.maximum(
                max_lift_ep,
                foot_lift,
            )

            max_forward_ep = torch.maximum(
                max_forward_ep,
                foot_forward,
            )

            # Loose retraction diagnostic only.
            # Official Recovery success remains the
            # environment stage-success contract.
            retract_candidate = (
                swing_seen_ep
                & (
                    torch.abs(
                        foot_forward
                    ) <= 0.045
                )
                & (
                    torch.abs(
                        foot_lift
                    ) <= 0.015
                )
                & (
                    selected_fz >= 8.0
                )
                & (
                    support3 >= 100.0
                )
            )

            candidate_seen_ep |= (
                retract_candidate
            )

            if swing_seen_ep.any():

                mask = swing_seen_ep

                n = int(mask.sum())

                post_swing_samples += n

                post_swing_abs_forward += float(
                    torch.abs(
                        foot_forward[mask]
                    ).sum()
                )

                post_swing_abs_lift += float(
                    torch.abs(
                        foot_lift[mask]
                    ).sum()
                )

                post_swing_fz += float(
                    selected_fz[
                        mask
                    ].sum()
                )

                post_swing_support3 += float(
                    support3[
                        mask
                    ].sum()
                )

            official_stage_success += int(
                diag[
                    "stage_successful_episodes"
                ]
            )

            completed_step = int(
                diag[
                    "completed_episodes"
                ]
            )

            completed += completed_step

            done_mask = (
                done.to(
                    env.device
                )
                .bool()
            )

            if done_mask.any():

                done_lift4 += int(
                    (
                        max_lift_ep[
                            done_mask
                        ] >= 0.040
                    ).sum()
                )

                done_bridge += int(
                    bridge_seen_ep[
                        done_mask
                    ].sum()
                )

                # Correct R8.24 bookkeeping:
                # calculate forward >=6 cm ourselves.
                done_fwd6 += int(
                    (
                        max_forward_ep[
                            done_mask
                        ] >= 0.060
                    ).sum()
                )

                done_swing += int(
                    swing_seen_ep[
                        done_mask
                    ].sum()
                )

                done_candidate += int(
                    candidate_seen_ep[
                        done_mask
                    ].sum()
                )

                done_max_lift_sum += float(
                    max_lift_ep[
                        done_mask
                    ].sum()
                )

                done_max_forward_sum += float(
                    max_forward_ep[
                        done_mask
                    ].sum()
                )

                done_abs_forward_sum += float(
                    torch.abs(
                        foot_forward[
                            done_mask
                        ]
                    ).sum()
                )

                done_abs_lift_sum += float(
                    torch.abs(
                        foot_lift[
                            done_mask
                        ]
                    ).sum()
                )

                done_fz_sum += float(
                    selected_fz[
                        done_mask
                    ].sum()
                )

                done_support3_sum += float(
                    support3[
                        done_mask
                    ].sum()
                )

                bridge_age[
                    done_mask
                ] = 0

                swing_age[
                    done_mask
                ] = 0

                bridge_seen_ep[
                    done_mask
                ] = False

                swing_seen_ep[
                    done_mask
                ] = False

                candidate_seen_ep[
                    done_mask
                ] = False

                max_lift_ep[
                    done_mask
                ] = 0.0

                max_forward_ep[
                    done_mask
                ] = 0.0

            steps += 1

            if steps > safety:
                raise RuntimeError(
                    "R8.25 evaluation safety limit."
                )

    denom = max(
        completed,
        1,
    )

    return {
        "stage":
            stage,

        "episodes":
            completed,

        "lift4_rate":
            done_lift4 / denom,

        "bridge_rate":
            done_bridge / denom,

        "forward6_rate":
            done_fwd6 / denom,

        "swing_rate":
            done_swing / denom,

        "candidate_rate":
            done_candidate / denom,

        "official_stage_success_rate":
            official_stage_success / denom,

        "mean_max_lift_m":
            done_max_lift_sum / denom,

        "mean_max_forward_m":
            done_max_forward_sum / denom,

        "done_abs_forward_m":
            done_abs_forward_sum / denom,

        "done_abs_lift_m":
            done_abs_lift_sum / denom,

        "done_selected_fz_N":
            done_fz_sum / denom,

        "done_support3_fz_N":
            done_support3_sum / denom,

        "post_swing_samples":
            post_swing_samples,

        "post_swing_abs_forward_m":
            (
                post_swing_abs_forward
                / post_swing_samples
                if post_swing_samples
                else 0.0
            ),

        "post_swing_abs_lift_m":
            (
                post_swing_abs_lift
                / post_swing_samples
                if post_swing_samples
                else 0.0
            ),

        "post_swing_selected_fz_N":
            (
                post_swing_fz
                / post_swing_samples
                if post_swing_samples
                else 0.0
            ),

        "post_swing_support3_fz_N":
            (
                post_swing_support3
                / post_swing_samples
                if post_swing_samples
                else 0.0
            ),
    }


def run_r825_recovery_sweep(
    *,
    runner: Any,
    env: IsolatedSupportTransferSkillEnv,
    lift_checkpoint: Path,
    lift_head_path: Path,
    swing_checkpoint: Path,
    recovery_checkpoints: dict[str, Path],
    summary_path: Path,
    sweep_episodes: int,
    confirm_episodes: int,
    seed: int,
) -> None:

    lift_policy = _r824_load_policy_copy(
        runner,
        lift_checkpoint,
    )

    swing_policy = _r824_load_policy_copy(
        runner,
        swing_checkpoint,
    )

    head_data = torch.load(
        lift_head_path,
        map_location=runner.device,
        weights_only=False,
    )

    lift_head = R820LiftResidualHead(
        obs_dim=70,
        hidden=64,
        action_dim=12,
        scale=float(
            head_data[
                "residual_scale"
            ]
        ),
    ).to(
        runner.device
    )

    lift_head.load_state_dict(
        head_data[
            "state_dict"
        ]
    )

    lift_head.eval()

    for parameter in lift_head.parameters():
        parameter.requires_grad_(False)

    recovery_policies = {}

    for name, ckpt in (
        recovery_checkpoints.items()
    ):

        recovery_policies[name] = (
            _r824_load_policy_copy(
                runner,
                ckpt,
            )
        )

        print(
            "[R825_POLICY_LOADED] "
            f"{name}={ckpt}",
            flush=True,
        )

    # ========================================================
    # Baselines:
    # confirm R8.24 and detect recovery-stage context effects.
    # ========================================================

    baselines = {}

    for stage in (
        "swing_6cm",
        "recovery",
    ):

        label = (
            f"{stage}__no_recovery"
        )

        result = _r825_eval_chain(
            runner=runner,
            env=env,
            lift_policy=lift_policy,
            lift_head=lift_head,
            swing_policy=swing_policy,
            recovery_policy=None,
            stage=stage,
            recovery_switch="none",
            episodes=sweep_episodes,
            seed=seed + 17000,
        )

        baselines[label] = result

        print(
            "[R825_BASELINE] "
            f"label={label} "
            f"lift4={result['lift4_rate']:.4f} "
            f"bridge={result['bridge_rate']:.4f} "
            f"fwd6={result['forward6_rate']:.4f} "
            f"swing={result['swing_rate']:.4f} "
            f"stage_success="
            f"{result['official_stage_success_rate']:.4f}",
            flush=True,
        )

    # ========================================================
    # Broad 12-way recovery sweep.
    # ========================================================

    sweep_cases = []

    switches = (
        "immediate",
        "hold2",
        "blend4",
    )

    for actor_name, actor in (
        recovery_policies.items()
    ):

        for switch in switches:

            label = (
                f"{actor_name}__{switch}"
            )

            result = _r825_eval_chain(
                runner=runner,
                env=env,
                lift_policy=lift_policy,
                lift_head=lift_head,
                swing_policy=swing_policy,
                recovery_policy=actor,
                stage="recovery",
                recovery_switch=switch,
                episodes=sweep_episodes,
                seed=seed + 17000,
            )

            result[
                "label"
            ] = label

            result[
                "recovery_actor"
            ] = actor_name

            result[
                "recovery_switch"
            ] = switch

            sweep_cases.append(
                result
            )

            print(
                "[R825_SWEEP] "
                f"label={label} "
                f"recovery="
                f"{result['official_stage_success_rate']:.4f} "
                f"candidate="
                f"{result['candidate_rate']:.4f} "
                f"swing="
                f"{result['swing_rate']:.4f} "
                f"bridge="
                f"{result['bridge_rate']:.4f} "
                f"fwd6="
                f"{result['forward6_rate']:.4f} "
                f"done_fwd="
                f"{result['done_abs_forward_m']:.4f}",
                flush=True,
            )

    # Recovery first; if none, rank by physical retraction.
    ranked = sorted(
        sweep_cases,
        key=lambda x: (
            x[
                "official_stage_success_rate"
            ],
            x["candidate_rate"],
            x["swing_rate"],
            x["bridge_rate"],
            -x["done_abs_forward_m"],
            -x["done_abs_lift_m"],
            x["done_selected_fz_N"],
        ),
        reverse=True,
    )

    top_labels = [
        ranked[0]["label"],
        ranked[1]["label"],
    ]

    # ========================================================
    # Confirm top 2 at 512 episodes.
    # ========================================================

    confirmations = []

    by_label = {
        x["label"]: x
        for x in sweep_cases
    }

    for label in top_labels:

        source_case = by_label[
            label
        ]

        actor_name = source_case[
            "recovery_actor"
        ]

        switch = source_case[
            "recovery_switch"
        ]

        result = _r825_eval_chain(
            runner=runner,
            env=env,
            lift_policy=lift_policy,
            lift_head=lift_head,
            swing_policy=swing_policy,
            recovery_policy=recovery_policies[
                actor_name
            ],
            stage="recovery",
            recovery_switch=switch,
            episodes=confirm_episodes,
            seed=seed + 28000,
        )

        result[
            "label"
        ] = label

        result[
            "recovery_actor"
        ] = actor_name

        result[
            "recovery_switch"
        ] = switch

        confirmations.append(
            result
        )

        print(
            "[R825_CONFIRM] "
            f"label={label} "
            f"recovery="
            f"{result['official_stage_success_rate']:.4f} "
            f"candidate="
            f"{result['candidate_rate']:.4f} "
            f"swing="
            f"{result['swing_rate']:.4f} "
            f"bridge="
            f"{result['bridge_rate']:.4f} "
            f"fwd6="
            f"{result['forward6_rate']:.4f}",
            flush=True,
        )

    winner = max(
        confirmations,
        key=lambda x: (
            x[
                "official_stage_success_rate"
            ],
            x["candidate_rate"],
            x["swing_rate"],
            -x["done_abs_forward_m"],
        ),
    )

    recovery_stage_baseline = baselines[
        "recovery__no_recovery"
    ]

    rec = winner[
        "official_stage_success_rate"
    ]

    if rec >= 0.50:

        verdict = (
            "COMPOSED_RECOVERY_STRONG"
        )

    elif rec > 0.0:

        verdict = (
            "COMPOSED_RECOVERY_EMERGED"
        )

    elif (
        winner["candidate_rate"] >= 0.50
    ):

        verdict = (
            "RETRACTION_EXISTS_BUT_"
            "FULL_RECOVERY_CONTRACT_BLOCKED"
        )

    elif (
        recovery_stage_baseline[
            "swing_rate"
        ] < 0.50
    ):

        verdict = (
            "RECOVERY_STAGE_CONTEXT_"
            "BREAKS_LIFT_SWING"
        )

    elif (
        winner["swing_rate"] >= 0.80
        and winner["candidate_rate"] < 0.10
    ):

        verdict = (
            "RETRACTION_IS_NEXT_BLOCKER"
        )

    else:

        verdict = (
            "RECOVERY_TRANSITION_UNRESOLVED"
        )

    summary = {
        "version":
            "r8_25_recovery_composition_sweep",

        "training":
            False,

        "reward_change":
            False,

        "video":
            False,

        "lift_checkpoint":
            str(lift_checkpoint),

        "lift_head":
            str(lift_head_path),

        "swing_checkpoint":
            str(swing_checkpoint),

        "recovery_checkpoints": {
            k: str(v)
            for k, v in recovery_checkpoints.items()
        },

        "sweep_episodes":
            sweep_episodes,

        "confirm_episodes":
            confirm_episodes,

        "baselines":
            baselines,

        "sweep_cases":
            sweep_cases,

        "top2":
            top_labels,

        "confirmations":
            confirmations,

        "winner":
            winner["label"],

        "verdict":
            verdict,
    }

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(
        "[R825_RESULT] "
        f"winner={winner['label']} "
        f"recovery="
        f"{winner['official_stage_success_rate']:.6f} "
        f"candidate="
        f"{winner['candidate_rate']:.6f} "
        f"swing="
        f"{winner['swing_rate']:.6f} "
        f"verdict={verdict}",
        flush=True,
    )

    print(
        "R825_RECOVERY_SWEEP_OK",
        flush=True,
    )



def _r826_set_std(
    policy: Any,
    value: float,
) -> float:

    value = float(value)

    std = getattr(
        policy,
        "std",
        None,
    )

    if torch.is_tensor(std):

        with torch.no_grad():
            std.data.fill_(value)

        return float(
            std.detach()
            .float()
            .mean()
        )

    log_std = getattr(
        policy,
        "log_std",
        None,
    )

    if torch.is_tensor(log_std):

        with torch.no_grad():
            log_std.data.fill_(
                float(
                    np.log(value)
                )
            )

        return float(
            torch.exp(
                log_std.detach()
            )
            .float()
            .mean()
        )

    raise RuntimeError(
        "R826 cannot set policy std."
    )


def _r826_eval(
    *,
    runner: Any,
    env: IsolatedSupportTransferSkillEnv,
    lift_policy: Any,
    lift_head: Any,
    swing_policy: Any,
    recovery_policy: Any,
    recovery_std: float,
    episodes: int,
    seed: int,
) -> dict[str, Any]:

    actual_std = _r826_set_std(
        recovery_policy,
        recovery_std,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)

    generator = getattr(
        env,
        "generator",
        None,
    )

    if isinstance(
        generator,
        torch.Generator,
    ):
        generator.manual_seed(seed)

    env.set_curriculum_stage(
        "recovery"
    )

    env.reset()

    N = env.num_envs

    bridge_age = torch.zeros(
        N,
        dtype=torch.long,
        device=env.device,
    )

    swing_seen = torch.zeros(
        N,
        dtype=torch.bool,
        device=env.device,
    )

    bridge_seen = torch.zeros_like(
        swing_seen
    )

    x10_seen = torch.zeros_like(
        swing_seen
    )

    x8_seen = torch.zeros_like(
        swing_seen
    )

    x6_seen = torch.zeros_like(
        swing_seen
    )

    x45_seen = torch.zeros_like(
        swing_seen
    )

    touchdown_seen = torch.zeros_like(
        swing_seen
    )

    candidate_seen = torch.zeros_like(
        swing_seen
    )

    min_abs_forward = torch.full(
        (N,),
        float("inf"),
        dtype=torch.float32,
        device=env.device,
    )

    min_abs_lift = torch.full_like(
        min_abs_forward,
        float("inf"),
    )

    swing_start_forward = torch.zeros(
        N,
        dtype=torch.float32,
        device=env.device,
    )

    completed = 0
    steps = 0

    official_success = 0

    total_swing = 0
    total_bridge = 0

    total_x10 = 0
    total_x8 = 0
    total_x6 = 0
    total_x45 = 0
    total_touchdown = 0
    total_candidate = 0

    min_forward_sum = 0.0
    min_lift_sum = 0.0
    swing_start_sum = 0.0

    swing_episode_count = 0

    safety = max(
        4000,
        episodes
        * (
            env.total_episode_steps
            + 100
        ),
    )

    with torch.no_grad():

        while completed < episodes:

            obs = TensorDict(
                {
                    "policy":
                        env.low_obs_buf
                },
                batch_size=[N],
                device=env.device,
            )

            obs_runner = obs.to(
                runner.device
            )

            # ================================================
            # Deterministic Lift primitive.
            # ================================================

            lift_base = (
                lift_policy.act_inference(
                    obs_runner
                )
            )

            normalized = _r820_actor_input(
                lift_policy,
                obs_runner,
            )

            lift_action = (
                lift_base
                + lift_head(
                    normalized
                )
            )

            # ================================================
            # R8.24 winner:
            # bridge -> hold2 -> R8.15 swing.
            # ================================================

            bridge_state = (
                env.bridge_seen_r852
                .detach()
                .bool()
                .clone()
            )

            bridge_age = torch.where(
                bridge_state,
                bridge_age + 1,
                torch.zeros_like(
                    bridge_age
                ),
            )

            swing_action = (
                swing_policy.act_inference(
                    obs_runner
                )
            )

            alpha = (
                bridge_state
                & (bridge_age >= 3)
            ).float()

            chain_action = (
                (
                    1.0
                    - alpha.unsqueeze(1)
                )
                * lift_action
                + alpha.unsqueeze(1)
                * swing_action
            )

            # ================================================
            # Only AFTER a real swing event do we sample the
            # candidate recovery policy.
            # ================================================

            stochastic_recovery = (
                recovery_policy.act(
                    obs_runner
                )
            )

            action = torch.where(
                swing_seen.to(
                    runner.device
                ).unsqueeze(1),
                stochastic_recovery,
                chain_action,
            )

            (
                _,
                _,
                done,
                _,
                diag,
            ) = env.kick_step(
                action.to(
                    env.device
                )
            )

            foot_forward = (
                env.r817_foot_forward
                .detach()
            )

            foot_lift = (
                env.r817_foot_lift
                .detach()
            )

            selected_fz = (
                env.r817_selected_fz
                .detach()
            )

            support3 = (
                env.r817_support3_fz
                .detach()
            )

            bridge_now = (
                env.r817_bridge_now
                .detach()
                .bool()
            )

            swing_now = (
                env.r817_swing_now
                .detach()
                .bool()
            )

            new_swing = (
                swing_now
                & (~swing_seen)
            )

            if new_swing.any():

                swing_start_forward[
                    new_swing
                ] = torch.abs(
                    foot_forward[
                        new_swing
                    ]
                )

            bridge_seen |= bridge_now
            swing_seen |= swing_now

            # All following metrics are explicitly masked
            # by a real prior/current swing event.
            post = swing_seen

            if post.any():

                abs_f = torch.abs(
                    foot_forward
                )

                abs_z = torch.abs(
                    foot_lift
                )

                min_abs_forward = torch.where(
                    post,
                    torch.minimum(
                        min_abs_forward,
                        abs_f,
                    ),
                    min_abs_forward,
                )

                min_abs_lift = torch.where(
                    post,
                    torch.minimum(
                        min_abs_lift,
                        abs_z,
                    ),
                    min_abs_lift,
                )

                x10_seen |= (
                    post
                    & (abs_f <= 0.100)
                )

                x8_seen |= (
                    post
                    & (abs_f <= 0.080)
                )

                x6_seen |= (
                    post
                    & (abs_f <= 0.060)
                )

                x45_seen |= (
                    post
                    & (abs_f <= 0.045)
                )

                touchdown_now = (
                    post
                    & (abs_z <= 0.015)
                    & (selected_fz >= 8.0)
                )

                touchdown_seen |= (
                    touchdown_now
                )

                candidate_seen |= (
                    touchdown_now
                    & (abs_f <= 0.045)
                    & (support3 >= 100.0)
                )

            official_success += int(
                diag[
                    "stage_successful_episodes"
                ]
            )

            completed += int(
                diag[
                    "completed_episodes"
                ]
            )

            done_mask = (
                done.to(
                    env.device
                )
                .bool()
            )

            if done_mask.any():

                total_bridge += int(
                    bridge_seen[
                        done_mask
                    ].sum()
                )

                total_swing += int(
                    swing_seen[
                        done_mask
                    ].sum()
                )

                total_x10 += int(
                    x10_seen[
                        done_mask
                    ].sum()
                )

                total_x8 += int(
                    x8_seen[
                        done_mask
                    ].sum()
                )

                total_x6 += int(
                    x6_seen[
                        done_mask
                    ].sum()
                )

                total_x45 += int(
                    x45_seen[
                        done_mask
                    ].sum()
                )

                total_touchdown += int(
                    touchdown_seen[
                        done_mask
                    ].sum()
                )

                total_candidate += int(
                    candidate_seen[
                        done_mask
                    ].sum()
                )

                swung_done = (
                    done_mask
                    & swing_seen
                )

                if swung_done.any():

                    count = int(
                        swung_done.sum()
                    )

                    swing_episode_count += count

                    min_forward_sum += float(
                        min_abs_forward[
                            swung_done
                        ].sum()
                    )

                    min_lift_sum += float(
                        min_abs_lift[
                            swung_done
                        ].sum()
                    )

                    swing_start_sum += float(
                        swing_start_forward[
                            swung_done
                        ].sum()
                    )

                bridge_age[
                    done_mask
                ] = 0

                swing_seen[
                    done_mask
                ] = False

                bridge_seen[
                    done_mask
                ] = False

                x10_seen[
                    done_mask
                ] = False

                x8_seen[
                    done_mask
                ] = False

                x6_seen[
                    done_mask
                ] = False

                x45_seen[
                    done_mask
                ] = False

                touchdown_seen[
                    done_mask
                ] = False

                candidate_seen[
                    done_mask
                ] = False

                min_abs_forward[
                    done_mask
                ] = float("inf")

                min_abs_lift[
                    done_mask
                ] = float("inf")

                swing_start_forward[
                    done_mask
                ] = 0.0

            steps += 1

            if steps > safety:
                raise RuntimeError(
                    "R826 evaluation safety limit."
                )

    denom = max(
        completed,
        1,
    )

    swing_denom = max(
        swing_episode_count,
        1,
    )

    return {
        "episodes":
            completed,

        "actual_std":
            actual_std,

        "official_recovery_rate":
            official_success / denom,

        "bridge_rate":
            total_bridge / denom,

        "swing_rate":
            total_swing / denom,

        "x10_rate":
            total_x10 / denom,

        "x8_rate":
            total_x8 / denom,

        "x6_rate":
            total_x6 / denom,

        "x45_rate":
            total_x45 / denom,

        "touchdown_rate":
            total_touchdown / denom,

        "candidate_rate":
            total_candidate / denom,

        "swing_episode_count":
            swing_episode_count,

        "mean_swing_start_abs_forward_m":
            swing_start_sum / swing_denom,

        "mean_min_abs_forward_m":
            min_forward_sum / swing_denom,

        "mean_min_abs_lift_m":
            min_lift_sum / swing_denom,
    }


def run_r826_sweep(
    *,
    runner: Any,
    env: IsolatedSupportTransferSkillEnv,
    lift_checkpoint: Path,
    lift_head_path: Path,
    swing_checkpoint: Path,
    policy_checkpoints: dict[str, Path],
    summary_path: Path,
    sweep_episodes: int,
    confirm_episodes: int,
    seed: int,
) -> None:

    lift_policy = _r824_load_policy_copy(
        runner,
        lift_checkpoint,
    )

    swing_policy = _r824_load_policy_copy(
        runner,
        swing_checkpoint,
    )

    head_data = torch.load(
        lift_head_path,
        map_location=runner.device,
        weights_only=False,
    )

    lift_head = R820LiftResidualHead(
        obs_dim=70,
        hidden=64,
        action_dim=12,
        scale=float(
            head_data[
                "residual_scale"
            ]
        ),
    ).to(
        runner.device
    )

    lift_head.load_state_dict(
        head_data[
            "state_dict"
        ]
    )

    lift_head.eval()

    for p in lift_head.parameters():
        p.requires_grad_(False)

    policies = {}

    for name, checkpoint in (
        policy_checkpoints.items()
    ):

        policies[name] = (
            _r824_load_policy_copy(
                runner,
                checkpoint,
            )
        )

    std_values = (
        0.20,
        0.35,
        0.50,
        0.70,
    )

    cases = []

    for policy_name, policy in (
        policies.items()
    ):

        for std in std_values:

            label = (
                f"{policy_name}"
                f"__std{std:.2f}"
            )

            result = _r826_eval(
                runner=runner,
                env=env,
                lift_policy=lift_policy,
                lift_head=lift_head,
                swing_policy=swing_policy,
                recovery_policy=policy,
                recovery_std=std,
                episodes=sweep_episodes,
                seed=seed + 31000,
            )

            result["label"] = label
            result[
                "policy"
            ] = policy_name

            result["requested_std"] = std

            cases.append(result)

            print(
                "[R826_SWEEP] "
                f"label={label} "
                f"rec={result['official_recovery_rate']:.4f} "
                f"cand={result['candidate_rate']:.4f} "
                f"x45={result['x45_rate']:.4f} "
                f"x6={result['x6_rate']:.4f} "
                f"x8={result['x8_rate']:.4f} "
                f"touch={result['touchdown_rate']:.4f} "
                f"minF={result['mean_min_abs_forward_m']:.4f}",
                flush=True,
            )

    ranked = sorted(
        cases,
        key=lambda x: (
            x[
                "official_recovery_rate"
            ],
            x["candidate_rate"],
            x["x45_rate"],
            x["x6_rate"],
            x["x8_rate"],
            x["touchdown_rate"],
            -x[
                "mean_min_abs_forward_m"
            ],
        ),
        reverse=True,
    )

    top3 = [
        row["label"]
        for row in ranked[:3]
    ]

    by_label = {
        row["label"]: row
        for row in cases
    }

    confirmations = []

    for label in top3:

        source = by_label[
            label
        ]

        result = _r826_eval(
            runner=runner,
            env=env,
            lift_policy=lift_policy,
            lift_head=lift_head,
            swing_policy=swing_policy,
            recovery_policy=policies[
                source["policy"]
            ],
            recovery_std=float(
                source[
                    "requested_std"
                ]
            ),
            episodes=confirm_episodes,
            seed=seed + 47000,
        )

        result["label"] = label
        result[
            "policy"
        ] = source["policy"]

        result[
            "requested_std"
        ] = source[
            "requested_std"
        ]

        confirmations.append(
            result
        )

        print(
            "[R826_CONFIRM] "
            f"label={label} "
            f"rec={result['official_recovery_rate']:.4f} "
            f"cand={result['candidate_rate']:.4f} "
            f"x45={result['x45_rate']:.4f} "
            f"x6={result['x6_rate']:.4f} "
            f"x8={result['x8_rate']:.4f}",
            flush=True,
        )

    winner = max(
        confirmations,
        key=lambda x: (
            x[
                "official_recovery_rate"
            ],
            x["candidate_rate"],
            x["x45_rate"],
            x["x6_rate"],
            x["x8_rate"],
            x["touchdown_rate"],
            -x[
                "mean_min_abs_forward_m"
            ],
        ),
    )

    if winner[
        "official_recovery_rate"
    ] > 0:

        verdict = (
            "STOCHASTIC_RECOVERY_EMERGED"
        )

    elif winner[
        "candidate_rate"
    ] > 0:

        verdict = (
            "STOCHASTIC_RETRACTION_CANDIDATE_FOUND"
        )

    elif winner["x45_rate"] > 0:

        verdict = (
            "FULL_RETRACTION_REACHABLE_"
            "TOUCHDOWN_BLOCKER"
        )

    elif winner["x6_rate"] >= 0.10:

        verdict = (
            "RETRACTION_REACHABLE"
        )

    elif winner["x8_rate"] > 0:

        verdict = (
            "PARTIAL_RETRACTION_REACHABLE"
        )

    else:

        verdict = (
            "NO_RETRACTION_IN_EXISTING_POLICY_TAILS"
        )

    summary = {
        "version":
            "r8_26_retraction_reachability_sweep",

        "training":
            False,

        "reward_change":
            False,

        "cases":
            cases,

        "ranking":
            [
                row["label"]
                for row in ranked
            ],

        "top3":
            top3,

        "confirmations":
            confirmations,

        "winner":
            winner["label"],

        "verdict":
            verdict,
    }

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(
        "[R826_RESULT] "
        f"winner={winner['label']} "
        f"rec={winner['official_recovery_rate']:.6f} "
        f"cand={winner['candidate_rate']:.6f} "
        f"x45={winner['x45_rate']:.6f} "
        f"x6={winner['x6_rate']:.6f} "
        f"x8={winner['x8_rate']:.6f} "
        f"verdict={verdict}",
        flush=True,
    )

    print(
        "R826_RETRACTION_SWEEP_OK",
        flush=True,
    )



def _r827_k2(
    trajectories: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:

    x = trajectories.reshape(
        trajectories.shape[0],
        -1,
    ).astype(np.float64)

    norms = np.linalg.norm(
        x,
        axis=1,
    )

    unit = x / np.maximum(
        norms[:, None],
        1.0e-12,
    )

    rng = np.random.default_rng(seed)

    first = int(
        rng.integers(
            0,
            len(unit),
        )
    )

    c0 = unit[first].copy()

    cosine0 = unit @ c0

    second = int(
        np.argmin(cosine0)
    )

    centroids = np.stack(
        [
            c0,
            unit[second],
        ],
        axis=0,
    )

    for _ in range(100):

        sim = unit @ centroids.T

        labels = np.argmax(
            sim,
            axis=1,
        )

        new = np.zeros_like(
            centroids
        )

        for k in (0, 1):

            ids = np.flatnonzero(
                labels == k
            )

            if len(ids) == 0:

                new[k] = unit[
                    int(
                        rng.integers(
                            0,
                            len(unit),
                        )
                    )
                ]

                continue

            c = unit[
                ids
            ].mean(
                axis=0
            )

            n = np.linalg.norm(c)

            new[k] = (
                c
                / max(
                    n,
                    1.0e-12,
                )
            )

        if np.max(
            np.abs(
                new - centroids
            )
        ) < 1.0e-7:

            centroids = new
            break

        centroids = new

    labels = np.argmax(
        unit @ centroids.T,
        axis=1,
    )

    sizes = [
        int(
            np.sum(
                labels == k
            )
        )
        for k in (0, 1)
    ]

    centroid_cos = float(
        centroids[0]
        @ centroids[1]
    )

    cluster_mean_norm = []

    coherence = []

    for k in (0, 1):

        ids = labels == k

        raw_mean = x[
            ids
        ].mean(
            axis=0
        )

        cluster_mean_norm.append(
            float(
                np.linalg.norm(
                    raw_mean
                )
            )
        )

        coherence.append(
            float(
                (
                    unit[ids]
                    @ centroids[k]
                ).mean()
            )
        )

    info = {
        "sizes":
            sizes,

        "fractions":
            [
                s / len(labels)
                for s in sizes
            ],

        "centroid_cosine":
            centroid_cos,

        "cluster_mean_norm":
            cluster_mean_norm,

        "coherence":
            coherence,
    }

    return (
        labels,
        centroids,
        info,
    )


def _r827_collect(
    *,
    runner: Any,
    env: IsolatedSupportTransferSkillEnv,
    lift_policy: Any,
    lift_head: Any,
    swing_policy: Any,
    recovery_policy: Any,
    episodes: int,
    history_steps: int,
    seed: int,
) -> dict[str, Any]:

    _r826_set_std(
        recovery_policy,
        0.70,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)

    generator = getattr(
        env,
        "generator",
        None,
    )

    if isinstance(
        generator,
        torch.Generator,
    ):
        generator.manual_seed(seed)

    env.set_curriculum_stage(
        "recovery"
    )

    env.reset()

    K = int(history_steps)
    N = env.num_envs

    ring_obs = torch.zeros(
        K,
        N,
        70,
        dtype=torch.float32,
        device=env.device,
    )

    ring_residual = torch.zeros(
        K,
        N,
        12,
        dtype=torch.float32,
        device=env.device,
    )

    valid = torch.zeros(
        N,
        dtype=torch.long,
        device=env.device,
    )

    bridge_age = torch.zeros(
        N,
        dtype=torch.long,
        device=env.device,
    )

    swing_seen = torch.zeros(
        N,
        dtype=torch.bool,
        device=env.device,
    )

    x45_captured = torch.zeros_like(
        swing_seen
    )

    candidate_captured = (
        torch.zeros_like(
            swing_seen
        )
    )

    x45_obs = []
    x45_residual = []

    candidate_obs = []
    candidate_residual = []

    completed = 0
    steps = 0

    physical_x45 = 0
    physical_candidate = 0

    max_x45 = int(
        os.environ.get(
            "R827_MAX_X45",
            "800",
        )
    )

    max_candidate = int(
        os.environ.get(
            "R827_MAX_CANDIDATE",
            "500",
        )
    )

    safety = max(
        7000,
        episodes
        * (
            env.total_episode_steps
            + 120
        ),
    )

    def capture(
        ids: torch.Tensor,
        obs_store: list,
        residual_store: list,
        slot: int,
    ) -> None:

        for idx in ids.tolist():

            chronological = [
                (
                    slot
                    - K
                    + 1
                    + j
                ) % K

                for j in range(K)
            ]

            obs_store.append(
                torch.stack(
                    [
                        ring_obs[
                            s,
                            idx,
                        ]
                        for s in chronological
                    ],
                    dim=0,
                )
                .detach()
                .cpu()
            )

            residual_store.append(
                torch.stack(
                    [
                        ring_residual[
                            s,
                            idx,
                        ]
                        for s in chronological
                    ],
                    dim=0,
                )
                .detach()
                .cpu()
            )

    with torch.no_grad():

        while completed < episodes:

            obs = TensorDict(
                {
                    "policy":
                        env.low_obs_buf
                },
                batch_size=[N],
                device=env.device,
            )

            obs_runner = obs.to(
                runner.device
            )

            # ================================================
            # Fixed deterministic Lift -> Swing.
            # ================================================

            lift_base = (
                lift_policy.act_inference(
                    obs_runner
                )
            )

            lift_norm = (
                _r820_actor_input(
                    lift_policy,
                    obs_runner,
                )
            )

            lift_action = (
                lift_base
                + lift_head(
                    lift_norm
                )
            )

            bridge_state = (
                env.bridge_seen_r852
                .detach()
                .bool()
                .clone()
            )

            bridge_age = torch.where(
                bridge_state,
                bridge_age + 1,
                torch.zeros_like(
                    bridge_age
                ),
            )

            swing_action = (
                swing_policy.act_inference(
                    obs_runner
                )
            )

            alpha = (
                bridge_state
                & (bridge_age >= 3)
            ).float()

            chain_action = (
                (
                    1.0
                    - alpha.unsqueeze(1)
                )
                * lift_action

                + alpha.unsqueeze(1)
                * swing_action
            )

            # ================================================
            # Recovery stochastic exploration.
            # ================================================

            recovery_mean = (
                recovery_policy.act_inference(
                    obs_runner
                )
            )

            recovery_sample = (
                recovery_policy.act(
                    obs_runner
                )
            )

            recovery_norm = (
                _r820_actor_input(
                    recovery_policy,
                    obs_runner,
                )
            )

            recovery_residual = (
                recovery_sample
                - recovery_mean
            )

            slot = steps % K

            post_before = (
                swing_seen
            )

            if post_before.any():

                ring_obs[
                    slot,
                    post_before,
                ] = recovery_norm[
                    post_before.to(
                        runner.device
                    )
                ].to(
                    env.device
                )

                ring_residual[
                    slot,
                    post_before,
                ] = recovery_residual[
                    post_before.to(
                        runner.device
                    )
                ].to(
                    env.device
                )

                valid = torch.where(
                    post_before,
                    torch.clamp(
                        valid + 1,
                        max=K,
                    ),
                    valid,
                )

            action = torch.where(
                post_before.to(
                    runner.device
                ).unsqueeze(1),
                recovery_sample,
                chain_action,
            )

            (
                _,
                _,
                done,
                _,
                diag,
            ) = env.kick_step(
                action.to(
                    env.device
                )
            )

            foot_forward = (
                env.r817_foot_forward
                .detach()
            )

            foot_lift = (
                env.r817_foot_lift
                .detach()
            )

            selected_fz = (
                env.r817_selected_fz
                .detach()
            )

            support3 = (
                env.r817_support3_fz
                .detach()
            )

            swing_now = (
                env.r817_swing_now
                .detach()
                .bool()
            )

            new_swing = (
                swing_now
                & (~swing_seen)
            )

            swing_seen |= swing_now

            # First post-swing step becomes valid on NEXT
            # control step. This avoids mixing Swing action
            # into Recovery-history targets.
            valid[
                new_swing
            ] = 0

            abs_f = torch.abs(
                foot_forward
            )

            abs_z = torch.abs(
                foot_lift
            )

            x45_now = (
                swing_seen
                & (abs_f <= 0.045)
            )

            candidate_now = (
                x45_now
                & (abs_z <= 0.015)
                & (selected_fz >= 8.0)
                & (support3 >= 100.0)
            )

            physical_x45 += int(
                x45_now.sum()
            )

            physical_candidate += int(
                candidate_now.sum()
            )

            enough = (
                valid >= K
            )

            x45_capture = (
                x45_now
                & enough
                & (~x45_captured)
            )

            candidate_capture = (
                candidate_now
                & enough
                & (~candidate_captured)
            )

            if (
                len(x45_obs)
                < max_x45
                and x45_capture.any()
            ):

                ids = torch.nonzero(
                    x45_capture,
                    as_tuple=False,
                ).squeeze(-1)

                remaining = (
                    max_x45
                    - len(x45_obs)
                )

                ids = ids[
                    :remaining
                ]

                capture(
                    ids,
                    x45_obs,
                    x45_residual,
                    slot,
                )

                x45_captured[
                    ids
                ] = True

            if (
                len(candidate_obs)
                < max_candidate
                and candidate_capture.any()
            ):

                ids = torch.nonzero(
                    candidate_capture,
                    as_tuple=False,
                ).squeeze(-1)

                remaining = (
                    max_candidate
                    - len(
                        candidate_obs
                    )
                )

                ids = ids[
                    :remaining
                ]

                capture(
                    ids,
                    candidate_obs,
                    candidate_residual,
                    slot,
                )

                candidate_captured[
                    ids
                ] = True

            completed += int(
                diag[
                    "completed_episodes"
                ]
            )

            done_mask = (
                done.to(
                    env.device
                )
                .bool()
            )

            if done_mask.any():

                bridge_age[
                    done_mask
                ] = 0

                swing_seen[
                    done_mask
                ] = False

                x45_captured[
                    done_mask
                ] = False

                candidate_captured[
                    done_mask
                ] = False

                valid[
                    done_mask
                ] = 0

            steps += 1

            if steps > safety:
                raise RuntimeError(
                    "R827 collection safety limit."
                )

    result = {
        "episodes":
            completed,

        "steps":
            steps,

        "x45_trajectories":
            len(x45_obs),

        "candidate_trajectories":
            len(candidate_obs),

        "physical_x45_step_events":
            physical_x45,

        "physical_candidate_step_events":
            physical_candidate,
    }

    print(
        "[R827_COLLECTION] "
        + json.dumps(
            result
        ),
        flush=True,
    )

    return {
        "summary":
            result,

        "x45_obs":
            (
                torch.stack(
                    x45_obs
                ).numpy()
                if x45_obs
                else np.zeros(
                    (0, K, 70),
                    dtype=np.float32,
                )
            ),

        "x45_residual":
            (
                torch.stack(
                    x45_residual
                ).numpy()
                if x45_residual
                else np.zeros(
                    (0, K, 12),
                    dtype=np.float32,
                )
            ),

        "candidate_obs":
            (
                torch.stack(
                    candidate_obs
                ).numpy()
                if candidate_obs
                else np.zeros(
                    (0, K, 70),
                    dtype=np.float32,
                )
            ),

        "candidate_residual":
            (
                torch.stack(
                    candidate_residual
                ).numpy()
                if candidate_residual
                else np.zeros(
                    (0, K, 12),
                    dtype=np.float32,
                )
            ),
    }


def _r827_train_head(
    *,
    obs_history: np.ndarray,
    residual_history: np.ndarray,
    labels: np.ndarray,
    mode: int,
    runner: Any,
    tag: str,
) -> tuple[Any, dict[str, Any]]:

    ids = np.flatnonzero(
        labels == mode
    )

    if len(ids) < 20:
        raise RuntimeError(
            f"R827 {tag} mode {mode} "
            f"only has {len(ids)} trajectories."
        )

    flat = residual_history[
        ids
    ].reshape(
        len(ids),
        -1,
    ).astype(
        np.float64
    )

    norms = np.linalg.norm(
        flat,
        axis=1,
    )

    unit = flat / np.maximum(
        norms[:, None],
        1.0e-12,
    )

    centroid = unit.mean(
        axis=0
    )

    centroid /= max(
        np.linalg.norm(
            centroid
        ),
        1.0e-12,
    )

    cosine = unit @ centroid

    core_fraction = float(
        os.environ.get(
            "R827_CORE_FRACTION",
            "0.60",
        )
    )

    keep = max(
        16,
        int(
            round(
                len(ids)
                * core_fraction
            )
        ),
    )

    local_order = np.argsort(
        -cosine
    )

    core_ids = ids[
        local_order[:keep]
    ]

    core_cos = cosine[
        local_order[:keep]
    ]

    obs = obs_history[
        core_ids
    ].reshape(
        -1,
        70,
    ).astype(
        np.float32
    )

    target = residual_history[
        core_ids
    ].reshape(
        -1,
        12,
    ).astype(
        np.float32
    )

    K = residual_history.shape[1]

    time_weights = np.linspace(
        0.20,
        1.60,
        K,
        dtype=np.float32,
    )

    weights = np.tile(
        time_weights,
        len(core_ids),
    )

    traj_weight = (
        0.5
        + 0.5
        * (
            (
                core_cos
                - core_cos.min()
            )
            / max(
                float(
                    core_cos.max()
                    - core_cos.min()
                ),
                1.0e-6,
            )
        )
    ).astype(
        np.float32
    )

    weights *= np.repeat(
        traj_weight,
        K,
    )

    obs_t = torch.from_numpy(
        obs
    ).to(
        runner.device
    )

    target_t = torch.from_numpy(
        target
    ).to(
        runner.device
    )

    weight_t = torch.from_numpy(
        weights
    ).to(
        runner.device
    )

    head = R820LiftResidualHead(
        obs_dim=70,
        hidden=64,
        action_dim=12,
        scale=float(
            os.environ.get(
                "R827_HEAD_SCALE",
                "0.90",
            )
        ),
    ).to(
        runner.device
    )

    optimizer = torch.optim.Adam(
        head.parameters(),
        lr=float(
            os.environ.get(
                "R827_LR",
                "0.0005",
            )
        ),
        weight_decay=1.0e-5,
    )

    steps = int(
        os.environ.get(
            "R827_OPT_STEPS",
            "500",
        )
    )

    batch = int(
        os.environ.get(
            "R827_BATCH",
            "512",
        )
    )

    initial = None
    final = None

    for update in range(steps):

        idx = torch.randint(
            0,
            len(obs_t),
            (
                min(
                    batch,
                    len(obs_t),
                ),
            ),
            device=runner.device,
        )

        pred = head(
            obs_t[idx]
        )

        per_sample = (
            (
                pred
                - target_t[idx]
            )
            .square()
            .mean(dim=1)
        )

        loss = (
            per_sample
            * weight_t[idx]
        ).sum() / (
            weight_t[idx].sum()
            + 1.0e-8
        )

        loss = (
            loss
            + 0.005
            * pred.square().mean()
        )

        if initial is None:
            initial = float(
                loss.detach()
            )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            head.parameters(),
            1.0,
        )

        optimizer.step()

        final = float(
            loss.detach()
        )

    head.eval()

    with torch.no_grad():

        predicted = head(
            obs_t
        )

    info = {
        "tag":
            tag,

        "mode":
            int(mode),

        "mode_trajectories":
            int(len(ids)),

        "core_trajectories":
            int(len(core_ids)),

        "core_cosine_mean":
            float(
                core_cos.mean()
            ),

        "target_abs_mean":
            float(
                target_t.abs().mean()
            ),

        "pred_abs_mean":
            float(
                predicted.abs().mean()
            ),

        "initial_loss":
            initial,

        "final_loss":
            final,
    }

    return (
        head,
        info,
    )


def _r827_eval(
    *,
    runner: Any,
    env: IsolatedSupportTransferSkillEnv,
    lift_policy: Any,
    lift_head: Any,
    swing_policy: Any,
    recovery_policy: Any,
    recovery_head: Any,
    gain: float,
    release_mode: str,
    episodes: int,
    seed: int,
) -> dict[str, Any]:

    torch.manual_seed(seed)
    np.random.seed(seed)

    generator = getattr(
        env,
        "generator",
        None,
    )

    if isinstance(
        generator,
        torch.Generator,
    ):
        generator.manual_seed(seed)

    env.set_curriculum_stage(
        "recovery"
    )

    env.reset()

    N = env.num_envs

    bridge_age = torch.zeros(
        N,
        dtype=torch.long,
        device=env.device,
    )

    swing_seen = torch.zeros(
        N,
        dtype=torch.bool,
        device=env.device,
    )

    candidate_seen = torch.zeros_like(
        swing_seen
    )

    x45_seen = torch.zeros_like(
        swing_seen
    )

    touchdown_seen = torch.zeros_like(
        swing_seen
    )

    min_forward = torch.full(
        (N,),
        float("inf"),
        dtype=torch.float32,
        device=env.device,
    )

    completed = 0
    steps = 0

    official = 0
    swing_total = 0
    candidate_total = 0
    x45_total = 0
    touchdown_total = 0

    min_forward_sum = 0.0
    swung_done_total = 0

    safety = max(
        4000,
        episodes
        * (
            env.total_episode_steps
            + 100
        ),
    )

    with torch.no_grad():

        while completed < episodes:

            obs = TensorDict(
                {
                    "policy":
                        env.low_obs_buf
                },
                batch_size=[N],
                device=env.device,
            )

            obs_runner = obs.to(
                runner.device
            )

            # ================================================
            # Fixed Lift -> Swing.
            # ================================================

            lift_base = (
                lift_policy.act_inference(
                    obs_runner
                )
            )

            lift_norm = (
                _r820_actor_input(
                    lift_policy,
                    obs_runner,
                )
            )

            lift_action = (
                lift_base
                + lift_head(
                    lift_norm
                )
            )

            bridge_state = (
                env.bridge_seen_r852
                .detach()
                .bool()
                .clone()
            )

            bridge_age = torch.where(
                bridge_state,
                bridge_age + 1,
                torch.zeros_like(
                    bridge_age
                ),
            )

            swing_action = (
                swing_policy.act_inference(
                    obs_runner
                )
            )

            alpha = (
                bridge_state
                & (bridge_age >= 3)
            ).float()

            chain_action = (
                (
                    1.0
                    - alpha.unsqueeze(1)
                )
                * lift_action

                + alpha.unsqueeze(1)
                * swing_action
            )

            # ================================================
            # Deterministic recovery base + learned mode head.
            # ================================================

            recovery_mean = (
                recovery_policy.act_inference(
                    obs_runner
                )
            )

            recovery_norm = (
                _r820_actor_input(
                    recovery_policy,
                    obs_runner,
                )
            )

            correction = (
                float(gain)
                * recovery_head(
                    recovery_norm
                )
            )

            if release_mode == "continuous":

                head_active = (
                    swing_seen
                )

            elif release_mode == "release":

                head_active = (
                    swing_seen
                    & (~candidate_seen)
                )

            else:
                raise RuntimeError(
                    f"bad release mode "
                    f"{release_mode}"
                )

            recovery_action = (
                recovery_mean
                + torch.where(
                    head_active.to(
                        runner.device
                    ).unsqueeze(1),
                    correction,
                    torch.zeros_like(
                        correction
                    ),
                )
            )

            action = torch.where(
                swing_seen.to(
                    runner.device
                ).unsqueeze(1),
                recovery_action,
                chain_action,
            )

            (
                _,
                _,
                done,
                _,
                diag,
            ) = env.kick_step(
                action.to(
                    env.device
                )
            )

            foot_forward = (
                env.r817_foot_forward
                .detach()
            )

            foot_lift = (
                env.r817_foot_lift
                .detach()
            )

            selected_fz = (
                env.r817_selected_fz
                .detach()
            )

            support3 = (
                env.r817_support3_fz
                .detach()
            )

            swing_now = (
                env.r817_swing_now
                .detach()
                .bool()
            )

            swing_seen |= swing_now

            abs_f = torch.abs(
                foot_forward
            )

            abs_z = torch.abs(
                foot_lift
            )

            post = swing_seen

            min_forward = torch.where(
                post,
                torch.minimum(
                    min_forward,
                    abs_f,
                ),
                min_forward,
            )

            x45_now = (
                post
                & (abs_f <= 0.045)
            )

            touchdown_now = (
                post
                & (abs_z <= 0.015)
                & (selected_fz >= 8.0)
            )

            candidate_now = (
                x45_now
                & touchdown_now
                & (support3 >= 100.0)
            )

            x45_seen |= x45_now
            touchdown_seen |= touchdown_now
            candidate_seen |= candidate_now

            official += int(
                diag[
                    "stage_successful_episodes"
                ]
            )

            completed += int(
                diag[
                    "completed_episodes"
                ]
            )

            done_mask = (
                done.to(
                    env.device
                )
                .bool()
            )

            if done_mask.any():

                swing_total += int(
                    swing_seen[
                        done_mask
                    ].sum()
                )

                x45_total += int(
                    x45_seen[
                        done_mask
                    ].sum()
                )

                touchdown_total += int(
                    touchdown_seen[
                        done_mask
                    ].sum()
                )

                candidate_total += int(
                    candidate_seen[
                        done_mask
                    ].sum()
                )

                swung_done = (
                    done_mask
                    & swing_seen
                )

                if swung_done.any():

                    swung_done_total += int(
                        swung_done.sum()
                    )

                    min_forward_sum += float(
                        min_forward[
                            swung_done
                        ].sum()
                    )

                bridge_age[
                    done_mask
                ] = 0

                swing_seen[
                    done_mask
                ] = False

                candidate_seen[
                    done_mask
                ] = False

                x45_seen[
                    done_mask
                ] = False

                touchdown_seen[
                    done_mask
                ] = False

                min_forward[
                    done_mask
                ] = float("inf")

            steps += 1

            if steps > safety:
                raise RuntimeError(
                    "R827 eval safety limit."
                )

    denom = max(
        completed,
        1,
    )

    swing_denom = max(
        swung_done_total,
        1,
    )

    return {
        "episodes":
            completed,

        "official_recovery_rate":
            official / denom,

        "swing_rate":
            swing_total / denom,

        "x45_rate":
            x45_total / denom,

        "touchdown_rate":
            touchdown_total / denom,

        "candidate_rate":
            candidate_total / denom,

        "mean_min_abs_forward_m":
            min_forward_sum
            / swing_denom,

        "gain":
            float(gain),

        "release_mode":
            release_mode,
    }


def run_r827(
    *,
    runner: Any,
    env: IsolatedSupportTransferSkillEnv,
    lift_checkpoint: Path,
    lift_head_path: Path,
    swing_checkpoint: Path,
    recovery_checkpoint: Path,
    output_dir: Path,
    collection_episodes: int,
    broad_episodes: int,
    confirm_episodes: int,
    seed: int,
) -> None:

    lift_policy = (
        _r824_load_policy_copy(
            runner,
            lift_checkpoint,
        )
    )

    swing_policy = (
        _r824_load_policy_copy(
            runner,
            swing_checkpoint,
        )
    )

    recovery_policy = (
        _r824_load_policy_copy(
            runner,
            recovery_checkpoint,
        )
    )

    lift_bundle = torch.load(
        lift_head_path,
        map_location=runner.device,
        weights_only=False,
    )

    lift_head = R820LiftResidualHead(
        obs_dim=70,
        hidden=64,
        action_dim=12,
        scale=float(
            lift_bundle[
                "residual_scale"
            ]
        ),
    ).to(
        runner.device
    )

    lift_head.load_state_dict(
        lift_bundle[
            "state_dict"
        ]
    )

    lift_head.eval()

    for parameter in (
        lift_head.parameters()
    ):
        parameter.requires_grad_(False)

    # ========================================================
    # COLLECTION
    # ========================================================

    collection = _r827_collect(
        runner=runner,
        env=env,
        lift_policy=lift_policy,
        lift_head=lift_head,
        swing_policy=swing_policy,
        recovery_policy=recovery_policy,
        episodes=collection_episodes,
        history_steps=8,
        seed=seed + 10000,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_dir
        / "recovery_trajectories.npz",

        x45_obs=collection[
            "x45_obs"
        ],

        x45_residual=collection[
            "x45_residual"
        ],

        candidate_obs=collection[
            "candidate_obs"
        ],

        candidate_residual=collection[
            "candidate_residual"
        ],
    )

    datasets = {}

    if (
        len(
            collection[
                "x45_residual"
            ]
        ) >= 50
    ):

        labels, centers, info = (
            _r827_k2(
                collection[
                    "x45_residual"
                ],
                seed + 20000,
            )
        )

        datasets[
            "x45"
        ] = {
            "obs":
                collection[
                    "x45_obs"
                ],

            "residual":
                collection[
                    "x45_residual"
                ],

            "labels":
                labels,

            "cluster_info":
                info,
        }

    if (
        len(
            collection[
                "candidate_residual"
            ]
        ) >= 40
    ):

        labels, centers, info = (
            _r827_k2(
                collection[
                    "candidate_residual"
                ],
                seed + 30000,
            )
        )

        datasets[
            "candidate"
        ] = {
            "obs":
                collection[
                    "candidate_obs"
                ],

            "residual":
                collection[
                    "candidate_residual"
                ],

            "labels":
                labels,

            "cluster_info":
                info,
        }

    if not datasets:
        raise RuntimeError(
            "R827 no usable recovery dataset."
        )

    # ========================================================
    # Train mode-specific heads.
    # ========================================================

    heads = {}
    training_info = {}

    for tag, data in datasets.items():

        for mode in (0, 1):

            key = (
                f"{tag}_mode{mode}"
            )

            head, info = (
                _r827_train_head(
                    obs_history=data["obs"],
                    residual_history=data[
                        "residual"
                    ],
                    labels=data["labels"],
                    mode=mode,
                    runner=runner,
                    tag=tag,
                )
            )

            heads[key] = head
            training_info[key] = info

            torch.save(
                {
                    "version":
                        "r8_27_recovery_mode_core",

                    "tag":
                        tag,

                    "mode":
                        mode,

                    "scale":
                        float(
                            os.environ.get(
                                "R827_HEAD_SCALE",
                                "0.90",
                            )
                        ),

                    "state_dict":
                        head.state_dict(),
                },
                output_dir
                / f"{key}_head.pt",
            )

    # ========================================================
    # 24-way broad deterministic sweep.
    # ========================================================

    gains = (
        0.8,
        1.0,
        1.2,
    )

    releases = (
        "continuous",
        "release",
    )

    broad = []

    for head_name, head in (
        heads.items()
    ):

        for gain in gains:

            for release in releases:

                label = (
                    f"{head_name}"
                    f"__g{gain:.1f}"
                    f"__{release}"
                )

                result = _r827_eval(
                    runner=runner,
                    env=env,
                    lift_policy=lift_policy,
                    lift_head=lift_head,
                    swing_policy=swing_policy,
                    recovery_policy=recovery_policy,
                    recovery_head=head,
                    gain=gain,
                    release_mode=release,
                    episodes=broad_episodes,
                    seed=seed + 40000,
                )

                result[
                    "label"
                ] = label

                result[
                    "head"
                ] = head_name

                broad.append(
                    result
                )

                print(
                    "[R827_BROAD] "
                    f"label={label} "
                    f"rec="
                    f"{result['official_recovery_rate']:.4f} "
                    f"cand="
                    f"{result['candidate_rate']:.4f} "
                    f"x45="
                    f"{result['x45_rate']:.4f} "
                    f"touch="
                    f"{result['touchdown_rate']:.4f} "
                    f"minF="
                    f"{result['mean_min_abs_forward_m']:.4f}",
                    flush=True,
                )

    ranked = sorted(
        broad,
        key=lambda x: (
            x[
                "official_recovery_rate"
            ],
            x["candidate_rate"],
            x["x45_rate"],
            x["touchdown_rate"],
            -x[
                "mean_min_abs_forward_m"
            ],
        ),
        reverse=True,
    )

    top4 = [
        x["label"]
        for x in ranked[:4]
    ]

    by_label = {
        x["label"]: x
        for x in broad
    }

    confirmations = []

    for label in top4:

        source = by_label[
            label
        ]

        head = heads[
            source["head"]
        ]

        result = _r827_eval(
            runner=runner,
            env=env,
            lift_policy=lift_policy,
            lift_head=lift_head,
            swing_policy=swing_policy,
            recovery_policy=recovery_policy,
            recovery_head=head,
            gain=source["gain"],
            release_mode=source[
                "release_mode"
            ],
            episodes=confirm_episodes,
            seed=seed + 50000,
        )

        result[
            "label"
        ] = label

        result[
            "head"
        ] = source[
            "head"
        ]

        confirmations.append(
            result
        )

        print(
            "[R827_CONFIRM] "
            f"label={label} "
            f"rec="
            f"{result['official_recovery_rate']:.4f} "
            f"cand="
            f"{result['candidate_rate']:.4f} "
            f"x45="
            f"{result['x45_rate']:.4f} "
            f"touch="
            f"{result['touchdown_rate']:.4f} "
            f"minF="
            f"{result['mean_min_abs_forward_m']:.4f}",
            flush=True,
        )

    winner = max(
        confirmations,
        key=lambda x: (
            x[
                "official_recovery_rate"
            ],
            x["candidate_rate"],
            x["x45_rate"],
            x["touchdown_rate"],
            -x[
                "mean_min_abs_forward_m"
            ],
        ),
    )

    if winner[
        "official_recovery_rate"
    ] >= 0.50:

        verdict = (
            "DET_RECOVERY_STRONG"
        )

    elif winner[
        "official_recovery_rate"
    ] > 0:

        verdict = (
            "DET_RECOVERY_EMERGED"
        )

    elif winner[
        "candidate_rate"
    ] >= 0.50:

        verdict = (
            "DET_RETRACTION_STRONG_"
            "FULL_CONTRACT_BLOCKER"
        )

    elif winner[
        "candidate_rate"
    ] > 0:

        verdict = (
            "DET_RETRACTION_CANDIDATE_EMERGED"
        )

    elif winner[
        "x45_rate"
    ] > 0:

        verdict = (
            "DET_X45_RETRACTION_EMERGED"
        )

    else:

        verdict = (
            "MODE_CORE_RECOVERY_NOT_SUFFICIENT"
        )

    summary = {
        "version":
            "r8_27_recovery_mode_core",

        "training":
            "offline_mode_specific_self_generated",

        "ppo_update":
            False,

        "reward_change":
            False,

        "collection":
            collection[
                "summary"
            ],

        "datasets": {
            tag: {
                "count":
                    int(
                        len(
                            data[
                                "residual"
                            ]
                        )
                    ),

                "cluster_info":
                    data[
                        "cluster_info"
                    ],
            }

            for tag, data
            in datasets.items()
        },

        "training_info":
            training_info,

        "broad_cases":
            broad,

        "top4":
            top4,

        "confirmations":
            confirmations,

        "winner":
            winner[
                "label"
            ],

        "verdict":
            verdict,
    }

    (
        output_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(
        "[R827_RESULT] "
        f"winner={winner['label']} "
        f"rec="
        f"{winner['official_recovery_rate']:.6f} "
        f"cand="
        f"{winner['candidate_rate']:.6f} "
        f"x45="
        f"{winner['x45_rate']:.6f} "
        f"minF="
        f"{winner['mean_min_abs_forward_m']:.6f} "
        f"verdict={verdict}",
        flush=True,
    )

    print(
        "R827_RECOVERY_MODE_CORE_OK",
        flush=True,
    )



def _r828_eval(
    *,
    runner: Any,
    env: IsolatedSupportTransferSkillEnv,
    lift_policy: Any,
    lift_head: Any,
    swing_policy: Any,
    recovery_policy: Any,
    recovery_head: Any,
    settle_policy: Any,
    transition: str,
    episodes: int,
    seed: int,
) -> dict[str, Any]:

    torch.manual_seed(seed)
    np.random.seed(seed)

    generator = getattr(
        env,
        "generator",
        None,
    )

    if isinstance(
        generator,
        torch.Generator,
    ):
        generator.manual_seed(seed)

    env.set_curriculum_stage(
        "recovery"
    )

    env.reset()

    N = env.num_envs

    bridge_age = torch.zeros(
        N,
        dtype=torch.long,
        device=env.device,
    )

    swing_seen = torch.zeros(
        N,
        dtype=torch.bool,
        device=env.device,
    )

    bridge_seen = torch.zeros_like(
        swing_seen
    )

    candidate_seen = torch.zeros_like(
        swing_seen
    )

    candidate_age = torch.zeros(
        N,
        dtype=torch.long,
        device=env.device,
    )

    candidate_hold = torch.zeros_like(
        candidate_age
    )

    candidate_hold_max = torch.zeros_like(
        candidate_age
    )

    touchdown_hold = torch.zeros_like(
        candidate_age
    )

    touchdown_hold_max = torch.zeros_like(
        candidate_age
    )

    completed = 0
    steps = 0

    official = 0

    done_bridge = 0
    done_swing = 0
    done_candidate = 0
    done_candidate_hold2 = 0
    done_candidate_hold3 = 0
    done_touchdown_hold3 = 0

    done_abs_forward = 0.0
    done_abs_lift = 0.0
    done_fz = 0.0
    done_support3 = 0.0

    safety = max(
        4000,
        episodes
        * (
            env.total_episode_steps
            + 100
        ),
    )

    with torch.no_grad():

        while completed < episodes:

            obs = TensorDict(
                {
                    "policy":
                        env.low_obs_buf
                },
                batch_size=[N],
                device=env.device,
            )

            obs_runner = obs.to(
                runner.device
            )

            # =================================================
            # PHASE 1:
            # deterministic Lift.
            # =================================================

            lift_base = (
                lift_policy.act_inference(
                    obs_runner
                )
            )

            lift_norm = (
                _r820_actor_input(
                    lift_policy,
                    obs_runner,
                )
            )

            lift_action = (
                lift_base
                + lift_head(
                    lift_norm
                )
            )

            bridge_state = (
                env.bridge_seen_r852
                .detach()
                .bool()
                .clone()
            )

            bridge_age = torch.where(
                bridge_state,
                bridge_age + 1,
                torch.zeros_like(
                    bridge_age
                ),
            )

            # =================================================
            # PHASE 2:
            # R8.24 winner:
            # bridge + hold2 -> R8.15 Swing.
            # =================================================

            swing_action = (
                swing_policy.act_inference(
                    obs_runner
                )
            )

            swing_alpha = (
                bridge_state
                & (bridge_age >= 3)
            ).float()

            pre_recovery_action = (
                (
                    1.0
                    - swing_alpha.unsqueeze(1)
                )
                * lift_action

                + swing_alpha.unsqueeze(1)
                * swing_action
            )

            # =================================================
            # PHASE 3:
            # deterministic R8.27 winner retraction.
            # =================================================

            recovery_mean = (
                recovery_policy.act_inference(
                    obs_runner
                )
            )

            recovery_norm = (
                _r820_actor_input(
                    recovery_policy,
                    obs_runner,
                )
            )

            recovery_action = (
                recovery_mean
                + 1.20
                * recovery_head(
                    recovery_norm
                )
            )

            post_swing_action = torch.where(
                swing_seen.to(
                    runner.device
                ).unsqueeze(1),
                recovery_action,
                pre_recovery_action,
            )

            # =================================================
            # PHASE 4:
            # once candidate has appeared, test multiple
            # learned policies as settling/recovery policies.
            # =================================================

            settle_action = (
                settle_policy.act_inference(
                    obs_runner
                )
            )

            candidate_age = torch.where(
                candidate_seen,
                candidate_age + 1,
                torch.zeros_like(
                    candidate_age
                ),
            )

            if transition == "immediate":

                settle_alpha = (
                    candidate_seen.float()
                )

            elif transition == "hold2":

                settle_alpha = (
                    candidate_seen
                    & (candidate_age >= 3)
                ).float()

            elif transition == "blend4":

                settle_alpha = torch.where(
                    candidate_seen,
                    torch.clamp(
                        (
                            candidate_age.float()
                            - 1.0
                        ) / 4.0,
                        0.0,
                        1.0,
                    ),
                    torch.zeros(
                        N,
                        dtype=torch.float32,
                        device=env.device,
                    ),
                )

            else:
                raise RuntimeError(
                    f"bad transition: {transition}"
                )

            action = (
                (
                    1.0
                    - settle_alpha.unsqueeze(1)
                )
                * post_swing_action

                + settle_alpha.unsqueeze(1)
                * settle_action
            )

            (
                _,
                _,
                done,
                _,
                diag,
            ) = env.kick_step(
                action.to(
                    env.device
                )
            )

            foot_forward = (
                env.r817_foot_forward
                .detach()
            )

            foot_lift = (
                env.r817_foot_lift
                .detach()
            )

            selected_fz = (
                env.r817_selected_fz
                .detach()
            )

            support3 = (
                env.r817_support3_fz
                .detach()
            )

            bridge_now = (
                env.r817_bridge_now
                .detach()
                .bool()
            )

            swing_now = (
                env.r817_swing_now
                .detach()
                .bool()
            )

            bridge_seen |= bridge_now
            swing_seen |= swing_now

            abs_f = torch.abs(
                foot_forward
            )

            abs_z = torch.abs(
                foot_lift
            )

            touchdown_now = (
                swing_seen
                & (abs_z <= 0.015)
                & (selected_fz >= 8.0)
            )

            candidate_now = (
                touchdown_now
                & (abs_f <= 0.045)
                & (support3 >= 100.0)
            )

            candidate_seen |= (
                candidate_now
            )

            candidate_hold = torch.where(
                candidate_now,
                candidate_hold + 1,
                torch.zeros_like(
                    candidate_hold
                ),
            )

            candidate_hold_max = torch.maximum(
                candidate_hold_max,
                candidate_hold,
            )

            touchdown_hold = torch.where(
                touchdown_now,
                touchdown_hold + 1,
                torch.zeros_like(
                    touchdown_hold
                ),
            )

            touchdown_hold_max = torch.maximum(
                touchdown_hold_max,
                touchdown_hold,
            )

            official += int(
                diag[
                    "stage_successful_episodes"
                ]
            )

            completed += int(
                diag[
                    "completed_episodes"
                ]
            )

            done_mask = (
                done.to(
                    env.device
                )
                .bool()
            )

            if done_mask.any():

                done_bridge += int(
                    bridge_seen[
                        done_mask
                    ].sum()
                )

                done_swing += int(
                    swing_seen[
                        done_mask
                    ].sum()
                )

                done_candidate += int(
                    candidate_seen[
                        done_mask
                    ].sum()
                )

                done_candidate_hold2 += int(
                    (
                        candidate_hold_max[
                            done_mask
                        ] >= 2
                    ).sum()
                )

                done_candidate_hold3 += int(
                    (
                        candidate_hold_max[
                            done_mask
                        ] >= 3
                    ).sum()
                )

                done_touchdown_hold3 += int(
                    (
                        touchdown_hold_max[
                            done_mask
                        ] >= 3
                    ).sum()
                )

                done_abs_forward += float(
                    abs_f[
                        done_mask
                    ].sum()
                )

                done_abs_lift += float(
                    abs_z[
                        done_mask
                    ].sum()
                )

                done_fz += float(
                    selected_fz[
                        done_mask
                    ].sum()
                )

                done_support3 += float(
                    support3[
                        done_mask
                    ].sum()
                )

                bridge_age[
                    done_mask
                ] = 0

                swing_seen[
                    done_mask
                ] = False

                bridge_seen[
                    done_mask
                ] = False

                candidate_seen[
                    done_mask
                ] = False

                candidate_age[
                    done_mask
                ] = 0

                candidate_hold[
                    done_mask
                ] = 0

                candidate_hold_max[
                    done_mask
                ] = 0

                touchdown_hold[
                    done_mask
                ] = 0

                touchdown_hold_max[
                    done_mask
                ] = 0

            steps += 1

            if steps > safety:
                raise RuntimeError(
                    "R828 eval safety limit."
                )

    denom = max(
        completed,
        1,
    )

    return {
        "episodes":
            completed,

        "official_recovery_rate":
            official / denom,

        "bridge_rate":
            done_bridge / denom,

        "swing_rate":
            done_swing / denom,

        "candidate_rate":
            done_candidate / denom,

        "candidate_hold2_rate":
            done_candidate_hold2 / denom,

        "candidate_hold3_rate":
            done_candidate_hold3 / denom,

        "touchdown_hold3_rate":
            done_touchdown_hold3 / denom,

        "done_abs_forward_m":
            done_abs_forward / denom,

        "done_abs_lift_m":
            done_abs_lift / denom,

        "done_selected_fz_N":
            done_fz / denom,

        "done_support3_fz_N":
            done_support3 / denom,
    }


def run_r828(
    *,
    runner: Any,
    env: IsolatedSupportTransferSkillEnv,
    lift_checkpoint: Path,
    lift_head_path: Path,
    swing_checkpoint: Path,
    recovery_checkpoint: Path,
    recovery_head_path: Path,
    settle_checkpoints: dict[str, Path],
    summary_path: Path,
    broad_episodes: int,
    confirm_episodes: int,
    seed: int,
) -> None:

    lift_policy = _r824_load_policy_copy(
        runner,
        lift_checkpoint,
    )

    swing_policy = _r824_load_policy_copy(
        runner,
        swing_checkpoint,
    )

    recovery_policy = _r824_load_policy_copy(
        runner,
        recovery_checkpoint,
    )

    lift_bundle = torch.load(
        lift_head_path,
        map_location=runner.device,
        weights_only=False,
    )

    lift_head = R820LiftResidualHead(
        obs_dim=70,
        hidden=64,
        action_dim=12,
        scale=float(
            lift_bundle["residual_scale"]
        ),
    ).to(
        runner.device
    )

    lift_head.load_state_dict(
        lift_bundle["state_dict"]
    )

    lift_head.eval()

    recovery_bundle = torch.load(
        recovery_head_path,
        map_location=runner.device,
        weights_only=False,
    )

    recovery_head = R820LiftResidualHead(
        obs_dim=70,
        hidden=64,
        action_dim=12,
        scale=float(
            recovery_bundle["scale"]
        ),
    ).to(
        runner.device
    )

    recovery_head.load_state_dict(
        recovery_bundle[
            "state_dict"
        ]
    )

    recovery_head.eval()

    for p in lift_head.parameters():
        p.requires_grad_(False)

    for p in recovery_head.parameters():
        p.requires_grad_(False)

    settle_policies = {}

    for name, checkpoint in (
        settle_checkpoints.items()
    ):

        settle_policies[name] = (
            _r824_load_policy_copy(
                runner,
                checkpoint,
            )
        )

    transitions = (
        "immediate",
        "hold2",
        "blend4",
    )

    broad = []

    for name, settle_policy in (
        settle_policies.items()
    ):

        for transition in transitions:

            label = (
                f"{name}__{transition}"
            )

            result = _r828_eval(
                runner=runner,
                env=env,
                lift_policy=lift_policy,
                lift_head=lift_head,
                swing_policy=swing_policy,
                recovery_policy=recovery_policy,
                recovery_head=recovery_head,
                settle_policy=settle_policy,
                transition=transition,
                episodes=broad_episodes,
                seed=seed + 61000,
            )

            result["label"] = label
            result["settle_actor"] = name
            result["transition"] = transition

            broad.append(result)

            print(
                "[R828_BROAD] "
                f"label={label} "
                f"rec={result['official_recovery_rate']:.4f} "
                f"cand={result['candidate_rate']:.4f} "
                f"h2={result['candidate_hold2_rate']:.4f} "
                f"h3={result['candidate_hold3_rate']:.4f} "
                f"touch3={result['touchdown_hold3_rate']:.4f} "
                f"swing={result['swing_rate']:.4f}",
                flush=True,
            )

    ranked = sorted(
        broad,
        key=lambda x: (
            x[
                "official_recovery_rate"
            ],
            x[
                "candidate_hold3_rate"
            ],
            x[
                "candidate_hold2_rate"
            ],
            x[
                "candidate_rate"
            ],
            x[
                "touchdown_hold3_rate"
            ],
            -x[
                "done_abs_forward_m"
            ],
        ),
        reverse=True,
    )

    top3 = [
        x["label"]
        for x in ranked[:3]
    ]

    by_label = {
        x["label"]: x
        for x in broad
    }

    confirmations = []

    for label in top3:

        source = by_label[label]

        result = _r828_eval(
            runner=runner,
            env=env,
            lift_policy=lift_policy,
            lift_head=lift_head,
            swing_policy=swing_policy,
            recovery_policy=recovery_policy,
            recovery_head=recovery_head,
            settle_policy=settle_policies[
                source[
                    "settle_actor"
                ]
            ],
            transition=source[
                "transition"
            ],
            episodes=confirm_episodes,
            seed=seed + 72000,
        )

        result["label"] = label
        result[
            "settle_actor"
        ] = source[
            "settle_actor"
        ]
        result[
            "transition"
        ] = source[
            "transition"
        ]

        confirmations.append(
            result
        )

        print(
            "[R828_CONFIRM] "
            f"label={label} "
            f"rec={result['official_recovery_rate']:.4f} "
            f"cand={result['candidate_rate']:.4f} "
            f"h3={result['candidate_hold3_rate']:.4f} "
            f"touch3={result['touchdown_hold3_rate']:.4f}",
            flush=True,
        )

    winner = max(
        confirmations,
        key=lambda x: (
            x[
                "official_recovery_rate"
            ],
            x[
                "candidate_hold3_rate"
            ],
            x[
                "candidate_rate"
            ],
        ),
    )

    rec = winner[
        "official_recovery_rate"
    ]

    hold3 = winner[
        "candidate_hold3_rate"
    ]

    cand = winner[
        "candidate_rate"
    ]

    if rec >= 0.50:

        verdict = (
            "DET_FULL_RECOVERY_STRONG"
        )

    elif rec > 0:

        verdict = (
            "DET_FULL_RECOVERY_EMERGED"
        )

    elif hold3 >= 0.20:

        verdict = (
            "RETRACTION_HOLD_ESTABLISHED_"
            "BODY_OR_EXACT_SUPPORT_BLOCKER"
        )

    elif (
        cand >= 0.20
        and hold3 < 0.05
    ):

        verdict = (
            "RETRACTION_TRANSIENT_"
            "HOLD_IS_BLOCKER"
        )

    elif cand > 0:

        verdict = (
            "PARTIAL_RETRACTION_STABILIZATION"
        )

    else:

        verdict = (
            "SETTLING_SWITCH_DESTROYS_RETRACTION"
        )

    summary = {
        "version":
            "r8_28_recovery_stabilization_sweep",

        "training":
            False,

        "reward_change":
            False,

        "recovery_head":
            str(
                recovery_head_path
            ),

        "broad_cases":
            broad,

        "top3":
            top3,

        "confirmations":
            confirmations,

        "winner":
            winner["label"],

        "verdict":
            verdict,
    }

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(
        "[R828_RESULT] "
        f"winner={winner['label']} "
        f"rec={rec:.6f} "
        f"candidate={cand:.6f} "
        f"hold3={hold3:.6f} "
        f"verdict={verdict}",
        flush=True,
    )

    print(
        "R828_STABILIZATION_SWEEP_OK",
        flush=True,
    )



def _r829_eval(
    *,
    runner: Any,
    env: IsolatedSupportTransferSkillEnv,
    lift_policy: Any,
    lift_head: Any,
    swing_policy: Any,
    recovery_policy: Any,
    recovery_head: Any,
    start_phase: float,
    gain: float,
    release_mode: str,
    episodes: int,
    seed: int,
) -> dict[str, Any]:

    torch.manual_seed(seed)
    np.random.seed(seed)

    generator = getattr(
        env,
        "generator",
        None,
    )

    if isinstance(
        generator,
        torch.Generator,
    ):
        generator.manual_seed(seed)

    env.set_curriculum_stage(
        "recovery"
    )

    env.reset()

    N = env.num_envs

    bridge_age = torch.zeros(
        N,
        dtype=torch.long,
        device=env.device,
    )

    swing_seen = torch.zeros(
        N,
        dtype=torch.bool,
        device=env.device,
    )

    approx_candidate_seen = (
        torch.zeros_like(
            swing_seen
        )
    )

    approx_age = torch.zeros(
        N,
        dtype=torch.long,
        device=env.device,
    )

    names = (
        "candidate",
        "forward",
        "lift",
        "return",
        "contact",
        "support4",
        "base",
        "height",
        "upright",
        "body",
        "touchdown",
        "all_now",
    )

    seen = {
        name:
            torch.zeros(
                N,
                dtype=torch.bool,
                device=env.device,
            )
        for name in names
    }

    max_hold = torch.zeros(
        N,
        dtype=torch.long,
        device=env.device,
    )

    first_approx_phase = torch.full(
        (N,),
        float("nan"),
        dtype=torch.float32,
        device=env.device,
    )

    first_all_phase = torch.full_like(
        first_approx_phase,
        float("nan"),
    )

    completed = 0
    steps = 0

    totals = {
        name: 0
        for name in names
    }

    hold1 = 0
    hold2 = 0
    hold3 = 0
    recovery_seen_total = 0
    stage_success_total = 0

    stage_keys = (
        "load",
        "prelift",
        "bridge",
        "swing",
        "base",
        "height",
        "upright",
        "support3",
        "no_violation",
    )

    stage_totals = {
        key: 0
        for key in stage_keys
    }

    approx_phase_sum = 0.0
    approx_phase_n = 0

    all_phase_sum = 0.0
    all_phase_n = 0

    safety = max(
        4000,
        episodes
        * (
            env.total_episode_steps
            + 100
        ),
    )

    with torch.no_grad():

        while completed < episodes:

            obs = TensorDict(
                {
                    "policy":
                        env.low_obs_buf
                },
                batch_size=[N],
                device=env.device,
            )

            obs_runner = obs.to(
                runner.device
            )

            phase_before = (
                env._phase()
                .detach()
                .clone()
            )

            # ================================================
            # Lift primitive.
            # ================================================

            lift_base = (
                lift_policy.act_inference(
                    obs_runner
                )
            )

            lift_norm = (
                _r820_actor_input(
                    lift_policy,
                    obs_runner,
                )
            )

            lift_action = (
                lift_base
                + lift_head(
                    lift_norm
                )
            )

            # ================================================
            # Bridge + hold2 -> R8.15 Swing.
            # ================================================

            bridge_state = (
                env.bridge_seen_r852
                .detach()
                .bool()
                .clone()
            )

            bridge_age = torch.where(
                bridge_state,
                bridge_age + 1,
                torch.zeros_like(
                    bridge_age
                ),
            )

            swing_action = (
                swing_policy.act_inference(
                    obs_runner
                )
            )

            swing_alpha = (
                bridge_state
                & (bridge_age >= 3)
            ).float()

            chain_action = (
                (
                    1.0
                    - swing_alpha.unsqueeze(1)
                )
                * lift_action

                + swing_alpha.unsqueeze(1)
                * swing_action
            )

            # ================================================
            # Recovery head.
            #
            # Key sweep variable:
            # do NOT necessarily retract immediately after
            # swing; optionally align it with phase 0.88.
            # ================================================

            recovery_mean = (
                recovery_policy.act_inference(
                    obs_runner
                )
            )

            recovery_norm = (
                _r820_actor_input(
                    recovery_policy,
                    obs_runner,
                )
            )

            correction = (
                float(gain)
                * recovery_head(
                    recovery_norm
                )
            )

            approx_age = torch.where(
                approx_candidate_seen,
                approx_age + 1,
                torch.zeros_like(
                    approx_age
                ),
            )

            head_active = (
                swing_seen
                & (
                    phase_before
                    >= float(start_phase)
                )
            )

            if release_mode == "hold2_release":

                head_active &= ~(
                    approx_candidate_seen
                    & (approx_age >= 3)
                )

            elif release_mode != "continuous":

                raise RuntimeError(
                    f"bad release mode "
                    f"{release_mode}"
                )

            recovery_action = (
                recovery_mean
                + torch.where(
                    head_active.to(
                        runner.device
                    ).unsqueeze(1),
                    correction,
                    torch.zeros_like(
                        correction
                    ),
                )
            )

            action = torch.where(
                swing_seen.to(
                    runner.device
                ).unsqueeze(1),
                recovery_action,
                chain_action,
            )

            (
                _,
                _,
                done,
                _,
                diag,
            ) = env.kick_step(
                action.to(
                    env.device
                )
            )

            swing_now = (
                env.r817_swing_now
                .detach()
                .bool()
            )

            swing_seen |= swing_now

            foot_forward = (
                env.r817_foot_forward
                .detach()
            )

            foot_lift = (
                env.r817_foot_lift
                .detach()
            )

            selected_fz = (
                env.r817_selected_fz
                .detach()
            )

            support3 = (
                env.r817_support3_fz
                .detach()
            )

            # Same approximate candidate used in R8.27/28.
            approx_now = (
                swing_seen
                & (
                    torch.abs(
                        foot_forward
                    ) <= 0.045
                )
                & (
                    torch.abs(
                        foot_lift
                    ) <= 0.015
                )
                & (selected_fz >= 8.0)
                & (support3 >= 100.0)
            )

            new_approx = (
                approx_now
                & torch.isnan(
                    first_approx_phase
                )
            )

            first_approx_phase = torch.where(
                new_approx,
                env.r829_phase,
                first_approx_phase,
            )

            approx_candidate_seen |= (
                approx_now
            )

            exact = {
                "candidate":
                    env.r829_rec_candidate,

                "forward":
                    env.r829_forward_ok,

                "lift":
                    env.r829_lift_ok,

                "return":
                    env.r829_return_ok,

                "contact":
                    env.r829_contact_ok,

                "support4":
                    env.r829_support4_ok,

                "base":
                    env.r829_base_ok,

                "height":
                    env.r829_height_ok,

                "upright":
                    env.r829_upright_ok,

                "body":
                    env.r829_body_ok,

                "touchdown":
                    env.r829_touchdown_ok,

                "all_now":
                    env.r829_all_now,
            }

            for name, mask in exact.items():
                seen[name] |= (
                    mask.detach().bool()
                )

            new_all = (
                env.r829_all_now
                & torch.isnan(
                    first_all_phase
                )
            )

            first_all_phase = torch.where(
                new_all,
                env.r829_phase,
                first_all_phase,
            )

            max_hold = torch.maximum(
                max_hold,
                env.r829_recovery_hold
                .long(),
            )

            completed += int(
                diag[
                    "completed_episodes"
                ]
            )

            done_mask = (
                done.to(
                    env.device
                )
                .bool()
            )

            if done_mask.any():

                for name in names:

                    totals[name] += int(
                        seen[name][
                            done_mask
                        ].sum()
                    )

                hold1 += int(
                    (
                        max_hold[
                            done_mask
                        ] >= 1
                    ).sum()
                )

                hold2 += int(
                    (
                        max_hold[
                            done_mask
                        ] >= 2
                    ).sum()
                )

                hold3 += int(
                    (
                        max_hold[
                            done_mask
                        ] >= 3
                    ).sum()
                )

                recovery_seen_total += int(
                    env.r829_recovery_seen[
                        done_mask
                    ].sum()
                )

                stage_success_total += int(
                    env.r829_stage_success[
                        done_mask
                    ].sum()
                )

                stage_masks = {
                    "load":
                        env.r829_stage_load_ok,

                    "prelift":
                        env.r829_stage_prelift_ok,

                    "bridge":
                        env.r829_stage_bridge_ok,

                    "swing":
                        env.r829_stage_swing_ok,

                    "base":
                        env.r829_stage_base_ok,

                    "height":
                        env.r829_stage_height_ok,

                    "upright":
                        env.r829_stage_upright_ok,

                    "support3":
                        env.r829_stage_support3_ok,

                    "no_violation":
                        env.r829_stage_no_violation,
                }

                for key, mask in (
                    stage_masks.items()
                ):

                    stage_totals[key] += int(
                        mask[
                            done_mask
                        ].sum()
                    )

                approx_values = (
                    first_approx_phase[
                        done_mask
                    ]
                )

                approx_valid = (
                    torch.isfinite(
                        approx_values
                    )
                )

                if approx_valid.any():

                    approx_phase_sum += float(
                        approx_values[
                            approx_valid
                        ].sum()
                    )

                    approx_phase_n += int(
                        approx_valid.sum()
                    )

                all_values = (
                    first_all_phase[
                        done_mask
                    ]
                )

                all_valid = torch.isfinite(
                    all_values
                )

                if all_valid.any():

                    all_phase_sum += float(
                        all_values[
                            all_valid
                        ].sum()
                    )

                    all_phase_n += int(
                        all_valid.sum()
                    )

                bridge_age[
                    done_mask
                ] = 0

                swing_seen[
                    done_mask
                ] = False

                approx_candidate_seen[
                    done_mask
                ] = False

                approx_age[
                    done_mask
                ] = 0

                max_hold[
                    done_mask
                ] = 0

                first_approx_phase[
                    done_mask
                ] = float("nan")

                first_all_phase[
                    done_mask
                ] = float("nan")

                for name in names:
                    seen[name][
                        done_mask
                    ] = False

            steps += 1

            if steps > safety:
                raise RuntimeError(
                    "R829 eval safety limit."
                )

    denom = max(
        completed,
        1,
    )

    result = {
        "episodes":
            completed,

        "start_phase":
            float(start_phase),

        "gain":
            float(gain),

        "release_mode":
            release_mode,

        "hold1_rate":
            hold1 / denom,

        "hold2_rate":
            hold2 / denom,

        "hold3_rate":
            hold3 / denom,

        "recovery_seen_rate":
            recovery_seen_total / denom,

        "stage_success_rate":
            stage_success_total / denom,

        "mean_first_approx_phase":
            (
                approx_phase_sum
                / approx_phase_n
                if approx_phase_n
                else None
            ),

        "mean_first_all_now_phase":
            (
                all_phase_sum
                / all_phase_n
                if all_phase_n
                else None
            ),
    }

    for name in names:
        result[
            f"{name}_rate"
        ] = (
            totals[name]
            / denom
        )

    for key in stage_keys:
        result[
            f"stage_{key}_rate"
        ] = (
            stage_totals[key]
            / denom
        )

    return result


def run_r829(
    *,
    runner: Any,
    env: IsolatedSupportTransferSkillEnv,
    lift_checkpoint: Path,
    lift_head_path: Path,
    swing_checkpoint: Path,
    recovery_checkpoint: Path,
    recovery_head_path: Path,
    summary_path: Path,
    broad_episodes: int,
    confirm_episodes: int,
    seed: int,
) -> None:

    lift_policy = _r824_load_policy_copy(
        runner,
        lift_checkpoint,
    )

    swing_policy = _r824_load_policy_copy(
        runner,
        swing_checkpoint,
    )

    recovery_policy = _r824_load_policy_copy(
        runner,
        recovery_checkpoint,
    )

    lift_bundle = torch.load(
        lift_head_path,
        map_location=runner.device,
        weights_only=False,
    )

    lift_head = R820LiftResidualHead(
        obs_dim=70,
        hidden=64,
        action_dim=12,
        scale=float(
            lift_bundle[
                "residual_scale"
            ]
        ),
    ).to(
        runner.device
    )

    lift_head.load_state_dict(
        lift_bundle[
            "state_dict"
        ]
    )

    lift_head.eval()

    recovery_bundle = torch.load(
        recovery_head_path,
        map_location=runner.device,
        weights_only=False,
    )

    recovery_head = R820LiftResidualHead(
        obs_dim=70,
        hidden=64,
        action_dim=12,
        scale=float(
            recovery_bundle[
                "scale"
            ]
        ),
    ).to(
        runner.device
    )

    recovery_head.load_state_dict(
        recovery_bundle[
            "state_dict"
        ]
    )

    recovery_head.eval()

    starts = (
        0.00,
        0.72,
        0.80,
        0.86,
    )

    gains = (
        1.0,
        1.2,
        1.4,
    )

    releases = (
        "continuous",
        "hold2_release",
    )

    broad = []

    for start in starts:

        for gain in gains:

            for release in releases:

                label = (
                    f"p{start:.2f}"
                    f"__g{gain:.1f}"
                    f"__{release}"
                )

                result = _r829_eval(
                    runner=runner,
                    env=env,
                    lift_policy=lift_policy,
                    lift_head=lift_head,
                    swing_policy=swing_policy,
                    recovery_policy=recovery_policy,
                    recovery_head=recovery_head,
                    start_phase=start,
                    gain=gain,
                    release_mode=release,
                    episodes=broad_episodes,
                    seed=seed + 81000,
                )

                result[
                    "label"
                ] = label

                broad.append(
                    result
                )

                print(
                    "[R829_BROAD] "
                    f"label={label} "
                    f"stage="
                    f"{result['stage_success_rate']:.4f} "
                    f"rec="
                    f"{result['recovery_seen_rate']:.4f} "
                    f"all="
                    f"{result['all_now_rate']:.4f} "
                    f"h3="
                    f"{result['hold3_rate']:.4f} "
                    f"return="
                    f"{result['return_rate']:.4f} "
                    f"touch="
                    f"{result['touchdown_rate']:.4f} "
                    f"body="
                    f"{result['body_rate']:.4f}",
                    flush=True,
                )

    ranked = sorted(
        broad,
        key=lambda x: (
            x["stage_success_rate"],
            x["recovery_seen_rate"],
            x["hold3_rate"],
            x["all_now_rate"],
            x["return_rate"],
            x["touchdown_rate"],
            x["body_rate"],
        ),
        reverse=True,
    )

    top4 = [
        x["label"]
        for x in ranked[:4]
    ]

    by_label = {
        x["label"]: x
        for x in broad
    }

    confirmations = []

    for label in top4:

        source = by_label[label]

        result = _r829_eval(
            runner=runner,
            env=env,
            lift_policy=lift_policy,
            lift_head=lift_head,
            swing_policy=swing_policy,
            recovery_policy=recovery_policy,
            recovery_head=recovery_head,
            start_phase=source[
                "start_phase"
            ],
            gain=source["gain"],
            release_mode=source[
                "release_mode"
            ],
            episodes=confirm_episodes,
            seed=seed + 92000,
        )

        result["label"] = label

        confirmations.append(
            result
        )

        print(
            "[R829_CONFIRM] "
            f"label={label} "
            f"stage="
            f"{result['stage_success_rate']:.4f} "
            f"rec="
            f"{result['recovery_seen_rate']:.4f} "
            f"all="
            f"{result['all_now_rate']:.4f} "
            f"h3="
            f"{result['hold3_rate']:.4f}",
            flush=True,
        )

    winner = max(
        confirmations,
        key=lambda x: (
            x["stage_success_rate"],
            x["recovery_seen_rate"],
            x["hold3_rate"],
            x["all_now_rate"],
            x["return_rate"],
        ),
    )

    if winner[
        "stage_success_rate"
    ] >= 0.50:

        verdict = (
            "BALL_FREE_RECOVERY_STRONG"
        )

    elif winner[
        "stage_success_rate"
    ] > 0:

        verdict = (
            "BALL_FREE_RECOVERY_EMERGED"
        )

    elif winner[
        "recovery_seen_rate"
    ] > 0:

        verdict = (
            "RECOVERY_CONTRACT_PASSES_"
            "STAGE_GLOBAL_CONSTRAINT_BLOCKER"
        )

    elif winner[
        "all_now_rate"
    ] > 0:

        verdict = (
            "EXACT_RECOVERY_NOW_EMERGED_"
            "HOLD3_BLOCKER"
        )

    else:

        # Find smallest exact component rate.
        components = {
            key:
                winner[
                    f"{key}_rate"
                ]
            for key in (
                "return",
                "contact",
                "support4",
                "base",
                "height",
                "upright",
                "body",
                "touchdown",
            )
        }

        blocker = min(
            components,
            key=components.get,
        )

        verdict = (
            "EXACT_COMPONENT_BLOCKER__"
            + blocker.upper()
        )

    summary = {
        "version":
            "r8_29_exact_recovery_timing",

        "training":
            False,

        "reward_change":
            False,

        "broad_cases":
            broad,

        "top4":
            top4,

        "confirmations":
            confirmations,

        "winner":
            winner["label"],

        "verdict":
            verdict,
    }

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(
        "[R829_RESULT] "
        f"winner={winner['label']} "
        f"stage={winner['stage_success_rate']:.6f} "
        f"recovery={winner['recovery_seen_rate']:.6f} "
        f"all={winner['all_now_rate']:.6f} "
        f"hold3={winner['hold3_rate']:.6f} "
        f"verdict={verdict}",
        flush=True,
    )

    print(
        "R829_EXACT_SWEEP_OK",
        flush=True,
    )



def _r830_eval(
    *,
    runner,
    env,
    lift_policy,
    lift_head,
    swing_policy,
    recovery_policy,
    recovery_head,
    lift_gain,
    swing_mode,
    recovery_start,
    recovery_gain,
    episodes,
    seed,
):

    torch.manual_seed(seed)
    np.random.seed(seed)

    generator = getattr(
        env,
        "generator",
        None,
    )

    if isinstance(generator, torch.Generator):
        generator.manual_seed(seed)

    env.set_curriculum_stage("recovery")
    env.reset()

    spec = env._stage_spec()

    max_base_limit = float(
        spec["max_base"]
    )
    max_height_limit = float(
        spec["max_height"]
    )
    max_upright_limit = float(
        spec["max_upright"]
    )

    N = env.num_envs

    bridge_age = torch.zeros(
        N,
        dtype=torch.long,
        device=env.device,
    )

    swing_seen = torch.zeros(
        N,
        dtype=torch.bool,
        device=env.device,
    )

    # ---------------------------------------------
    # Per-phase maximum stability excursions.
    #
    # phase0: pre-swing / Lift
    # phase1: Swing
    # phase2: Recovery window
    # ---------------------------------------------

    phase_max_base = torch.zeros(
        3, N,
        dtype=torch.float32,
        device=env.device,
    )

    phase_max_height = torch.zeros_like(
        phase_max_base
    )

    phase_max_upright = torch.zeros_like(
        phase_max_base
    )

    phase_visited = torch.zeros(
        3, N,
        dtype=torch.bool,
        device=env.device,
    )

    exact_all_seen = torch.zeros(
        N,
        dtype=torch.bool,
        device=env.device,
    )

    max_recovery_hold = torch.zeros(
        N,
        dtype=torch.long,
        device=env.device,
    )

    completed = 0
    steps = 0

    stage_success_total = 0
    recovery_seen_total = 0

    bridge_total = 0
    swing_total = 0

    exact_all_total = 0
    hold1_total = 0
    hold2_total = 0
    hold3_total = 0

    stage_base_total = 0
    stage_height_total = 0
    stage_upright_total = 0

    phase_pass = {
        "lift_base": 0,
        "lift_height": 0,
        "lift_upright": 0,

        "swing_base": 0,
        "swing_height": 0,
        "swing_upright": 0,

        "recovery_base": 0,
        "recovery_height": 0,
        "recovery_upright": 0,
    }

    phase_max_sums = {
        key: 0.0
        for key in (
            "lift_base",
            "lift_height",
            "lift_upright",
            "swing_base",
            "swing_height",
            "swing_upright",
            "recovery_base",
            "recovery_height",
            "recovery_upright",
        )
    }

    safety = max(
        4000,
        episodes
        * (
            env.total_episode_steps
            + 100
        ),
    )

    with torch.no_grad():

        while completed < episodes:

            obs = TensorDict(
                {"policy": env.low_obs_buf},
                batch_size=[N],
                device=env.device,
            )

            obs_runner = obs.to(
                runner.device
            )

            phase_before = (
                env._phase()
                .detach()
                .clone()
            )

            # =============================================
            # Lift
            # =============================================

            lift_base = (
                lift_policy.act_inference(
                    obs_runner
                )
            )

            lift_norm = _r820_actor_input(
                lift_policy,
                obs_runner,
            )

            lift_action = (
                lift_base
                + float(lift_gain)
                * lift_head(
                    lift_norm
                )
            )

            # =============================================
            # Swing handoff
            # =============================================

            bridge_state = (
                env.bridge_seen_r852
                .detach()
                .bool()
                .clone()
            )

            bridge_age = torch.where(
                bridge_state,
                bridge_age + 1,
                torch.zeros_like(
                    bridge_age
                ),
            )

            swing_action = (
                swing_policy.act_inference(
                    obs_runner
                )
            )

            if swing_mode == "hold2":

                alpha = (
                    bridge_state
                    & (bridge_age >= 3)
                ).float()

            elif swing_mode == "blend4":

                alpha = torch.where(
                    bridge_state,
                    torch.clamp(
                        (
                            bridge_age.float()
                            - 1.0
                        ) / 4.0,
                        0.0,
                        1.0,
                    ),
                    torch.zeros(
                        N,
                        device=env.device,
                    ),
                )

            elif swing_mode == "blend6":

                alpha = torch.where(
                    bridge_state,
                    torch.clamp(
                        (
                            bridge_age.float()
                            - 1.0
                        ) / 6.0,
                        0.0,
                        1.0,
                    ),
                    torch.zeros(
                        N,
                        device=env.device,
                    ),
                )

            else:
                raise RuntimeError(
                    f"bad swing mode {swing_mode}"
                )

            chain_action = (
                (
                    1.0
                    - alpha.unsqueeze(1)
                )
                * lift_action

                + alpha.unsqueeze(1)
                * swing_action
            )

            # =============================================
            # Recovery
            # =============================================

            rec_mean = (
                recovery_policy.act_inference(
                    obs_runner
                )
            )

            rec_norm = _r820_actor_input(
                recovery_policy,
                obs_runner,
            )

            rec_action = (
                rec_mean
                + float(recovery_gain)
                * recovery_head(
                    rec_norm
                )
            )

            rec_active = (
                swing_seen
                & (
                    phase_before
                    >= float(
                        recovery_start
                    )
                )
            )

            action = torch.where(
                rec_active.to(
                    runner.device
                ).unsqueeze(1),
                rec_action,
                chain_action,
            )

            (
                _,
                _,
                done,
                _,
                diag,
            ) = env.kick_step(
                action.to(
                    env.device
                )
            )

            swing_seen |= (
                env.r817_swing_now
                .detach()
                .bool()
            )

            phase = (
                env.r829_phase
                .detach()
            )

            base = (
                env.r829_base_displacement
                .detach()
            )

            height = (
                env.r829_height_error
                .detach()
            )

            upright = (
                env.r829_upright_error
                .detach()
            )

            # phase bins
            masks = (
                phase < 0.58,
                (
                    (phase >= 0.58)
                    & (phase < 0.88)
                ),
                phase >= 0.88,
            )

            for i, mask in enumerate(masks):

                phase_visited[i] |= mask

                phase_max_base[i] = torch.where(
                    mask,
                    torch.maximum(
                        phase_max_base[i],
                        base,
                    ),
                    phase_max_base[i],
                )

                phase_max_height[i] = torch.where(
                    mask,
                    torch.maximum(
                        phase_max_height[i],
                        height,
                    ),
                    phase_max_height[i],
                )

                phase_max_upright[i] = torch.where(
                    mask,
                    torch.maximum(
                        phase_max_upright[i],
                        upright,
                    ),
                    phase_max_upright[i],
                )

            exact_all_seen |= (
                env.r829_all_now
                .detach()
                .bool()
            )

            max_recovery_hold = torch.maximum(
                max_recovery_hold,
                env.r829_recovery_hold
                .detach()
                .long(),
            )

            completed += int(
                diag[
                    "completed_episodes"
                ]
            )

            done_mask = (
                done.to(
                    env.device
                )
                .bool()
            )

            if done_mask.any():

                stage_success_total += int(
                    env.r829_stage_success[
                        done_mask
                    ].sum()
                )

                recovery_seen_total += int(
                    env.r829_recovery_seen[
                        done_mask
                    ].sum()
                )

                bridge_total += int(
                    env.r829_stage_bridge_ok[
                        done_mask
                    ].sum()
                )

                swing_total += int(
                    env.r829_stage_swing_ok[
                        done_mask
                    ].sum()
                )

                exact_all_total += int(
                    exact_all_seen[
                        done_mask
                    ].sum()
                )

                hold1_total += int(
                    (
                        max_recovery_hold[
                            done_mask
                        ] >= 1
                    ).sum()
                )

                hold2_total += int(
                    (
                        max_recovery_hold[
                            done_mask
                        ] >= 2
                    ).sum()
                )

                hold3_total += int(
                    (
                        max_recovery_hold[
                            done_mask
                        ] >= 3
                    ).sum()
                )

                stage_base_total += int(
                    env.r829_stage_base_ok[
                        done_mask
                    ].sum()
                )

                stage_height_total += int(
                    env.r829_stage_height_ok[
                        done_mask
                    ].sum()
                )

                stage_upright_total += int(
                    env.r829_stage_upright_ok[
                        done_mask
                    ].sum()
                )

                labels = (
                    "lift",
                    "swing",
                    "recovery",
                )

                for i, label in enumerate(
                    labels
                ):

                    valid = (
                        done_mask
                        & phase_visited[i]
                    )

                    phase_pass[
                        f"{label}_base"
                    ] += int(
                        (
                            valid
                            & (
                                phase_max_base[i]
                                <= max_base_limit
                            )
                        ).sum()
                    )

                    phase_pass[
                        f"{label}_height"
                    ] += int(
                        (
                            valid
                            & (
                                phase_max_height[i]
                                <= max_height_limit
                            )
                        ).sum()
                    )

                    phase_pass[
                        f"{label}_upright"
                    ] += int(
                        (
                            valid
                            & (
                                phase_max_upright[i]
                                <= max_upright_limit
                            )
                        ).sum()
                    )

                    phase_max_sums[
                        f"{label}_base"
                    ] += float(
                        phase_max_base[i][
                            done_mask
                        ].sum()
                    )

                    phase_max_sums[
                        f"{label}_height"
                    ] += float(
                        phase_max_height[i][
                            done_mask
                        ].sum()
                    )

                    phase_max_sums[
                        f"{label}_upright"
                    ] += float(
                        phase_max_upright[i][
                            done_mask
                        ].sum()
                    )

                bridge_age[
                    done_mask
                ] = 0

                swing_seen[
                    done_mask
                ] = False

                exact_all_seen[
                    done_mask
                ] = False

                max_recovery_hold[
                    done_mask
                ] = 0

                phase_visited[
                    :,
                    done_mask
                ] = False

                phase_max_base[
                    :,
                    done_mask
                ] = 0.0

                phase_max_height[
                    :,
                    done_mask
                ] = 0.0

                phase_max_upright[
                    :,
                    done_mask
                ] = 0.0

            steps += 1

            if steps > safety:
                raise RuntimeError(
                    "R830 eval safety limit"
                )

    denom = max(
        completed,
        1,
    )

    result = {
        "episodes": completed,

        "lift_gain":
            float(lift_gain),

        "swing_mode":
            swing_mode,

        "recovery_start":
            float(recovery_start),

        "recovery_gain":
            float(recovery_gain),

        "bridge_rate":
            bridge_total / denom,

        "swing_rate":
            swing_total / denom,

        "exact_all_rate":
            exact_all_total / denom,

        "hold1_rate":
            hold1_total / denom,

        "hold2_rate":
            hold2_total / denom,

        "hold3_rate":
            hold3_total / denom,

        "recovery_seen_rate":
            recovery_seen_total / denom,

        "stage_success_rate":
            stage_success_total / denom,

        "stage_base_rate":
            stage_base_total / denom,

        "stage_height_rate":
            stage_height_total / denom,

        "stage_upright_rate":
            stage_upright_total / denom,

        "spec_max_base":
            max_base_limit,

        "spec_max_height":
            max_height_limit,

        "spec_max_upright":
            max_upright_limit,
    }

    for key, value in phase_pass.items():
        result[
            f"{key}_pass_rate"
        ] = value / denom

    for key, value in phase_max_sums.items():
        result[
            f"{key}_mean_max"
        ] = value / denom

    return result


def run_r830(
    *,
    runner,
    env,
    lift_checkpoint,
    lift_head_path,
    swing_checkpoint,
    recovery_checkpoint,
    recovery_head_path,
    summary_path,
    broad_episodes,
    confirm_episodes,
    seed,
):

    lift_policy = _r824_load_policy_copy(
        runner,
        lift_checkpoint,
    )

    swing_policy = _r824_load_policy_copy(
        runner,
        swing_checkpoint,
    )

    recovery_policy = _r824_load_policy_copy(
        runner,
        recovery_checkpoint,
    )

    lift_bundle = torch.load(
        lift_head_path,
        map_location=runner.device,
        weights_only=False,
    )

    lift_head = R820LiftResidualHead(
        obs_dim=70,
        hidden=64,
        action_dim=12,
        scale=float(
            lift_bundle[
                "residual_scale"
            ]
        ),
    ).to(
        runner.device
    )

    lift_head.load_state_dict(
        lift_bundle[
            "state_dict"
        ]
    )

    lift_head.eval()

    rec_bundle = torch.load(
        recovery_head_path,
        map_location=runner.device,
        weights_only=False,
    )

    recovery_head = R820LiftResidualHead(
        obs_dim=70,
        hidden=64,
        action_dim=12,
        scale=float(
            rec_bundle["scale"]
        ),
    ).to(
        runner.device
    )

    recovery_head.load_state_dict(
        rec_bundle[
            "state_dict"
        ]
    )

    recovery_head.eval()

    lift_gains = (
        0.80,
        0.90,
        1.00,
    )

    swing_modes = (
        "hold2",
        "blend4",
        "blend6",
    )

    recovery_starts = (
        0.00,
        0.72,
    )

    recovery_gains = (
        1.30,
        1.40,
        1.50,
    )

    broad = []

    for lg in lift_gains:

        for sm in swing_modes:

            for rs in recovery_starts:

                for rg in recovery_gains:

                    label = (
                        f"L{lg:.2f}"
                        f"__{sm}"
                        f"__P{rs:.2f}"
                        f"__R{rg:.2f}"
                    )

                    result = _r830_eval(
                        runner=runner,
                        env=env,
                        lift_policy=lift_policy,
                        lift_head=lift_head,
                        swing_policy=swing_policy,
                        recovery_policy=recovery_policy,
                        recovery_head=recovery_head,
                        lift_gain=lg,
                        swing_mode=sm,
                        recovery_start=rs,
                        recovery_gain=rg,
                        episodes=broad_episodes,
                        seed=seed + 101000,
                    )

                    result[
                        "label"
                    ] = label

                    broad.append(result)

                    print(
                        "[R830_BROAD] "
                        f"{label} "
                        f"stage="
                        f"{result['stage_success_rate']:.4f} "
                        f"rec="
                        f"{result['recovery_seen_rate']:.4f} "
                        f"h3="
                        f"{result['hold3_rate']:.4f} "
                        f"all="
                        f"{result['exact_all_rate']:.4f} "
                        f"bridge="
                        f"{result['bridge_rate']:.4f} "
                        f"swing="
                        f"{result['swing_rate']:.4f} "
                        f"globalH="
                        f"{result['stage_height_rate']:.4f} "
                        f"globalU="
                        f"{result['stage_upright_rate']:.4f}",
                        flush=True,
                    )

    ranked = sorted(
        broad,
        key=lambda x: (
            x["stage_success_rate"],
            x["recovery_seen_rate"],
            x["hold3_rate"],
            x["exact_all_rate"],
            x["stage_upright_rate"],
            x["stage_height_rate"],
            x["swing_rate"],
            x["bridge_rate"],
        ),
        reverse=True,
    )

    top4_labels = [
        x["label"]
        for x in ranked[:4]
    ]

    by_label = {
        x["label"]: x
        for x in broad
    }

    confirmations = []

    for label in top4_labels:

        src = by_label[label]

        result = _r830_eval(
            runner=runner,
            env=env,
            lift_policy=lift_policy,
            lift_head=lift_head,
            swing_policy=swing_policy,
            recovery_policy=recovery_policy,
            recovery_head=recovery_head,
            lift_gain=src["lift_gain"],
            swing_mode=src["swing_mode"],
            recovery_start=src[
                "recovery_start"
            ],
            recovery_gain=src[
                "recovery_gain"
            ],
            episodes=confirm_episodes,
            seed=seed + 112000,
        )

        result["label"] = label
        confirmations.append(result)

        print(
            "[R830_CONFIRM] "
            f"{label} "
            f"stage="
            f"{result['stage_success_rate']:.4f} "
            f"rec="
            f"{result['recovery_seen_rate']:.4f} "
            f"h3="
            f"{result['hold3_rate']:.4f} "
            f"globalH="
            f"{result['stage_height_rate']:.4f} "
            f"globalU="
            f"{result['stage_upright_rate']:.4f}",
            flush=True,
        )

    winner = max(
        confirmations,
        key=lambda x: (
            x["stage_success_rate"],
            x["recovery_seen_rate"],
            x["hold3_rate"],
            x["exact_all_rate"],
            x["stage_upright_rate"],
            x["stage_height_rate"],
        ),
    )

    if winner[
        "stage_success_rate"
    ] >= 0.50:

        verdict = (
            "BALL_FREE_CHAIN_STRONG"
        )

    elif winner[
        "stage_success_rate"
    ] > 0:

        verdict = (
            "BALL_FREE_CHAIN_EMERGED"
        )

    elif winner[
        "recovery_seen_rate"
    ] > 0:

        if (
            winner[
                "stage_upright_rate"
            ]
            < winner[
                "stage_height_rate"
            ]
        ):
            verdict = (
                "GLOBAL_UPRIGHT_IS_FINAL_BLOCKER"
            )
        else:
            verdict = (
                "GLOBAL_HEIGHT_IS_FINAL_BLOCKER"
            )

    else:

        verdict = (
            "RECOVERY_HOLD_AND_STABILITY_"
            "JOINT_BLOCKER"
        )

    summary = {
        "version":
            "r8_30_stability_budget_sweep",

        "training":
            False,

        "cases":
            broad,

        "top4":
            top4_labels,

        "confirmations":
            confirmations,

        "winner":
            winner["label"],

        "verdict":
            verdict,
    }

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(
        "[R830_RESULT] "
        f"winner={winner['label']} "
        f"stage="
        f"{winner['stage_success_rate']:.6f} "
        f"rec="
        f"{winner['recovery_seen_rate']:.6f} "
        f"globalH="
        f"{winner['stage_height_rate']:.6f} "
        f"globalU="
        f"{winner['stage_upright_rate']:.6f} "
        f"verdict={verdict}",
        flush=True,
    )

    print(
        "R830_STABILITY_SWEEP_OK",
        flush=True,
    )



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

    if os.environ.get(
        "R830_STABILITY_ONLY",
        "0",
    ) == "1":

        values = {
            key:
                os.environ.get(
                    key,
                    "",
                ).strip()

            for key in (
                "R830_LIFT",
                "R830_LIFT_HEAD",
                "R830_SWING",
                "R830_RECOVERY",
                "R830_RECOVERY_HEAD",
                "R830_SUMMARY",
            )
        }

        missing = [
            k
            for k, v in values.items()
            if not v
        ]

        if missing:
            raise RuntimeError(
                "R830 missing: "
                + ",".join(missing)
            )

        run_r830(
            runner=runner,
            env=env,
            lift_checkpoint=Path(
                values["R830_LIFT"]
            ).resolve(),
            lift_head_path=Path(
                values[
                    "R830_LIFT_HEAD"
                ]
            ).resolve(),
            swing_checkpoint=Path(
                values["R830_SWING"]
            ).resolve(),
            recovery_checkpoint=Path(
                values[
                    "R830_RECOVERY"
                ]
            ).resolve(),
            recovery_head_path=Path(
                values[
                    "R830_RECOVERY_HEAD"
                ]
            ).resolve(),
            summary_path=Path(
                values[
                    "R830_SUMMARY"
                ]
            ).resolve(),
            broad_episodes=int(
                os.environ.get(
                    "R830_BROAD_EPISODES",
                    "64",
                )
            ),
            confirm_episodes=int(
                os.environ.get(
                    "R830_CONFIRM_EPISODES",
                    "512",
                )
            ),
            seed=args_cli.seed,
        )

        return

    if os.environ.get(
        "R829_EXACT_ONLY",
        "0",
    ) == "1":

        values = {
            key:
                os.environ.get(
                    key,
                    "",
                ).strip()

            for key in (
                "R829_LIFT",
                "R829_LIFT_HEAD",
                "R829_SWING",
                "R829_RECOVERY",
                "R829_RECOVERY_HEAD",
                "R829_SUMMARY",
            )
        }

        missing = [
            k
            for k, v in values.items()
            if not v
        ]

        if missing:
            raise RuntimeError(
                "R829 missing: "
                + ",".join(missing)
            )

        run_r829(
            runner=runner,
            env=env,
            lift_checkpoint=Path(
                values["R829_LIFT"]
            ).resolve(),
            lift_head_path=Path(
                values[
                    "R829_LIFT_HEAD"
                ]
            ).resolve(),
            swing_checkpoint=Path(
                values["R829_SWING"]
            ).resolve(),
            recovery_checkpoint=Path(
                values[
                    "R829_RECOVERY"
                ]
            ).resolve(),
            recovery_head_path=Path(
                values[
                    "R829_RECOVERY_HEAD"
                ]
            ).resolve(),
            summary_path=Path(
                values[
                    "R829_SUMMARY"
                ]
            ).resolve(),
            broad_episodes=int(
                os.environ.get(
                    "R829_BROAD_EPISODES",
                    "96",
                )
            ),
            confirm_episodes=int(
                os.environ.get(
                    "R829_CONFIRM_EPISODES",
                    "512",
                )
            ),
            seed=args_cli.seed,
        )

        return

    if os.environ.get(
        "R828_STABILIZE_ONLY",
        "0",
    ) == "1":

        keys = (
            "R828_LIFT",
            "R828_LIFT_HEAD",
            "R828_SWING",
            "R828_RECOVERY",
            "R828_RECOVERY_HEAD",
            "R828_R821",
            "R828_R815",
            "R828_R89",
            "R828_R811PRE",
            "R828_R811POST",
            "R828_SUMMARY",
        )

        values = {
            k:
                os.environ.get(
                    k,
                    "",
                ).strip()

            for k in keys
        }

        missing = [
            k
            for k, v in values.items()
            if not v
        ]

        if missing:
            raise RuntimeError(
                "R828 missing: "
                + ",".join(missing)
            )

        run_r828(
            runner=runner,
            env=env,
            lift_checkpoint=Path(
                values["R828_LIFT"]
            ).resolve(),
            lift_head_path=Path(
                values[
                    "R828_LIFT_HEAD"
                ]
            ).resolve(),
            swing_checkpoint=Path(
                values["R828_SWING"]
            ).resolve(),
            recovery_checkpoint=Path(
                values["R828_RECOVERY"]
            ).resolve(),
            recovery_head_path=Path(
                values[
                    "R828_RECOVERY_HEAD"
                ]
            ).resolve(),
            settle_checkpoints={
                "r821":
                    Path(
                        values[
                            "R828_R821"
                        ]
                    ).resolve(),

                "r815":
                    Path(
                        values[
                            "R828_R815"
                        ]
                    ).resolve(),

                "r89":
                    Path(
                        values[
                            "R828_R89"
                        ]
                    ).resolve(),

                "r811pre":
                    Path(
                        values[
                            "R828_R811PRE"
                        ]
                    ).resolve(),

                "r811post":
                    Path(
                        values[
                            "R828_R811POST"
                        ]
                    ).resolve(),
            },
            summary_path=Path(
                values[
                    "R828_SUMMARY"
                ]
            ).resolve(),
            broad_episodes=int(
                os.environ.get(
                    "R828_BROAD_EPISODES",
                    "128",
                )
            ),
            confirm_episodes=int(
                os.environ.get(
                    "R828_CONFIRM_EPISODES",
                    "512",
                )
            ),
            seed=args_cli.seed,
        )

        return

    if os.environ.get(
        "R827_MODE_CORE_ONLY",
        "0",
    ) == "1":

        required = (
            "R827_LIFT",
            "R827_LIFT_HEAD",
            "R827_SWING",
            "R827_RECOVERY",
            "R827_OUTPUT",
        )

        values = {
            key:
                os.environ.get(
                    key,
                    "",
                ).strip()

            for key in required
        }

        missing = [
            k
            for k, v in values.items()
            if not v
        ]

        if missing:
            raise RuntimeError(
                "R827 missing: "
                + ",".join(missing)
            )

        run_r827(
            runner=runner,
            env=env,
            lift_checkpoint=Path(
                values[
                    "R827_LIFT"
                ]
            ).resolve(),
            lift_head_path=Path(
                values[
                    "R827_LIFT_HEAD"
                ]
            ).resolve(),
            swing_checkpoint=Path(
                values[
                    "R827_SWING"
                ]
            ).resolve(),
            recovery_checkpoint=Path(
                values[
                    "R827_RECOVERY"
                ]
            ).resolve(),
            output_dir=Path(
                values[
                    "R827_OUTPUT"
                ]
            ).resolve(),
            collection_episodes=int(
                os.environ.get(
                    "R827_COLLECTION_EPISODES",
                    "8192",
                )
            ),
            broad_episodes=int(
                os.environ.get(
                    "R827_BROAD_EPISODES",
                    "96",
                )
            ),
            confirm_episodes=int(
                os.environ.get(
                    "R827_CONFIRM_EPISODES",
                    "512",
                )
            ),
            seed=args_cli.seed,
        )

        return

    if os.environ.get(
        "R826_RETRACTION_ONLY",
        "0",
    ) == "1":

        names = (
            "R826_LIFT",
            "R826_HEAD",
            "R826_SWING",
            "R826_R821",
            "R826_R815",
            "R826_R89",
            "R826_R811PRE",
            "R826_R811POST",
            "R826_SUMMARY",
        )

        values = {
            name:
                os.environ.get(
                    name,
                    "",
                ).strip()
            for name in names
        }

        missing = [
            k
            for k, v in values.items()
            if not v
        ]

        if missing:
            raise RuntimeError(
                "R826 missing paths: "
                + ",".join(missing)
            )

        run_r826_sweep(
            runner=runner,
            env=env,
            lift_checkpoint=Path(
                values["R826_LIFT"]
            ).resolve(),
            lift_head_path=Path(
                values["R826_HEAD"]
            ).resolve(),
            swing_checkpoint=Path(
                values["R826_SWING"]
            ).resolve(),
            policy_checkpoints={
                "r821":
                    Path(
                        values["R826_R821"]
                    ).resolve(),

                "r815":
                    Path(
                        values["R826_R815"]
                    ).resolve(),

                "r89":
                    Path(
                        values["R826_R89"]
                    ).resolve(),

                "r811pre":
                    Path(
                        values["R826_R811PRE"]
                    ).resolve(),

                "r811post":
                    Path(
                        values["R826_R811POST"]
                    ).resolve(),
            },
            summary_path=Path(
                values["R826_SUMMARY"]
            ).resolve(),
            sweep_episodes=int(
                os.environ.get(
                    "R826_SWEEP_EPISODES",
                    "96",
                )
            ),
            confirm_episodes=int(
                os.environ.get(
                    "R826_CONFIRM_EPISODES",
                    "512",
                )
            ),
            seed=args_cli.seed,
        )

        return

    if os.environ.get(
        "R825_RECOVERY_ONLY",
        "0",
    ) == "1":

        keys = (
            "R825_LIFT",
            "R825_HEAD",
            "R825_SWING",
            "R825_REC_R811_POST",
            "R825_REC_R811_PRE",
            "R825_REC_R821",
            "R825_REC_R815",
            "R825_SUMMARY",
        )

        values = {
            k:
                os.environ.get(
                    k,
                    "",
                ).strip()
            for k in keys
        }

        missing = [
            k
            for k, v in values.items()
            if not v
        ]

        if missing:
            raise RuntimeError(
                "R825 missing paths: "
                + ",".join(missing)
            )

        run_r825_recovery_sweep(
            runner=runner,
            env=env,
            lift_checkpoint=Path(
                values["R825_LIFT"]
            ).resolve(),
            lift_head_path=Path(
                values["R825_HEAD"]
            ).resolve(),
            swing_checkpoint=Path(
                values["R825_SWING"]
            ).resolve(),
            recovery_checkpoints={
                "r811post":
                    Path(
                        values[
                            "R825_REC_R811_POST"
                        ]
                    ).resolve(),

                "r811pre":
                    Path(
                        values[
                            "R825_REC_R811_PRE"
                        ]
                    ).resolve(),

                "r821":
                    Path(
                        values[
                            "R825_REC_R821"
                        ]
                    ).resolve(),

                "r815":
                    Path(
                        values[
                            "R825_REC_R815"
                        ]
                    ).resolve(),
            },
            summary_path=Path(
                values[
                    "R825_SUMMARY"
                ]
            ).resolve(),
            sweep_episodes=int(
                os.environ.get(
                    "R825_SWEEP_EPISODES",
                    "128",
                )
            ),
            confirm_episodes=int(
                os.environ.get(
                    "R825_CONFIRM_EPISODES",
                    "512",
                )
            ),
            seed=args_cli.seed,
        )

        return

    if os.environ.get(
        "R824_COMPOSITION_ONLY",
        "0",
    ) == "1":

        required = {
            "lift":
                os.environ.get(
                    "R824_LIFT_CHECKPOINT",
                    "",
                ).strip(),

            "head":
                os.environ.get(
                    "R824_LIFT_HEAD",
                    "",
                ).strip(),

            "r821":
                os.environ.get(
                    "R824_POST_R821",
                    "",
                ).strip(),

            "r815":
                os.environ.get(
                    "R824_POST_R815",
                    "",
                ).strip(),

            "r89":
                os.environ.get(
                    "R824_POST_R89",
                    "",
                ).strip(),

            "r811":
                os.environ.get(
                    "R824_POST_R811",
                    "",
                ).strip(),

            "summary":
                os.environ.get(
                    "R824_SUMMARY",
                    "",
                ).strip(),
        }

        missing = [
            key
            for key, value in required.items()
            if not value
        ]

        if missing:
            raise RuntimeError(
                "R824 missing paths: "
                + ",".join(missing)
            )

        run_r824_composition_sweep(
            runner=runner,
            env=env,
            lift_checkpoint=Path(
                required["lift"]
            ).resolve(),
            lift_head_path=Path(
                required["head"]
            ).resolve(),
            post_checkpoints={
                "r821":
                    Path(
                        required["r821"]
                    ).resolve(),

                "r815":
                    Path(
                        required["r815"]
                    ).resolve(),

                "r89":
                    Path(
                        required["r89"]
                    ).resolve(),

                "r811":
                    Path(
                        required["r811"]
                    ).resolve(),
            },
            summary_path=Path(
                required["summary"]
            ).resolve(),
            episodes=int(
                os.environ.get(
                    "R824_EPISODES",
                    "256",
                )
            ),
            seed=args_cli.seed,
        )

        return

    if os.environ.get(
        "R823_MODE_CORE_ONLY",
        "0",
    ) == "1":

        dataset_raw = os.environ.get(
            "R823_DATASET",
            "",
        ).strip()

        output_raw = os.environ.get(
            "R823_OUTPUT_DIR",
            "",
        ).strip()

        if (
            not dataset_raw
            or not output_raw
        ):
            raise RuntimeError(
                "R823 paths required."
            )

        run_r823_mode_core_ab(
            runner=runner,
            env=env,
            checkpoint=source_checkpoint,
            dataset_path=Path(
                dataset_raw
            ).resolve(),
            output_dir=Path(
                output_raw
            ).resolve(),
            evaluation_episodes=int(
                os.environ.get(
                    "R823_EVAL_EPISODES",
                    "512",
                )
            ),
            seed=args_cli.seed,
        )

        return

    if os.environ.get(
        "R820_RESIDUAL_ONLY",
        "0",
    ) == "1":

        bundle_raw = os.environ.get(
            "R820_BUNDLE",
            "",
        ).strip()

        summary_raw = os.environ.get(
            "R820_SUMMARY",
            "",
        ).strip()

        if (
            not bundle_raw
            or not summary_raw
        ):
            raise RuntimeError(
                "R820 output paths required."
            )

        run_r820_lift_residual(
            runner=runner,
            env=env,
            checkpoint=source_checkpoint,
            bundle_path=Path(
                bundle_raw
            ).resolve(),
            summary_path=Path(
                summary_raw
            ).resolve(),
            episodes=int(
                os.environ.get(
                    "R820_EPISODES",
                    "4096",
                )
            ),
            history_steps=int(
                os.environ.get(
                    "R820_HISTORY_STEPS",
                    "8",
                )
            ),
            optimization_steps=int(
                os.environ.get(
                    "R820_OPT_STEPS",
                    "300",
                )
            ),
            evaluation_episodes=int(
                os.environ.get(
                    "R820_EVAL_EPISODES",
                    "1024",
                )
            ),
            seed=args_cli.seed,
        )

        return

    if os.environ.get(
        "R817_ATTR_ONLY",
        "0",
    ) == "1":

        summary_raw = os.environ.get(
            "R817_SUMMARY",
            "",
        ).strip()

        if not summary_raw:
            raise RuntimeError(
                "R817_SUMMARY is required."
            )

        episodes = int(
            os.environ.get(
                "R817_EPISODES",
                "2048",
            )
        )

        run_r817_action_attribution(
            runner=runner,
            env=env,
            checkpoint=source_checkpoint,
            summary_path=Path(
                summary_raw
            ).resolve(),
            episodes=episodes,
            seed=args_cli.seed,
        )

        return

    if os.environ.get(
        "R814_AUDIT_ONLY",
        "0",
    ) == "1":

        name = os.environ.get(
            "R814_CHECKPOINT_NAME",
            "unknown",
        ).strip()

        summary_raw = os.environ.get(
            "R814_SUMMARY",
            "",
        ).strip()

        if not summary_raw:
            raise RuntimeError(
                "R814_SUMMARY is required."
            )

        episodes = int(
            os.environ.get(
                "R814_EPISODES",
                "256",
            )
        )

        run_r814_root_audit(
            runner=runner,
            env=env,
            checkpoint=source_checkpoint,
            checkpoint_name=name,
            summary_path=Path(
                summary_raw
            ).resolve(),
            episodes=episodes,
            seed=args_cli.seed,
        )

        return

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
