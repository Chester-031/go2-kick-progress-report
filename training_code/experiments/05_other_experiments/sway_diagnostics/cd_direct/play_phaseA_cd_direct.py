"""Train a high-level soccer PPO policy over a frozen Go2 locomotion policy."""

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

    parser.add_argument(
        "--preflight_only",
        action="store_true",
        help="Run the environment/interface preflight and exit before PPO.",
    )

    # PHASEA_DETERMINISTIC_VIDEO_EVAL_V2
    parser.add_argument(
        "--eval_checkpoint",
        type=str,
        default="",
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
        default=30,
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = build_parser()
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

print("[PHASEA_BOOT] app_launcher_returned", flush=True)

import isaac_go2_low_level_rl.tasks  # noqa: E402,F401

# PHASEA_SOCCER_REGISTRATION_IMPORT_V1
import isaac_go2_low_level_rl.tasks.manager_based.soccer # noqa: E402,F401

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import (  # noqa: E402
    load_cfg_from_registry,
    parse_env_cfg,
)
from rsl_rl.runners import OnPolicyRunner  # noqa: E402


def instantiate_cfg(value: Any) -> Any:
    return value() if isinstance(value, type) else value


def extract_observation(value: Any) -> torch.Tensor:
    """Extract the policy tensor from tensors, mappings, or TensorDicts."""

    original_type = type(value)
    visited_types: list[str] = []

    # Isaac Lab/RSL-RL versions may return:
    #   Tensor
    #   tuple(TensorDict, ...)
    #   TensorDict({"policy": Tensor})
    #   nested TensorDict({"policy": {"obs": Tensor}})
    for _ in range(8):
        visited_types.append(
            f"{type(value).__module__}.{type(value).__name__}"
        )

        if torch.is_tensor(value):
            # Keep persistent environment observations as ordinary,
            # detached tensors. This also breaks aliases to TensorDict
            # or simulator-owned storage.
            return value.detach().clone()

        if isinstance(value, tuple):
            if not value:
                break
            value = value[0]
            continue

        keys_method = getattr(value, "keys", None)

        if callable(keys_method):
            try:
                keys = list(keys_method())
            except TypeError:
                # Compatibility with TensorDict versions whose keys()
                # method exposes optional keyword-only arguments.
                keys = list(
                    keys_method(
                        include_nested=False,
                        leaves_only=False,
                    )
                )

            preferred_keys = (
                "policy",
                ("policy",),
                "observations",
                ("observations",),
                "obs",
                ("obs",),
            )

            selected_key = None

            for candidate in preferred_keys:
                if candidate in keys:
                    selected_key = candidate
                    break

            if selected_key is None and len(keys) == 1:
                selected_key = keys[0]

            if selected_key is not None:
                value = value[selected_key]
                continue

            raise RuntimeError(
                "Observation container has multiple unsupported keys: "
                f"type={type(value)!r}, keys={keys!r}"
            )

        break

    raise RuntimeError(
        "Unable to extract a tensor observation. "
        f"original_type={original_type!r}, "
        f"visited_types={visited_types!r}, "
        f"final_type={type(value)!r}"
    )


# PHASEA_INFERENCE_TENSOR_LIFETIME_V4
# PHASEA_LOW_POLICY_SHAPE_CONTRACT_V3
def prepare_low_policy_input(
    value: Any,
    *,
    num_envs: int,
    expected_features: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Create the observation container expected by RSL-RL act_inference.

    The frozen actor was configured with:
        obs_groups["policy"] == ["policy"]

    Consequently, it expects a mapping containing a two-dimensional
    policy tensor rather than the bare tensor itself.
    """

    tensor = extract_observation(value)

    if tensor.ndim == 1 and num_envs == 1:
        tensor = tensor.unsqueeze(0)

    if tensor.ndim != 2:
        raise RuntimeError(
            "Low-level policy observation must be rank 2. "
            f"received_shape={tuple(tensor.shape)}"
        )

    if tensor.shape[0] != num_envs:
        raise RuntimeError(
            "Low-level policy observation batch mismatch. "
            f"expected_num_envs={num_envs}, "
            f"received_shape={tuple(tensor.shape)}"
        )

    if expected_features is not None and tensor.shape[1] != expected_features:
        raise RuntimeError(
            "Low-level policy observation feature mismatch. "
            f"expected_features={expected_features}, "
            f"received_shape={tuple(tensor.shape)}"
        )

    if not torch.isfinite(tensor).all():
        raise RuntimeError(
            "Non-finite values detected in low-level observations."
        )

    # A normal dictionary is sufficient for actor_critic.act_inference:
    # it performs obs["policy"] and then concatenates the selected tensors.
    return tensor, {"policy": tensor}


def normalize_done_tensor(
    value: Any,
    *,
    num_envs: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Normalize scalar, [N], or [N, K] termination data to bool [N]."""

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
            f"expected_num_envs={num_envs}, "
            f"received_shape={tuple(tensor.shape)}"
        )

    if tensor.ndim == 1:
        return tensor

    return tensor.reshape(num_envs, -1).any(dim=1)


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



# PHASEA_RSLRL_TENSORDICT_ADAPTER_V5
class RslRlHighLevelEnvAdapter:
    """Expose the soccer environment using RSL-RL 3.x VecEnv semantics."""

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

    def _pack_observations(self, value: Any) -> TensorDict:
        observation = extract_observation(value).detach().clone()

        if observation.ndim != 2:
            raise RuntimeError(
                "High-level runner observation must be rank 2. "
                f"received_shape={tuple(observation.shape)}"
            )

        expected_shape = (
            self.num_envs,
            self.env.num_obs,
        )

        if observation.shape != expected_shape:
            raise RuntimeError(
                "High-level runner observation shape mismatch. "
                f"expected={expected_shape} "
                f"received={tuple(observation.shape)}"
            )

        if not torch.isfinite(observation).all():
            raise RuntimeError(
                "High-level runner observation contains non-finite values."
            )

        if torch.is_inference(observation):
            raise RuntimeError(
                "High-level runner observation remained an inference tensor."
            )

        return TensorDict(
            {"policy": observation},
            batch_size=[self.num_envs],
            device=observation.device,
        )

    @torch.inference_mode(False)
    def get_observations(self) -> TensorDict:
        return self._pack_observations(
            self.env.get_observations()
        )

    @torch.inference_mode(False)
    def reset(self) -> TensorDict:
        return self._pack_observations(
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
        if not torch.is_tensor(actions):
            raise RuntimeError(
                "RSL-RL high-level actions must be a tensor. "
                f"received_type={type(actions)!r}"
            )

        # RSL-RL samples actions inside inference_mode().
        # Clone under inference_mode(False) before passing them into
        # the mutable simulator.
        safe_actions = actions.detach().clone()

        if torch.is_inference(safe_actions):
            raise RuntimeError(
                "High-level actions remained an inference tensor "
                "after clone."
            )

        observations, rewards, dones, extras = self.env.step(
            safe_actions
        )

        rewards = rewards.detach().clone()
        dones = dones.detach().clone()

        return (
            self._pack_observations(observations),
            rewards,
            dones,
            extras,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


class HighLevelSoccerVecEnv:
    """RSL-RL vector environment driving a frozen low-level policy."""

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

        expected_observation_shape = (
            self.num_envs,
            17,
        )

        if observation.shape != expected_observation_shape:
            raise RuntimeError(
                "High-level observation shape mismatch. "
                f"expected={expected_observation_shape}, "
                f"received={tuple(observation.shape)}"
            )

        if not torch.isfinite(observation).all():
            raise RuntimeError(
                "Non-finite values detected in high-level observations."
            )

        return observation

    def _action_to_command(
        self,
        action: torch.Tensor,
    ) -> torch.Tensor:
        action = torch.clamp(action, -1.0, 1.0)

        command = torch.empty_like(action)

        # Zero actor output maps to zero command.
        command[:, 0] = torch.where(
            action[:, 0] >= 0.0,
            0.65 * action[:, 0],
            0.25 * action[:, 0],
        )
        command[:, 1] = 0.35 * action[:, 1]
        command[:, 2] = 0.90 * action[:, 2]

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
                "High-level actions must be a tensor. "
                f"received_type={type(actions)!r}"
            )

        if actions.ndim == 1 and self.num_envs == 1:
            actions = actions.unsqueeze(0)

        if actions.ndim != 2:
            raise RuntimeError(
                "High-level actions must be rank 2. "
                f"received_shape={tuple(actions.shape)}"
            )

        if actions.shape != (self.num_envs, 3):
            raise RuntimeError(
                "High-level action shape mismatch. "
                f"expected={(self.num_envs, 3)}, "
                f"received={tuple(actions.shape)}"
            )

        actions = torch.clamp(
            actions.to(self.device),
            -1.0,
            1.0,
        )

        command = self._action_to_command(actions)

        base_done = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )

        with torch.no_grad():
            for _ in range(self.low_level_steps_per_action):
                self._set_command(command)

                # Recompute the low-level observation after installing
                # the current high-level command.
                self._low_obs = extract_observation(
                    self.low_env.get_observations()
                )

                expected_low_obs_features = getattr(
                    self,
                    "_low_obs_features",
                    None,
                )

                (
                    self._low_obs,
                    low_policy_input,
                ) = prepare_low_policy_input(
                    self._low_obs,
                    num_envs=self.num_envs,
                    expected_features=expected_low_obs_features,
                )

                if expected_low_obs_features is None:
                    self._low_obs_features = int(
                        self._low_obs.shape[1]
                    )

                if not getattr(
                    self,
                    "_low_policy_contract_logged",
                    False,
                ):
                    print(
                        "[PHASEA_PREFLIGHT] "
                        "low_policy_input_container=dict "
                        f"policy_shape={tuple(self._low_obs.shape)} "
                        f"policy_dtype={self._low_obs.dtype}",
                        flush=True,
                    )
                    self._low_policy_contract_logged = True

                low_actions = self.low_policy(low_policy_input)
                low_actions = low_actions.detach().clone()

                if torch.is_inference(low_actions):
                    raise RuntimeError(
                        "Low-level policy output remained an "
                        "inference tensor after clone."
                    )

                if not torch.is_tensor(low_actions):
                    raise RuntimeError(
                        "Frozen low-level policy returned a non-tensor. "
                        f"received_type={type(low_actions)!r}"
                    )

                if low_actions.ndim == 1 and self.num_envs == 1:
                    low_actions = low_actions.unsqueeze(0)

                if low_actions.ndim != 2:
                    raise RuntimeError(
                        "Low-level actions must be rank 2. "
                        f"received_shape={tuple(low_actions.shape)}"
                    )

                if low_actions.shape[0] != self.num_envs:
                    raise RuntimeError(
                        "Low-level action batch mismatch. "
                        f"expected_num_envs={self.num_envs}, "
                        f"received_shape={tuple(low_actions.shape)}"
                    )

                expected_low_action_features = getattr(
                    self,
                    "_low_action_features",
                    None,
                )

                if expected_low_action_features is None:
                    self._low_action_features = int(
                        low_actions.shape[1]
                    )
                elif (
                    low_actions.shape[1]
                    != expected_low_action_features
                ):
                    raise RuntimeError(
                        "Low-level action feature count changed. "
                        f"expected={expected_low_action_features}, "
                        f"received={low_actions.shape[1]}"
                    )

                if not torch.isfinite(low_actions).all():
                    raise RuntimeError(
                        "Frozen low-level policy produced non-finite actions."
                    )

                result = self.low_env.step(low_actions)

                low_obs, _, dones, _ = normalize_step_result(
                    result
                )

                self._low_obs = extract_observation(low_obs)

                base_done |= normalize_done_tensor(
                    dones,
                    num_envs=self.num_envs,
                    device=self.device,
                )

        state = self._state()

        goal_progress = (
            self.previous_ball_goal_distance
            - state["ball_goal_distance"]
        )
        goal_progress = torch.clamp(
            goal_progress,
            -0.25,
            0.25,
        )

        robot_ball_progress = (
            self.previous_robot_ball_distance
            - state["robot_ball_distance"]
        )
        robot_ball_progress = torch.clamp(
            robot_ball_progress,
            -0.20,
            0.20,
        )

        staging_reward = torch.exp(
            -1.5 * state["staging_distance"].square()
        )

        proximity_reward = torch.exp(
            -2.0
            * (state["robot_ball_distance"] - 0.48).square()
        )

        ball_velocity_reward = torch.clamp(
            state["ball_velocity_toward_goal"],
            -1.5,
            2.0,
        )

        useful_approach = robot_ball_progress * torch.clamp(
            state["robot_ball_distance"] - 0.45,
            min=0.0,
            max=2.0,
        )

        action_rate_penalty = (
            actions - self.previous_action
        ).square().sum(dim=1)

        ball_lateral_from_center = torch.abs(
            state["ball_position"][:, 1] - state["origin_y"]
        )

        wall_stagnation = (
            (ball_lateral_from_center > 2.05)
            & (
                torch.linalg.vector_norm(
                    state["ball_velocity_world"][:, :2],
                    dim=1,
                )
                < 0.08
            )
        )

        reward = (
            5.0 * goal_progress
            + 0.16 * ball_velocity_reward
            + 0.08 * staging_reward
            + 0.05 * proximity_reward
            + 0.60 * useful_approach
            - 0.010 * action_rate_penalty
            - 0.05 * wall_stagnation.float()
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

        reward = reward + 10.0 * goal_scored.float()
        reward = reward - 10.0 * own_goal.float()

        # A base termination is normally a fall or illegal contact.
        reward = torch.where(
            base_done,
            torch.full_like(reward, -2.0),
            reward,
        )

        self.episode_length_buf += 1

        time_out = (
            self.episode_length_buf
            >= self.max_episode_length
        )

        dones = base_done | goal_scored | own_goal | time_out

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

        self.rew_buf = torch.clamp(
            reward,
            -10.0,
            10.0,
        )
        self.reset_buf = dones

        if done_ids.numel() > 0:
            self._reset_idx(done_ids)

        self.obs_buf = self._build_observation()

        self.vector_steps += 1

        if self.vector_steps % 100 == 0:
            print(
                "[PHASEA_PROGRESS] "
                f"vector_step={self.vector_steps} "
                f"mean_reward={float(self.rew_buf.mean()):.4f} "
                f"goal_fraction={float(goal_scored.float().mean()):.4f} "
                f"mean_ball_goal={float(state['ball_goal_distance'].mean()):.3f} "
                f"difficulty={min(1.0, self.total_resets / max(1.0, float(self.num_envs * 300))):.3f}",
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


def create_runner_cfg(
    *,
    base_cfg: Any,
    run_name: str,
) -> dict[str, Any]:
    cfg = base_cfg.to_dict()

    cfg["num_steps_per_env"] = 24
    cfg["save_interval"] = 100
    cfg["experiment_name"] = (
        "go2_soccer_high_level_phaseA"
    )
    cfg["run_name"] = run_name

    cfg["policy"]["init_noise_std"] = 0.8
    cfg["policy"]["actor_hidden_dims"] = [256, 256, 128]
    cfg["policy"]["critic_hidden_dims"] = [256, 256, 128]

    cfg["algorithm"]["entropy_coef"] = 0.02
    cfg["algorithm"]["learning_rate"] = 5.0e-4
    cfg["algorithm"]["num_learning_epochs"] = 5
    cfg["algorithm"]["num_mini_batches"] = 4
    cfg["algorithm"]["gamma"] = 0.99
    cfg["algorithm"]["lam"] = 0.95
    cfg["algorithm"]["desired_kl"] = 0.01

    return cfg


def main() -> None:
    if args_cli.num_envs <= 0:
        raise ValueError("--num_envs must be positive.")

    if args_cli.max_iterations <= 0:
        raise ValueError("--max_iterations must be positive.")

    checkpoint = Path(
        args_cli.low_level_checkpoint
    ).resolve()
    log_dir = Path(args_cli.log_dir).resolve()

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Low-level checkpoint not found: {checkpoint}"
        )

    log_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args_cli.seed)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )

    # Keep the base locomotion timeout longer than a high-level
    # Phase-A episode.
    env_cfg.episode_length_s = 40.0

    try:
        env_cfg.observations.policy.enable_corruption = False
    except AttributeError:
        pass

    # Phase A focuses on learning soccer control rather than
    # recovery from artificial pushes.
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

    print(
        f"[PHASEA_BOOT] task={args_cli.task}",
        flush=True,
    )
    print(
        f"[PHASEA_BOOT] num_envs={args_cli.num_envs}",
        flush=True,
    )
    print(
        f"[PHASEA_BOOT] iterations={args_cli.max_iterations}",
        flush=True,
    )
    print(
        f"[PHASEA_BOOT] low_level_checkpoint={checkpoint}",
        flush=True,
    )
    print(
        "[PHASEA_BOOT] creating_soccer_environment",
        flush=True,
    )

    gym_env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array",
    )

    low_env = RslRlVecEnvWrapper(
        gym_env,
        clip_actions=low_agent_cfg.clip_actions,
    )

    print(
        "[PHASEA_BOOT] creating_frozen_low_level_runner",
        flush=True,
    )

    low_runner = OnPolicyRunner(
        low_env,
        low_agent_cfg.to_dict(),
        log_dir=None,
        device=low_agent_cfg.device,
    )

    low_runner.load(
        str(checkpoint),
        load_optimizer=False,
    )

    low_policy = low_runner.get_inference_policy(
        device=low_env.device,
    )

    print(
        "[PHASEA_BOOT] frozen_low_level_policy_loaded",
        flush=True,
    )

    high_env = HighLevelSoccerVecEnv(
        low_env=low_env,
        low_policy=low_policy,
        low_level_steps_per_action=(
            args_cli.low_level_steps_per_action
        ),
        episode_steps=args_cli.high_level_episode_steps,
        goal_line_x=args_cli.goal_line_x,
        goal_center_x=args_cli.goal_center_x,
        goal_half_width=args_cli.goal_half_width,
        seed=args_cli.seed,
    )

    print(
        "[PHASEA_BOOT] running_in_allocation_preflight",
        flush=True,
    )

    with torch.no_grad():
        test_actions = torch.zeros(
            high_env.num_envs,
            high_env.num_actions,
            device=high_env.device,
        )

        for preflight_step in range(4):
            obs, reward, dones, _ = high_env.step(
                test_actions
            )

            if not torch.isfinite(obs).all():
                raise RuntimeError(
                    "Preflight produced non-finite observations."
                )

            if not torch.isfinite(reward).all():
                raise RuntimeError(
                    "Preflight produced non-finite rewards."
                )

            print(
                "[PHASEA_PREFLIGHT] "
                f"step={preflight_step} "
                f"obs_shape={tuple(obs.shape)} "
                f"reward_mean={float(reward.mean()):.5f} "
                f"done_count={int(dones.sum())}",
                flush=True,
            )

    high_env.reset()

    print("HIGH_LEVEL_PHASE_A_PREFLIGHT_OK", flush=True)

    (log_dir / "HIGH_LEVEL_PHASE_A_PREFLIGHT_OK").touch()

    if args_cli.preflight_only:
        preflight_summary = {
            "task": args_cli.task,
            "low_level_checkpoint": str(checkpoint),
            "num_envs": args_cli.num_envs,
            "observation_shape": list(high_env.obs_buf.shape),
            "high_level_action_dim": high_env.num_actions,
            "low_level_steps_per_action": (
                args_cli.low_level_steps_per_action
            ),
            "status": "passed",
        }

        (log_dir / "preflight_summary.json").write_text(
            json.dumps(preflight_summary, indent=2),
            encoding="utf-8",
        )

        print(
            "HIGH_LEVEL_PHASE_A_PREFLIGHT_ONLY_COMPLETE",
            flush=True,
        )

        high_env.close()
        return

    runner_cfg = create_runner_cfg(
        base_cfg=low_agent_cfg,
        run_name=args_cli.run_name,
    )


    # Explicit RSL-RL observation-set contract.
    runner_cfg["obs_groups"] = {
        "policy": ["policy"],
        "critic": ["policy"],
    }

    # Keep the serialized runner config synchronized with the CLI.
    runner_cfg["max_iterations"] = args_cli.max_iterations

    contract = {
        "task": args_cli.task,
        "low_level_checkpoint": str(checkpoint),
        "low_level_frozen": True,
        "high_level_actions": ["vx", "vy", "wz"],
        "num_envs": args_cli.num_envs,
        "max_iterations": args_cli.max_iterations,
        "low_level_steps_per_action": (
            args_cli.low_level_steps_per_action
        ),
        "high_level_episode_steps": (
            args_cli.high_level_episode_steps
        ),
        "goal_line_x": args_cli.goal_line_x,
        "goal_center_x": args_cli.goal_center_x,
        "goal_half_width": args_cli.goal_half_width,
        "runner_cfg": runner_cfg,
    }

    (log_dir / "phaseA_contract.json").write_text(
        json.dumps(contract, indent=2),
        encoding="utf-8",
    )

    print(
        "[PHASEA_BOOT] creating_high_level_runner",
        flush=True,
    )

    runner_env = RslRlHighLevelEnvAdapter(high_env)
    runner_obs = runner_env.get_observations()

    print(
        "[PHASEA_RUNNER_PREFLIGHT] "
        f"obs_container={type(runner_obs).__name__} "
        f"obs_keys={list(runner_obs.keys())} "
        f"policy_shape={tuple(runner_obs['policy'].shape)}",
        flush=True,
    )

    high_runner = OnPolicyRunner(
        runner_env,
        runner_cfg,
        log_dir=str(log_dir),
        device=args_cli.device,
    )

    print(
        "[PHASEA_RUNNER_PREFLIGHT] high_level_runner_constructed",
        flush=True,
    )

    if hasattr(high_runner, "add_git_repo_to_log"):
        try:
            high_runner.add_git_repo_to_log(__file__)
        except Exception as error:
            print(
                "[PHASEA_WARNING] "
                f"add_git_repo_to_log failed: {error}",
                flush=True,
            )


    if args_cli.eval_checkpoint:
        from pathlib import Path as _EvalPath

        import imageio.v2 as _imageio
        import numpy as _np

        eval_checkpoint = _EvalPath(
            args_cli.eval_checkpoint
        ).expanduser().resolve()

        eval_video = _EvalPath(
            args_cli.eval_video
        ).expanduser().resolve()

        if not eval_checkpoint.is_file():
            raise RuntimeError(
                "High-level checkpoint not found: "
                f"{eval_checkpoint}"
            )

        if not args_cli.eval_video:
            raise RuntimeError(
                "--eval_video is required in evaluation mode."
            )

        if args_cli.eval_steps <= 0:
            raise RuntimeError(
                "--eval_steps must be positive."
            )

        if args_cli.eval_fps <= 0:
            raise RuntimeError(
                "--eval_fps must be positive."
            )

        eval_video.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            "[PHASEA_EVAL] "
            f"loading_high_level_checkpoint={eval_checkpoint}",
            flush=True,
        )

        try:
            high_runner.load(
                str(eval_checkpoint),
                load_optimizer=False,
            )
        except TypeError:
            high_runner.load(
                str(eval_checkpoint)
            )

        policy = high_runner.get_inference_policy(
            device=args_cli.device
        )

        observations = runner_env.reset()
        render_env = high_env.low_env

        writer = _imageio.get_writer(
            str(eval_video),
            fps=args_cli.eval_fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
            ffmpeg_log_level="error",
        )

        frame_count = 0
        reward_total = 0.0
        done_total = 0

        print(
            "[PHASEA_EVAL] deterministic_policy_started "
            f"steps={args_cli.eval_steps} "
            f"fps={args_cli.eval_fps}",
            flush=True,
        )

        try:
            for step_index in range(
                args_cli.eval_steps
            ):
                with torch.no_grad():
                    actions = policy(
                        observations
                    )

                (
                    observations,
                    rewards,
                    dones,
                    extras,
                ) = runner_env.step(actions)

                reward_total += float(
                    rewards.mean()
                )
                done_total += int(
                    dones.sum()
                )

                frame = _np.zeros((16, 16, 3), dtype=_np.uint8)

                if isinstance(frame, dict):
                    if not frame:
                        raise RuntimeError(
                            "Rendered-frame dictionary is empty."
                        )

                    frame = next(
                        iter(frame.values())
                    )

                if isinstance(frame, (list, tuple)):
                    if not frame:
                        raise RuntimeError(
                            "Rendered-frame sequence is empty."
                        )

                    frame = frame[0]

                if frame is None:
                    raise RuntimeError(
                        "Environment render() returned None. "
                        "Verify render_mode and --enable_cameras."
                    )

                if torch.is_tensor(frame):
                    frame = (
                        frame
                        .detach()
                        .cpu()
                        .numpy()
                    )

                frame = _np.asarray(frame)

                if frame.ndim == 4:
                    frame = frame[0]

                if frame.ndim != 3:
                    raise RuntimeError(
                        "Unexpected rendered-frame shape: "
                        f"{frame.shape}"
                    )

                if frame.shape[-1] == 4:
                    frame = frame[..., :3]

                if frame.dtype != _np.uint8:
                    maximum = float(
                        _np.nanmax(frame)
                    )

                    if maximum <= 1.0:
                        frame = frame * 255.0

                    frame = _np.clip(
                        frame,
                        0.0,
                        255.0,
                    ).astype(_np.uint8)

                writer.append_data(frame)
                frame_count += 1

                if (
                    step_index == 0
                    or (step_index + 1) % 100 == 0
                    or step_index + 1
                    == args_cli.eval_steps
                ):
                    print(
                        "[PHASEA_EVAL] "
                        f"step={step_index + 1}/"
                        f"{args_cli.eval_steps} "
                        f"reward_mean="
                        f"{float(rewards.mean()):.5f} "
                        f"done_count="
                        f"{int(dones.sum())}",
                        flush=True,
                    )
        finally:
            writer.close()

        if not eval_video.is_file():
            raise RuntimeError(
                "Evaluation video was not created."
            )

        if eval_video.stat().st_size <= 0:
            raise RuntimeError(
                "Evaluation video is empty."
            )

        print(
            "[PHASEA_EVAL] "
            f"frames_written={frame_count} "
            f"average_step_reward="
            f"{reward_total / max(frame_count, 1):.5f} "
            f"total_done_count={done_total}",
            flush=True,
        )

        print(
            f"PHASEA_EVAL_VIDEO={eval_video}",
            flush=True,
        )
        print(
            "PHASEA_EVAL_VIDEO_OK",
            flush=True,
        )

        high_env.close()
        return

    print(
        "[PHASEA_TRAIN] high_level_ppo_started",
        flush=True,
    )

    high_runner.learn(
        num_learning_iterations=args_cli.max_iterations,
        init_at_random_ep_len=True,
    )

    final_checkpoint = log_dir / "model_final.pt"

    if hasattr(high_runner, "save"):
        high_runner.save(str(final_checkpoint))

    checkpoints = sorted(
        log_dir.glob("model_*.pt"),
        key=lambda path: path.stat().st_mtime,
    )

    if not checkpoints:
        raise RuntimeError(
            "Training completed without producing a checkpoint."
        )

    latest_checkpoint = checkpoints[-1]

    (log_dir / "latest_checkpoint.txt").write_text(
        str(latest_checkpoint) + "\n",
        encoding="utf-8",
    )

    (log_dir / "HIGH_LEVEL_PHASE_A_OK").touch()

    print("HIGH_LEVEL_PHASE_A_TRAINING_COMPLETE", flush=True)
    print(
        f"HIGH_LEVEL_LATEST_CHECKPOINT={latest_checkpoint}",
        flush=True,
    )
    print(f"HIGH_LEVEL_LOG_DIR={log_dir}", flush=True)

    high_env.close()


# PHASEA_SKIP_CLOSE_AFTER_FAILURE_V1
_phasea_failed = False


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


try:
    main()
except BaseException as error:
    _phasea_failed = True
    print(
        f"[PHASEA_FATAL] {type(error).__name__}: {error}",
        flush=True,
    )
    raise
finally:
    if _phasea_failed:
        # A failed Isaac/PhysX startup can hang indefinitely inside
        # simulation_app.close(). The process exit will release resources.
        print(
            "[PHASEA_BOOT] skipping_simulation_close_after_failure",
            flush=True,
        )
    else:
        print(
            "[PHASEA_BOOT] closing_simulation_app",
            flush=True,
        )
        simulation_app.close()
