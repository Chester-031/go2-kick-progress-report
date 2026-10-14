"""A small deterministic stand-in for the Isaac Lab scene, for CPU tests.

``robot_soccer.environment`` talks to Isaac Lab through a handful of objects:
the scene's robot, ball and two contact sensors, the command and action
managers, the observation manager and ``step``/``_reset_idx``.  This module
fakes exactly those, with toy kinematics in place of PhysX: the robot moves
at the commanded body-frame velocity, the ball is shoved when the robot is
close and rolls with friction, and falls and joint-limit faults fire on a
fixed (env, tick) schedule.  Everything is deterministic, so two versions of
the environment wrapper can be run on the same fake and their episodes
compared, and the wrapper's own Python overhead can be timed with physics
that costs almost nothing.

``install_isaaclab_stub()`` must run before ``robot_soccer.environment`` is
imported: the environment imports the guarded joint action, which subclasses
an Isaac Lab class.
"""

from __future__ import annotations

import math
import sys
import types
from types import SimpleNamespace
from typing import Any

import torch


def install_isaaclab_stub() -> None:
    """Register a fake ``isaaclab...joint_actions`` module (idempotent)."""

    name = "isaaclab.envs.mdp.actions.joint_actions"
    if name in sys.modules:
        return

    class JointPositionAction:  # pylint: disable=too-few-public-methods
        def __init__(self, cfg: Any, env: Any) -> None:
            del cfg, env

        def reset(self, env_ids: Any = None) -> None:
            del env_ids

        def process_actions(self, actions: torch.Tensor) -> None:
            self._processed_actions = actions.clone()

    parts = name.split(".")
    for index in range(1, len(parts) + 1):
        module_name = ".".join(parts[:index])
        if module_name not in sys.modules:
            sys.modules[module_name] = types.ModuleType(module_name)
    sys.modules[name].JointPositionAction = JointPositionAction  # type: ignore[attr-defined]


ROBOT_Z_M = 0.30
BALL_Z_M = 0.11
ENV_SPACING_M = 12.0
PUSH_RADIUS_M = 0.35
FALL_PERIOD_TICKS = 2_311
JOINT_FAULT_PERIOD_TICKS = 3_137


def _rows(env_ids: Any, count: int) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(count)
    return torch.as_tensor(env_ids, dtype=torch.long).flatten().cpu()


class _Articulation:
    def __init__(self, num_envs: int, body_names: tuple[str, ...], device: str) -> None:
        self.body_names = list(body_names)
        quaternion = torch.zeros(num_envs, 4, device=device)
        quaternion[:, 0] = 1.0
        self.data = SimpleNamespace(
            root_pos_w=torch.zeros(num_envs, 3, device=device),
            root_quat_w=quaternion,
            root_lin_vel_w=torch.zeros(num_envs, 3, device=device),
            root_ang_vel_w=torch.zeros(num_envs, 3, device=device),
            default_joint_pos=torch.zeros(num_envs, 12, device=device),
            joint_pos=torch.zeros(num_envs, 12, device=device),
            joint_pos_limits=torch.stack(
                (
                    torch.full((num_envs, 12), -1.0, device=device),
                    torch.full((num_envs, 12), 1.0, device=device),
                ),
                dim=-1,
            ),
            default_root_state=torch.zeros(num_envs, 13, device=device),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]], device=device).repeat(num_envs, 1),
        )

    def write_root_pose_to_sim(self, pose: torch.Tensor, env_ids: Any = None) -> None:
        rows = _rows(env_ids, self.data.root_pos_w.shape[0]).to(pose.device)
        self.data.root_pos_w[rows] = pose[:, :3]
        self.data.root_quat_w[rows] = pose[:, 3:7]

    def write_root_velocity_to_sim(self, velocity: torch.Tensor, env_ids: Any = None) -> None:
        rows = _rows(env_ids, self.data.root_pos_w.shape[0]).to(velocity.device)
        self.data.root_lin_vel_w[rows] = velocity[:, :3]
        self.data.root_ang_vel_w[rows] = velocity[:, 3:6]

    def write_joint_state_to_sim(
        self, position: torch.Tensor, velocity: torch.Tensor, env_ids: Any = None
    ) -> None:
        del velocity
        rows = _rows(env_ids, self.data.root_pos_w.shape[0]).to(position.device)
        self.data.joint_pos[rows] = position


class _Sensor:
    def __init__(self, data: SimpleNamespace, body_names: tuple[str, ...]) -> None:
        self.data = data
        self.body_names = list(body_names)

    def reset(self, env_ids: Any = None) -> None:
        for value in vars(self.data).values():
            rows = _rows(env_ids, value.shape[0]).to(value.device)
            value[rows] = 0.0


class _Manager:
    def __init__(self, terms: dict[str, Any], **extra: Any) -> None:
        self._terms = terms
        for key, value in extra.items():
            setattr(self, key, value)

    def get_term(self, name: str) -> Any:
        return self._terms[name]


class FakeSoccerBaseEnv:
    """The unwrapped ManagerBasedEnv surface the soccer wrapper uses."""

    def __init__(self, num_envs: int, *, device: str = "cpu") -> None:
        from robot_soccer.contact import ROBOT_BODY_NAMES
        from robot_soccer.native_safe_joint_action import NativeSafeJointPositionAction

        self.num_envs = int(num_envs)
        self.device = device
        self.cfg = SimpleNamespace(sim=SimpleNamespace(dt=0.005), decimation=4)
        self.tick = 0
        side = math.ceil(math.sqrt(self.num_envs))
        index = torch.arange(self.num_envs, device=device)
        self.scene_origins = torch.zeros(self.num_envs, 3, device=device)
        self.scene_origins[:, 0] = (index % side).float() * ENV_SPACING_M
        self.scene_origins[:, 1] = (index // side).float() * ENV_SPACING_M
        # Isaac Lab centres the grid of env origins on the world origin.
        self.scene_origins[:, :2] -= self.scene_origins[:, :2].mean(dim=0)

        robot = _Articulation(self.num_envs, ROBOT_BODY_NAMES, device)
        robot.data.root_pos_w[:, :2] = self.scene_origins[:, :2]
        robot.data.root_pos_w[:, 2] = ROBOT_Z_M
        ball = _Articulation(self.num_envs, ("ball",), device)
        ball.data.default_root_state[:, 2] = BALL_Z_M
        ball.data.default_root_state[:, 3] = 1.0
        ball.data.root_pos_w[:] = ball.data.default_root_state[:, :3] + self.scene_origins
        ball_contact = _Sensor(
            SimpleNamespace(
                force_matrix_w_history=torch.zeros(
                    self.num_envs, 4, 1, len(ROBOT_BODY_NAMES), 3, device=device
                )
            ),
            ("ball",),
        )
        base_contact = _Sensor(
            SimpleNamespace(net_forces_w_history=torch.zeros(self.num_envs, 4, 2, 3, device=device)),
            ("base", "FL_foot"),
        )
        self.scene = _Scene(
            {"robot": robot, "ball": ball, "robot_ball_contact": ball_contact, "contact_forces": base_contact},
            self.scene_origins,
        )
        self.ball_velocity_xy = torch.zeros(self.num_envs, 2, device=device)

        action_term = object.__new__(NativeSafeJointPositionAction)
        action_term._last_guard_correction = torch.zeros(self.num_envs, 12, device=device)
        action_term._unlimited_processed_actions = torch.zeros(self.num_envs, 12, device=device)
        self.action_manager = _Manager(
            {"joint_pos": action_term},
            prev_action=torch.zeros(self.num_envs, 12, device=device),
        )
        self.command_tensor = torch.zeros(self.num_envs, 3, device=device)
        self.command_manager = _Manager(
            {"base_velocity": SimpleNamespace(vel_command_b=self.command_tensor)}
        )
        self.observation_manager = SimpleNamespace(compute=self._observations)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=device)

    @property
    def unwrapped(self) -> "FakeSoccerBaseEnv":
        return self

    def _observations(self) -> dict[str, torch.Tensor]:
        robot = self.scene["robot"].data
        quaternion = robot.root_quat_w
        yaw = torch.atan2(
            2.0 * (quaternion[:, 0] * quaternion[:, 3] + quaternion[:, 1] * quaternion[:, 2]),
            1.0 - 2.0 * (quaternion[:, 2].square() + quaternion[:, 3].square()),
        )
        observation = torch.zeros(self.num_envs, 48, device=self.device)
        observation[:, 0] = torch.cos(yaw)
        observation[:, 1] = torch.sin(yaw)
        observation[:, 9:12] = self.command_tensor
        observation[:, 12:24] = robot.joint_pos
        observation[:, 36:48] = self.action_manager.prev_action
        return {"policy": observation}

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        rows = env_ids.to(self.device)
        self.action_manager.prev_action[rows] = 0.0
        self.ball_velocity_xy[rows] = 0.0
        self.episode_length_buf[rows] = 0

    def step(self, actions: torch.Tensor) -> tuple[Any, ...]:
        robot = self.scene["robot"].data
        ball = self.scene["ball"].data
        dt = 0.02
        quaternion = robot.root_quat_w
        yaw = torch.atan2(
            2.0 * (quaternion[:, 0] * quaternion[:, 3] + quaternion[:, 1] * quaternion[:, 2]),
            1.0 - 2.0 * (quaternion[:, 2].square() + quaternion[:, 3].square()),
        )
        command = self.command_tensor
        cosine, sine = torch.cos(yaw), torch.sin(yaw)
        velocity_w = torch.stack(
            (cosine * command[:, 0] - sine * command[:, 1], sine * command[:, 0] + cosine * command[:, 1]),
            dim=1,
        )
        robot.root_pos_w[:, :2] += velocity_w * dt
        yaw = yaw + command[:, 2] * dt
        robot.root_quat_w[:, 0] = torch.cos(yaw / 2.0)
        robot.root_quat_w[:, 3] = torch.sin(yaw / 2.0)

        offset = ball.root_pos_w[:, :2] - robot.root_pos_w[:, :2]
        distance = torch.linalg.vector_norm(offset, dim=1)
        touching = distance < PUSH_RADIUS_M
        direction = offset / distance.clamp_min(1.0e-6).unsqueeze(1)
        speed = torch.linalg.vector_norm(velocity_w, dim=1) + 0.3
        self.ball_velocity_xy = torch.where(
            touching.unsqueeze(1), direction * speed.unsqueeze(1), self.ball_velocity_xy * 0.97
        )
        ball.root_pos_w[:, :2] += self.ball_velocity_xy * dt
        ball.root_lin_vel_w[:, :2] = self.ball_velocity_xy

        contact = self.scene["robot_ball_contact"].data.force_matrix_w_history
        contact[:] = torch.roll(contact, 1, dims=1)
        contact[:, 0] = 0.0
        contact[:, 0, 0, 15, 2] = torch.where(touching, 5.0, 0.0)

        env_index = torch.arange(self.num_envs, device=self.device)
        base = self.scene["contact_forces"].data.net_forces_w_history
        base[:] = torch.roll(base, 1, dims=1)
        base[:, 0] = 0.0
        fall = (env_index * 7_919 + self.tick * 104_729) % FALL_PERIOD_TICKS == 0
        base[:, 0, 0, 2] = torch.where(fall, 50.0, 0.0)

        action_term = self.action_manager.get_term("joint_pos")
        target = 0.25 * actions
        limited = target.clamp(-0.02, 0.02)
        action_term._last_guard_correction.copy_(limited - target)
        joint = robot.default_joint_pos + 0.05 * torch.tanh(actions)
        fault = (env_index * 6_151 + self.tick * 15_485_863) % JOINT_FAULT_PERIOD_TICKS == 0
        joint[fault, 0] = 1.01
        robot.joint_pos[:] = joint
        self.action_manager.prev_action[:] = actions
        self.tick += 1
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return self._observations(), torch.zeros(self.num_envs, device=self.device), done, done, {}

    def close(self) -> None:
        return None


class _Scene(dict):
    def __init__(self, entries: dict[str, Any], origins: torch.Tensor) -> None:
        super().__init__(entries)
        self.env_origins = origins


class FakeGymEnv:
    """What ``gym.make`` returns: a wrapper whose ``unwrapped`` is the base env."""

    def __init__(self, base_env: FakeSoccerBaseEnv) -> None:
        self.unwrapped = base_env

    def close(self) -> None:
        return None


def frozen_low_actor(device: str = "cpu") -> torch.nn.Module:
    """A fixed, untrainable 48 -> 12 stand-in for the qualified low policy."""

    generator = torch.Generator().manual_seed(0)
    actor = torch.nn.Linear(48, 12)
    with torch.no_grad():
        actor.weight.copy_(torch.randn(12, 48, generator=generator) * 0.1)
        actor.bias.zero_()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    return actor.to(device)


def scripted_latent_action(observation: torch.Tensor) -> torch.Tensor:
    """A deterministic chase-the-ball high action computed from the 70-D
    observation (entity features 48.. hold the ball in the body frame)."""

    ball_x = observation[:, 48]
    ball_y = observation[:, 49]
    heading = torch.atan2(ball_y, ball_x)
    latent = torch.stack((2.0 * ball_x, 2.0 * ball_y, 2.0 * heading), dim=1)
    return latent.clamp(-3.0, 3.0)
