"""The no-Kick soccer task over the Isaac Lab Go2 scene.

The policy acts through a control (``controls``): either 12 joint actions
every native 50 Hz tick, or a body-velocity command held for five ticks and
executed by a frozen low-level policy.  Goal and safety termination are
evaluated after every native tick and before reset.  Episode starts come
from a reset sampler (``resets.ResetSampler``).
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import torch
from tensordict import TensorDict

from .resets import ResetSampler
from .observations import build_entity_observation
from .config import FieldGeometry, POLICY_OBSERVATION_DIM
from .controls import Control
from .contact import ROBOT_BODY_NAMES, validate_live_robot_body_names
from .shaping import PotentialShaping, ShapingState
from .task_limits import (
    BALL_SIDE_BOUNDARY_M,
    BASE_FALL_FORCE_THRESHOLD_N,
    CONTACT_FORCE_THRESHOLD_N,
    GOAL_CENTER_X_M,
    ROBOT_FIELD_HALF_LENGTH_M,
    ROBOT_FIELD_HALF_WIDTH_M,
    TERMINAL_GOAL_HALF_WIDTH_M,
)


LIVE_RESET_XY_TOLERANCE_M = 5.0e-6
LIVE_RESET_YAW_TOLERANCE_RAD = 1.0e-6
PROGRAMMED_LOCAL_SAMPLE_TOLERANCE_M = 2.0e-5


TRUE_CONTACT_FORCE_THRESHOLD_N = CONTACT_FORCE_THRESHOLD_N
# Episode outcomes, by code (the ``outcome`` episode column).  When several
# events happen on one tick the earliest listed wins: a goal is only a goal
# if nothing unsafe happened on that tick.
OUTCOMES = ("goal", "own_goal", "fall", "boundary", "timeout")
# The shaping potentials, in the order of the per-episode shaping columns.
SHAPING_COMPONENTS = ("ball_goal", "robot_ball", "base_height", "upright")


def extract_policy_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, tuple):
        value = value[0]
    while not torch.is_tensor(value):
        if not hasattr(value, "keys") or not hasattr(value, "__getitem__"):
            raise RuntimeError(f"unsupported observation type: {type(value)}")
        try:
            keys = list(value.keys())
        except TypeError:
            keys = list(value.keys(include_nested=False))
        if "policy" in keys:
            value = value["policy"]
        elif len(keys) == 1:
            value = value[keys[0]]
        else:
            raise RuntimeError(f"unsupported observation mapping: {keys}")
    return value.detach().clone()


def normalize_done(value: Any, *, num_envs: int, device: torch.device) -> torch.Tensor:
    if not isinstance(value, tuple):
        raise RuntimeError("Isaac step did not return a tuple")
    if len(value) == 4:
        done = value[2]
    elif len(value) == 5:
        done = torch.logical_or(value[2], value[3])
    else:
        raise RuntimeError(f"unexpected Isaac step tuple length: {len(value)}")
    result = torch.as_tensor(done, dtype=torch.bool, device=device)
    if result.ndim == 0:
        return result.expand(num_envs).clone()
    if result.shape[0] != num_envs:
        raise RuntimeError("native termination batch differs from environment")
    return result if result.ndim == 1 else result.reshape(num_envs, -1).any(dim=1)


def writable_command(base_env: Any) -> torch.Tensor:
    manager = base_env.command_manager
    term = manager.get_term("base_velocity") if hasattr(manager, "get_term") else manager._terms["base_velocity"]
    for name in ("vel_command_b", "command", "_command", "command_b"):
        value = getattr(term, name, None)
        if torch.is_tensor(value) and value.ndim == 2 and value.shape[1] >= 3:
            return value
    raise RuntimeError("no writable low-level velocity-command tensor exists")


def live_action_term(base_env: Any) -> Any:
    """The scene's one action term (joint position or joint velocity)."""

    terms = list(base_env.action_manager._terms.values())
    if len(terms) != 1:
        raise RuntimeError(f"expected one action term, found {len(terms)}")
    return terms[0]


def yaw_from_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quaternion.unbind(dim=1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))


def boundary_violation(
    robot_xy: torch.Tensor,
    ball_xy: torch.Tensor,
    geometry: FieldGeometry,
) -> torch.Tensor:
    inside_mouth = ball_xy[:, 1].abs() <= TERMINAL_GOAL_HALF_WIDTH_M
    ball_side = ball_xy[:, 1].abs() > BALL_SIDE_BOUNDARY_M
    ball_end_outside_mouth = (
        ball_xy[:, 0].abs() > GOAL_CENTER_X_M
    ) & ~inside_mouth
    robot = (robot_xy[:, 0].abs() > ROBOT_FIELD_HALF_LENGTH_M) | (
        robot_xy[:, 1].abs() > ROBOT_FIELD_HALF_WIDTH_M
    )
    return robot | ball_side | ball_end_outside_mouth


class FrozenLowInferenceEnv:
    """Expose the unwrapped ManagerBasedEnv without an implicit reset."""

    def __init__(self, gym_env: Any) -> None:
        self.env = gym_env
        self.unwrapped = gym_env.unwrapped
        self.device = self.unwrapped.device
        self.num_envs = int(self.unwrapped.num_envs)

    def get_observations(self) -> Any:
        return self.unwrapped.observation_manager.compute()

    def step(self, actions: torch.Tensor) -> Any:
        return self.unwrapped.step(actions)

    def close(self) -> Any:
        return self.env.close()


class CurrentTaskProgressHighGoalEnv:
    """The soccer task, stepped through a ``control``.

    One policy action is held for ``control.ticks_per_action`` native
    ticks.  After every tick the task checks goal, own goal, boundary and
    fall; an unsafe event on the same tick as a goal cancels the goal.
    Joint hard limits end nothing (the simulator enforces them; hitting a
    stop matters only on a real robot); each episode records its largest
    overshoot past one.  A slot that terminates is frozen at that tick (its
    joint action is zeroed and its post state stops updating) while the
    other slots finish the held action.
    The 20 s horizon is a true finite horizon, so the RSL adapter exposes
    no ``time_outs`` and PPO never bootstraps across it.

    The reward is +1 for a goal, -1 for an own goal, plus the
    potential-based shaping of ``shaping`` between the state before the
    held action and the state it ended in (the terminal tick's state for a
    slot that terminated), with no potential after any terminal
    transition, timeout included, plus ``native_reward_scale`` times the
    scene's own reward-manager reward over the active ticks (the motion
    regularizers of the direct-joint scene; 0 ignores it).

    Where each episode starts comes from ``reset_sampler`` (see
    ``resets.ResetBatch``); the environment only writes those poses and
    carries the batch's tags into the finished-episode records.  Finished
    episodes are reported per step as one columnar batch
    (``extras["episodes"]``: field name -> CPU tensor, one row per episode,
    rows in env-id order), gathered after the held action in two
    device-to-host copies.  Per-step safety checks are folded into one
    deferred check so the step synchronizes with the device only a few
    times however many slots reset or terminate.
    """

    num_obs = POLICY_OBSERVATION_DIM
    num_privileged_obs = None

    def __init__(
        self,
        *,
        low_env: FrozenLowInferenceEnv,
        control: Control,
        reset_sampler: ResetSampler,
        shaping: PotentialShaping,
        native_reward_scale: float = 0.0,
        episode_seconds: float = 20.0,
        geometry: FieldGeometry = FieldGeometry(),
    ) -> None:
        if episode_seconds <= 0.0:
            raise ValueError("episode duration must be positive")
        self.low_env = low_env
        self.base_env = low_env.unwrapped
        self.device = torch.device(low_env.device)
        self.num_envs = int(low_env.num_envs)
        self.control = control
        self.num_actions = int(control.num_actions)
        self.native_reward_scale = float(native_reward_scale)

        self.robot = self.base_env.scene["robot"]
        self.ball = self.base_env.scene["ball"]
        self.ball_contact = self.base_env.scene["robot_ball_contact"]
        self.base_contact = self.base_env.scene["contact_forces"]
        validate_live_robot_body_names(self.robot.body_names)
        base_names = tuple(str(name) for name in self.base_contact.body_names)
        if "base" not in base_names:
            raise RuntimeError("base contact sensor lacks base")
        self.base_body_index = base_names.index("base")
        self.action_term = live_action_term(self.base_env)
        # The guarded joint action reports how much it clamped; others don't.
        self.guard_correction = getattr(self.action_term, "last_guard_correction", None)
        self.command = writable_command(self.base_env)
        self.env_origins = self.base_env.scene.env_origins
        self.low_control_dt = float(self.base_env.cfg.sim.dt) * int(self.base_env.cfg.decimation)
        if not math.isclose(self.low_control_dt, 0.02, abs_tol=1.0e-12):
            raise RuntimeError(f"live low control period changed: {self.low_control_dt}")
        self.low_steps_per_high_action = int(control.ticks_per_action)
        self.high_control_dt = self.low_control_dt * self.low_steps_per_high_action
        self.max_episode_length = round(episode_seconds / self.high_control_dt)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # This is a true finite horizon.  The adapter intentionally omits the
        # RSL `time_outs` key so PPO never bootstraps across the 20-second end.
        self.cfg = SimpleNamespace(is_finite_horizon=True)
        self.geometry = geometry
        self.reset_sampler = reset_sampler
        self.shaping = shaping

        self.goal_xy_w = torch.zeros(self.num_envs, 2, device=self.device)
        self.contact_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.maximum_contact_force = torch.zeros(self.num_envs, device=self.device)
        self.first_contact_native_tick = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.episode_native_tick = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.goal_side = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        # Per-slot copies of the reset batch's tags, allocated on first use.
        self.reset_tags: dict[str, torch.Tensor] = {}
        # Record the simulator state after each reset for episode metrics.
        self.live_reset_robot_xy_yaw = torch.zeros(self.num_envs, 3, device=self.device)
        self.live_reset_ball_xy = torch.zeros(self.num_envs, 2, device=self.device)
        # Undiscounted shaping collected so far this episode, in total and
        # per potential (in SHAPING_COMPONENTS order).
        self.episode_shaping_return = torch.zeros(self.num_envs, device=self.device)
        self.episode_shaping_components = torch.zeros(self.num_envs, len(SHAPING_COMPONENTS), device=self.device)
        self.episode_native_reward = torch.zeros(self.num_envs, device=self.device)
        # Largest distance any joint went past its hard limit this episode.
        self.episode_max_joint_overshoot = torch.zeros(self.num_envs, device=self.device)
        self.vector_steps = 0

        self._observation = torch.zeros(self.num_envs, self.num_obs, device=self.device)
        self._reset_idx(torch.arange(self.num_envs, dtype=torch.long, device=self.device))

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """Start new episodes in ``env_ids`` (sorted, unique) from one batch
        of the reset sampler, in env-id order."""

        ids = env_ids.to(device=self.device, dtype=torch.long).flatten()
        count = int(ids.numel())
        if count == 0:
            return
        self.base_env._reset_idx(ids)
        batch = self.reset_sampler(count)
        if batch.robot_xy.shape != (count, 2) or batch.ball_xy.shape != (count, 2):
            raise RuntimeError("reset sampler returned the wrong batch size")
        robot_xy = batch.robot_xy.to(self.device, torch.float32)
        robot_yaw = batch.robot_yaw.to(self.device, torch.float32)
        ball_xy = batch.ball_xy.to(self.device, torch.float32)
        goal_side = batch.goal_side.to(self.device, torch.long)

        defaults = self.robot.data.default_joint_pos[ids].clone()
        self.robot.write_joint_state_to_sim(defaults, torch.zeros_like(defaults), env_ids=ids)
        origins = self.env_origins[ids]
        robot_pose = torch.zeros(count, 7, device=self.device)
        robot_pose[:, :2] = origins[:, :2] + robot_xy
        robot_pose[:, 2] = self.robot.data.root_pos_w[ids, 2]
        robot_pose[:, 3] = torch.cos(robot_yaw / 2.0)
        robot_pose[:, 6] = torch.sin(robot_yaw / 2.0)
        ball_default_root_state = self.ball.data.default_root_state[ids].clone()
        ball_pose = ball_default_root_state[:, :7].clone()
        ball_pose[:, :3] += origins
        ball_pose[:, :2] = origins[:, :2] + ball_xy
        ball_velocity = ball_default_root_state[:, 7:].clone()
        self.goal_xy_w[ids, 0] = origins[:, 0] + goal_side.to(torch.float32) * self.geometry.field_half_length
        self.goal_xy_w[ids, 1] = origins[:, 1]
        self.goal_side[ids] = goal_side
        for name, value in batch.tags.items():
            if name not in self.reset_tags:
                self.reset_tags[name] = torch.zeros(
                    (self.num_envs, *value.shape[1:]), dtype=value.dtype, device=self.device
                )
            self.reset_tags[name][ids] = value.to(self.device)

        self.robot.write_root_pose_to_sim(robot_pose, env_ids=ids)
        self.robot.write_root_velocity_to_sim(torch.zeros(count, 6, device=self.device), env_ids=ids)
        self.ball.write_root_pose_to_sim(ball_pose, env_ids=ids)
        self.ball.write_root_velocity_to_sim(ball_velocity, env_ids=ids)
        self.command[ids, :3] = 0.0
        self.control.reset(ids)
        self.action_term.reset(ids)
        self.contact_seen[ids] = False
        self.maximum_contact_force[ids] = 0.0
        self.first_contact_native_tick[ids] = -1
        self.episode_native_tick[ids] = 0
        self.episode_length_buf[ids] = 0
        self.episode_shaping_return[ids] = 0.0
        self.episode_shaping_components[ids] = 0.0
        self.episode_native_reward[ids] = 0.0
        self.episode_max_joint_overshoot[ids] = 0.0
        if hasattr(self.base_env, "episode_length_buf"):
            self.base_env.episode_length_buf[ids] = 0

        # One check that every write reached the simulator: the live root
        # states equal what was written and the action buffers are clear.
        live_robot_xy = self.robot.data.root_pos_w[ids, :2] - origins[:, :2]
        live_robot_yaw = yaw_from_quaternion(self.robot.data.root_quat_w[ids])
        live_ball_root_state = torch.cat(
            (
                self.ball.data.root_pos_w[ids],
                self.ball.data.root_quat_w[ids],
                self.ball.data.root_lin_vel_w[ids],
                self.ball.data.root_ang_vel_w[ids],
            ),
            dim=1,
        )
        previous_action = getattr(self.base_env.action_manager, "prev_action", None)
        if not torch.is_tensor(previous_action) or tuple(previous_action.shape) != (self.num_envs, 12):
            raise RuntimeError("previous action buffer changed")
        yaw_error = torch.atan2(torch.sin(live_robot_yaw - robot_yaw), torch.cos(live_robot_yaw - robot_yaw))
        write_failed = (
            ((self.robot.data.root_pos_w[ids, :2] - robot_pose[:, :2]).abs() > LIVE_RESET_XY_TOLERANCE_M).any()
            | ((live_ball_root_state - torch.cat((ball_pose, ball_velocity), dim=1)).abs() > LIVE_RESET_XY_TOLERANCE_M).any()
            | (yaw_error.abs() > LIVE_RESET_YAW_TOLERANCE_RAD).any()
            | (previous_action[ids].abs() > 1.0e-8).any()
            | (self.command[ids, :3].abs() > 1.0e-8).any()
        )
        if bool(write_failed):
            raise RuntimeError("reset state did not reach the simulator")
        live_ball_xy = self.ball.data.root_pos_w[ids, :2] - origins[:, :2]
        self.live_reset_robot_xy_yaw[ids] = torch.cat((live_robot_xy, live_robot_yaw.unsqueeze(1)), dim=1)
        self.live_reset_ball_xy[ids] = live_ball_xy
        # Keep sensor resets last: reading sensor.data after them would
        # lazily restore the terminal contact of the previous episode.
        self.ball_contact.reset(ids)
        self.base_contact.reset(ids)

    def _build_observation(self) -> torch.Tensor:
        # Round2 may clear the command-manager tensor after base.step when it
        # refreshes standing flags.  That clearing was not the command the low
        # executed and must not corrupt the next high observation.  Reinstall
        # the held command without advancing physics or processing an action.
        self.command[:, :3] = self.control.held_command()
        low_observation = extract_policy_tensor(self.low_env.get_observations())
        if low_observation.shape != (self.num_envs, 48):
            raise RuntimeError(f"low observation changed: {tuple(low_observation.shape)}")
        remaining = 1.0 - self.episode_length_buf.float() / float(self.max_episode_length)
        observation, _ = build_entity_observation(
            low_observation=low_observation,
            robot_position_w=self.robot.data.root_pos_w,
            robot_quaternion_wxyz=self.robot.data.root_quat_w,
            ball_position_w=self.ball.data.root_pos_w,
            ball_velocity_w=self.ball.data.root_lin_vel_w,
            goal_xy_w=self.goal_xy_w,
            env_origin_xy_w=self.env_origins[:, :2],
            time_remaining=remaining,
            contact_seen=self.contact_seen,
        )
        if observation.shape != (self.num_envs, POLICY_OBSERVATION_DIM):
            raise RuntimeError(f"high observation shape changed: {tuple(observation.shape)}")
        if not bool(torch.allclose(observation[:, 9:12], self.control.held_command(), rtol=0.0, atol=1.0e-6)):
            raise RuntimeError("70-D high observation lost the held applied command")
        return observation

    def get_observations(self) -> torch.Tensor:
        self._observation = self._build_observation()
        return self._observation

    def reset(self) -> torch.Tensor:
        """Start a new episode in every slot, drawn from the reset sampler as
        it is now.  Training never calls this (construction already filled
        every slot); playback scripts do after changing the ADR ranges."""

        self._reset_idx(torch.arange(self.num_envs, dtype=torch.long, device=self.device))
        return self.get_observations()

    @torch.inference_mode(False)
    def step(self, latent_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        if latent_action.shape != (self.num_envs, self.num_actions):
            raise RuntimeError(f"policy action shape changed: {tuple(latent_action.shape)}")
        if not bool(torch.isfinite(latent_action).all()):
            raise RuntimeError("policy action is non-finite")

        # Freeze physical pre-state before any command or reset.
        pre_robot_xy = (self.robot.data.root_pos_w[:, :2] - self.env_origins[:, :2]).detach().clone()
        pre_ball_xy = (self.ball.data.root_pos_w[:, :2] - self.env_origins[:, :2]).detach().clone()
        pre_goal_side = self.goal_side.detach().clone()
        pre_height = self.robot.data.root_pos_w[:, 2].detach().clone()
        pre_up = -self.robot.data.projected_gravity_b[:, 2].detach().clone()

        bounded_action = self.control.start(latent_action.detach().to(self.device))

        active = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        terminal_goal = torch.zeros_like(active)
        terminal_own = torch.zeros_like(active)
        terminal_boundary = torch.zeros_like(active)
        terminal_fall = torch.zeros_like(active)
        terminal_requested = torch.zeros_like(active)
        contact_this_high = torch.zeros_like(active)
        # Any per-tick failure is recorded here and raised once after the loop.
        failed = torch.zeros((), dtype=torch.bool, device=self.device)

        # Post-state fields are updated only while a slot is active.  Therefore
        # a terminal slot is frozen at the exact native tick that triggered
        # termination rather than drifting through the remaining held ticks.
        post_robot_xy = pre_robot_xy.clone()
        post_ball_xy = pre_ball_xy.clone()
        post_contact_force = torch.zeros(self.num_envs, device=self.device)
        post_base_force = torch.zeros(self.num_envs, device=self.device)
        post_guard_changed = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        post_guard_max = torch.zeros(self.num_envs, device=self.device)
        post_height = pre_height.clone()
        post_up = pre_up.clone()
        native_reward = torch.zeros(self.num_envs, device=self.device)

        def native_observation() -> torch.Tensor:
            value = extract_policy_tensor(self.low_env.get_observations())
            if value.shape != (self.num_envs, 48):
                raise RuntimeError(f"low observation changed: {tuple(value.shape)}")
            return value

        for _low_tick in range(self.low_steps_per_high_action):
            tick_active = active.clone()
            joint_action, control_failed = self.control.tick_action(tick_active, native_observation)
            failed |= control_failed | ~torch.isfinite(joint_action).all()
            native_step = self.low_env.step(joint_action.detach())
            native_done = normalize_done(native_step, num_envs=self.num_envs, device=self.device)
            failed |= native_done.any()
            if self.native_reward_scale != 0.0:
                native_reward += torch.where(
                    tick_active,
                    self.native_reward_scale * torch.as_tensor(native_step[1], device=self.device).float(),
                    torch.zeros_like(native_reward),
                )

            contact_force = self._ball_contact_force()
            base_force = self._base_contact_force()
            hard_excess = self._joint_hard_excess()
            guard = (
                self.guard_correction.abs()
                if self.guard_correction is not None
                else torch.zeros(self.num_envs, 12, device=self.device)
            )
            robot_xy = self.robot.data.root_pos_w[:, :2] - self.env_origins[:, :2]
            ball_xy = self.ball.data.root_pos_w[:, :2] - self.env_origins[:, :2]
            keep = tick_active.unsqueeze(1)
            post_robot_xy = torch.where(keep, robot_xy, post_robot_xy)
            post_ball_xy = torch.where(keep, ball_xy, post_ball_xy)
            post_contact_force = torch.where(tick_active, contact_force, post_contact_force)
            post_base_force = torch.where(tick_active, base_force, post_base_force)
            self.episode_max_joint_overshoot = torch.where(
                tick_active,
                torch.maximum(self.episode_max_joint_overshoot, hard_excess),
                self.episode_max_joint_overshoot,
            )
            post_guard_changed = torch.where(tick_active, (guard > 1.0e-9).sum(dim=1), post_guard_changed)
            post_guard_max = torch.where(tick_active, guard.amax(dim=1), post_guard_max)
            post_height = torch.where(tick_active, self.robot.data.root_pos_w[:, 2], post_height)
            post_up = torch.where(tick_active, -self.robot.data.projected_gravity_b[:, 2], post_up)
            contact = contact_force >= TRUE_CONTACT_FORCE_THRESHOLD_N
            new_contact = contact & ~self.contact_seen & tick_active
            self.first_contact_native_tick = torch.where(
                new_contact, self.episode_native_tick, self.first_contact_native_tick
            )
            self.contact_seen |= contact & tick_active
            self.maximum_contact_force = torch.maximum(
                self.maximum_contact_force,
                torch.where(tick_active, contact_force, torch.zeros_like(contact_force)),
            )
            contact_this_high |= contact & tick_active
            self.episode_native_tick += tick_active.long()

            selected = self.goal_side.to(ball_xy.dtype) * ball_xy[:, 0]
            inside = ball_xy[:, 1].abs() <= TERMINAL_GOAL_HALF_WIDTH_M
            requested_line = self.contact_seen & (selected >= GOAL_CENTER_X_M) & inside & tick_active
            own = (selected <= -GOAL_CENTER_X_M) & inside & tick_active
            boundary = boundary_violation(robot_xy, ball_xy, self.geometry) & tick_active
            fall = (base_force > BASE_FALL_FORCE_THRESHOLD_N) & tick_active
            unsafe = own | boundary | fall

            terminal_goal |= requested_line & ~unsafe
            terminal_own |= own
            terminal_boundary |= boundary
            terminal_fall |= fall
            terminal_requested |= requested_line
            active &= ~(requested_line | unsafe)

        self.episode_length_buf += 1
        timeout = (self.episode_length_buf >= self.max_episode_length) & active

        done = terminal_goal | terminal_own | terminal_boundary | terminal_fall | timeout
        goal_xy = torch.zeros_like(pre_robot_xy)
        goal_xy[:, 0] = pre_goal_side.to(goal_xy.dtype) * GOAL_CENTER_X_M
        shaping, components = self.shaping(
            ShapingState(pre_robot_xy, pre_ball_xy, goal_xy, pre_height, pre_up),
            ShapingState(post_robot_xy, post_ball_xy, goal_xy, post_height, post_up),
            done,
        )
        reward = shaping + native_reward + terminal_goal.to(shaping.dtype) - terminal_own.to(shaping.dtype)
        failed |= ~torch.isfinite(reward).all()
        if bool(failed):
            raise RuntimeError(
                "step failed a safety check: non-finite low observation, action or reward, "
                "a lost command, or a native termination"
            )
        self.episode_shaping_return += shaping
        self.episode_native_reward += native_reward
        self.episode_shaping_components += torch.stack([components[name] for name in SHAPING_COMPONENTS], dim=1)
        done_ids = torch.nonzero(done, as_tuple=False).flatten()
        episodes: dict[str, torch.Tensor] = {}
        if done_ids.numel():
            outcome = torch.full_like(done_ids, OUTCOMES.index("timeout"))
            for name, flag in reversed(
                (
                    ("goal", terminal_goal),
                    ("own_goal", terminal_own),
                    ("fall", terminal_fall),
                    ("boundary", terminal_boundary),
                )
            ):
                outcome = torch.where(flag[done_ids], OUTCOMES.index(name), outcome)
            reset_ball = self.live_reset_ball_xy[done_ids]
            terminal_ball = post_ball_xy[done_ids]
            side = self.goal_side[done_ids]
            float_columns = {
                "reset_robot_xy_m": self.live_reset_robot_xy_yaw[done_ids, :2],
                "reset_robot_yaw_rad": self.live_reset_robot_xy_yaw[done_ids, 2],
                "reset_ball_xy_m": reset_ball,
                "terminal_robot_xy_m": post_robot_xy[done_ids],
                "terminal_ball_xy_m": terminal_ball,
                "maximum_contact_force_n": self.maximum_contact_force[done_ids],
                "terminal_contact_force_n": post_contact_force[done_ids],
                "terminal_base_contact_force_n": post_base_force[done_ids],
                "max_joint_overshoot_rad": self.episode_max_joint_overshoot[done_ids],
                "native_safe_guard_max_correction_rad": post_guard_max[done_ids],
                "shaping_return": self.episode_shaping_return[done_ids],
                "native_reward_return": self.episode_native_reward[done_ids],
                **{
                    f"shaping_{name}": self.episode_shaping_components[done_ids, index]
                    for index, name in enumerate(SHAPING_COMPONENTS)
                },
                "requested_goal_axis_ball_displacement_m": side.to(torch.float32) * (terminal_ball[:, 0] - reset_ball[:, 0]),
                "abs_lateral_ball_drift_m": (terminal_ball[:, 1] - reset_ball[:, 1]).abs(),
            }
            integer_columns = {
                "goal_side": side,
                "outcome": outcome,
                "goal": terminal_goal[done_ids],
                "own_goal": terminal_own[done_ids],
                "boundary": terminal_boundary[done_ids],
                "fall": terminal_fall[done_ids],
                "timeout": timeout[done_ids],
                "requested_goal_line_reached": terminal_requested[done_ids],
                "contact_seen": self.contact_seen[done_ids],
                "first_true_contact_native_tick": self.first_contact_native_tick[done_ids],
                "native_safe_guard_changed_elements": post_guard_changed[done_ids],
                "episode_high_ticks": self.episode_length_buf[done_ids],
                "episode_native_ticks": self.episode_native_tick[done_ids],
            }
            episodes = _to_host_columns(float_columns, torch.float32)
            episodes.update(_to_host_columns(integer_columns, torch.long))
            for name in ("goal", "own_goal", "boundary", "fall", "timeout",
                         "requested_goal_line_reached", "contact_seen"):
                episodes[name] = episodes[name].bool()
            episodes["unsafe_pre_goal"] = episodes["own_goal"] | episodes["boundary"] | episodes["fall"]
            for name, value in self.reset_tags.items():
                episodes[name] = value[done_ids].cpu()
            self._reset_idx(done_ids)
        self.vector_steps += 1
        observation = self.get_observations()
        extras = {
            "episodes": episodes,
            "task": {
                "contact_fraction": contact_this_high.float().mean(),
                "bounded_action_abs_mean": bounded_action.abs().mean(),
            },
        }
        return observation, reward, done, extras

    def _ball_contact_force(self) -> torch.Tensor:
        """The largest robot-ball contact force over the last native tick:
        over every physics substep when the sensor keeps a filtered-force
        history (Isaac Lab >= 2.1), otherwise at the last substep."""

        data = self.ball_contact.data
        history = getattr(data, "force_matrix_w_history", None)
        if history is None and torch.is_tensor(getattr(data, "force_matrix_w", None)):
            history = data.force_matrix_w.unsqueeze(1)
        if (
            not torch.is_tensor(history)
            or history.ndim != 5
            or tuple(history.shape[i] for i in (0, 2, 3, 4)) != (self.num_envs, 1, len(ROBOT_BODY_NAMES), 3)
        ):
            actual = None if not torch.is_tensor(history) else tuple(history.shape)
            raise RuntimeError(f"robot-ball ContactSensor shape changed: {actual}")
        return torch.linalg.vector_norm(history, dim=-1).amax(dim=(1, 2, 3))

    def _base_contact_force(self) -> torch.Tensor:
        history = getattr(self.base_contact.data, "net_forces_w_history", None)
        if not torch.is_tensor(history) or history.ndim != 4 or history.shape[0] != self.num_envs:
            raise RuntimeError("base contact history is unavailable")
        return torch.linalg.vector_norm(history[:, :, self.base_body_index, :], dim=-1).amax(dim=1)

    def _joint_hard_excess(self) -> torch.Tensor:
        """The largest distance any joint is past its hard limit (rad)."""

        position = self.robot.data.joint_pos
        # Isaac Lab 2.0 calls the hard limits joint_limits.
        limits = getattr(self.robot.data, "joint_pos_limits", None)
        if limits is None:
            limits = self.robot.data.joint_limits
        below = torch.clamp(limits[..., 0] - position, min=0.0)
        above = torch.clamp(position - limits[..., 1], min=0.0)
        return torch.maximum(below, above).amax(dim=1)

    def close(self) -> None:
        self.low_env.close()


def _to_host_columns(columns: dict[str, torch.Tensor], dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Copy same-length columns to the host in one transfer and split them
    back by name ([n] columns stay [n], [n,k] columns stay [n,k])."""

    widths = [1 if value.ndim == 1 else int(value.shape[1]) for value in columns.values()]
    packed = torch.cat(
        [value.reshape(value.shape[0], -1).to(dtype) for value in columns.values()], dim=1
    ).cpu()
    result: dict[str, torch.Tensor] = {}
    start = 0
    for (name, value), width in zip(columns.items(), widths):
        piece = packed[:, start:start + width]
        result[name] = piece[:, 0] if value.ndim == 1 else piece
        start += width
    return result


class CurrentTaskProgressHighRslAdapter:
    def __init__(self, env: CurrentTaskProgressHighGoalEnv) -> None:
        self.env = env
        self.num_envs = env.num_envs
        self.num_actions = env.num_actions
        self.device = env.device
        self.max_episode_length = env.max_episode_length
        self.cfg = env.cfg

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.env.episode_length_buf = value

    def _pack(self, observation: torch.Tensor) -> TensorDict:
        if observation.shape != (self.num_envs, POLICY_OBSERVATION_DIM):
            raise RuntimeError(f"policy observation shape changed: {tuple(observation.shape)}")
        return TensorDict({"policy": observation}, batch_size=[self.num_envs], device=self.device)

    @torch.inference_mode(False)
    def get_observations(self) -> TensorDict:
        return self._pack(self.env.get_observations())

    @torch.inference_mode(False)
    def reset(self) -> TensorDict:
        return self._pack(self.env.reset())

    @torch.inference_mode(False)
    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, Any]]:
        observation, reward, done, extras = self.env.step(actions.detach().clone())
        if "time_outs" in extras:
            raise RuntimeError("finite-horizon task must not expose timeout bootstrap")
        return self._pack(observation), reward, done, extras

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)
