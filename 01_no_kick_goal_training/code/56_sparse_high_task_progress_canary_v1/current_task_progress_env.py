"""V56 task-progress high interface over V53 strict terminal semantics.

The behavioral task is intentionally small: a fresh recurrent high emits one
three-axis command at 10 Hz and the exact frozen current low executes it for
five native 50 Hz ticks.  Native Isaac terminations are disabled by the
runtime builder.  This wrapper evaluates contact, strict goal and safety after
every native step, before any selected environment is reset.
"""

from __future__ import annotations

from collections import Counter
import math
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch
from tensordict import TensorDict

from continuous_reset import (
    ResetSample,
    fixed_evaluation_samples,
    is_hard_goal_facing_ball_behind,
    relative_bucket,
    sample_reset,
)
from entity_observation import build_entity_observation
from high_command_interface import limit_latent_high_command
from interface_contract import FieldGeometry, POLICY_ACTION_DIM, POLICY_OBSERVATION_DIM
from canary_protocol import EPISODE_ROW_SCHEMA, TRANSITION_ROW_SCHEMA, LIVE_RESET_XY_TOLERANCE_M, LIVE_RESET_YAW_TOLERANCE_RAD
from native_safe_joint_action import NativeSafeJointPositionAction
from staging_contact_contract import ROBOT_BODY_NAMES, validate_live_robot_body_names
from task_progress_reward import (
    BALL_SIDE_BOUNDARY_M,
    BASE_FALL_FORCE_THRESHOLD_N,
    CONTACT_FORCE_THRESHOLD_N,
    GOAL_CENTER_X_M,
    JOINT_HARD_LIMIT_TOLERANCE_RAD,
    PATH_NORMALIZER_M,
    ROBOT_FIELD_HALF_LENGTH_M,
    ROBOT_FIELD_HALF_WIDTH_M,
    TERMINAL_GOAL_HALF_WIDTH_M,
    scalar_task_geometry,
    torch_progress,
    torch_task_geometry,
)


TRUE_CONTACT_FORCE_THRESHOLD_N = CONTACT_FORCE_THRESHOLD_N

BALL_CONTACT_RESET_CACHE_FIELDS = (
    "net_forces_w",
    "net_forces_w_history",
    "force_matrix_w",
    "force_matrix_w_history",
)
BASE_CONTACT_RESET_CACHE_FIELDS = (
    "net_forces_w",
    "net_forces_w_history",
    "current_air_time",
    "last_air_time",
    "current_contact_time",
    "last_contact_time",
)
CONTACT_SENSOR_RESET_TIMER_FIELDS = (
    "_timestamp",
    "_timestamp_last_update",
    "_is_outdated",
)


def cached_contact_sensor_state(
    sensor: Any,
    env_ids: torch.Tensor,
    zero_fields: Sequence[str],
    *,
    sensor_name: str,
) -> dict[str, torch.Tensor]:
    """Copy ContactSensor reset state without triggering a lazy PhysX refresh."""

    data = getattr(sensor, "_data", None)
    if data is None:
        raise RuntimeError(f"{sensor_name} ContactSensor cache is unavailable")
    expected_envs = int(getattr(sensor, "_num_envs", -1))
    if expected_envs <= 0:
        raise RuntimeError(f"{sensor_name} ContactSensor environment count is unavailable")
    reference = getattr(data, zero_fields[0], None) if zero_fields else None
    if not torch.is_tensor(reference):
        raise RuntimeError(f"{sensor_name} ContactSensor reference cache is unavailable")
    ids = env_ids.to(device=reference.device, dtype=torch.long).flatten()
    if ids.numel() and (int(ids.min().item()) < 0 or int(ids.max().item()) >= expected_envs):
        raise RuntimeError(f"{sensor_name} ContactSensor receipt IDs are out of range")

    result: dict[str, torch.Tensor] = {}
    for field in (*zero_fields, *CONTACT_SENSOR_RESET_TIMER_FIELDS):
        owner = sensor if field.startswith("_") else data
        value = getattr(owner, field, None)
        if not torch.is_tensor(value) or value.ndim < 1 or int(value.shape[0]) != expected_envs:
            raise RuntimeError(f"{sensor_name} ContactSensor cached field changed: {field}")
        result[field] = value[ids].detach().clone()
    return result


def cached_contact_reset_receipt(
    ball_contact: Any,
    base_contact: Any,
    env_ids: torch.Tensor,
) -> dict[str, Any]:
    """Return JSON-safe postconditions from reset caches, never ``sensor.data``."""

    sensor_states = {
        "robot_ball_contact": cached_contact_sensor_state(
            ball_contact,
            env_ids,
            BALL_CONTACT_RESET_CACHE_FIELDS,
            sensor_name="robot_ball_contact",
        ),
        "base_contact": cached_contact_sensor_state(
            base_contact,
            env_ids,
            BASE_CONTACT_RESET_CACHE_FIELDS,
            sensor_name="base_contact",
        ),
    }
    sensor_checks: dict[str, dict[str, Any]] = {}
    all_pass = True
    for sensor_name, state in sensor_states.items():
        zero_fields = (
            BALL_CONTACT_RESET_CACHE_FIELDS
            if sensor_name == "robot_ball_contact"
            else BASE_CONTACT_RESET_CACHE_FIELDS
        )
        zero_values = [state[field] for field in zero_fields]
        finite_cache = all(bool(torch.isfinite(value).all()) for value in zero_values)
        cache_zero = finite_cache and all(bool((value.abs() <= 1.0e-8).all()) for value in zero_values)
        timestamp_zero = bool(torch.isfinite(state["_timestamp"]).all()) and bool(
            (state["_timestamp"].abs() <= 1.0e-8).all()
        )
        timestamp_last_update_zero = bool(torch.isfinite(state["_timestamp_last_update"]).all()) and bool(
            (state["_timestamp_last_update"].abs() <= 1.0e-8).all()
        )
        outdated = state["_is_outdated"].dtype == torch.bool and bool(state["_is_outdated"].all())
        checks = {
            "cached_reset_buffers_finite": finite_cache,
            "cached_reset_buffers_zero": cache_zero,
            "timestamp_zero": timestamp_zero,
            "timestamp_last_update_zero": timestamp_last_update_zero,
            "outdated_until_next_physics_update": outdated,
        }
        checks["pass"] = all(checks.values())
        sensor_checks[sensor_name] = checks
        all_pass = all_pass and bool(checks["pass"])
    return {
        "schema": "sparse_high_task_progress_canary_v1/contact_sensor_reset_receipt/v1",
        "status": "PASS" if all_pass else "FAIL",
        "read_path": "PRIVATE_CACHE_NONLAZY_NO_PUBLIC_DATA_ACCESS",
        "selected_env_ids": env_ids.detach().cpu().to(torch.long).flatten().tolist(),
        "selected_env_count": int(env_ids.numel()),
        "sensor_checks": sensor_checks,
    }


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
    manager = base_env.action_manager
    return manager.get_term("joint_pos") if hasattr(manager, "get_term") else manager._terms["joint_pos"]


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
    """Finite-horizon task with V53 same-tick safety precedence."""

    num_obs = POLICY_OBSERVATION_DIM
    num_actions = POLICY_ACTION_DIM
    num_privileged_obs = None

    def __init__(
        self,
        *,
        low_env: FrozenLowInferenceEnv,
        frozen_low_actor: torch.nn.Module,
        seed: int,
        command_vertices: Sequence[Sequence[float]],
        command_axis_ranges: Sequence[Sequence[float]],
        command_rate_limits: Sequence[float],
        episode_seconds: float = 20.0,
        low_steps_per_high_action: int = 5,
        evaluation_case_count: int | None = None,
        evaluation_hard_case_count: int = 0,
        capture_transition_rows: bool = False,
        geometry: FieldGeometry = FieldGeometry(),
    ) -> None:
        if episode_seconds <= 0.0 or low_steps_per_high_action <= 0:
            raise ValueError("episode duration and action hold must be positive")
        self.low_env = low_env
        self.base_env = low_env.unwrapped
        self.device = torch.device(low_env.device)
        self.num_envs = int(low_env.num_envs)
        self.low_actor = frozen_low_actor
        self.low_actor.eval()
        if any(parameter.requires_grad for parameter in self.low_actor.parameters()):
            raise RuntimeError("qualified low actor is unexpectedly trainable")
        actor_devices = {
            value.device
            for value in tuple(self.low_actor.parameters()) + tuple(self.low_actor.buffers())
        }
        if actor_devices != {self.device}:
            raise RuntimeError(f"low actor and environment devices differ: {actor_devices}")

        self.robot = self.base_env.scene["robot"]
        self.ball = self.base_env.scene["ball"]
        self.ball_contact = self.base_env.scene["robot_ball_contact"]
        self.base_contact = self.base_env.scene["contact_forces"]
        validate_live_robot_body_names(self.robot.body_names)
        self.action_term = live_action_term(self.base_env)
        if not isinstance(self.action_term, NativeSafeJointPositionAction):
            raise RuntimeError(f"unexpected action term: {type(self.action_term).__name__}")
        self.command = writable_command(self.base_env)
        self.env_origins = self.base_env.scene.env_origins
        self.low_control_dt = float(self.base_env.cfg.sim.dt) * int(self.base_env.cfg.decimation)
        if not math.isclose(self.low_control_dt, 0.02, abs_tol=1.0e-12):
            raise RuntimeError(f"live low control period changed: {self.low_control_dt}")
        self.low_steps_per_high_action = int(low_steps_per_high_action)
        self.high_control_dt = self.low_control_dt * self.low_steps_per_high_action
        if not math.isclose(self.high_control_dt, 0.1, abs_tol=1.0e-12):
            raise RuntimeError(f"high control period changed: {self.high_control_dt}")
        self.max_episode_length = math.ceil(episode_seconds / self.high_control_dt)
        if self.max_episode_length != 200:
            raise RuntimeError("V56 episode horizon must be exactly 200 high ticks")
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # This is a true finite horizon.  The adapter intentionally omits the
        # RSL `time_outs` key so PPO never bootstraps across the 20-second end.
        self.cfg = SimpleNamespace(is_finite_horizon=True)
        self.geometry = geometry
        self.seed = int(seed)
        self.sample_serial = 0

        self.command_vertices = torch.as_tensor(command_vertices, dtype=torch.float32, device=self.device)
        self.command_axis_ranges = torch.as_tensor(command_axis_ranges, dtype=torch.float32, device=self.device)
        self.command_rate_limits = torch.as_tensor(command_rate_limits, dtype=torch.float32, device=self.device)
        if self.command_vertices.ndim != 2 or self.command_vertices.shape[1] != 3:
            raise RuntimeError("qualified command vertices changed")
        if self.command_axis_ranges.shape != (3, 2) or self.command_rate_limits.shape != (3,):
            raise RuntimeError("qualified command range/rate changed")
        self.previous_command = torch.zeros(self.num_envs, 3, device=self.device)

        self.goal_xy_w = torch.zeros(self.num_envs, 2, device=self.device)
        self.contact_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.maximum_contact_force = torch.zeros(self.num_envs, device=self.device)
        self.first_contact_native_tick = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.episode_native_tick = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.assignment = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.counts_for_evaluation = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.hard_case = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.goal_side = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.current_sample_serial = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.current_item_seed = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.bucket = [""] * self.num_envs
        self.reset_samples: list[ResetSample | None] = [None] * self.num_envs
        # These tensors are populated from the live simulator state *after*
        # each write.  Episode evidence must never substitute the intended
        # ResetSample for what Isaac actually installed.
        self.live_reset_robot_xy_yaw = torch.zeros(self.num_envs, 3, device=self.device)
        self.live_reset_ball_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.episode_initial_l_m = torch.zeros(self.num_envs, device=self.device)
        self.episode_cumulative_progress = torch.zeros(self.num_envs, device=self.device)
        self.evaluation_samples: Sequence[ResetSample] | None = None
        self.evaluation_cursor = 0
        self.completed_evaluation_cases = 0
        if evaluation_case_count is not None:
            self.evaluation_samples = fixed_evaluation_samples(
                evaluation_case_count,
                seed,
                hard_case_count=evaluation_hard_case_count,
            )

        self.training_initial_resets: list[dict[str, Any]] = []
        self.outcome_counts: Counter[str] = Counter()
        self.bucket_outcome_counts: Counter[str] = Counter()
        self.reward_term_sums: Counter[str] = Counter()
        self.reward_min = math.inf
        self.reward_max = -math.inf
        self.capture_transition_rows = bool(capture_transition_rows)
        self.vector_steps = 0
        self.low_vector_steps = 0
        self.contact_positive_low_ticks = 0
        self.contact_negative_low_ticks = 0
        self.native_safe_guard_changed_elements = 0
        self.native_safe_guard_max_correction_rad = 0.0
        self.command_abs_sum = torch.zeros(3, dtype=torch.float64)
        self.command_sample_count = 0
        self.last_contact_reset_receipt: dict[str, Any] | None = None
        self._observation = torch.zeros(self.num_envs, self.num_obs, device=self.device)
        self._reset_idx(torch.arange(self.num_envs, dtype=torch.long, device=self.device))

    def _next_sample(self) -> tuple[ResetSample, int, bool, int | None]:
        if self.evaluation_samples is not None:
            if self.evaluation_cursor < len(self.evaluation_samples):
                assignment = self.evaluation_cursor
                sample = self.evaluation_samples[assignment]
                self.evaluation_cursor += 1
                return sample, assignment, True, None
            return self.evaluation_samples[0], -1, False, None
        self.sample_serial += 1
        item_seed = self.seed * 1_000_003 + self.sample_serial
        return sample_reset(item_seed), -1, False, item_seed

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        ids = torch.sort(env_ids.to(device=self.device, dtype=torch.long).flatten()).values
        if int(torch.unique(ids).numel()) != int(ids.numel()):
            raise RuntimeError("reset environment IDs are not unique")
        self.base_env._reset_idx(ids)
        count = int(ids.numel())
        defaults = self.robot.data.default_joint_pos[ids].clone()
        self.robot.write_joint_state_to_sim(defaults, torch.zeros_like(defaults), env_ids=ids)
        origins = self.env_origins[ids]
        robot_pose = torch.zeros(count, 7, device=self.device)
        robot_velocity = torch.zeros(count, 6, device=self.device)
        robot_z = self.robot.data.root_pos_w[ids, 2].clone()
        ball_default_root_state = self.ball.data.default_root_state[ids].clone()
        if tuple(ball_default_root_state.shape) != (count, 13) or not bool(torch.isfinite(ball_default_root_state).all()):
            raise RuntimeError("canonical ball default root state changed")
        ball_pose = ball_default_root_state[:, :7].clone()
        ball_pose[:, :3] += origins
        ball_velocity = ball_default_root_state[:, 7:].clone()
        for local_index, env_id in enumerate(ids.detach().cpu().tolist()):
            sample, assignment, counted, item_seed = self._next_sample()
            robot_pose[local_index, 0] = origins[local_index, 0] + sample.robot_xy[0]
            robot_pose[local_index, 1] = origins[local_index, 1] + sample.robot_xy[1]
            robot_pose[local_index, 2] = robot_z[local_index]
            robot_pose[local_index, 3] = math.cos(sample.robot_yaw / 2.0)
            robot_pose[local_index, 6] = math.sin(sample.robot_yaw / 2.0)
            ball_pose[local_index, 0] = origins[local_index, 0] + sample.ball_xy[0]
            ball_pose[local_index, 1] = origins[local_index, 1] + sample.ball_xy[1]
            self.goal_xy_w[env_id, 0] = origins[local_index, 0] + sample.goal_side * self.geometry.field_half_length
            self.goal_xy_w[env_id, 1] = origins[local_index, 1]
            self.goal_side[env_id] = sample.goal_side
            self.current_sample_serial[env_id] = self.sample_serial if item_seed is not None else -1
            self.current_item_seed[env_id] = -1 if item_seed is None else item_seed
            self.assignment[env_id] = assignment
            self.counts_for_evaluation[env_id] = counted
            self.hard_case[env_id] = is_hard_goal_facing_ball_behind(sample)
            self.bucket[env_id] = relative_bucket(sample)
            self.reset_samples[env_id] = sample
            if item_seed is not None and len(self.training_initial_resets) < self.num_envs:
                self.training_initial_resets.append(
                    {
                        "sample_serial": self.sample_serial,
                        "item_seed": item_seed,
                        "robot_xy": list(sample.robot_xy),
                        "robot_yaw": sample.robot_yaw,
                        "ball_xy": list(sample.ball_xy),
                        "goal_side": sample.goal_side,
                        "bucket": relative_bucket(sample),
                    }
                )
        self.robot.write_root_pose_to_sim(robot_pose, env_ids=ids)
        self.robot.write_root_velocity_to_sim(robot_velocity, env_ids=ids)
        self.ball.write_root_pose_to_sim(ball_pose, env_ids=ids)
        self.ball.write_root_velocity_to_sim(ball_velocity, env_ids=ids)
        self.command[ids, :3] = 0.0
        self.previous_command[ids] = 0.0
        self.action_term.reset(ids)
        self.contact_seen[ids] = False
        self.maximum_contact_force[ids] = 0.0
        self.first_contact_native_tick[ids] = -1
        self.episode_native_tick[ids] = 0
        self.episode_length_buf[ids] = 0
        self.episode_cumulative_progress[ids] = 0.0
        if hasattr(self.base_env, "episode_length_buf"):
            self.base_env.episode_length_buf[ids] = 0
        if not bool((self.command[ids, :3].abs() <= 1.0e-8).all()):
            raise RuntimeError("command buffer did not reset")
        previous_action = getattr(self.base_env.action_manager, "prev_action", None)
        if not torch.is_tensor(previous_action) or tuple(previous_action.shape) != (self.num_envs, 12):
            raise RuntimeError("previous action buffer changed")
        if not bool((previous_action[ids].abs() <= 1.0e-8).all()):
            raise RuntimeError("previous action did not reset")

        live_robot_xy = self.robot.data.root_pos_w[ids, :2] - origins[:, :2]
        live_robot_yaw = yaw_from_quaternion(self.robot.data.root_quat_w[ids])
        live_robot = torch.cat((live_robot_xy, live_robot_yaw.unsqueeze(1)), dim=1)
        live_ball = self.ball.data.root_pos_w[ids, :2] - origins[:, :2]
        live_ball_root_state = torch.cat(
            (
                self.ball.data.root_pos_w[ids],
                self.ball.data.root_quat_w[ids],
                self.ball.data.root_lin_vel_w[ids],
                self.ball.data.root_ang_vel_w[ids],
            ),
            dim=1,
        )
        expected_ball_root_state = torch.cat((ball_pose, ball_velocity), dim=1)
        if not bool(
            torch.allclose(
                live_ball_root_state,
                expected_ball_root_state,
                rtol=0.0,
                atol=LIVE_RESET_XY_TOLERANCE_M,
            )
        ):
            raise RuntimeError("live ball reset state differs from canonical default state plus sampled XY")
        for local_index, env_id in enumerate(ids.detach().cpu().tolist()):
            sample = self.reset_samples[env_id]
            if sample is None:
                raise RuntimeError("reset sample disappeared before live receipt")
            expected_robot = torch.tensor(
                [*sample.robot_xy, sample.robot_yaw], dtype=torch.float64
            )
            actual_robot = live_robot[local_index].detach().cpu().to(torch.float64)
            expected_ball = torch.tensor(sample.ball_xy, dtype=torch.float64)
            actual_ball = live_ball[local_index].detach().cpu().to(torch.float64)
            if not bool(torch.allclose(actual_robot[:2], expected_robot[:2], rtol=0.0, atol=LIVE_RESET_XY_TOLERANCE_M)):
                raise RuntimeError("live robot reset XY differs from fixed reset sample")
            yaw_error = math.atan2(
                math.sin(float(actual_robot[2] - expected_robot[2])),
                math.cos(float(actual_robot[2] - expected_robot[2])),
            )
            if abs(yaw_error) > LIVE_RESET_YAW_TOLERANCE_RAD:
                raise RuntimeError("live robot reset yaw differs from fixed reset sample")
            if not bool(torch.allclose(actual_ball, expected_ball, rtol=0.0, atol=LIVE_RESET_XY_TOLERANCE_M)):
                raise RuntimeError("live ball reset XY differs from fixed reset sample")
            self.live_reset_robot_xy_yaw[env_id] = live_robot[local_index]
            self.live_reset_ball_xy[env_id] = live_ball[local_index]
        initial_l, _, _, _ = torch_task_geometry(live_robot[:, :2], live_ball, self.goal_side[ids])
        self.episode_initial_l_m[ids] = initial_l
        self._observation = self._build_observation()
        # SensorBase.reset marks these slots outdated.  Calling the public
        # ``data`` property before the next physics update would lazily refill
        # terminal contact into the just-cleared history, so these are the
        # final public sensor operations in this reset path.  The receipt below
        # deliberately reads only the already-cleared private cache.
        self.ball_contact.reset(ids)
        self.base_contact.reset(ids)
        receipt = self.contact_reset_receipt(ids)
        if receipt["status"] != "PASS":
            raise RuntimeError("ContactSensor cached reset postconditions failed")
        self.last_contact_reset_receipt = receipt

    def contact_reset_cache_snapshot(self, env_ids: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        """Snapshot selected sensor caches for a live selected/unselected reset probe."""

        return {
            "robot_ball_contact": cached_contact_sensor_state(
                self.ball_contact,
                env_ids,
                BALL_CONTACT_RESET_CACHE_FIELDS,
                sensor_name="robot_ball_contact",
            ),
            "base_contact": cached_contact_sensor_state(
                self.base_contact,
                env_ids,
                BASE_CONTACT_RESET_CACHE_FIELDS,
                sensor_name="base_contact",
            ),
        }

    def contact_reset_receipt(self, env_ids: torch.Tensor) -> dict[str, Any]:
        """Expose the non-lazy selected-slot reset postconditions to a real probe."""

        return cached_contact_reset_receipt(self.ball_contact, self.base_contact, env_ids)

    def _low_observation(self) -> torch.Tensor:
        value = extract_policy_tensor(self.low_env.get_observations())
        if value.shape != (self.num_envs, 48) or not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"low observation changed: {tuple(value.shape)}")
        return value

    def _build_observation(self) -> torch.Tensor:
        # Round2 may clear the command-manager tensor after base.step when it
        # refreshes standing flags.  That clearing was not the command the low
        # executed and must not corrupt the next high observation.  Reinstall
        # the held command without advancing physics or processing an action.
        self.command[:, :3] = self.previous_command
        low_observation = self._low_observation()
        if not bool(torch.allclose(low_observation[:, 9:12], self.previous_command, rtol=0.0, atol=1.0e-6)):
            raise RuntimeError("next-high native command slot differs from held applied command")
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
        if not bool(torch.allclose(observation[:, 9:12], self.previous_command, rtol=0.0, atol=1.0e-6)):
            raise RuntimeError("70-D high observation lost the held applied command")
        return observation

    def _ball_contact_force(self) -> torch.Tensor:
        history = getattr(self.ball_contact.data, "force_matrix_w_history", None)
        expected = (self.num_envs, 4, 1, len(ROBOT_BODY_NAMES), 3)
        if not torch.is_tensor(history) or tuple(history.shape) != expected:
            actual = None if not torch.is_tensor(history) else tuple(history.shape)
            raise RuntimeError(f"robot-ball ContactSensor shape changed: {actual}")
        return torch.linalg.vector_norm(history, dim=-1).amax(dim=(1, 2, 3))

    def _base_contact_force(self) -> torch.Tensor:
        names = tuple(str(name) for name in self.base_contact.body_names)
        if "base" not in names:
            raise RuntimeError("base contact sensor lacks base")
        history = getattr(self.base_contact.data, "net_forces_w_history", None)
        if not torch.is_tensor(history) or history.ndim != 4 or history.shape[0] != self.num_envs:
            raise RuntimeError("base contact history is unavailable")
        return torch.linalg.vector_norm(history[:, :, names.index("base"), :], dim=-1).amax(dim=1)

    def _joint_hard_excess(self) -> torch.Tensor:
        position = self.robot.data.joint_pos
        limits = self.robot.data.joint_pos_limits
        below = torch.clamp(limits[..., 0] - position, min=0.0)
        above = torch.clamp(position - limits[..., 1], min=0.0)
        return torch.maximum(below, above).amax(dim=1)

    def get_observations(self) -> torch.Tensor:
        self._observation = self._build_observation()
        return self._observation

    def reset(self) -> torch.Tensor:
        # Construction already installs the first complete reset batch.  A
        # framework-level initial reset must be idempotent or fixed evaluation
        # would silently consume assignments 0..num_envs-1 twice.
        if self.vector_steps == 0:
            return self.get_observations()
        self._reset_idx(torch.arange(self.num_envs, dtype=torch.long, device=self.device))
        return self.get_observations()

    def _terminal_event(
        self,
        env_id: int,
        *,
        goal: bool,
        own_goal: bool,
        boundary: bool,
        fall: bool,
        joint_hard_limit: bool,
        timeout: bool,
        requested_line: bool,
        robot_xy: torch.Tensor,
        ball_xy: torch.Tensor,
        contact_force: float,
        base_force: float,
        hard_excess: float,
        completed_high_ticks: int,
        transition_pre_robot_xy: torch.Tensor,
        transition_pre_ball_xy: torch.Tensor,
    ) -> dict[str, Any]:
        sample = self.reset_samples[env_id]
        if sample is None:
            raise RuntimeError("terminal row lost reset sample")
        unsafe = bool(own_goal or boundary or fall or joint_hard_limit)
        outcome = (
            "goal" if goal else "own_goal" if own_goal else "fall" if fall
            else "boundary" if boundary else "joint_hard_limit" if joint_hard_limit
            else "timeout"
        )
        pre_geometry = scalar_task_geometry(
            transition_pre_robot_xy.detach().cpu().tolist(),
            transition_pre_ball_xy.detach().cpu().tolist(),
            int(self.goal_side[env_id].item()),
        )
        final_geometry = scalar_task_geometry(
            robot_xy.detach().cpu().tolist(),
            ball_xy.detach().cpu().tolist(),
            int(self.goal_side[env_id].item()),
        )
        transition_progress = (pre_geometry.path_length_m - final_geometry.path_length_m) / PATH_NORMALIZER_M
        cumulative_progress = float(self.episode_cumulative_progress[env_id].item()) + transition_progress
        reset_ball = self.live_reset_ball_xy[env_id].detach().cpu().tolist()
        terminal_ball = ball_xy.detach().cpu().tolist()
        side = int(self.goal_side[env_id].item())
        return {
            "schema": EPISODE_ROW_SCHEMA,
            "evaluation_assignment": int(self.assignment[env_id].item()),
            "counts_for_evaluation": bool(self.counts_for_evaluation[env_id].item()),
            "hard_goal_facing_ball_behind": bool(self.hard_case[env_id].item()),
            "bucket": self.bucket[env_id],
            "goal_side": int(self.goal_side[env_id].item()),
            "sample_serial": int(self.current_sample_serial[env_id].item()),
            "item_seed": int(self.current_item_seed[env_id].item()),
            "reset_robot_xy_m": self.live_reset_robot_xy_yaw[env_id, :2].detach().cpu().tolist(),
            "reset_robot_yaw_rad": float(self.live_reset_robot_xy_yaw[env_id, 2].item()),
            "reset_ball_xy_m": self.live_reset_ball_xy[env_id].detach().cpu().tolist(),
            "outcome": outcome,
            "goal": bool(goal),
            "own_goal": bool(own_goal),
            "boundary": bool(boundary),
            "fall": bool(fall),
            "joint_hard_limit": bool(joint_hard_limit),
            "timeout": bool(timeout),
            "unsafe_pre_goal": unsafe,
            "requested_goal_line_reached": bool(requested_line),
            "contact_seen": bool(self.contact_seen[env_id].item()),
            "first_true_contact_native_tick": (
                None
                if int(self.first_contact_native_tick[env_id].item()) < 0
                else int(self.first_contact_native_tick[env_id].item())
            ),
            "maximum_contact_force_n": float(self.maximum_contact_force[env_id].item()),
            "terminal_contact_force_n": float(contact_force),
            "terminal_base_contact_force_n": float(base_force),
            "terminal_joint_hard_limit_excess_rad": float(hard_excess),
            "native_safe_guard_changed_elements": int(
                (self.action_term.last_guard_correction[env_id].abs() > 1.0e-9).sum().item()
            ),
            "native_safe_guard_max_correction_rad": float(
                self.action_term.last_guard_correction[env_id].abs().max().item()
            ),
            "episode_high_ticks": int(completed_high_ticks),
            "episode_native_ticks": int(self.episode_native_tick[env_id].item()),
            "terminal_robot_xy_m": robot_xy.detach().cpu().tolist(),
            "terminal_ball_xy_m": terminal_ball,
            "initial_l_m": float(self.episode_initial_l_m[env_id].item()),
            "final_l_m": final_geometry.path_length_m,
            "cumulative_progress_reward": cumulative_progress,
            "requested_goal_axis_ball_displacement_m": side * (terminal_ball[0] - reset_ball[0]),
            "abs_lateral_ball_drift_m": abs(terminal_ball[1] - reset_ball[1]),
        }
    def disable_transition_capture(self) -> None:
        self.capture_transition_rows = False

    @torch.inference_mode(False)
    def step(self, latent_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        if latent_action.shape != (self.num_envs, POLICY_ACTION_DIM):
            raise RuntimeError(f"policy action shape changed: {tuple(latent_action.shape)}")
        if not bool(torch.isfinite(latent_action).all()):
            raise RuntimeError("policy action is non-finite")

        # Freeze every pre-transition identity before any command or physics
        # step.  These tensors remain valid even when _reset_idx runs below.
        pre_robot_xy = (self.robot.data.root_pos_w[:, :2] - self.env_origins[:, :2]).detach().clone()
        pre_ball_xy = (self.ball.data.root_pos_w[:, :2] - self.env_origins[:, :2]).detach().clone()
        pre_goal_side = self.goal_side.detach().clone()
        pre_sample_serial = self.current_sample_serial.detach().clone()
        pre_item_seed = self.current_item_seed.detach().clone()
        pre_episode_step = self.episode_length_buf.detach().clone()
        pre_reset_robot = self.live_reset_robot_xy_yaw.detach().clone()
        pre_reset_ball = self.live_reset_ball_xy.detach().clone()

        applied_command, bounded_action = limit_latent_high_command(
            latent_action=latent_action.detach().to(self.device),
            previous_command=self.previous_command,
            vertices=self.command_vertices,
            axis_ranges=self.command_axis_ranges,
            rate_limits_per_second=self.command_rate_limits,
            high_control_dt=self.high_control_dt,
        )
        if not bool(torch.isfinite(applied_command).all()):
            raise RuntimeError("applied command is non-finite")
        self.previous_command = applied_command.clone()
        self.command_abs_sum += applied_command.detach().abs().sum(dim=0).cpu().double()
        self.command_sample_count += self.num_envs

        active = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        terminal_goal = torch.zeros_like(active)
        terminal_own = torch.zeros_like(active)
        terminal_boundary = torch.zeros_like(active)
        terminal_fall = torch.zeros_like(active)
        terminal_joint = torch.zeros_like(active)
        terminal_events_by_env: dict[int, dict[str, Any]] = {}
        contact_this_high = torch.zeros_like(active)
        native_ticks_executed = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Post-state fields are updated only while a slot is active.  Therefore
        # a terminal slot is frozen at the exact native tick that triggered
        # termination rather than drifting through the remaining held ticks.
        post_robot_xy = pre_robot_xy.clone()
        post_ball_xy = pre_ball_xy.clone()
        post_contact_force = torch.zeros(self.num_envs, device=self.device)
        post_base_force = torch.zeros(self.num_envs, device=self.device)
        post_hard_excess = torch.zeros(self.num_envs, device=self.device)

        for _low_tick in range(self.low_steps_per_high_action):
            tick_active = active.clone()
            self.command[:, :3] = 0.0
            self.command[tick_active, :3] = applied_command[tick_active]
            native_observation = self._low_observation()
            if not bool(torch.allclose(native_observation[:, 9:12], self.command[:, :3], rtol=0.0, atol=1.0e-6)):
                raise RuntimeError("native command slice differs from reinstalled command")
            low_action = torch.zeros(self.num_envs, 12, device=self.device)
            if bool(tick_active.any()):
                with torch.no_grad():
                    low_action[tick_active] = self.low_actor(native_observation[tick_active])
            if not bool(torch.isfinite(low_action).all()):
                raise RuntimeError("low action is non-finite")
            native_done = normalize_done(
                self.low_env.step(low_action.detach()),
                num_envs=self.num_envs,
                device=self.device,
            )
            if bool(native_done.any()):
                raise RuntimeError("disabled native termination emitted done before V56 snapshot")

            contact_force = self._ball_contact_force()
            base_force = self._base_contact_force()
            hard_excess = self._joint_hard_excess()
            robot_xy = self.robot.data.root_pos_w[:, :2] - self.env_origins[:, :2]
            ball_xy = self.ball.data.root_pos_w[:, :2] - self.env_origins[:, :2]
            post_robot_xy[tick_active] = robot_xy[tick_active]
            post_ball_xy[tick_active] = ball_xy[tick_active]
            post_contact_force[tick_active] = contact_force[tick_active]
            post_base_force[tick_active] = base_force[tick_active]
            post_hard_excess[tick_active] = hard_excess[tick_active]
            native_ticks_executed[tick_active] += 1

            correction = self.action_term.last_guard_correction.detach()
            self.native_safe_guard_changed_elements += int((correction.abs() > 1.0e-9).sum().item())
            self.native_safe_guard_max_correction_rad = max(
                self.native_safe_guard_max_correction_rad,
                float(correction.abs().max().item()),
            )
            contact = contact_force >= TRUE_CONTACT_FORCE_THRESHOLD_N
            new_contact = contact & ~self.contact_seen & tick_active
            self.first_contact_native_tick[new_contact] = self.episode_native_tick[new_contact]
            self.contact_seen |= contact & tick_active
            self.maximum_contact_force = torch.maximum(
                self.maximum_contact_force,
                torch.where(tick_active, contact_force, torch.zeros_like(contact_force)),
            )
            contact_this_high |= contact & tick_active
            self.contact_positive_low_ticks += int((contact & tick_active).sum().item())
            self.contact_negative_low_ticks += int((~contact & tick_active).sum().item())
            self.episode_native_tick[tick_active] += 1

            selected = self.goal_side.to(ball_xy.dtype) * ball_xy[:, 0]
            inside = ball_xy[:, 1].abs() <= TERMINAL_GOAL_HALF_WIDTH_M
            requested_line = self.contact_seen & (selected >= GOAL_CENTER_X_M) & inside & tick_active
            own = (selected <= -GOAL_CENTER_X_M) & inside & tick_active
            boundary = boundary_violation(robot_xy, ball_xy, self.geometry) & tick_active
            fall = (base_force > BASE_FALL_FORCE_THRESHOLD_N) & tick_active
            joint = (hard_excess > JOINT_HARD_LIMIT_TOLERANCE_RAD) & tick_active
            unsafe = own | boundary | fall | joint
            goal = requested_line & ~unsafe
            terminal_now = requested_line | unsafe

            terminal_goal |= goal
            terminal_own |= own
            terminal_boundary |= boundary
            terminal_fall |= fall
            terminal_joint |= joint
            for env_id in torch.nonzero(terminal_now, as_tuple=False).flatten().detach().cpu().tolist():
                terminal_events_by_env[env_id] = self._terminal_event(
                    env_id,
                    goal=bool(goal[env_id].item()),
                    own_goal=bool(own[env_id].item()),
                    boundary=bool(boundary[env_id].item()),
                    fall=bool(fall[env_id].item()),
                    joint_hard_limit=bool(joint[env_id].item()),
                    timeout=False,
                    requested_line=bool(requested_line[env_id].item()),
                    robot_xy=robot_xy[env_id],
                    ball_xy=ball_xy[env_id],
                    contact_force=float(contact_force[env_id].item()),
                    base_force=float(base_force[env_id].item()),
                    hard_excess=float(hard_excess[env_id].item()),
                    completed_high_ticks=int(self.episode_length_buf[env_id].item()) + 1,
                    transition_pre_robot_xy=pre_robot_xy[env_id],
                    transition_pre_ball_xy=pre_ball_xy[env_id],
                )
            active &= ~terminal_now
            self.low_vector_steps += 1

        self.episode_length_buf += 1
        timeout = (self.episode_length_buf >= self.max_episode_length) & active
        if bool(timeout.any()):
            for env_id in torch.nonzero(timeout, as_tuple=False).flatten().detach().cpu().tolist():
                terminal_events_by_env[env_id] = self._terminal_event(
                    env_id,
                    goal=False,
                    own_goal=False,
                    boundary=False,
                    fall=False,
                    joint_hard_limit=False,
                    timeout=True,
                    requested_line=False,
                    robot_xy=post_robot_xy[env_id],
                    ball_xy=post_ball_xy[env_id],
                    contact_force=float(post_contact_force[env_id].item()),
                    base_force=float(post_base_force[env_id].item()),
                    hard_excess=float(post_hard_excess[env_id].item()),
                    completed_high_ticks=int(self.episode_length_buf[env_id].item()),
                    transition_pre_robot_xy=pre_robot_xy[env_id],
                    transition_pre_ball_xy=pre_ball_xy[env_id],
                )

        progress, geometry = torch_progress(
            pre_robot_xy,
            pre_ball_xy,
            post_robot_xy,
            post_ball_xy,
            pre_goal_side,
        )
        strict_reward = terminal_goal.to(progress.dtype) - terminal_own.to(progress.dtype)
        reward = progress + strict_reward
        if not bool(torch.isfinite(reward).all()):
            raise RuntimeError("V56 task-progress reward is non-finite")
        if bool(terminal_goal.any()) and not bool((reward[terminal_goal] > 0.0).all()):
            raise RuntimeError("goal transition total reward lost positive sign")
        if bool(terminal_own.any()) and not bool((reward[terminal_own] < 0.0).all()):
            raise RuntimeError("own-goal transition total reward lost negative sign")
        self.episode_cumulative_progress += progress
        self.reward_term_sums["progress"] += float(progress.sum().item())
        self.reward_term_sums["strict_goal"] += float(strict_reward.sum().item())
        self.reward_term_sums["total"] += float(reward.sum().item())
        self.reward_min = min(self.reward_min, float(reward.min().item()))
        self.reward_max = max(self.reward_max, float(reward.max().item()))

        done = terminal_goal | terminal_own | terminal_boundary | terminal_fall | terminal_joint | timeout
        done_ids = torch.sort(torch.nonzero(done, as_tuple=False).flatten()).values
        terminal_events = [terminal_events_by_env[int(env_id)] for env_id in done_ids.detach().cpu().tolist()]
        for event in terminal_events:
            outcome = str(event["outcome"])
            self.outcome_counts[outcome] += 1
            self.bucket_outcome_counts[f"{event['bucket']}/{outcome}"] += 1
            if event["counts_for_evaluation"]:
                self.completed_evaluation_cases += 1

        transition_rows: list[dict[str, Any]] = []
        if self.capture_transition_rows:
            rollout_iteration, high_step_index = divmod(self.vector_steps, self.max_episode_length)
            for env_id in range(self.num_envs):
                is_done = bool(done[env_id].item())
                reason = (
                    "GOAL" if bool(terminal_goal[env_id].item())
                    else "OWN_GOAL" if bool(terminal_own[env_id].item())
                    else "FALL" if bool(terminal_fall[env_id].item())
                    else "BOUNDARY" if bool(terminal_boundary[env_id].item())
                    else "JOINT_HARD_LIMIT" if bool(terminal_joint[env_id].item())
                    else "TIMEOUT" if bool(timeout[env_id].item())
                    else "NONE"
                )
                transition_rows.append({
                    "schema": TRANSITION_ROW_SCHEMA,
                    "global_transition_index": high_step_index * self.num_envs + env_id,
                    "rollout_iteration": rollout_iteration,
                    "high_step_index": high_step_index,
                    "env_id": env_id,
                    "sample_serial": int(pre_sample_serial[env_id].item()),
                    "episode_serial_before_step": int(pre_sample_serial[env_id].item()),
                    "item_seed": int(pre_item_seed[env_id].item()),
                    "episode_high_step_index_before_step": int(pre_episode_step[env_id].item()),
                    "goal_side": int(pre_goal_side[env_id].item()),
                    "reset_robot_xy_m": pre_reset_robot[env_id, :2].detach().cpu().tolist(),
                    "reset_robot_yaw_rad": float(pre_reset_robot[env_id, 2].item()),
                    "reset_ball_xy_m": pre_reset_ball[env_id].detach().cpu().tolist(),
                    "pre_robot_xy_m": pre_robot_xy[env_id].detach().cpu().tolist(),
                    "pre_ball_xy_m": pre_ball_xy[env_id].detach().cpu().tolist(),
                    "post_robot_xy_pre_reset_m": post_robot_xy[env_id].detach().cpu().tolist(),
                    "post_ball_xy_pre_reset_m": post_ball_xy[env_id].detach().cpu().tolist(),
                    "goal_xy_m": geometry["goal_xy"][env_id].detach().cpu().tolist(),
                    "raw_staging_target_pre_xy_m": geometry["raw_staging_target_pre_xy"][env_id].detach().cpu().tolist(),
                    "raw_staging_target_post_xy_m": geometry["raw_staging_target_post_xy"][env_id].detach().cpu().tolist(),
                    "staging_target_pre_xy_m": geometry["staging_target_pre_xy"][env_id].detach().cpu().tolist(),
                    "staging_target_post_xy_m": geometry["staging_target_post_xy"][env_id].detach().cpu().tolist(),
                    "staging_projection_pre_applied": bool(geometry["staging_projection_pre_applied"][env_id].item()),
                    "staging_projection_post_applied": bool(geometry["staging_projection_post_applied"][env_id].item()),
                    "l_pre_m": float(geometry["l_pre_m"][env_id].item()),
                    "l_post_m": float(geometry["l_post_m"][env_id].item()),
                    "normalizer_m": PATH_NORMALIZER_M,
                    "progress_reward": float(progress[env_id].item()),
                    "strict_goal_reward": float(strict_reward[env_id].item()),
                    "total_reward": float(reward[env_id].item()),
                    "raw_action": latent_action[env_id].detach().cpu().tolist(),
                    "applied_command": applied_command[env_id].detach().cpu().tolist(),
                    "native_ticks_executed": int(native_ticks_executed[env_id].item()),
                    "terminal_reason": reason,
                    "goal": bool(terminal_goal[env_id].item()),
                    "own_goal": bool(terminal_own[env_id].item()),
                    "boundary": bool(terminal_boundary[env_id].item()),
                    "fall": bool(terminal_fall[env_id].item()),
                    "joint_hard_limit": bool(terminal_joint[env_id].item()),
                    "timeout": bool(timeout[env_id].item()),
                    "unsafe_same_tick": bool(
                        terminal_own[env_id].item()
                        or terminal_boundary[env_id].item()
                        or terminal_fall[env_id].item()
                        or terminal_joint[env_id].item()
                    ),
                    "done": is_done,
                    "reset_after_step": is_done,
                    "contact_seen": bool(self.contact_seen[env_id].item()),
                    "maximum_contact_force_n": float(self.maximum_contact_force[env_id].item()),
                    "terminal_contact_force_n": float(post_contact_force[env_id].item()),
                    "terminal_base_contact_force_n": float(post_base_force[env_id].item()),
                    "terminal_joint_hard_limit_excess_rad": float(post_hard_excess[env_id].item()),
                    "post_state_source": "SAME_TICK_PHYSICAL_PRE_RESET",
                })

        step_receipt = {
            "bounded_action": bounded_action.detach().clone(),
            "applied_command": applied_command.detach().clone(),
            "contact_this_high": contact_this_high.detach().clone(),
            "done": done.detach().clone(),
            "reward": reward.detach().clone(),
            "progress_reward": progress.detach().clone(),
            "strict_goal_reward": strict_reward.detach().clone(),
            "evaluation_assignment_before_reset": self.assignment.detach().clone(),
            "counts_for_evaluation_before_reset": self.counts_for_evaluation.detach().clone(),
        }
        self.vector_steps += 1
        if done_ids.numel():
            self._reset_idx(done_ids)
        observation = self.get_observations()
        extras = {
            "terminal_events": terminal_events,
            "finite_horizon_timeout_diagnostic": timeout,
            "v56_pre_reset_step_receipt": step_receipt,
            "v56_transition_rows": transition_rows,
            "task": {
                "contact_fraction": float(contact_this_high.float().mean().item()),
                "bounded_action_abs_mean": float(bounded_action.abs().mean().item()),
            },
        }
        return observation, reward, done, extras

    def summary(self) -> dict[str, Any]:
        denominator = max(1, self.vector_steps * self.num_envs)
        return {
            "status": "CURRENT_TASK_PROGRESS_HIGH_ENV_COMPLETE",
            "observation_dim": self.num_obs,
            "action_dim": self.num_actions,
            "high_policy_rate_hz": round(1.0 / self.high_control_dt),
            "low_policy_rate_hz": round(1.0 / self.low_control_dt),
            "low_steps_per_high_action": self.low_steps_per_high_action,
            "finite_horizon": True,
            "time_outs_bootstrap_key_emitted": False,
            "terminal_semantics": "V53_STRICT_UNSAFE_SAME_TICK_REJECTS_GOAL_PRE_RESET_SNAPSHOT",
            "actual_joint_hard_limit_is_terminal": True,
            "native_safe_guard_correction_is_terminal": False,
            "curriculum_registered": False,
            "teacher_registered": False,
            "runtime_router_registered": False,
            "scripted_phase_transition": False,
            "kick_registered": False,
            "native_base_reward_may_be_registered": True,
            "native_base_reward_discarded": True,
            "native_base_reward_consumed_by_high_or_ppo": False,
            "high_ppo_reward_kind": "TASK_PROGRESS_PLUS_STRICT_TERMINAL",
            "strict_terminal_reward_values": [-1.0, 0.0, 1.0],
            "high_ppo_reward_min": None if self.vector_steps == 0 else self.reward_min,
            "high_ppo_reward_max": None if self.vector_steps == 0 else self.reward_max,
            "dense_return_authorizes_behavior": False,
            "contact_is_terminal": False,
            "contact_positive_low_ticks": self.contact_positive_low_ticks,
            "contact_negative_low_ticks": self.contact_negative_low_ticks,
            "outcome_counts": dict(sorted(self.outcome_counts.items())),
            "bucket_outcome_counts": dict(sorted(self.bucket_outcome_counts.items())),
            "reward_term_means": {
                name: value / denominator for name, value in sorted(self.reward_term_sums.items())
            },
            "mean_abs_applied_command": (
                self.command_abs_sum / max(1, self.command_sample_count)
            ).tolist(),
            "evaluation_cases_completed": self.completed_evaluation_cases,
            "high_vector_steps": self.vector_steps,
            "low_vector_steps": self.low_vector_steps,
            "native_safe_guard_changed_elements": self.native_safe_guard_changed_elements,
            "native_safe_guard_max_correction_rad": self.native_safe_guard_max_correction_rad,
            "joint_guard_telemetry": self.action_term.telemetry(),
            "training_initial_reset_count": len(self.training_initial_resets),
            "training_initial_resets": self.training_initial_resets,
        }

    def close(self) -> None:
        self.low_env.close()


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
            raise RuntimeError("finite-horizon V56 must not expose RSL timeout bootstrap key")
        return self._pack(observation), reward, done, extras

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)
