#!/usr/bin/env python3
"""Real-Isaac causal gate for the V56 ContactSensor reset lifecycle."""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import math
import os
from pathlib import Path
import traceback
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--contract", type=Path, required=True)
parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=Path, required=True)
parser.add_argument("--task", default="Isaac-Go2-Soccer-Milestone2-v0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

from source_bootstrap import configure_source_roots  # noqa: E402

SOURCE_RECEIPTS = configure_source_roots()

import torch  # noqa: E402
import isaac_go2_low_level_rl.tasks  # noqa: E402,F401
import isaac_go2_low_level_rl.tasks.manager_based.soccer  # noqa: E402,F401

from canary_protocol import load_contract  # noqa: E402
from runtime_common import atomic_bytes, atomic_json, make_runtime, runtime_provenance  # noqa: E402
from staging_contact_contract import (  # noqa: E402
    CONTACT_BODY_CONTRACT_SHA256,
    CONTACT_QUALIFICATION_JOB,
    ROBOT_BODY_NAMES,
)


SCHEMA = "sparse_high_task_progress_canary_v1/contact_reset_lifecycle_probe/v2"
NUM_ENVS = 8
PARTIAL_IDS = (1, 4, 6)
CALF_CANDIDATES = ("FL_calf", "FR_calf", "RL_calf", "RR_calf")
MAX_RAW_TICKS_PER_CANDIDATE = 4
BALL_FIELDS = (
    "net_forces_w",
    "net_forces_w_history",
    "force_matrix_w",
    "force_matrix_w_history",
)
BASE_FIELDS = (
    "net_forces_w",
    "net_forces_w_history",
    "current_air_time",
    "last_air_time",
    "current_contact_time",
    "last_contact_time",
)
ZERO_TOLERANCE = 0.0
STATE_TOLERANCE = 1.0e-6
NON_CONTACT_FORCE_LIMIT_N = 1.0
EXPECTED_PHYSICS_DT_S = 0.005

CURRENT_LANE = "runtime_construction"
COMPLETED_LANES: list[str] = []
LANE_RECEIPTS: dict[str, Any] = {}

FAILURE_CLASS_BY_LANE = {
    "runtime_construction": "RUNTIME_CONSTRUCTION_ERROR",
    "contact_qualification": "CONTACT_CONTRACT_ERROR",
    "physical_partial_reset": "PHYSICAL_PARTIAL_RESET_ERROR",
    "post_reset_raw_query": "POST_RESET_RAW_QUERY_ERROR",
    "first_post_reset_physics_tick": "FIRST_POST_RESET_PHYSICS_TICK_ERROR",
    "production_low_step": "PRODUCTION_LOW_STEP_ERROR",
    "sentinel_partial": "SENTINEL_PARTIAL_ERROR",
    "sentinel_full": "SENTINEL_FULL_ERROR",
    "final_checks": "FINAL_CHECK_ERROR",
}


class ProbeFailure(RuntimeError):
    def __init__(self, classification: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.classification = classification
        self.details = {} if details is None else details


def _begin_lane(name: str) -> None:
    global CURRENT_LANE
    CURRENT_LANE = name


def _publish_lane(name: str, receipt: dict[str, Any], *, complete: bool = False) -> None:
    LANE_RECEIPTS[name] = receipt
    if complete and name not in COMPLETED_LANES:
        COMPLETED_LANES.append(name)


def _fail(classification: str, message: str, **details: Any) -> None:
    raise ProbeFailure(classification, message, details)


def _json_safe(value: Any) -> Any:
    """Make failure diagnostics durable even when the failed value is non-finite."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if torch.is_tensor(value):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _raw_tensor(sensor: Any, field: str) -> torch.Tensor:
    data = getattr(sensor, "_data", None)
    value = None if data is None else getattr(data, field, None)
    if not torch.is_tensor(value) or value.shape[0] != NUM_ENVS:
        actual = None if not torch.is_tensor(value) else tuple(value.shape)
        raise RuntimeError(f"ContactSensor raw field changed: {field}={actual}")
    return value


def _sensor_snapshot(sensor: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    data = getattr(sensor, "_data", None)
    tensor_fields = tuple(sorted(name for name, value in vars(data).items() if torch.is_tensor(value)))
    if tensor_fields != tuple(sorted(fields)):
        raise RuntimeError(
            f"ContactSensor private tensor field set changed: expected={tuple(sorted(fields))} "
            f"actual={tensor_fields}"
        )
    outdated = getattr(sensor, "_is_outdated", None)
    timestamp = getattr(sensor, "_timestamp", None)
    last_update = getattr(sensor, "_timestamp_last_update", None)
    if not all(torch.is_tensor(value) and value.shape == (NUM_ENVS,) for value in (outdated, timestamp, last_update)):
        raise RuntimeError("ContactSensor lifecycle metadata changed")
    return {
        "fields": {name: _raw_tensor(sensor, name).clone() for name in fields},
        "outdated": outdated.clone(),
        "timestamp": timestamp.clone(),
        "timestamp_last_update": last_update.clone(),
    }


def _snapshot_equal(before: dict[str, Any], after: dict[str, Any], ids: torch.Tensor) -> bool:
    return bool(
        all(torch.equal(before["fields"][name][ids], after["fields"][name][ids]) for name in before["fields"])
        and torch.equal(before["outdated"][ids], after["outdated"][ids])
        and torch.equal(before["timestamp"][ids], after["timestamp"][ids])
        and torch.equal(before["timestamp_last_update"][ids], after["timestamp_last_update"][ids])
    )


def _snapshot_delta(before: dict[str, Any], after: dict[str, Any], ids: torch.Tensor) -> float:
    values: list[float] = []
    pairs = [(before["fields"][name][ids], after["fields"][name][ids]) for name in before["fields"]]
    pairs.extend(
        [
            (before["outdated"][ids], after["outdated"][ids]),
            (before["timestamp"][ids], after["timestamp"][ids]),
            (before["timestamp_last_update"][ids], after["timestamp_last_update"][ids]),
        ]
    )
    for left, right in pairs:
        if left.dtype == torch.bool:
            values.append(float(torch.ne(left, right).any().item()))
        else:
            values.append(float((left - right).abs().max().item()))
    return max(values, default=0.0)


def _snapshot_digest(
    ball: dict[str, Any], base: dict[str, Any], ids: torch.Tensor
) -> str:
    digest = hashlib.sha256()
    for sensor_name, snapshot in (("ball", ball), ("base", base)):
        items = [(f"field:{name}", value) for name, value in sorted(snapshot["fields"].items())]
        items.extend(
            [
                ("outdated", snapshot["outdated"]),
                ("timestamp", snapshot["timestamp"]),
                ("timestamp_last_update", snapshot["timestamp_last_update"]),
            ]
        )
        for item_name, value in items:
            selected = value[ids].detach().contiguous().cpu()
            digest.update(f"{sensor_name}:{item_name}:{selected.dtype}:{tuple(selected.shape)}\n".encode())
            digest.update(selected.numpy().tobytes())
    return digest.hexdigest()


def _inject_sensor_sentinel(sensor: Any, fields: tuple[str, ...], base: float) -> None:
    for offset, name in enumerate(fields, 1):
        value = _raw_tensor(sensor, name)
        slot_shape = (NUM_ENVS,) + (1,) * (value.ndim - 1)
        slots = torch.arange(NUM_ENVS, dtype=value.dtype, device=value.device).reshape(slot_shape)
        value.copy_(torch.full_like(value, base + float(offset)) + slots * 0.125)
    slots = torch.arange(NUM_ENVS, dtype=sensor._timestamp.dtype, device=sensor._timestamp.device)
    sensor._timestamp.copy_(torch.full_like(sensor._timestamp, base + 20.0) + slots * 0.125)
    sensor._timestamp_last_update.copy_(torch.full_like(sensor._timestamp_last_update, base + 19.0) + slots * 0.125)
    sensor._is_outdated.fill_(False)


def _selected_cache_zero(sensor: Any, fields: tuple[str, ...], ids: torch.Tensor) -> tuple[bool, float]:
    maxima = [float(_raw_tensor(sensor, name)[ids].abs().max().item()) for name in fields]
    maximum = max(maxima, default=0.0)
    return maximum <= ZERO_TOLERANCE, maximum


def _selected_cache_dirty(sensor: Any, fields: tuple[str, ...], ids: torch.Tensor) -> bool:
    return all(bool((_raw_tensor(sensor, name)[ids].abs() > 0.0).all()) for name in fields)


def _selected_metadata_reset(sensor: Any, ids: torch.Tensor) -> bool:
    return bool(
        sensor._is_outdated[ids].all()
        and (sensor._timestamp[ids] == 0.0).all()
        and (sensor._timestamp_last_update[ids] == 0.0).all()
    )


def _unselected_unchanged(
    sensor: Any,
    fields: tuple[str, ...],
    before: dict[str, Any],
    ids: torch.Tensor,
) -> bool:
    return bool(
        all(torch.equal(_raw_tensor(sensor, name)[ids], before["fields"][name][ids]) for name in fields)
        and torch.equal(sensor._is_outdated[ids], before["outdated"][ids])
        and torch.equal(sensor._timestamp[ids], before["timestamp"][ids])
        and torch.equal(sensor._timestamp_last_update[ids], before["timestamp_last_update"][ids])
    )


def _live_ball_root_state(env: Any, ids: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            env.ball.data.root_pos_w[ids],
            env.ball.data.root_quat_w[ids],
            env.ball.data.root_lin_vel_w[ids],
            env.ball.data.root_ang_vel_w[ids],
        ),
        dim=1,
    )


def _expected_ball_root_state(env: Any, ids: torch.Tensor) -> torch.Tensor:
    expected = env.ball.data.default_root_state[ids].clone()
    if tuple(expected.shape) != (int(ids.numel()), 13):
        _fail("CANONICAL_BALL_STATE_ERROR", "canonical ball default root state shape changed", shape=list(expected.shape))
    expected[:, :3] += env.env_origins[ids]
    for local_index, env_id in enumerate(ids.detach().cpu().tolist()):
        sample = env.reset_samples[env_id]
        if sample is None:
            _fail("CANONICAL_BALL_STATE_ERROR", "reset sample missing", env_id=env_id)
        expected[local_index, 0] = env.env_origins[env_id, 0] + float(sample.ball_xy[0])
        expected[local_index, 1] = env.env_origins[env_id, 1] + float(sample.ball_xy[1])
    return expected


def _dirty_ball_state(env: Any, ids: torch.Tensor, *, z_offset: float) -> None:
    pose = torch.cat((env.ball.data.root_pos_w[ids], env.ball.data.root_quat_w[ids]), dim=1).clone()
    pose[:, 2] = _expected_ball_root_state(env, ids)[:, 2] + z_offset
    velocity = torch.tensor(
        [0.37, -0.23, 0.41, 0.19, -0.17, 0.29],
        dtype=pose.dtype,
        device=pose.device,
    ).repeat(int(ids.numel()), 1)
    env.ball.write_root_pose_to_sim(pose, env_ids=ids)
    env.ball.write_root_velocity_to_sim(velocity, env_ids=ids)


def _ball_reset_metrics(env: Any, ids: torch.Tensor) -> dict[str, Any]:
    actual = _live_ball_root_state(env, ids)
    expected = _expected_ball_root_state(env, ids)
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    maximum = float((actual - expected).abs().max().item()) if finite else None
    return {
        "selected_ball_root_state_shape": list(actual.shape),
        "expected_ball_root_state_13d": expected.detach().cpu().tolist(),
        "actual_ball_root_state_13d": actual.detach().cpu().tolist(),
        "selected_ball_root_state_13d_max_error": maximum,
        "selected_ball_root_state_13d_canonical": bool(
            finite and maximum is not None and maximum <= STATE_TOLERANCE
        ),
    }


def _raw_physics_tick(env: Any) -> None:
    base = env.base_env
    base.scene.write_data_to_sim()
    base.sim.step(render=False)
    base.scene.update(dt=base.physics_dt)


def _normal_low_step(env: Any) -> dict[str, float]:
    env.command[:, :3] = 0.0
    native_observation = env._low_observation()
    with torch.no_grad():
        low_action = env.low_actor(native_observation)
    if tuple(low_action.shape) != (NUM_ENVS, 12) or not bool(torch.isfinite(low_action).all()):
        raise RuntimeError("probe low action changed")
    env.low_env.step(low_action.detach())
    return {
        "low_action_max_abs": float(low_action.abs().max().item()),
        "low_action_min": float(low_action.min().item()),
        "low_action_max": float(low_action.max().item()),
    }


def _selected_reset_metrics(env: Any, ids: torch.Tensor) -> dict[str, Any]:
    ball_zero, ball_max = _selected_cache_zero(env.ball_contact, BALL_FIELDS, ids)
    base_zero, base_max = _selected_cache_zero(env.base_contact, BASE_FIELDS, ids)
    return {
        "selected_ball_cache_zero_after": ball_zero,
        "selected_base_cache_zero_after": base_zero,
        "selected_ball_cache_max_abs_after": ball_max,
        "selected_base_cache_max_abs_after": base_max,
        "selected_ball_metadata_reset": _selected_metadata_reset(env.ball_contact, ids),
        "selected_base_metadata_reset": _selected_metadata_reset(env.base_contact, ids),
        **_ball_reset_metrics(env, ids),
    }


def _contact_contract_receipt(env: Any) -> dict[str, Any]:
    live_names = tuple(str(name) for name in env.robot.body_names)
    base_names = tuple(str(name) for name in env.base_contact.body_names)
    base_names_exact = bool(
        len(base_names) == len(ROBOT_BODY_NAMES)
        and len(set(base_names)) == len(base_names)
        and set(base_names) == set(ROBOT_BODY_NAMES)
    )
    base_body_index_by_accepted_body = (
        [base_names.index(name) for name in ROBOT_BODY_NAMES] if base_names_exact else []
    )
    ball_shape = tuple(_raw_tensor(env.ball_contact, "force_matrix_w").shape)
    base_shape = tuple(_raw_tensor(env.base_contact, "net_forces_w").shape)
    filter_count = int(env.ball_contact.contact_physx_view.filter_count)
    receipt = {
        "qualification_job": CONTACT_QUALIFICATION_JOB,
        "contact_body_contract_sha256": CONTACT_BODY_CONTRACT_SHA256,
        "accepted_body_names": list(ROBOT_BODY_NAMES),
        "live_body_names": list(live_names),
        "base_contact_body_names": list(base_names),
        "base_body_index_by_accepted_body": base_body_index_by_accepted_body,
        "body_count": len(live_names),
        "ball_sensor_body_count": int(env.ball_contact.num_bodies),
        "base_sensor_body_count": int(env.base_contact.num_bodies),
        "filter_count": filter_count,
        "ball_force_matrix_shape": list(ball_shape),
        "base_net_force_shape": list(base_shape),
        "candidate_order": list(CALF_CANDIDATES),
        "max_raw_physics_ticks_per_candidate": MAX_RAW_TICKS_PER_CANDIDATE,
        "contact_threshold_n": NON_CONTACT_FORCE_LIMIT_N,
        "current_same_tick_same_body_required": True,
        "history_used_as_contact_truth": False,
        "ball_displacement_used_as_contact_truth": False,
        "low_actor_used": False,
        "low_env_step_used": False,
        "action_manager_processing_used": False,
        "trials": [],
        "winning_body_name": None,
        "winning_body_index": None,
        "winning_filter_index": None,
        "winning_base_body_index": None,
        "winning_tick": None,
        "pass": False,
    }
    _publish_lane("contact_qualification", receipt)
    expected_ball_shape = (NUM_ENVS, 1, len(ROBOT_BODY_NAMES), 3)
    expected_base_shape = (NUM_ENVS, len(ROBOT_BODY_NAMES), 3)
    if live_names != ROBOT_BODY_NAMES:
        _fail("CONTACT_CONTRACT_ERROR", "live robot body order differs from qualified contract")
    if not base_names_exact:
        _fail(
            "CONTACT_CONTRACT_ERROR",
            "base contact body names differ from the qualified body-name set",
        )
    if ball_shape != expected_ball_shape or base_shape != expected_base_shape:
        _fail(
            "CONTACT_CONTRACT_ERROR",
            "current contact tensor shape differs from qualified contract",
            ball_shape=list(ball_shape),
            base_shape=list(base_shape),
        )
    if int(env.ball_contact.num_bodies) != 1 or int(env.base_contact.num_bodies) != len(ROBOT_BODY_NAMES):
        _fail("CONTACT_CONTRACT_ERROR", "contact sensor body counts changed")
    if filter_count != len(ROBOT_BODY_NAMES):
        _fail("CONTACT_CONTRACT_ERROR", "ball contact filter count changed", filter_count=filter_count)
    if abs(float(env.base_env.physics_dt) - EXPECTED_PHYSICS_DT_S) > 1.0e-12:
        _fail("CONTACT_CONTRACT_ERROR", "physics dt changed", physics_dt=float(env.base_env.physics_dt))
    return receipt


def _establish_contact_and_reset(
    env: Any,
    ids: torch.Tensor,
    unselected_ids: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _begin_lane("contact_qualification")
    qualification = _contact_contract_receipt(env)
    names = tuple(qualification["live_body_names"])
    base_names = tuple(qualification["base_contact_body_names"])
    env_id_values = ids.detach().cpu().tolist()

    for body_name in CALF_CANDIDATES:
        body_index = ROBOT_BODY_NAMES.index(body_name)
        filter_index = body_index
        base_body_index = base_names.index(body_name)
        env._reset_idx(ids)
        pose = torch.cat((env.ball.data.root_pos_w[ids], env.ball.data.root_quat_w[ids]), dim=1).clone()
        pose[:, :3] = env.robot.data.body_pos_w[ids, body_index, :]
        pose[:, 3:] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=pose.dtype, device=pose.device)
        velocity = torch.zeros(int(ids.numel()), 6, dtype=pose.dtype, device=pose.device)
        env.ball.write_root_pose_to_sim(pose, env_ids=ids)
        env.ball.write_root_velocity_to_sim(velocity, env_ids=ids)

        ever_hit = torch.zeros(int(ids.numel()), dtype=torch.bool, device=ids.device)
        first_hit = [-1] * int(ids.numel())
        ball_peaks = [0.0] * int(ids.numel())
        robot_peaks = [0.0] * int(ids.numel())
        trial: dict[str, Any] = {
            "body_name": body_name,
            "body_index": body_index,
            "filter_index": filter_index,
            "base_body_index": base_body_index,
            "ticks_run": 0,
            "ticks": [],
            "per_env": [],
            "all_selected_same_tick_hit": False,
            "witness_tick": None,
        }
        qualification["trials"].append(trial)

        for tick in range(1, MAX_RAW_TICKS_PER_CANDIDATE + 1):
            _raw_physics_tick(env)
            ball_current = env.ball_contact.data.force_matrix_w
            base_current = env.base_contact.data.net_forces_w
            if tuple(ball_current.shape) != (NUM_ENVS, 1, len(names), 3):
                _fail("CONTACT_CONTRACT_ERROR", "ball current shape changed", shape=list(ball_current.shape))
            if tuple(base_current.shape) != (NUM_ENVS, len(base_names), 3):
                _fail("CONTACT_CONTRACT_ERROR", "robot current shape changed", shape=list(base_current.shape))
            ball_force = torch.linalg.vector_norm(ball_current[ids, 0, filter_index, :], dim=-1)
            robot_force = torch.linalg.vector_norm(base_current[ids, base_body_index, :], dim=-1)
            if not bool(torch.isfinite(ball_force).all() and torch.isfinite(robot_force).all()):
                _fail("PHYSICAL_CONTACT_ESTABLISHMENT_ERROR", "non-finite current contact force")
            dual = (ball_force >= NON_CONTACT_FORCE_LIMIT_N) & (robot_force >= NON_CONTACT_FORCE_LIMIT_N)
            rows = []
            for local, env_id in enumerate(env_id_values):
                ball_value = float(ball_force[local].item())
                robot_value = float(robot_force[local].item())
                ball_peaks[local] = max(ball_peaks[local], ball_value)
                robot_peaks[local] = max(robot_peaks[local], robot_value)
                if bool(dual[local].item()) and first_hit[local] < 0:
                    first_hit[local] = tick
                rows.append(
                    {
                        "env_id": env_id,
                        "ball_current_n": ball_value,
                        "robot_current_n": robot_value,
                        "dual_threshold_hit": bool(dual[local].item()),
                    }
                )
            ever_hit |= dual
            trial["ticks_run"] = tick
            trial["ticks"].append({"tick": tick, "per_env": rows})
            trial["all_selected_same_tick_hit"] = bool(dual.all().item())
            if trial["all_selected_same_tick_hit"]:
                trial["witness_tick"] = tick
            _publish_lane("contact_qualification", qualification)
            if trial["all_selected_same_tick_hit"]:
                break

        trial["per_env"] = [
            {
                "env_id": env_id,
                "ball_current_peak_n": ball_peaks[local],
                "robot_current_peak_n": robot_peaks[local],
                "first_dual_hit_tick": first_hit[local],
                "ever_same_tick_dual_hit": bool(ever_hit[local].item()),
            }
            for local, env_id in enumerate(env_id_values)
        ]
        _publish_lane("contact_qualification", qualification)
        if not trial["all_selected_same_tick_hit"]:
            continue

        qualification["winning_body_name"] = body_name
        qualification["winning_body_index"] = body_index
        qualification["winning_filter_index"] = filter_index
        qualification["winning_base_body_index"] = base_body_index
        qualification["winning_tick"] = trial["witness_tick"]
        qualification["pass"] = True
        _publish_lane("contact_qualification", qualification, complete=True)

        _begin_lane("physical_partial_reset")
        ball_before = _sensor_snapshot(env.ball_contact, BALL_FIELDS)
        base_before = _sensor_snapshot(env.base_contact, BASE_FIELDS)
        unselected_ball_state_before = _live_ball_root_state(env, unselected_ids).clone()
        # This real reset is the next simulation-mutating operation after the
        # winning current-frame witness.  No selected cache is overwritten.
        env._reset_idx(ids)
        physical = {
            "selected_ids": env_id_values,
            "reset_called_immediately_after_witness": True,
            "selected_physical_caches_overwritten_before_reset": False,
            **_selected_reset_metrics(env, ids),
            "unselected_ball_sensor_unchanged": _unselected_unchanged(
                env.ball_contact, BALL_FIELDS, ball_before, unselected_ids
            ),
            "unselected_base_sensor_unchanged": _unselected_unchanged(
                env.base_contact, BASE_FIELDS, base_before, unselected_ids
            ),
            "unselected_ball_state_unchanged": torch.equal(
                _live_ball_root_state(env, unselected_ids), unselected_ball_state_before
            ),
            "pass": False,
        }
        physical["pass"] = bool(
            physical["selected_ball_cache_zero_after"]
            and physical["selected_base_cache_zero_after"]
            and physical["selected_ball_metadata_reset"]
            and physical["selected_base_metadata_reset"]
            and physical["selected_ball_root_state_13d_canonical"]
            and physical["unselected_ball_sensor_unchanged"]
            and physical["unselected_base_sensor_unchanged"]
            and physical["unselected_ball_state_unchanged"]
        )
        _publish_lane("physical_partial_reset", physical, complete=physical["pass"])
        if not physical["pass"]:
            _fail("PHYSICAL_PARTIAL_RESET_ERROR", "physical-contact partial reset postconditions failed")
        return qualification, physical

    _fail(
        "PHYSICAL_CONTACT_ESTABLISHMENT_ERROR",
        "no qualified calf established same-tick dual-sensor contact for every selected slot",
        candidates=list(CALF_CANDIDATES),
        ticks_per_candidate=MAX_RAW_TICKS_PER_CANDIDATE,
    )


def _post_reset_raw_query(
    env: Any,
    ids: torch.Tensor,
    unselected_ids: torch.Tensor,
    body_name: str,
    base_body_index: int,
) -> dict[str, Any]:
    _begin_lane("post_reset_raw_query")
    before_ball = _sensor_snapshot(env.ball_contact, BALL_FIELDS)
    before_base = _sensor_snapshot(env.base_contact, BASE_FIELDS)
    receipt: dict[str, Any] = {
        "boundary": "AFTER_RESET_BEFORE_PUBLIC_SENSOR_DATA_OR_PHYSICS",
        "physics_dt_s": float(env.base_env.physics_dt),
        "selected_private_digest_before_sha256": _snapshot_digest(before_ball, before_base, ids),
        "unselected_private_digest_before_sha256": _snapshot_digest(
            before_ball, before_base, unselected_ids
        ),
    }
    _publish_lane("post_reset_raw_query", receipt)
    try:
        raw_ball = env.ball_contact.contact_physx_view.get_contact_force_matrix(
            dt=env.base_env.physics_dt
        ).clone()
        raw_base = env.base_contact.contact_physx_view.get_net_contact_forces(
            dt=env.base_env.physics_dt
        ).clone()
    except BaseException as error:
        _fail(
            "POST_RESET_RAW_QUERY_ERROR",
            "raw PhysX diagnostic query failed",
            raw_exception_type=type(error).__name__,
            raw_exception_message=str(error),
        )
    if not torch.is_tensor(raw_ball) or not torch.is_tensor(raw_base):
        _fail("POST_RESET_RAW_QUERY_ERROR", "raw PhysX query did not return tensors")
    ball_native_shape = list(raw_ball.shape)
    base_native_shape = list(raw_base.shape)
    try:
        ball_view = raw_ball.reshape(NUM_ENVS, 1, len(ROBOT_BODY_NAMES), 3)
        base_view = raw_base.reshape(NUM_ENVS, len(ROBOT_BODY_NAMES), 3)
    except RuntimeError as error:
        _fail(
            "POST_RESET_RAW_QUERY_ERROR",
            "raw PhysX query shape changed",
            ball_native_shape=ball_native_shape,
            base_native_shape=base_native_shape,
            reshape_error=str(error),
        )
    after_ball = _sensor_snapshot(env.ball_contact, BALL_FIELDS)
    after_base = _sensor_snapshot(env.base_contact, BASE_FIELDS)
    finite = bool(torch.isfinite(ball_view).all() and torch.isfinite(base_view).all())
    receipt.update(
        {
            "ball_raw_native_shape": ball_native_shape,
            "ball_raw_reshaped_shape": list(ball_view.shape),
            "base_raw_native_shape": base_native_shape,
            "base_raw_reshaped_shape": list(base_view.shape),
            "ball_sensor_body_count": int(env.ball_contact.num_bodies),
            "ball_filter_count": int(env.ball_contact.contact_physx_view.filter_count),
            "base_sensor_body_count": int(env.base_contact.num_bodies),
            "base_body_name": body_name,
            "base_body_index": base_body_index,
            "ball_raw_selected_max_n": (
                float(torch.linalg.vector_norm(ball_view[ids], dim=-1).max().item()) if finite else None
            ),
            "base_raw_selected_body_max_n": (
                float(torch.linalg.vector_norm(base_view[ids, base_body_index], dim=-1).max().item())
                if finite
                else None
            ),
            "raw_values_finite": finite,
            "selected_private_digest_after_sha256": _snapshot_digest(after_ball, after_base, ids),
            "unselected_private_digest_after_sha256": _snapshot_digest(
                after_ball, after_base, unselected_ids
            ),
            "selected_private_max_delta": max(
                _snapshot_delta(before_ball, after_ball, ids),
                _snapshot_delta(before_base, after_base, ids),
            ),
            "unselected_private_max_delta": max(
                _snapshot_delta(before_ball, after_ball, unselected_ids),
                _snapshot_delta(before_base, after_base, unselected_ids),
            ),
            "private_cache_unchanged_selected": bool(
                _snapshot_equal(before_ball, after_ball, ids)
                and _snapshot_equal(before_base, after_base, ids)
            ),
            "private_cache_unchanged_unselected": bool(
                _snapshot_equal(before_ball, after_ball, unselected_ids)
                and _snapshot_equal(before_base, after_base, unselected_ids)
            ),
            "raw_force_magnitude_used_as_pass_criterion": False,
            "public_contact_sensor_data_reads": 0,
            "pass": False,
        }
    )
    receipt["pass"] = bool(
        receipt["raw_values_finite"]
        and receipt["private_cache_unchanged_selected"]
        and receipt["private_cache_unchanged_unselected"]
        and receipt["selected_private_max_delta"] == 0.0
        and receipt["unselected_private_max_delta"] == 0.0
    )
    _publish_lane("post_reset_raw_query", receipt, complete=receipt["pass"])
    if not receipt["pass"]:
        _fail(
            "POST_RESET_RAW_QUERY_ERROR",
            "raw PhysX query returned non-finite data or mutated ContactSensor private cache",
        )
    return receipt


def _first_post_reset_tick(
    env: Any,
    ids: torch.Tensor,
    body_name: str,
    body_index: int,
    base_body_index: int,
) -> dict[str, Any]:
    _begin_lane("first_post_reset_physics_tick")
    dt = float(env.base_env.physics_dt)
    _raw_physics_tick(env)
    ball_timestamp = env.ball_contact._timestamp[ids].clone()
    ball_last = env.ball_contact._timestamp_last_update[ids].clone()
    base_timestamp = env.base_contact._timestamp[ids].clone()
    base_last = env.base_contact._timestamp_last_update[ids].clone()
    expected_ball_time = torch.full_like(ball_timestamp, dt)
    expected_base_time = torch.full_like(base_timestamp, dt)
    ball_data = env.ball_contact.data
    base_data = env.base_contact.data
    ball_current = ball_data.force_matrix_w
    base_current = base_data.net_forces_w
    ball_history = ball_data.force_matrix_w_history
    base_history = base_data.net_forces_w_history
    expected_shapes = (
        tuple(ball_current.shape) == (NUM_ENVS, 1, len(ROBOT_BODY_NAMES), 3)
        and tuple(base_current.shape) == (NUM_ENVS, len(ROBOT_BODY_NAMES), 3)
        and tuple(ball_history.shape) == (NUM_ENVS, 4, 1, len(ROBOT_BODY_NAMES), 3)
        and tuple(base_history.shape) == (NUM_ENVS, 3, len(ROBOT_BODY_NAMES), 3)
    )
    finite = bool(
        torch.isfinite(ball_current[ids]).all()
        and torch.isfinite(base_current[ids]).all()
        and torch.isfinite(ball_history[ids]).all()
        and torch.isfinite(base_history[ids]).all()
    )
    ball_tail = float(ball_history[ids, 1:].abs().max().item())
    base_tail = float(base_history[ids, 1:].abs().max().item())
    receipt = {
        "boundary": "EXACTLY_ONE_RAW_PHYSICS_TICK_AFTER_PHYSICAL_PARTIAL_RESET",
        "raw_physics_ticks": 1,
        "physics_dt_s": dt,
        "public_contact_sensor_data_reads_before_tick": 0,
        "public_contact_sensor_data_reads_after_tick": 2,
        "contact_body_name": body_name,
        "contact_body_index": body_index,
        "base_body_name": body_name,
        "base_body_index": base_body_index,
        "ball_force_matrix_shape": list(ball_current.shape),
        "base_net_force_shape": list(base_current.shape),
        "ball_history_shape": list(ball_history.shape),
        "base_history_shape": list(base_history.shape),
        "ball_history_tail_max_abs": ball_tail,
        "base_history_tail_max_abs": base_tail,
        "ball_index_zero_selected_max_n": float(
            torch.linalg.vector_norm(ball_history[ids, 0], dim=-1).max().item()
        ),
        "base_index_zero_selected_body_max_n": float(
            torch.linalg.vector_norm(base_history[ids, 0, base_body_index], dim=-1).max().item()
        ),
        "index_zero_allowed_any_contact": True,
        "ball_timestamp_s": ball_timestamp.detach().cpu().tolist(),
        "ball_timestamp_last_update_s": ball_last.detach().cpu().tolist(),
        "base_timestamp_s": base_timestamp.detach().cpu().tolist(),
        "base_timestamp_last_update_s": base_last.detach().cpu().tolist(),
        "ball_outdated_after_tick": bool(env.ball_contact._is_outdated[ids].any().item()),
        "base_outdated_after_tick": bool(env.base_contact._is_outdated[ids].any().item()),
        "finite": finite,
        "pass": False,
    }
    receipt["pass"] = bool(
        expected_shapes
        and finite
        and ball_tail == 0.0
        and base_tail == 0.0
        and torch.equal(ball_timestamp, expected_ball_time)
        and torch.equal(ball_last, expected_ball_time)
        and torch.equal(base_timestamp, expected_base_time)
        and torch.equal(base_last, expected_base_time)
        and not receipt["ball_outdated_after_tick"]
        and not receipt["base_outdated_after_tick"]
    )
    _publish_lane("first_post_reset_physics_tick", receipt, complete=receipt["pass"])
    if not receipt["pass"]:
        _fail("FIRST_POST_RESET_PHYSICS_TICK_ERROR", "one-tick reset history boundary failed")
    return receipt


def _production_low_step(env: Any, ids: torch.Tensor) -> dict[str, Any]:
    _begin_lane("production_low_step")
    env._reset_idx(ids)
    fresh_reset = _selected_reset_metrics(env, ids)
    fresh_reset["pass"] = bool(
        fresh_reset["selected_ball_cache_zero_after"]
        and fresh_reset["selected_base_cache_zero_after"]
        and fresh_reset["selected_ball_metadata_reset"]
        and fresh_reset["selected_base_metadata_reset"]
        and fresh_reset["selected_ball_root_state_13d_canonical"]
    )
    decimation = int(env.base_env.cfg.decimation)
    low_metrics = _normal_low_step(env)
    ball_data = env.ball_contact.data
    base_data = env.base_contact.data
    ball_history = ball_data.force_matrix_w_history
    base_history = base_data.net_forces_w_history
    ball_timestamp = env.ball_contact._timestamp[ids].clone()
    ball_last = env.ball_contact._timestamp_last_update[ids].clone()
    base_timestamp = env.base_contact._timestamp[ids].clone()
    base_last = env.base_contact._timestamp_last_update[ids].clone()
    dt = float(env.base_env.physics_dt)
    expected_ball_time = torch.full_like(ball_timestamp, decimation * dt)
    expected_base_time = torch.full_like(base_timestamp, decimation * dt)
    shapes_ok = bool(
        tuple(ball_history.shape) == (NUM_ENVS, 4, 1, len(ROBOT_BODY_NAMES), 3)
        and tuple(base_history.shape) == (NUM_ENVS, 3, len(ROBOT_BODY_NAMES), 3)
    )
    finite = bool(
        torch.isfinite(ball_history[ids]).all()
        and torch.isfinite(base_history[ids]).all()
        and torch.isfinite(_live_ball_root_state(env, ids)).all()
    )
    receipt = {
        "boundary": "INDEPENDENT_FRESH_RESET_THEN_ONE_NORMAL_LOW_ENV_STEP",
        "fresh_reset": fresh_reset,
        "configured_decimation": decimation,
        "normal_low_steps": 1,
        "raw_physics_ticks": decimation,
        "physics_dt_s": dt,
        **low_metrics,
        "ball_history_shape": list(ball_history.shape),
        "base_history_shape": list(base_history.shape),
        "ball_timestamp_s": ball_timestamp.detach().cpu().tolist(),
        "ball_timestamp_last_update_s": ball_last.detach().cpu().tolist(),
        "base_timestamp_s": base_timestamp.detach().cpu().tolist(),
        "base_timestamp_last_update_s": base_last.detach().cpu().tolist(),
        "ball_outdated_after_step": bool(env.ball_contact._is_outdated[ids].any().item()),
        "base_outdated_after_step": bool(env.base_contact._is_outdated[ids].any().item()),
        "finite": finite,
        "pass": False,
    }
    receipt["pass"] = bool(
        fresh_reset["pass"]
        and decimation == 4
        and shapes_ok
        and finite
        and torch.allclose(ball_timestamp, expected_ball_time, rtol=0.0, atol=1.0e-7)
        and torch.allclose(ball_last, expected_ball_time, rtol=0.0, atol=1.0e-7)
        and torch.allclose(base_timestamp, expected_base_time, rtol=0.0, atol=1.0e-7)
        and torch.allclose(base_last, expected_base_time, rtol=0.0, atol=1.0e-7)
        and not receipt["ball_outdated_after_step"]
        and not receipt["base_outdated_after_step"]
    )
    _publish_lane("production_low_step", receipt, complete=receipt["pass"])
    if not receipt["pass"]:
        _fail("PRODUCTION_LOW_STEP_ERROR", "fresh-reset normal-low integration failed")
    return receipt


def _sentinel_partial_lane(
    env: Any, ids: torch.Tensor, unselected_ids: torch.Tensor
) -> dict[str, Any]:
    _begin_lane("sentinel_partial")
    _inject_sensor_sentinel(env.ball_contact, BALL_FIELDS, 17.0)
    _inject_sensor_sentinel(env.base_contact, BASE_FIELDS, 41.0)
    _dirty_ball_state(env, ids, z_offset=0.27)
    ball_before = _sensor_snapshot(env.ball_contact, BALL_FIELDS)
    base_before = _sensor_snapshot(env.base_contact, BASE_FIELDS)
    unselected_ball_state_before = _live_ball_root_state(env, unselected_ids).clone()
    dirty = bool(
        _selected_cache_dirty(env.ball_contact, BALL_FIELDS, ids)
        and _selected_cache_dirty(env.base_contact, BASE_FIELDS, ids)
    )
    env._reset_idx(ids)
    receipt = {
        "selected_ids": ids.detach().cpu().tolist(),
        "sentinel_scheme": "DISTINCT_BASE_PLUS_FIELD_PLUS_ENV_SLOT",
        "selected_sensor_cache_dirty_before": dirty,
        **_selected_reset_metrics(env, ids),
        "unselected_ball_sensor_unchanged": _unselected_unchanged(
            env.ball_contact, BALL_FIELDS, ball_before, unselected_ids
        ),
        "unselected_base_sensor_unchanged": _unselected_unchanged(
            env.base_contact, BASE_FIELDS, base_before, unselected_ids
        ),
        "unselected_ball_state_unchanged": torch.equal(
            _live_ball_root_state(env, unselected_ids), unselected_ball_state_before
        ),
        "pass": False,
    }
    receipt["pass"] = bool(
        dirty
        and receipt["selected_ball_cache_zero_after"]
        and receipt["selected_base_cache_zero_after"]
        and receipt["selected_ball_metadata_reset"]
        and receipt["selected_base_metadata_reset"]
        and receipt["selected_ball_root_state_13d_canonical"]
        and receipt["unselected_ball_sensor_unchanged"]
        and receipt["unselected_base_sensor_unchanged"]
        and receipt["unselected_ball_state_unchanged"]
    )
    _publish_lane("sentinel_partial", receipt, complete=receipt["pass"])
    if not receipt["pass"]:
        _fail("SENTINEL_PARTIAL_ERROR", "partial sentinel reset lane failed")
    return receipt


def _sentinel_full_lane(env: Any, ids: torch.Tensor) -> dict[str, Any]:
    _begin_lane("sentinel_full")
    _inject_sensor_sentinel(env.ball_contact, BALL_FIELDS, 71.0)
    _inject_sensor_sentinel(env.base_contact, BASE_FIELDS, 101.0)
    _dirty_ball_state(env, ids, z_offset=0.31)
    dirty = bool(
        _selected_cache_dirty(env.ball_contact, BALL_FIELDS, ids)
        and _selected_cache_dirty(env.base_contact, BASE_FIELDS, ids)
    )
    env._reset_idx(ids)
    receipt = {
        "selected_ids": ids.detach().cpu().tolist(),
        "sentinel_scheme": "DISTINCT_BASE_PLUS_FIELD_PLUS_ENV_SLOT",
        "selected_sensor_cache_dirty_before": dirty,
        **_selected_reset_metrics(env, ids),
        "pass": False,
    }
    receipt["pass"] = bool(
        dirty
        and receipt["selected_ball_cache_zero_after"]
        and receipt["selected_base_cache_zero_after"]
        and receipt["selected_ball_metadata_reset"]
        and receipt["selected_base_metadata_reset"]
        and receipt["selected_ball_root_state_13d_canonical"]
    )
    _publish_lane("sentinel_full", receipt, complete=receipt["pass"])
    if not receipt["pass"]:
        _fail("SENTINEL_FULL_ERROR", "full sentinel reset lane failed")
    return receipt


def run() -> int:
    _begin_lane("runtime_construction")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract = load_contract(args.contract)
    reset_seed = int(contract["bounded_canary"]["training_reset_seed"])
    env, _adapter, _low, _low_before, runtime_identity = make_runtime(
        contract=contract,
        num_envs=NUM_ENVS,
        reset_seed=reset_seed,
        device=args.device,
        task=args.task,
    )
    runtime_identity.update(runtime_provenance(contract))
    device = env.device
    partial_ids = torch.tensor(PARTIAL_IDS, dtype=torch.long, device=device)
    all_ids = torch.arange(NUM_ENVS, dtype=torch.long, device=device)
    unselected_ids = torch.tensor(
        [item for item in range(NUM_ENVS) if item not in PARTIAL_IDS],
        dtype=torch.long,
        device=device,
    )

    qualification, physical = _establish_contact_and_reset(env, partial_ids, unselected_ids)
    winner_index = int(qualification["winning_body_index"])
    winner_base_index = int(qualification["winning_base_body_index"])
    winner_name = str(qualification["winning_body_name"])
    raw_query = _post_reset_raw_query(
        env, partial_ids, unselected_ids, winner_name, winner_base_index
    )
    first_tick = _first_post_reset_tick(
        env, partial_ids, winner_name, winner_index, winner_base_index
    )
    production = _production_low_step(env, partial_ids)
    sentinel_partial = _sentinel_partial_lane(env, partial_ids, unselected_ids)
    sentinel_full = _sentinel_full_lane(env, all_ids)

    _begin_lane("final_checks")
    checks = {
        "contact_qualification": qualification["pass"],
        "physical_partial_reset": physical["pass"],
        "post_reset_raw_query": raw_query["pass"],
        "first_post_reset_physics_tick": first_tick["pass"],
        "production_low_step": production["pass"],
        "sentinel_partial": sentinel_partial["pass"],
        "sentinel_full": sentinel_full["pass"],
    }
    if not all(checks.values()):
        _fail("FINAL_CHECK_ERROR", "contact reset lifecycle checks failed", checks=checks)
    report = {
        "schema": SCHEMA,
        "status": "PASS",
        "num_envs": NUM_ENVS,
        "partial_reset_ids": list(PARTIAL_IDS),
        "unselected_ids": unselected_ids.detach().cpu().tolist(),
        "zero_tolerance": ZERO_TOLERANCE,
        "state_tolerance": STATE_TOLERANCE,
        "contact_threshold_n": NON_CONTACT_FORCE_LIMIT_N,
        "contact_qualification": qualification,
        "physical_partial_reset": physical,
        "post_reset_raw_query": raw_query,
        "first_post_reset_physics_tick": first_tick,
        "production_low_step": production,
        "sentinel_partial": sentinel_partial,
        "sentinel_full": sentinel_full,
        "checks": checks,
        "ppo_constructed": False,
        "optimizer_constructed": False,
        "runtime_identity": runtime_identity,
        "source_receipts": SOURCE_RECEIPTS,
        "failures": [],
        "long_training_authorized": False,
    }
    atomic_json(output / "contact_reset_lifecycle_probe.json", report)
    atomic_bytes(output / "CONTACT_RESET_LIFECYCLE_PROBE_FINAL_GUARD", b"PASS\n")
    print("V56_CONTACT_RESET_LIFECYCLE_PROBE=PASS", flush=True)
    return 0


if __name__ == "__main__":
    faulthandler.enable(all_threads=True)
    try:
        return_code = run()
    except BaseException as error:
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        classification = getattr(
            error,
            "classification",
            FAILURE_CLASS_BY_LANE.get(CURRENT_LANE, "UNCLASSIFIED_ERROR"),
        )
        error_details = getattr(error, "details", {})
        atomic_json(
            output / "contact_reset_lifecycle_probe_failure.json",
            {
                "schema": SCHEMA,
                "status": "ERROR",
                "failure_classification": classification,
                "failed_stage": CURRENT_LANE,
                "exception_type": type(error).__name__,
                "message": str(error),
                "diagnostics": {
                    "completed_lanes": list(COMPLETED_LANES),
                    "current_lane": CURRENT_LANE,
                    "lane_receipts": _json_safe(LANE_RECEIPTS),
                    "error_details": _json_safe(error_details),
                },
                "traceback": traceback.format_exc(),
                "long_training_authorized": False,
            },
        )
        print(traceback.format_exc(), flush=True)
        os._exit(92)
    print("V56_CONTACT_RESET_LIFECYCLE_PROBE_PROCESS=COMPLETE", flush=True)
    os._exit(return_code)
