"""Independent stdlib-only verifier for one V56 single-job transaction.

This file intentionally imports no producer, simulator, Torch, or RSL module.
Reward, reset, terminal, branch, and result semantics are duplicated here so a
producer bug cannot certify its own evidence.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import stat
import struct
import sys
from typing import Any, Mapping, Sequence


def _canonical_float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


GOAL_CENTER_X_M = _canonical_float32(4.11)
GOAL_CENTER_Y_M = _canonical_float32(0.0)
STAGING_OFFSET_M = _canonical_float32(0.55)
STAGING_SAFE_MIN_X_M = _canonical_float32(-3.65)
STAGING_SAFE_MAX_X_M = _canonical_float32(3.65)
STAGING_SAFE_MIN_Y_M = _canonical_float32(-2.15)
STAGING_SAFE_MAX_Y_M = _canonical_float32(2.15)
PATH_NORMALIZER_M = _canonical_float32(math.hypot(8.0, 5.0))
TERMINAL_GOAL_HALF_WIDTH_M = _canonical_float32(0.49)
ROBOT_FIELD_HALF_LENGTH_M = _canonical_float32(4.0)
ROBOT_FIELD_HALF_WIDTH_M = _canonical_float32(2.5)
BALL_SIDE_BOUNDARY_M = _canonical_float32(2.61)
CONTACT_FORCE_THRESHOLD_N = _canonical_float32(1.0)
BASE_FALL_FORCE_THRESHOLD_N = _canonical_float32(1.0)
JOINT_HARD_LIMIT_TOLERANCE_RAD = _canonical_float32(1.0e-5)
PATH_ABS_TOLERANCE = 1.0e-5
REWARD_ABS_TOLERANCE = 2.0e-6
SUPPORT_EPSILON = 1.0e-5
CONTINUITY_ABS_TOLERANCE = 5.0e-6


@dataclass(frozen=True)
class ScalarGeometry:
    goal_xy: tuple[float, float]
    raw_staging_target_xy: tuple[float, float]
    staging_target_xy: tuple[float, float]
    staging_projection_applied: bool
    path_length_m: float


def _reward_xy(value: Sequence[float], label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be a two-vector")
    result = (float(value[0]), float(value[1]))
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} is non-finite")
    return result


def scalar_task_geometry(robot_xy: Sequence[float], ball_xy: Sequence[float], goal_side: int) -> ScalarGeometry:
    robot = _reward_xy(robot_xy, "robot_xy")
    ball = _reward_xy(ball_xy, "ball_xy")
    if type(goal_side) is not int or goal_side not in (-1, 1):
        raise ValueError("goal_side must be exactly -1 or 1")
    goal = (float(goal_side) * GOAL_CENTER_X_M, GOAL_CENTER_Y_M)
    dx, dy = goal[0] - ball[0], goal[1] - ball[1]
    distance = math.hypot(dx, dy)
    if not math.isfinite(distance):
        raise ValueError("goal distance is non-finite")
    if distance == 0.0:
        unit_x, unit_y = float(goal_side), 0.0
    else:
        unit_x, unit_y = dx / distance, dy / distance
    raw = (ball[0] - STAGING_OFFSET_M * unit_x, ball[1] - STAGING_OFFSET_M * unit_y)
    stage = (
        min(STAGING_SAFE_MAX_X_M, max(STAGING_SAFE_MIN_X_M, raw[0])),
        min(STAGING_SAFE_MAX_Y_M, max(STAGING_SAFE_MIN_Y_M, raw[1])),
    )
    path = math.hypot(robot[0] - stage[0], robot[1] - stage[1]) + distance
    if not all(math.isfinite(item) for item in (*raw, *stage, path)):
        raise ValueError("task geometry is non-finite")
    return ScalarGeometry(goal, raw, stage, raw != stage, path)


def reconstruct_transition(row: Mapping[str, Any]) -> dict[str, Any]:
    side = row["goal_side"]
    pre = scalar_task_geometry(row["pre_robot_xy_m"], row["pre_ball_xy_m"], side)
    post = scalar_task_geometry(row["post_robot_xy_pre_reset_m"], row["post_ball_xy_pre_reset_m"], side)
    progress = (pre.path_length_m - post.path_length_m) / PATH_NORMALIZER_M
    sparse = 1.0 if row["goal"] else -1.0 if row["own_goal"] else 0.0
    return {
        "goal_xy_m": list(pre.goal_xy),
        "raw_staging_target_pre_xy_m": list(pre.raw_staging_target_xy),
        "raw_staging_target_post_xy_m": list(post.raw_staging_target_xy),
        "staging_target_pre_xy_m": list(pre.staging_target_xy),
        "staging_target_post_xy_m": list(post.staging_target_xy),
        "staging_projection_pre_applied": pre.staging_projection_applied,
        "staging_projection_post_applied": post.staging_projection_applied,
        "l_pre_m": pre.path_length_m, "l_post_m": post.path_length_m,
        "normalizer_m": PATH_NORMALIZER_M, "progress_reward": progress,
        "strict_goal_reward": sparse, "total_reward": progress + sparse,
    }


def close_scalar(actual: Any, expected: float, tolerance: float) -> bool:
    try:
        value = float(actual)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and math.isclose(value, expected, rel_tol=0.0, abs_tol=tolerance)


def close_vector(actual: Any, expected: Sequence[float], tolerance: float) -> bool:
    return isinstance(actual, list) and len(actual) == len(expected) and all(
        close_scalar(left, float(right), tolerance)
        for left, right in zip(actual, expected, strict=True)
    )


CONTRACT_SCHEMA = "sparse_high_task_progress_canary_v1/contract/v1"
EVALUATION_SCHEMA = "sparse_high_task_progress_canary_v1/fixed_evaluation/v1"
EPISODE_ROW_SCHEMA = "sparse_high_task_progress_canary_v1/episode_row/v1"
TRANSITION_ROW_SCHEMA = "sparse_high_task_progress_canary_v1/first_rollout_transition/v1"
REWARD_SUPPORT_SCHEMA = "sparse_high_task_progress_canary_v1/reward_support/v1"
FIRST_UPDATE_SCHEMA = "sparse_high_task_progress_canary_v1/first_update/v1"
TRAINING_SCHEMA = "sparse_high_task_progress_canary_v1/bounded_training/v1"
RESULT_SCHEMA = "sparse_high_task_progress_canary_v1/result/v1"
CONTACT_RESET_PROBE_SCHEMA = "sparse_high_task_progress_canary_v1/contact_reset_lifecycle_probe/v2"

CONTACT_RESET_NUM_ENVS = 8
CONTACT_RESET_PARTIAL_IDS = (1, 4, 6)
CONTACT_RESET_UNSELECTED_IDS = (0, 2, 3, 5, 7)
CONTACT_RESET_PHYSICS_DT_S = 0.005
CONTACT_RESET_FORCE_THRESHOLD_N = 1.0
CONTACT_RESET_QUALIFICATION_JOB = "16315309"
CONTACT_RESET_BODY_CONTRACT_SHA256 = (
    "aaf95a4bdfd06913b34b23132bdfec1865ce4a5a83f41a6135336a8dc3109e14"
)
CONTACT_RESET_BODY_NAMES = (
    "base",
    "FL_hip", "FR_hip", "Head_upper", "RL_hip", "RR_hip",
    "FL_thigh", "FR_thigh", "Head_lower", "RL_thigh", "RR_thigh",
    "FL_calf", "FR_calf", "RL_calf", "RR_calf",
    "FL_foot", "FR_foot", "RL_foot", "RR_foot",
)
CONTACT_RESET_CALF_NAMES = ("FL_calf", "FR_calf", "RL_calf", "RR_calf")
CONTACT_RESET_CALF_INDICES = (11, 12, 13, 14)
CONTACT_RESET_LANES = (
    "runtime_construction",
    "contact_qualification",
    "physical_partial_reset",
    "post_reset_raw_query",
    "first_post_reset_physics_tick",
    "production_low_step",
    "sentinel_partial",
    "sentinel_full",
    "final_checks",
)
CONTACT_RESET_EVIDENCE_LANES = CONTACT_RESET_LANES[1:-1]
CONTACT_RESET_FAILURE_CLASSIFICATION_BY_STAGE = {
    "runtime_construction": "RUNTIME_CONSTRUCTION_ERROR",
    "physical_partial_reset": "PHYSICAL_PARTIAL_RESET_ERROR",
    "post_reset_raw_query": "POST_RESET_RAW_QUERY_ERROR",
    "first_post_reset_physics_tick": "FIRST_POST_RESET_PHYSICS_TICK_ERROR",
    "production_low_step": "PRODUCTION_LOW_STEP_ERROR",
    "sentinel_partial": "SENTINEL_PARTIAL_ERROR",
    "sentinel_full": "SENTINEL_FULL_ERROR",
    "final_checks": "FINAL_CHECK_ERROR",
}

PENDING_STATUS = "PENDING_EXACT_PARENT_V55_OWN_ARTIFACTS"
RELEASE_STATUS = "RELEASE_READY_EXACT_PARENT_V55_OWN_ARTIFACTS_PINNED"
PENDING_DECISION = "NO_GPU_AUTHORIZATION_WHILE_PARENT_PENDING"
RELEASE_DECISION = "AUTHORIZE_EXACTLY_ONE_FRESH_HIGH_TASK_PROGRESS_BOUNDED_CANARY"

DECISION_REWARD_SUPPORT_REJECTED = "V56_FIRST_ROLLOUT_REWARD_SUPPORT_REJECTED_STOP_BEFORE_PPO_UPDATE"
DECISION_NO_BREAKTHROUGH = "V56_TASK_PROGRESS_BOUNDED_CANARY_NO_BREAKTHROUGH_NO_ESCALATION"
DECISION_NONIMPROVING = "V56_FIXED_GOAL_NONIMPROVING_NOT_LEARNING_ATTRIBUTABLE"
DECISION_BREAKTHROUGH = "V56_FIRST_FIXED_GOAL_BREAKTHROUGH_MULTISEED_REQUIRED"
DECISION_HALF_GATE = "V56_SINGLE_SEED_50_PERCENT_GATE_PASS_MULTISEED_REQUIRED"
DECISION_INVALID = "V56_TERMINAL_EVIDENCE_INVALID"
DECISIONS = frozenset({
    DECISION_REWARD_SUPPORT_REJECTED,
    DECISION_NO_BREAKTHROUGH,
    DECISION_NONIMPROVING,
    DECISION_BREAKTHROUGH,
    DECISION_HALF_GATE,
    DECISION_INVALID,
})

FIXED_CASE_COUNT = 288
HARD_CASE_COUNT = 36
FIRST_ROLLOUT_STEPS = 200
FIRST_ROLLOUT_SAMPLES = 25_600
FIRST_UPDATE_OPTIMIZER_STEPS = 20
TRAINING_ITERATIONS = 12
TOTAL_ROLLOUT_STEPS = 2_400
TOTAL_ROLLOUT_SAMPLES = 307_200
TOTAL_OPTIMIZER_STEPS = 240
FIXED_EVALUATION_SEED = 75_001
TRAINING_SEED = 83_001
GOAL_SIDE_CASE_COUNTS = {-1: 132, 1: 156}
GOAL_SIDE_HALF_GATES = {-1: 66, 1: 78}
HARD_GOAL_SIDE_CASE_COUNTS = {-1: 18, 1: 18}
HARD_GOAL_SIDE_HALF_GATES = {-1: 9, 1: 9}
OVERALL_HALF_GATE = 144
HARD_HALF_GATE = 18
MAXIMUM_UNSAFE_HALF_GATE = 18
LIVE_RESET_XY_TOLERANCE_M = 5.0e-6
LIVE_RESET_YAW_TOLERANCE_RAD = 1.0e-6
EPISODE_PROGRESS_TELESCOPE_TOLERANCE = 5.0e-5
EXPECTED_LOW_CHECKPOINT_SHA256 = "f46d3b760c43664350aac8099ae5bbd32bc1207f97c8814f5d21c9b531c3d96c"
EXPECTED_LOW_MODULE_SHA256 = "3664ae556cdc895782d91400a35631353ae8be4d0386265ebbb9cd7379a1c7e4"
PARENT_PIN_KEYS = (
    "run_label",
    "result_sha256",
    "teacher_fixed_summary_sha256",
    "teacher_fixed_rows_sha256",
    "transaction_status_sha256",
    "task_manifest_sha256",
)
EXPECTED_PARENT_V55 = {
    "run_label": "sparse_high_offline_bc_warm_start_v1_17938567_20260918T045821635312091Z",
    "result_sha256": "30b1369ff05a084733d0278e7fb31dbae0261a243f9a678b3d595e8fe3f7b4e3",
    "teacher_fixed_summary_sha256": "00b54c399abb05cd34fc7f86a0f5caf62dc4dc3b6eb238aa80d8e0ca07444356",
    "teacher_fixed_rows_sha256": "248d7e158becb03ec8480cf687d5b9b368380b5c20bca86abfb4ae25fc15a496",
    "transaction_status_sha256": "09a842bc23895489113f104ec63b640087115619583a7cc08f1eea24680ebfcc",
    "task_manifest_sha256": "b3af08bf243fa793257ebf625012ef05a43eb38df8e533293be68dfda9b36e3f",
}
EXPECTED_CAUSAL_CHANGE = {
    "only_changed_mechanism": "HIGH_PPO_TASK_PROGRESS_REWARD",
    "hypothesis": "The frozen low and 10 Hz high-to-50 Hz low interface have physical contact and transport capacity, but a fresh high has no usable strict-goal credit. A same-tick task-progress signal over robot-to-staging plus ball-to-requested-goal path length can supply both-sign on-policy credit without changing the observation, action, reset, low policy, terminal rule, PPO budget or fixed evaluation.",
    "intentional_task_progress_shaping": True,
    "policy_invariant_pbrs": False,
    "reason_not_pbrs": "The reward is the undiscounted same-transition decrease L(s_pre)-L(s_post), not gamma*Phi(s_post)-Phi(s_pre).",
    "dense_return_authorizes_behavior": False,
}
EXPECTED_COMMAND_INTERFACE = {
    "mapping": "tanh_then_qualified_convex_hull_then_rate_limit",
    "axis_ranges": [[-0.45, 0.45], [-0.35, 0.35], [-0.6, 0.6]],
    "rate_limits_per_second": [1.0, 0.8, 1.5],
    "vertices": [
        [0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [-0.4, 0.0, 0.0],
        [0.0, 0.3, 0.0], [0.0, -0.3, 0.0], [0.0, 0.0, 0.55],
        [0.0, 0.0, -0.55], [0.3, 0.0, 0.35], [-0.3, 0.0, -0.35],
        [0.0, 0.25, 0.35], [0.0, -0.25, -0.35],
    ],
}
EXPECTED_FORBIDDEN = {
    key: True
    for key in (
        "online_curriculum", "difficulty_observation", "v4d_checkpoint_or_weights",
        "teacher_action_or_distillation", "offline_bc", "kick_action", "kick_reward",
        "kick_router_state", "runtime_router", "waypoint_controller",
        "scripted_phase_transition", "scripted_latch", "joint_residual",
        "contact_reward", "dwell_reward", "velocity_reward", "checkpoint_selection",
        "long_training",
    )
}
EXPECTED_AUTHORIZATION_PROVENANCE = {
    "scope": "EXACTLY_ONE_FRESH_HIGH_TASK_PROGRESS_BOUNDED_CANARY_AFTER_PARENT_RELEASE",
    "long_training_authorized": False,
}
EXPECTED_QUALIFIED_LOW = {
    "checkpoint_cluster_path": "/scratch/lj2634/runs/go2_native_low_lateral_deadzone_repair_v1/native_low_lateral_deadzone_repair_16543750_1_symmetric_lateral_10_20260829_012702/evidence/training/model_final.pt",
    "checkpoint_bytes": 4_584_447,
    "checkpoint_sha256": EXPECTED_LOW_CHECKPOINT_SHA256,
    "runtime_module_state_sha256": EXPECTED_LOW_MODULE_SHA256,
    "actor_dimensions": [48, 512, 256, 128, 12],
    "actor_activation": "elu",
    "frozen": True,
    "optimizer_constructed": False,
}


class ProtocolError(RuntimeError):
    """An artifact violates the frozen V56 contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def _exact_int(value: Any, expected: int, label: str) -> None:
    _require(type(value) is int and value == expected, f"{label} must equal {expected}")


def _exact_id_list(value: Any, expected: Sequence[int], label: str) -> None:
    expected_list = list(expected)
    _require(
        isinstance(value, list)
        and len(value) == len(expected_list)
        and all(type(item) is int for item in value)
        and value == expected_list,
        f"{label} changed",
    )


def _bounded_int(value: Any, lower: int, upper: int, label: str) -> int:
    _require(type(value) is int and lower <= value <= upper, f"{label} is invalid")
    return int(value)


def _finite(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)),
        f"{label} must be finite",
    )
    return float(value)


def _sha(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA256",
    )
    return str(value)


def _is_sha(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _checkpoint_receipt_valid(value: Any, expected_relative_path: str) -> bool:
    return bool(
        isinstance(value, Mapping)
        and value.get("relative_path") == expected_relative_path
        and type(value.get("bytes")) is int
        and value["bytes"] > 0
        and _is_sha(value.get("sha256"))
        and _is_sha(value.get("policy_state_sha256"))
    )


def _vector(value: Any, length: int, label: str) -> list[float]:
    _require(isinstance(value, list) and len(value) == length, f"{label} is invalid")
    return [_finite(item, f"{label}[{index}]") for index, item in enumerate(value)]


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _validate_diagnostic_json(value: Any, label: str) -> None:
    """Reject opaque/non-canonical values in a probe failure diagnostic."""

    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        _require(math.isfinite(value), f"{label} contains a non-finite float")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_diagnostic_json(item, f"{label}[{index}]")
        return
    if isinstance(value, Mapping):
        _require(
            all(isinstance(key, str) and bool(key) for key in value),
            f"{label} contains an invalid object key",
        )
        for key, item in value.items():
            _validate_diagnostic_json(item, f"{label}.{key}")
        return
    raise ProtocolError(f"{label} contains an unsupported value type")


def classify_contact_reset_probe_failure(
    failed_stage: Any,
    message: Any,
    diagnostics: Any,
) -> str:
    """Independently classify a v2 probe failure from its stage and facts."""

    if not isinstance(failed_stage, str) or not failed_stage:
        return "UNCLASSIFIED_ERROR"
    normalized_message = message.lower() if isinstance(message, str) else ""
    if any(
        token in normalized_message
        for token in ("canonical ball", "ball default root state", "reset sample missing")
    ):
        return "CANONICAL_BALL_STATE_ERROR"
    if failed_stage == "contact_qualification":
        details = diagnostics.get("error_details", {}) if isinstance(diagnostics, Mapping) else {}
        physical_miss = bool(
            isinstance(details, Mapping)
            and (
                details.get("actual_contact_established") is False
                or details.get("failure_kind") == "PHYSICAL_CONTACT_ESTABLISHMENT"
            )
        ) or any(
            token in normalized_message
            for token in (
                "could not establish actual",
                "physical contact establishment",
                "no qualified calf",
                "non-finite current contact force",
            )
        )
        return (
            "PHYSICAL_CONTACT_ESTABLISHMENT_ERROR"
            if physical_miss
            else "CONTACT_CONTRACT_ERROR"
        )
    return CONTACT_RESET_FAILURE_CLASSIFICATION_BY_STAGE.get(
        failed_stage, "UNCLASSIFIED_ERROR"
    )


def validate_contact_reset_probe_failure(value: Any) -> tuple[str, str]:
    """Validate a durable v2 failure receipt and return independent attribution."""

    _require(isinstance(value, Mapping), "contact reset probe failure is not an object")
    _require(
        set(value) == {
            "schema", "status", "failure_classification", "failed_stage",
            "exception_type", "message", "diagnostics", "traceback",
            "long_training_authorized",
        },
        "contact reset probe failure key set changed",
    )
    _require(
        value.get("schema") == CONTACT_RESET_PROBE_SCHEMA
        and value.get("status") == "ERROR"
        and value.get("long_training_authorized") is False,
        "contact reset probe failure receipt changed",
    )
    exception_type = value.get("exception_type")
    message = value.get("message")
    trace = value.get("traceback")
    _require(
        isinstance(exception_type, str) and bool(exception_type)
        and isinstance(message, str) and bool(message)
        and isinstance(trace, str) and bool(trace),
        "contact reset probe failure attribution is incomplete",
    )
    diagnostics = value.get("diagnostics")
    _require(
        isinstance(diagnostics, Mapping)
        and set(diagnostics) == {
            "completed_lanes", "current_lane", "lane_receipts", "error_details"
        },
        "contact reset probe failure diagnostics changed",
    )
    completed = diagnostics.get("completed_lanes")
    current = diagnostics.get("current_lane")
    receipts = diagnostics.get("lane_receipts")
    details = diagnostics.get("error_details")
    _require(
        isinstance(completed, list)
        and all(isinstance(item, str) and item in CONTACT_RESET_EVIDENCE_LANES for item in completed)
        and len(completed) == len(set(completed)),
        "contact reset probe completed-lane trace is invalid",
    )
    _require(
        completed == list(CONTACT_RESET_EVIDENCE_LANES[: len(completed)]),
        "contact reset probe completed lanes are not an ordered prefix",
    )
    _require(
        isinstance(current, str) and bool(current)
        and value.get("failed_stage") == current,
        "contact reset probe failed stage/current lane differ",
    )
    if current == "runtime_construction":
        _require(completed == [], "runtime construction failure claims completed probe lanes")
    elif current in CONTACT_RESET_EVIDENCE_LANES:
        _require(
            CONTACT_RESET_EVIDENCE_LANES.index(current) == len(completed),
            "contact reset probe current lane does not follow completed lanes",
        )
    elif current == "final_checks":
        _require(
            completed == list(CONTACT_RESET_EVIDENCE_LANES),
            "final probe checks ran before every evidence lane completed",
        )
    _require(isinstance(receipts, Mapping), "contact reset probe lane receipts are invalid")
    _require(
        set(receipts).issubset(set(completed) | {current}),
        "contact reset probe diagnostics contain a receipt for an unreached lane",
    )
    _require(isinstance(details, Mapping), "contact reset probe error details are invalid")
    _validate_diagnostic_json(diagnostics, "contact reset probe diagnostics")
    classification = classify_contact_reset_probe_failure(current, message, diagnostics)
    _require(
        value.get("failure_classification") == classification,
        "contact reset probe producer classification differs from independent classification",
    )
    trace_sha256 = hashlib.sha256(trace.encode("utf-8")).hexdigest()
    return classification, trace_sha256


def assignment_sha256(assignments: Sequence[int]) -> str:
    return hashlib.sha256((json.dumps(list(assignments), separators=(",", ":")) + "\n").encode()).hexdigest()


def _reject_duplicate_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ProtocolError(f"non-finite JSON constant is forbidden: {value}")


@dataclass(frozen=True)
class FixedResetCase:
    assignment: int
    robot_xy: tuple[float, float]
    robot_yaw: float
    ball_xy: tuple[float, float]
    goal_side: int
    hard: bool
    bucket: str


def _relative_bucket(robot_xy: tuple[float, float], yaw: float, ball_xy: tuple[float, float]) -> str:
    dx, dy = ball_xy[0] - robot_xy[0], ball_xy[1] - robot_xy[1]
    cosine, sine = math.cos(yaw), math.sin(yaw)
    body_x = cosine * dx + sine * dy
    body_y = -sine * dx + cosine * dy
    far = math.hypot(dx, dy) >= 1.45
    side = ("front" if body_x >= 0.0 else "back") if abs(body_x) >= abs(body_y) else ("left" if body_y >= 0.0 else "right")
    return f"{side}_{'far' if far else 'near'}"


def _sample_regular(seed: int) -> tuple[tuple[float, float], float, tuple[float, float], int]:
    rng = random.Random(int(seed))
    for _ in range(256):
        ball = (rng.uniform(-3.65, 3.65), rng.uniform(-2.15, 2.15))
        bearing = rng.uniform(-math.pi, math.pi)
        distance = rng.uniform(0.65, 2.40)
        robot = (ball[0] - distance * math.cos(bearing), ball[1] - distance * math.sin(bearing))
        if abs(robot[0]) <= 3.65 and abs(robot[1]) <= 2.15:
            return robot, rng.uniform(-math.pi, math.pi), ball, (-1 if rng.random() < 0.5 else 1)
    raise ProtocolError("fixed regular reset sampler exhausted")


def _sample_hard(seed: int, goal_side: int) -> tuple[tuple[float, float], float, tuple[float, float], int]:
    rng = random.Random(int(seed))
    for _ in range(256):
        robot = (rng.uniform(-2.20, 2.20), rng.uniform(-1.50, 1.50))
        yaw = math.atan2(-robot[1], float(goal_side) * 4.0 - robot[0])
        distance = rng.uniform(0.80, 1.80)
        ball = (robot[0] - distance * math.cos(yaw), robot[1] - distance * math.sin(yaw))
        if abs(ball[0]) <= 3.65 and abs(ball[1]) <= 2.15:
            return robot, yaw, ball, goal_side
    raise ProtocolError("fixed hard reset sampler exhausted")


def fixed_reset_cases(seed: int = FIXED_EVALUATION_SEED) -> tuple[FixedResetCase, ...]:
    hard_indices = {(ordinal * FIXED_CASE_COUNT) // HARD_CASE_COUNT for ordinal in range(HARD_CASE_COUNT)}
    hard_ordinal = 0
    cases: list[FixedResetCase] = []
    for assignment in range(FIXED_CASE_COUNT):
        item_seed = int(seed) * 1_000_003 + assignment
        hard = assignment in hard_indices
        if hard:
            robot, yaw, ball, goal_side = _sample_hard(item_seed, 1 if hard_ordinal % 2 == 0 else -1)
            hard_ordinal += 1
        else:
            robot, yaw, ball, goal_side = _sample_regular(item_seed)
        cases.append(FixedResetCase(assignment, robot, yaw, ball, goal_side, hard, _relative_bucket(robot, yaw, ball)))
    return tuple(cases)


def fixed_case_table_payload(seed: int = FIXED_EVALUATION_SEED) -> list[dict[str, Any]]:
    return [{
        "assignment": case.assignment,
        "robot_xy_m": list(case.robot_xy),
        "robot_yaw_rad": case.robot_yaw,
        "ball_xy_m": list(case.ball_xy),
        "goal_side": case.goal_side,
        "hard_goal_facing_ball_behind": case.hard,
        "bucket": case.bucket,
    } for case in fixed_reset_cases(seed)]


EXPECTED_FIXED_CASE_TABLE_SHA256 = semantic_sha256(fixed_case_table_payload())
EXPECTED_ASSIGNMENT_SHA256 = assignment_sha256(list(range(FIXED_CASE_COUNT)))


def expected_training_initial_resets(seed: int = TRAINING_SEED, count: int = 128) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for serial in range(1, count + 1):
        item_seed = int(seed) * 1_000_003 + serial
        robot, yaw, ball, side = _sample_regular(item_seed)
        values.append({
            "sample_serial": serial,
            "item_seed": item_seed,
            "robot_xy": list(robot),
            "robot_yaw": yaw,
            "ball_xy": list(ball),
            "goal_side": side,
            "bucket": _relative_bucket(robot, yaw, ball),
        })
    return values


def load_contract(path: str | Path, *, allow_unreleased_design: bool = False) -> dict[str, Any]:
    with Path(path).resolve(strict=True).open("r", encoding="utf-8") as stream:
        value = json.load(
            stream,
            object_pairs_hook=_reject_duplicate_json_object,
            parse_constant=_reject_nonfinite_json_constant,
        )
    _require(isinstance(value, dict), "contract root must be an object")
    _require(
        set(value) == {
            "contract_schema", "result_schema", "status", "decision", "parent_v55",
            "causal_change", "qualified_low", "architecture", "command_interface",
            "task", "reward_receipt", "ppo", "bounded_canary", "forbidden",
            "authorization_provenance",
        },
        "contract top-level key set changed",
    )
    _require(value.get("contract_schema") == CONTRACT_SCHEMA, "contract schema changed")
    _require(value.get("result_schema") == RESULT_SCHEMA, "result schema changed")
    status = value.get("status")
    parent = value.get("parent_v55", {})
    _require(isinstance(parent, Mapping), "parent V55 block missing")
    if status == PENDING_STATUS:
        _require(allow_unreleased_design, "V56 exact parent is pending; GPU/runtime release is forbidden")
        _require(value.get("decision") == PENDING_DECISION, "pending decision changed")
        _require(all(parent.get(key) is None for key in PARENT_PIN_KEYS), "pending design contains partial parent pins")
    else:
        _require(status == RELEASE_STATUS and value.get("decision") == RELEASE_DECISION, "V56 release status changed")
        _require(isinstance(parent.get("run_label"), str) and bool(parent["run_label"]), "parent run label missing")
        for key in PARENT_PIN_KEYS[1:]:
            _sha(parent.get(key), f"parent_v55.{key}")
        _require(all(parent.get(key) == expected for key, expected in EXPECTED_PARENT_V55.items()), "exact parent V55 own identity changed")
    _require(parent.get("required_decision") == "V55_GEOMETRIC_TEACHER_FIXED_GATE_REJECTED_STOP_PRE_BC", "parent decision changed")
    _require(parent.get("required_checker_rc") == 0, "parent checker requirement changed")
    _require(parent.get("required_engineering_integrity") == "PASS", "parent integrity requirement changed")
    _require(parent.get("exact_own_artifact_pin_required_before_gpu") is True, "parent exact-own gate weakened")
    expected_parent = {
        "required_decision": "V55_GEOMETRIC_TEACHER_FIXED_GATE_REJECTED_STOP_PRE_BC",
        "required_checker_rc": 0,
        "required_engineering_integrity": "PASS",
        "exact_own_artifact_pin_required_before_gpu": True,
        **({key: None for key in PARENT_PIN_KEYS} if status == PENDING_STATUS else EXPECTED_PARENT_V55),
    }
    _require(dict(parent) == expected_parent, "parent V55 block contains drift or extra fields")
    _require(value.get("causal_change") == EXPECTED_CAUSAL_CHANGE, "causal isolation contract changed")

    low = value.get("qualified_low", {})
    _require(low == EXPECTED_QUALIFIED_LOW, "qualified low identity or behavior changed")
    architecture = value.get("architecture", {})
    expected_architecture = {
        "policy_class": "ActorCriticRecurrent", "actor_observation_dim": 70,
        "critic_observation_dim": 70, "policy_action_dim": 3,
        "action_semantics": "continuous_vx_vy_wz",
        "high_policy_hz": 10, "low_policy_hz": 50, "low_steps_per_high_action": 5,
        "gru_hidden_dim": 256, "gru_layers": 1,
        "actor_hidden_dims": [256, 256, 128], "critic_hidden_dims": [256, 256, 128],
        "activation": "elu", "noise_std_type": "scalar", "state_dependent_std": False,
        "fresh_initialization": True, "checkpoint_warm_start": False,
        "v4d_weights_loaded": False, "teacher_loaded": False,
        "correlated_or_ar_exploration": False,
        "actor_observation_normalization": False, "critic_observation_normalization": False,
    }
    _require(architecture == expected_architecture, "high architecture changed")
    _require(value.get("command_interface") == EXPECTED_COMMAND_INTERFACE, "qualified command interface changed")
    task = value.get("task", {})
    reward = task.get("reward", {})
    exact_reward = {
        "kind": "INTENTIONAL_TASK_PROGRESS_NOT_POLICY_INVARIANT_PBRS",
        "goal_xy_m": "[goal_side*4.110000133514404,0.0]",
        "staging_offset_m": STAGING_OFFSET_M,
        "normalizer_m": PATH_NORMALIZER_M,
        "normalizer_definition": "IEEE_FLOAT32(hypot(8.0,5.0))",
        "raw_staging_target": "ball_xy-0.55*unit(goal_xy-ball_xy)",
        "staging_safe_box_xy_m": [[STAGING_SAFE_MIN_X_M, STAGING_SAFE_MAX_X_M], [STAGING_SAFE_MIN_Y_M, STAGING_SAFE_MAX_Y_M]],
        "staging_target": "componentwise Euclidean projection of raw_staging_target into staging_safe_box_xy_m",
        "staging_projection_applied_flag": "TELEMETRY_ONLY; independently verify projected coordinates, not exact Boolean at float32 bounds",
        "zero_goal_distance_unit_fallback": "[goal_side,0.0] only when canonical norm(goal_xy-ball_xy)==0",
        "potential_path_length": "norm(robot_xy-staging_target)+norm(ball_xy-goal_xy)",
        "progress": "(L_pre-L_post)/hypot(8.0,5.0)",
        "all_reward_constants_are_ieee_float32_canonical": True,
        "goal": 1.0, "own_goal": -1.0, "otherwise": 0.0,
        "total": "progress+strict_terminal_reward",
        "gamma_multiplier": False, "cross_reset_transition": False,
        "contact_reward": False, "dwell_reward": False, "velocity_reward": False,
        "fall_boundary_joint_extra_reward": False,
        "post_state": "SAME_TICK_PHYSICAL_PRE_RESET",
    }
    _require(reward == exact_reward, "task-progress reward contract changed")
    expected_task = {
        "reset_distribution": "V14_CONTINUOUS_FULL_RELATIVE_POSITION_FROM_ITERATION_ZERO",
        "curriculum": False,
        "episode_seconds": 20.0,
        "terminal_evaluation_frequency": "EVERY_NATIVE_50HZ_TICK",
        "terminal_semantics": "V53_STRICT_UNSAFE_SAME_TICK_REJECTS_GOAL_PRE_RESET_SNAPSHOT",
        "terminal_outcomes": ["goal", "own_goal", "boundary", "fall", "joint_hard_limit", "timeout"],
        "finite_horizon": True,
        "timeout_bootstrap": False,
        "rsl_time_outs_key_emitted": False,
        "strict_goal_center_x_m": GOAL_CENTER_X_M,
        "strict_goal_center_half_width_m": TERMINAL_GOAL_HALF_WIDTH_M,
        "true_contact_force_threshold_n": CONTACT_FORCE_THRESHOLD_N,
        "base_fall_force_threshold_n": BASE_FALL_FORCE_THRESHOLD_N,
        "joint_hard_limit_tolerance_rad": JOINT_HARD_LIMIT_TOLERANCE_RAD,
        "robot_field_half_length_m": ROBOT_FIELD_HALF_LENGTH_M,
        "robot_field_half_width_m": ROBOT_FIELD_HALF_WIDTH_M,
        "ball_side_boundary_m": BALL_SIDE_BOUNDARY_M,
        "all_terminal_comparison_constants_are_ieee_float32_canonical": True,
        "live_reset_xy_tolerance_m": LIVE_RESET_XY_TOLERANCE_M,
        "live_reset_yaw_tolerance_rad": LIVE_RESET_YAW_TOLERANCE_RAD,
        "native_base_reward_may_be_registered": True,
        "native_base_reward_discarded": True,
        "native_base_reward_consumed_by_high_or_ppo": False,
        "reward": exact_reward,
    }
    _require(task == expected_task, "task contract contains drift or extra fields")
    receipt = value.get("reward_receipt", {})
    expected_receipt = {
        "transition_schema": TRANSITION_ROW_SCHEMA, "expected_rows": FIRST_ROLLOUT_SAMPLES,
        "expected_high_steps": FIRST_ROLLOUT_STEPS, "expected_env_slots": 128,
        "support_epsilon": SUPPORT_EPSILON,
        "position_and_path_abs_tolerance": PATH_ABS_TOLERANCE,
        "progress_and_total_abs_tolerance": REWARD_ABS_TOLERANCE,
        "continuity_abs_tolerance": CONTINUITY_ABS_TOLERANCE,
        "minimum_global_nonzero": 1, "minimum_global_positive": 2,
        "minimum_global_negative": 1, "minimum_positive_per_goal_side": {"-1": 1, "1": 1},
        "must_precede_first_optimizer_update": True,
        "optimizer_may_be_constructed_before_gate": True,
        "optimizer_state_must_be_empty_before_gate": True,
        "optimizer_steps_before_gate": 0,
    }
    _require(receipt == expected_receipt, "reward receipt gate changed")
    ppo = value.get("ppo", {})
    expected_ppo = {
        "schedule": "fixed", "learning_rate": 0.0003, "desired_kl": None,
        "entropy_coef": 0.03, "gamma": 0.997, "lambda": 0.98,
        "num_learning_epochs": 5, "num_mini_batches": 4, "initial_noise_std": 1.0,
        "clip_param": 0.2, "value_loss_coef": 1.0, "use_clipped_value_loss": True,
        "max_grad_norm": 1.0, "normalize_advantage_per_mini_batch": False,
        "rnd_registered": False, "symmetry_registered": False,
    }
    _require(ppo == expected_ppo, "PPO contract changed")
    run = value.get("bounded_canary", {})
    expected_run = {
        "policy_initialization_seed": TRAINING_SEED, "training_reset_seed": TRAINING_SEED,
        "num_envs": 128, "learning_iterations": TRAINING_ITERATIONS,
        "num_steps_per_env": FIRST_ROLLOUT_STEPS,
        "first_update_rollout_steps": FIRST_ROLLOUT_STEPS,
        "first_update_rollout_samples": FIRST_ROLLOUT_SAMPLES,
        "first_update_optimizer_steps": FIRST_UPDATE_OPTIMIZER_STEPS,
        "total_rollout_steps": TOTAL_ROLLOUT_STEPS, "total_rollout_samples": TOTAL_ROLLOUT_SAMPLES,
        "total_optimizer_steps": TOTAL_OPTIMIZER_STEPS,
        "fixed_evaluation_seed": FIXED_EVALUATION_SEED,
        "fixed_evaluation_case_count": FIXED_CASE_COUNT, "hard_case_count": HARD_CASE_COUNT,
        "overall_half_gate": OVERALL_HALF_GATE, "hard_half_gate": HARD_HALF_GATE,
        "fixed_goal_side_case_counts": {"-1": 132, "1": 156},
        "per_goal_side_half_gate": {"-1": 66, "1": 78},
        "hard_goal_side_case_counts": {"-1": 18, "1": 18},
        "hard_per_goal_side_half_gate": {"-1": 9, "1": 9},
        "maximum_unsafe_fixed_cases_for_half_gate": MAXIMUM_UNSAFE_HALF_GATE,
        "long_training_authorized": False,
    }
    _require(run == expected_run, "bounded canary changed")
    _require(value.get("forbidden") == EXPECTED_FORBIDDEN, "forbidden mechanism set changed")
    _require(value.get("authorization_provenance") == EXPECTED_AUTHORIZATION_PROVENANCE, "authorization scope changed")
    return value


def _terminal_facts(row: Mapping[str, Any], label: str) -> dict[str, Any]:
    for key in ("contact_seen", "requested_goal_line_reached", "goal", "own_goal", "boundary", "fall", "joint_hard_limit", "timeout", "unsafe_pre_goal"):
        _require(type(row.get(key)) is bool, f"{label}.{key} is not Boolean")
    robot = _vector(row.get("terminal_robot_xy_m"), 2, f"{label}.terminal_robot_xy_m")
    ball = _vector(row.get("terminal_ball_xy_m"), 2, f"{label}.terminal_ball_xy_m")
    side = row.get("goal_side")
    _require(type(side) is int and side in (-1, 1), f"{label}.goal_side invalid")
    maximum_contact = _finite(row.get("maximum_contact_force_n"), f"{label}.maximum_contact_force_n")
    terminal_contact = _finite(row.get("terminal_contact_force_n"), f"{label}.terminal_contact_force_n")
    base_force = _finite(row.get("terminal_base_contact_force_n"), f"{label}.terminal_base_contact_force_n")
    hard_excess = _finite(row.get("terminal_joint_hard_limit_excess_rad"), f"{label}.terminal_joint_hard_limit_excess_rad")
    _require(min(maximum_contact, terminal_contact, base_force, hard_excess) >= 0.0, f"{label} physical witness is negative")
    _require(maximum_contact + 1.0e-6 >= terminal_contact, f"{label} contact maximum inconsistent")
    contact = maximum_contact >= CONTACT_FORCE_THRESHOLD_N
    _require(row["contact_seen"] is contact, f"{label} contact latch not derived")
    line = bool(contact and side * ball[0] >= GOAL_CENTER_X_M and abs(ball[1]) <= TERMINAL_GOAL_HALF_WIDTH_M)
    own = bool(side * (-ball[0]) >= GOAL_CENTER_X_M and abs(ball[1]) <= TERMINAL_GOAL_HALF_WIDTH_M)
    fall = base_force > BASE_FALL_FORCE_THRESHOLD_N
    joint = hard_excess > JOINT_HARD_LIMIT_TOLERANCE_RAD
    inside_mouth = abs(ball[1]) <= TERMINAL_GOAL_HALF_WIDTH_M
    boundary = bool(abs(robot[0]) > ROBOT_FIELD_HALF_LENGTH_M or abs(robot[1]) > ROBOT_FIELD_HALF_WIDTH_M or abs(ball[1]) > BALL_SIDE_BOUNDARY_M or (abs(ball[0]) > GOAL_CENTER_X_M and not inside_mouth))
    unsafe = bool(own or fall or joint or boundary)
    goal = bool(line and not unsafe)
    high_ticks = _bounded_int(row.get("episode_high_ticks"), 1, 200, f"{label}.episode_high_ticks")
    native_ticks = _bounded_int(row.get("episode_native_ticks"), 1, 1000, f"{label}.episode_native_ticks")
    timeout = bool(not line and not unsafe and high_ticks == 200 and native_ticks == 1000)
    expected = {"requested_goal_line_reached": line, "own_goal": own, "fall": fall, "joint_hard_limit": joint, "boundary": boundary, "unsafe_pre_goal": unsafe, "goal": goal, "timeout": timeout}
    _require(all(row.get(key) is value for key, value in expected.items()), f"{label} terminal facts changed")
    outcome = "goal" if goal else "own_goal" if own else "fall" if fall else "boundary" if boundary else "joint_hard_limit" if joint else "timeout"
    _require(row.get("outcome") == outcome, f"{label} outcome precedence changed")
    return {"goal": goal, "unsafe": unsafe, "contact": contact, "robot": robot, "ball": ball, "side": side, "outcome": outcome}


def validate_episode_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(len(rows) == FIXED_CASE_COUNT, "fixed evaluation must contain 288 rows")
    cases = fixed_reset_cases()
    assignments: list[int] = []
    goal_count = hard_goal_count = unsafe_count = contact_count = hard_count = 0
    case_by_side = {-1: 0, 1: 0}; hard_case_by_side = {-1: 0, 1: 0}
    goal_by_side = {-1: 0, 1: 0}; hard_goal_by_side = {-1: 0, 1: 0}
    outcome_counts: dict[str, int] = {
        name: 0 for name in ("goal", "own_goal", "boundary", "fall", "joint_hard_limit", "timeout")
    }
    diagnostics = {"initial_l_m": [], "final_l_m": [], "cumulative_progress_reward": [], "requested_goal_axis_ball_displacement_m": [], "abs_lateral_ball_drift_m": []}
    maximum_reset_error = 0.0
    for index, row in enumerate(rows):
        label = f"row[{index}]"
        _require(row.get("schema") == EPISODE_ROW_SCHEMA, f"{label} schema changed")
        assignment = row.get("evaluation_assignment")
        _require(type(assignment) is int and 0 <= assignment < FIXED_CASE_COUNT, f"{label} assignment invalid")
        case = cases[assignment]; assignments.append(assignment)
        reset_robot = _vector(row.get("reset_robot_xy_m"), 2, f"{label}.reset_robot_xy_m")
        reset_ball = _vector(row.get("reset_ball_xy_m"), 2, f"{label}.reset_ball_xy_m")
        reset_yaw = _finite(row.get("reset_robot_yaw_rad"), f"{label}.reset_robot_yaw_rad")
        for actual, expected in zip((*reset_robot, *reset_ball), (*case.robot_xy, *case.ball_xy), strict=True):
            maximum_reset_error = max(maximum_reset_error, abs(actual - expected))
            _require(abs(actual - expected) <= LIVE_RESET_XY_TOLERANCE_M, f"{label} reset position differs")
        yaw_error = abs(math.atan2(math.sin(reset_yaw - case.robot_yaw), math.cos(reset_yaw - case.robot_yaw)))
        maximum_reset_error = max(maximum_reset_error, yaw_error)
        _require(yaw_error <= LIVE_RESET_YAW_TOLERANCE_RAD, f"{label} reset yaw differs")
        _require(row.get("goal_side") == case.goal_side and row.get("bucket") == case.bucket, f"{label} fixed case changed")
        _require(row.get("hard_goal_facing_ball_behind") is case.hard and row.get("counts_for_evaluation") is True, f"{label} membership changed")
        facts = _terminal_facts(row, label)
        initial_geometry = scalar_task_geometry(reset_robot, reset_ball, facts["side"])
        final_geometry = scalar_task_geometry(facts["robot"], facts["ball"], facts["side"])
        cumulative = _finite(row.get("cumulative_progress_reward"), f"{label}.cumulative_progress_reward")
        telescope = (initial_geometry.path_length_m - final_geometry.path_length_m) / PATH_NORMALIZER_M
        _require(abs(cumulative - telescope) <= EPISODE_PROGRESS_TELESCOPE_TOLERANCE, f"{label} cumulative progress does not telescope")
        axis = facts["side"] * (facts["ball"][0] - reset_ball[0]); lateral = abs(facts["ball"][1] - reset_ball[1])
        _require(close_scalar(row.get("initial_l_m"), initial_geometry.path_length_m, PATH_ABS_TOLERANCE), f"{label} initial L changed")
        _require(close_scalar(row.get("final_l_m"), final_geometry.path_length_m, PATH_ABS_TOLERANCE), f"{label} final L changed")
        _require(close_scalar(row.get("requested_goal_axis_ball_displacement_m"), axis, PATH_ABS_TOLERANCE), f"{label} axis displacement changed")
        _require(close_scalar(row.get("abs_lateral_ball_drift_m"), lateral, PATH_ABS_TOLERANCE), f"{label} lateral drift changed")
        for key, value in (("initial_l_m", initial_geometry.path_length_m), ("final_l_m", final_geometry.path_length_m), ("cumulative_progress_reward", cumulative), ("requested_goal_axis_ball_displacement_m", axis), ("abs_lateral_ball_drift_m", lateral)):
            diagnostics[key].append(value)
        hard = case.hard; goal = facts["goal"]; side = facts["side"]
        hard_count += int(hard); goal_count += int(goal); hard_goal_count += int(goal and hard)
        unsafe_count += int(facts["unsafe"]); contact_count += int(facts["contact"])
        case_by_side[side] += 1; hard_case_by_side[side] += int(hard)
        goal_by_side[side] += int(goal); hard_goal_by_side[side] += int(goal and hard)
        outcome_counts[facts["outcome"]] = outcome_counts.get(facts["outcome"], 0) + 1
    _require(sorted(assignments) == list(range(FIXED_CASE_COUNT)), "fixed assignments are not exact-once")
    _require(hard_count == HARD_CASE_COUNT and case_by_side == GOAL_SIDE_CASE_COUNTS and hard_case_by_side == HARD_GOAL_SIDE_CASE_COUNTS, "fixed denominators changed")
    return {
        "assignment_sha256": assignment_sha256(sorted(assignments)), "fixed_case_table_sha256": EXPECTED_FIXED_CASE_TABLE_SHA256,
        "goal_count": goal_count, "hard_goal_count": hard_goal_count, "hard_case_count": hard_count,
        "unsafe_count": unsafe_count, "contact_count": contact_count,
        "case_counts_by_goal_side": {str(k): v for k, v in case_by_side.items()},
        "hard_case_counts_by_goal_side": {str(k): v for k, v in hard_case_by_side.items()},
        "goal_counts_by_goal_side": {str(k): v for k, v in goal_by_side.items()},
        "hard_goal_counts_by_goal_side": {str(k): v for k, v in hard_goal_by_side.items()},
        "maximum_live_reset_error": maximum_reset_error, "outcome_counts": dict(sorted(outcome_counts.items())),
        **{f"mean_{key}": sum(values) / len(values) for key, values in diagnostics.items()},
    }


def _summary_metric_match(value: Mapping[str, Any], metrics: Mapping[str, Any], key: str, tolerance: float = 1.0e-9) -> None:
    _require(close_scalar(value.get(key), float(metrics[key]), tolerance), f"summary metric differs: {key}")


def validate_evaluation_summary(value: Mapping[str, Any], *, phase: str) -> None:
    _require(phase in {"initial", "final"}, "evaluation phase changed")
    _require(value.get("schema") == EVALUATION_SCHEMA and value.get("phase") == phase, f"{phase} schema changed")
    _require(value.get("status") == "COMPLETE" and value.get("failures") == [], f"{phase} incomplete")
    _exact_int(value.get("fixed_evaluation_case_count"), FIXED_CASE_COUNT, f"{phase} cases")
    _exact_int(value.get("hard_case_count"), HARD_CASE_COUNT, f"{phase} hard cases")
    _require(value.get("assignments_exactly_once") is True and value.get("assignment_sha256") == EXPECTED_ASSIGNMENT_SHA256, f"{phase} assignments changed")
    for key, upper in (("goal_count", 288), ("hard_goal_count", 36), ("unsafe_count", 288), ("contact_count", 288)):
        _bounded_int(value.get(key), 0, upper, f"{phase}.{key}")
    _require(value.get("case_counts_by_goal_side") == {"-1": 132, "1": 156}, f"{phase} side denominator changed")
    _require(value.get("hard_case_counts_by_goal_side") == {"-1": 18, "1": 18}, f"{phase} hard denominator changed")
    goals_by_side = value.get("goal_counts_by_goal_side")
    hard_goals_by_side = value.get("hard_goal_counts_by_goal_side")
    _require(isinstance(goals_by_side, Mapping) and set(goals_by_side) == {"-1", "1"}, f"{phase} goal-side counts invalid")
    _require(isinstance(hard_goals_by_side, Mapping) and set(hard_goals_by_side) == {"-1", "1"}, f"{phase} hard goal-side counts invalid")
    for side, denominator in GOAL_SIDE_CASE_COUNTS.items():
        goal_count = _bounded_int(goals_by_side.get(str(side)), 0, denominator, f"{phase}.goal_side[{side}]")
        hard_goal_count = _bounded_int(hard_goals_by_side.get(str(side)), 0, HARD_GOAL_SIDE_CASE_COUNTS[side], f"{phase}.hard_goal_side[{side}]")
        _require(hard_goal_count <= goal_count, f"{phase} hard side goals exceed side goals")
    _require(sum(goals_by_side.values()) == value.get("goal_count"), f"{phase} side goals do not sum to overall goals")
    _require(sum(hard_goals_by_side.values()) == value.get("hard_goal_count"), f"{phase} hard side goals do not sum to hard goals")
    _require(value.get("hard_goal_count") <= value.get("goal_count"), f"{phase} hard goals exceed overall goals")
    outcome_counts = value.get("outcome_counts")
    allowed_outcomes = {"goal", "own_goal", "boundary", "fall", "joint_hard_limit", "timeout"}
    _require(isinstance(outcome_counts, Mapping) and set(outcome_counts) == allowed_outcomes, f"{phase} outcomes invalid")
    _require(all(type(count) is int and count >= 0 for count in outcome_counts.values()), f"{phase} outcome count invalid")
    _require(sum(outcome_counts.values()) == FIXED_CASE_COUNT, f"{phase} outcomes do not sum to fixed cases")
    _require(outcome_counts.get("goal", 0) == value.get("goal_count"), f"{phase} goal outcome count changed")
    _require(sum(outcome_counts.get(name, 0) for name in ("own_goal", "boundary", "fall", "joint_hard_limit")) == value.get("unsafe_count"), f"{phase} unsafe outcome count changed")
    _require(value.get("goal_count") <= value.get("contact_count"), f"{phase} goals exceed contact cases")
    _require(value.get("goal_count") + value.get("unsafe_count") <= FIXED_CASE_COUNT, f"{phase} goal/unsafe counts overlap")
    maximum_reset_error = _finite(value.get("maximum_live_reset_error"), f"{phase}.maximum_live_reset_error")
    _require(0.0 <= maximum_reset_error <= LIVE_RESET_XY_TOLERANCE_M, f"{phase} live reset error exceeded tolerance")
    for key in ("mean_initial_l_m", "mean_final_l_m", "mean_cumulative_progress_reward", "mean_requested_goal_axis_ball_displacement_m", "mean_abs_lateral_ball_drift_m"):
        _finite(value.get(key), f"{phase}.{key}")
    total_reward_min = _finite(value.get("total_reward_min"), f"{phase}.total_reward_min")
    total_reward_max = _finite(value.get("total_reward_max"), f"{phase}.total_reward_max")
    _require(total_reward_min <= total_reward_max, f"{phase} reward range changed")
    _require(value.get("strict_terminal_reward_values") == [-1.0, 0.0, 1.0], f"{phase} strict reward alphabet changed")
    _exact_int(value.get("ppo_optimizer_steps"), 0, f"{phase} optimizer steps")
    _require(value.get("policy_state_unchanged") is True and value.get("frozen_low_state_unchanged") is True, f"{phase} mutated model")
    _require(value.get("low_state_before_sha256") == EXPECTED_LOW_MODULE_SHA256 and value.get("low_state_after_sha256") == EXPECTED_LOW_MODULE_SHA256, f"{phase} low identity changed")
    _sha(value.get("policy_state_sha256"), f"{phase} policy state"); _sha(value.get("episode_rows_sha256"), f"{phase} rows")
    _require(value.get("fixed_case_table_sha256") == EXPECTED_FIXED_CASE_TABLE_SHA256, f"{phase} table changed")
    _require(value.get("finite_horizon") is True and value.get("time_outs_bootstrap_key_emitted") is False, f"{phase} timeout semantics changed")
    _require(value.get("dense_return_authorizes_behavior") is False and value.get("long_training_authorized") is False, f"{phase} evaluation authorized escalation")
    _exact_int(value.get("policy_initialization_seed"), TRAINING_SEED, f"{phase} policy seed")
    _require(value.get("policy_seed_reapplied_after_environment_construction") is True, f"{phase} policy reseed evidence changed")
    if phase == "initial":
        _require(value.get("checkpoint_loaded") is False and value.get("fresh_policy_initialized") is True, "initial policy provenance changed")
        _require(value.get("checkpoint_sha256") is None and value.get("loaded_policy_state_sha256") is None, "initial evaluation loaded a policy")
        checkpoint = value.get("initial_checkpoint", {})
        _require(_checkpoint_receipt_valid(checkpoint, "model_initial.pt"), "initial checkpoint invalid")
        _require(checkpoint.get("policy_state_sha256") == value.get("policy_state_sha256"), "initial checkpoint state changed")
    else:
        _require(value.get("checkpoint_loaded") is True and value.get("fresh_policy_initialized") is False, "final checkpoint provenance changed")
        _require(value.get("initial_checkpoint") is None, "final evaluation emitted an initial checkpoint")
        _sha(value.get("checkpoint_sha256"), "final checkpoint")
        _require(value.get("loaded_policy_state_sha256") == value.get("policy_state_sha256"), "final policy changed during evaluation")


def _transition_flags(row: Mapping[str, Any], label: str) -> None:
    for key in ("goal", "own_goal", "boundary", "fall", "joint_hard_limit", "timeout", "unsafe_same_tick", "done", "reset_after_step", "contact_seen", "staging_projection_pre_applied", "staging_projection_post_applied"):
        _require(type(row.get(key)) is bool, f"{label}.{key} is not Boolean")
    _require(row["done"] is row["reset_after_step"], f"{label} reset/done mismatch")
    _require(row["unsafe_same_tick"] is bool(row["own_goal"] or row["boundary"] or row["fall"] or row["joint_hard_limit"]), f"{label} unsafe flag changed")
    _require(not (row["goal"] and row["unsafe_same_tick"]), f"{label} unsafe same tick accepted goal")
    _require(not (row["goal"] and row["own_goal"]), f"{label} goal conflict")
    _require(row["done"] is bool(row["goal"] or row["own_goal"] or row["boundary"] or row["fall"] or row["joint_hard_limit"] or row["timeout"]), f"{label} done not derived")


def validate_first_rollout_transitions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(len(rows) == FIRST_ROLLOUT_SAMPLES, "first rollout must contain exactly 25600 rows")
    positive = negative = nonzero = finite_count = reconstructed = 0
    positive_by_side = {-1: 0, 1: 0}; row_count_by_side = {-1: 0, 1: 0}
    projection_pre = projection_post = terminal_count = 0
    previous_by_env: dict[int, Mapping[str, Any]] = {}
    progress_values: list[float] = []; total_values: list[float] = []
    expected_serial_by_env = {env_id: env_id + 1 for env_id in range(128)}
    next_reset_serial = 129
    for row_index, row in enumerate(rows):
        label = f"transition[{row_index}]"
        _require(row.get("schema") == TRANSITION_ROW_SCHEMA, f"{label} schema changed")
        expected_step, expected_env = divmod(row_index, 128)
        _exact_int(row.get("global_transition_index"), row_index, f"{label}.global_transition_index")
        _exact_int(row.get("rollout_iteration"), 0, f"{label}.rollout_iteration")
        _exact_int(row.get("high_step_index"), expected_step, f"{label}.high_step_index")
        _exact_int(row.get("env_id"), expected_env, f"{label}.env_id")
        serial = _bounded_int(row.get("sample_serial"), 1, 10_000_000, f"{label}.sample_serial")
        _exact_int(serial, expected_serial_by_env[expected_env], f"{label}.global_sample_serial")
        _exact_int(row.get("episode_serial_before_step"), serial, f"{label}.episode_serial_before_step")
        item_seed = _bounded_int(row.get("item_seed"), 1, 10**18, f"{label}.item_seed")
        _exact_int(item_seed, TRAINING_SEED * 1_000_003 + serial, f"{label}.item_seed")
        episode_step = _bounded_int(row.get("episode_high_step_index_before_step"), 0, 199, f"{label}.episode_high_step_index_before_step")
        side = row.get("goal_side"); _require(type(side) is int and side in (-1, 1), f"{label}.goal_side invalid")
        reset_robot = _vector(row.get("reset_robot_xy_m"), 2, f"{label}.reset_robot_xy_m")
        reset_ball = _vector(row.get("reset_ball_xy_m"), 2, f"{label}.reset_ball_xy_m")
        reset_yaw = _finite(row.get("reset_robot_yaw_rad"), f"{label}.reset_robot_yaw_rad")
        expected_robot, expected_yaw, expected_ball, expected_side = _sample_regular(item_seed)
        _require(side == expected_side, f"{label} reset side changed")
        _require(close_vector(reset_robot, expected_robot, LIVE_RESET_XY_TOLERANCE_M) and close_vector(reset_ball, expected_ball, LIVE_RESET_XY_TOLERANCE_M), f"{label} reset sample changed")
        yaw_error = abs(math.atan2(math.sin(reset_yaw - expected_yaw), math.cos(reset_yaw - expected_yaw)))
        _require(yaw_error <= LIVE_RESET_YAW_TOLERANCE_RAD, f"{label} reset yaw changed")
        pre_robot = _vector(row.get("pre_robot_xy_m"), 2, f"{label}.pre_robot_xy_m")
        pre_ball = _vector(row.get("pre_ball_xy_m"), 2, f"{label}.pre_ball_xy_m")
        post_robot = _vector(row.get("post_robot_xy_pre_reset_m"), 2, f"{label}.post_robot_xy_pre_reset_m")
        post_ball = _vector(row.get("post_ball_xy_pre_reset_m"), 2, f"{label}.post_ball_xy_pre_reset_m")
        _vector(row.get("raw_action"), 3, f"{label}.raw_action"); _vector(row.get("applied_command"), 3, f"{label}.applied_command")
        native_ticks = _bounded_int(row.get("native_ticks_executed"), 1, 5, f"{label}.native_ticks_executed")
        _transition_flags(row, label)
        _require(row.get("post_state_source") == "SAME_TICK_PHYSICAL_PRE_RESET", f"{label} post state source changed")
        maximum_contact = _finite(row.get("maximum_contact_force_n"), f"{label}.maximum_contact_force_n")
        terminal_contact = _finite(row.get("terminal_contact_force_n"), f"{label}.terminal_contact_force_n")
        base_force = _finite(row.get("terminal_base_contact_force_n"), f"{label}.terminal_base_contact_force_n")
        hard_excess = _finite(row.get("terminal_joint_hard_limit_excess_rad"), f"{label}.terminal_joint_hard_limit_excess_rad")
        _require(min(maximum_contact, terminal_contact, base_force, hard_excess) >= 0.0, f"{label} physical witness negative")
        _require(maximum_contact + 1.0e-6 >= terminal_contact, f"{label} contact maximum inconsistent")
        contact_seen = maximum_contact >= CONTACT_FORCE_THRESHOLD_N
        _require(row["contact_seen"] is contact_seen, f"{label} contact latch not reconstructed")
        requested_line = contact_seen and side * post_ball[0] >= GOAL_CENTER_X_M and abs(post_ball[1]) <= TERMINAL_GOAL_HALF_WIDTH_M
        own = side * (-post_ball[0]) >= GOAL_CENTER_X_M and abs(post_ball[1]) <= TERMINAL_GOAL_HALF_WIDTH_M
        fall = base_force > BASE_FALL_FORCE_THRESHOLD_N
        joint = hard_excess > JOINT_HARD_LIMIT_TOLERANCE_RAD
        inside_mouth = abs(post_ball[1]) <= TERMINAL_GOAL_HALF_WIDTH_M
        boundary = bool(abs(post_robot[0]) > ROBOT_FIELD_HALF_LENGTH_M or abs(post_robot[1]) > ROBOT_FIELD_HALF_WIDTH_M or abs(post_ball[1]) > BALL_SIDE_BOUNDARY_M or (abs(post_ball[0]) > GOAL_CENTER_X_M and not inside_mouth))
        unsafe = bool(own or fall or joint or boundary)
        goal = bool(requested_line and not unsafe)
        timeout = bool(not requested_line and not unsafe and episode_step == 199 and native_ticks == 5)
        expected_flags = {"goal": goal, "own_goal": own, "fall": fall, "joint_hard_limit": joint, "boundary": boundary, "unsafe_same_tick": unsafe, "timeout": timeout}
        _require(all(row[key] is expected for key, expected in expected_flags.items()), f"{label} terminal facts not reconstructed")
        expected_done = bool(requested_line or unsafe or timeout)
        _require(row["done"] is expected_done, f"{label} done changed")
        if not expected_done:
            _require(native_ticks == 5, f"{label} nonterminal transition did not execute five native ticks")
        reason = "GOAL" if goal else "OWN_GOAL" if own else "FALL" if fall else "BOUNDARY" if boundary else "JOINT_HARD_LIMIT" if joint else "TIMEOUT" if timeout else "NONE"
        _require(row.get("terminal_reason") == reason, f"{label} terminal reason changed")
        reconstruction = reconstruct_transition(row)
        serialized_goal = _vector(row.get("goal_xy_m"), 2, f"{label}.goal_xy_m")
        _require(serialized_goal == reconstruction["goal_xy_m"], f"{label} strict canonical goal coordinate changed")
        for key in ("raw_staging_target_pre_xy_m", "raw_staging_target_post_xy_m", "staging_target_pre_xy_m", "staging_target_post_xy_m"):
            _require(close_vector(row.get(key), reconstruction[key], PATH_ABS_TOLERANCE), f"{label}.{key} changed")
        for key in ("l_pre_m", "l_post_m", "normalizer_m"):
            _require(close_scalar(row.get(key), reconstruction[key], PATH_ABS_TOLERANCE), f"{label}.{key} changed")
        # The Boolean is telemetry only: an exact float32 boundary can differ
        # from the float64 reconstruction by one representable value.  The
        # projected coordinates above are the authoritative evidence.
        for key in ("progress_reward", "strict_goal_reward", "total_reward"):
            _require(close_scalar(row.get(key), reconstruction[key], REWARD_ABS_TOLERANCE), f"{label}.{key} changed")
        _require(row["strict_goal_reward"] in (-1.0, 0.0, 1.0), f"{label} sparse reward alphabet changed")
        progress = float(row["progress_reward"]); reconstructed_progress = float(reconstruction["progress_reward"]); total = float(row["total_reward"])
        if row["goal"]:
            _require(total > 0.0 and reconstruction["total_reward"] > 0.0, f"{label} goal total reward lost positive sign")
        if row["own_goal"]:
            _require(total < 0.0 and reconstruction["total_reward"] < 0.0, f"{label} own-goal total reward lost negative sign")
        finite_count += 1; reconstructed += 1; progress_values.append(progress); total_values.append(total)
        supported_nonzero = abs(progress) > SUPPORT_EPSILON and abs(reconstructed_progress) > SUPPORT_EPSILON
        supported_positive = progress > SUPPORT_EPSILON and reconstructed_progress > SUPPORT_EPSILON
        supported_negative = progress < -SUPPORT_EPSILON and reconstructed_progress < -SUPPORT_EPSILON
        nonzero += int(supported_nonzero); positive += int(supported_positive); negative += int(supported_negative)
        positive_by_side[side] += int(supported_positive); row_count_by_side[side] += 1
        projection_pre += int(row["staging_projection_pre_applied"]); projection_post += int(row["staging_projection_post_applied"]); terminal_count += int(row["done"])
        previous = previous_by_env.get(expected_env)
        if previous is None or previous["done"]:
            _require(episode_step == 0, f"{label} reset episode step is not zero")
            _require(close_vector(pre_robot, reset_robot, CONTINUITY_ABS_TOLERANCE) and close_vector(pre_ball, reset_ball, CONTINUITY_ABS_TOLERANCE), f"{label} reset-to-pre boundary changed")
            if previous is not None:
                _require(serial > int(previous["sample_serial"]), f"{label} sample serial did not advance after done")
        else:
            _require(serial == int(previous["sample_serial"]) and episode_step == int(previous["episode_high_step_index_before_step"]) + 1, f"{label} nonterminal episode identity changed")
            _require(close_vector(pre_robot, previous["post_robot_xy_pre_reset_m"], CONTINUITY_ABS_TOLERANCE) and close_vector(pre_ball, previous["post_ball_xy_pre_reset_m"], CONTINUITY_ABS_TOLERANCE), f"{label} nonterminal next-pre discontinuity")
        previous_by_env[expected_env] = row
        if row["done"]:
            expected_serial_by_env[expected_env] = next_reset_serial
            next_reset_serial += 1
    checks = {
        "row_count_exact": len(rows) == FIRST_ROLLOUT_SAMPLES,
        "complete_unique_grid": True,
        "all_finite": finite_count == FIRST_ROLLOUT_SAMPLES,
        "all_formula_reconstructed": reconstructed == FIRST_ROLLOUT_SAMPLES,
        "global_nonzero_support": nonzero >= 1,
        "global_positive_support": positive >= 2,
        "global_negative_support": negative >= 1,
        "positive_support_goal_side_negative": positive_by_side[-1] >= 1,
        "positive_support_goal_side_positive": positive_by_side[1] >= 1,
    }
    return {
        "schema": REWARD_SUPPORT_SCHEMA, "status": "COMPLETE", "failures": [],
        "transition_row_count": len(rows), "finite_transition_count": finite_count,
        "reconstructed_transition_count": reconstructed, "support_epsilon": SUPPORT_EPSILON,
        "nonzero_progress_count": nonzero, "positive_progress_count": positive, "negative_progress_count": negative,
        "positive_progress_count_by_goal_side": {str(k): v for k, v in positive_by_side.items()},
        "row_count_by_goal_side": {str(k): v for k, v in row_count_by_side.items()},
        "staging_projection_pre_count": projection_pre, "staging_projection_post_count": projection_post,
        "terminal_transition_count": terminal_count,
        "progress_reward_min": min(progress_values), "progress_reward_max": max(progress_values),
        "progress_reward_mean": sum(progress_values) / len(progress_values), "total_reward_mean": sum(total_values) / len(total_values),
        "checks": checks, "gate_pass": all(checks.values()),
        "optimizer_constructed_before_gate": True,
        "optimizer_state_empty_before_gate": True,
        "optimizer_steps_before_gate": 0,
    }


def reward_support_checks(value: Mapping[str, Any]) -> dict[str, bool]:
    checks = value.get("checks", {})
    names = ("row_count_exact", "complete_unique_grid", "all_finite", "all_formula_reconstructed", "global_nonzero_support", "global_positive_support", "global_negative_support", "positive_support_goal_side_negative", "positive_support_goal_side_positive")
    result = {
        "terminal_complete": value.get("schema") == REWARD_SUPPORT_SCHEMA and value.get("status") == "COMPLETE" and value.get("failures") == [],
        **{name: isinstance(checks, Mapping) and checks.get(name) is True for name in names},
        "pre_optimizer_update_gate": value.get("optimizer_constructed_before_gate") is True
        and value.get("optimizer_state_empty_before_gate") is True
        and value.get("optimizer_steps_before_gate") == 0,
        "transition_rows_receipt_valid": value.get("transition_rows_relative_path") == "first_rollout_transitions.jsonl"
        and type(value.get("transition_rows_bytes")) is int
        and value["transition_rows_bytes"] > 0
        and _is_sha(value.get("transition_rows_sha256")),
        "initial_identity_fields_well_formed": _is_sha(value.get("initial_policy_state_sha256"))
        and _is_sha(value.get("initial_checkpoint_sha256"))
        and value.get("frozen_low_state_sha256_before_gate") == EXPECTED_LOW_MODULE_SHA256,
    }
    if value.get("gate_pass") is False:
        rejection = value.get("rejection_integrity", {})
        result["rejected_branch_no_update_integrity"] = bool(
            isinstance(rejection, Mapping)
            and rejection.get("storage_cleared_without_update") is True
            and rejection.get("policy_state_unchanged_without_update") is True
            and rejection.get("low_state_unchanged_without_update") is True
            and rejection.get("optimizer_state_empty_after_rejection") is True
            and rejection.get("optimizer_steps_after_rejection") == 0
            and rejection.get("learned_checkpoint_count") == 0
            and isinstance(rejection.get("policy_state_sha256_after_rejection"), str)
            and rejection.get("policy_state_sha256_after_rejection") == value.get("initial_policy_state_sha256")
            and rejection.get("low_state_sha256_after_rejection") == EXPECTED_LOW_MODULE_SHA256
        )
    return result


def first_update_checks(value: Mapping[str, Any]) -> dict[str, bool]:
    checkpoint = value.get("first_update_checkpoint")
    iteration = value.get("iteration_receipt", {})
    return {
        "terminal_complete": value.get("schema") == FIRST_UPDATE_SCHEMA and value.get("status") == "COMPLETE" and value.get("failures") == [],
        "rollout_steps_exact": value.get("rollout_steps") == FIRST_ROLLOUT_STEPS,
        "rollout_samples_exact": value.get("rollout_samples") == FIRST_ROLLOUT_SAMPLES,
        "optimizer_steps_exact": value.get("ppo_optimizer_steps") == FIRST_UPDATE_OPTIMIZER_STEPS,
        "actor_changed": value.get("actor_update", {}).get("changed") is True,
        "critic_changed": value.get("critic_update", {}).get("changed") is True,
        "returns_all_finite": value.get("returns_complete") is True and value.get("returns_all_finite") is True,
        "losses_all_finite": value.get("losses_all_finite") is True,
        "storage_reset": value.get("storage_reset_after_update") is True,
        "checkpoint_provenance": value.get("checkpoint_loaded") is True
        and value.get("checkpoint_origin") == "RUN_LOCAL_INITIAL_EVALUATION"
        and value.get("optimizer_constructed") is True
        and value.get("optimizer_loaded") is False
        and value.get("fresh_optimizer_state_empty_before_first_update") is True,
        "policy_finite": value.get("policy_all_finite") is True,
        "low_unchanged": value.get("frozen_low_state_unchanged") is True and value.get("low_state_before_sha256") == EXPECTED_LOW_MODULE_SHA256 and value.get("low_state_after_sha256") == EXPECTED_LOW_MODULE_SHA256,
        "hidden_restored": value.get("recurrent_hidden_restore", {}).get("actor_equal_after_restore") is True
        and value.get("recurrent_hidden_restore", {}).get("critic_equal_after_restore") is True
        and value.get("recurrent_hidden_restore", {}).get("actor_unchanged_by_update") is True
        and value.get("recurrent_hidden_restore", {}).get("critic_unchanged_by_update") is True,
        "input_identity_fields_well_formed": _is_sha(value.get("initial_policy_state_sha256"))
        and _is_sha(value.get("initial_checkpoint_sha256"))
        and _is_sha(value.get("reward_support_rows_sha256")),
        "reward_support_bound": value.get("reward_support_gate_pass") is True,
        "first_update_checkpoint_valid": _checkpoint_receipt_valid(checkpoint, "model_first_update.pt"),
        "iteration_receipt_bound": isinstance(iteration, Mapping)
        and iteration.get("iteration") == 0
        and iteration.get("optimizer_steps") == FIRST_UPDATE_OPTIMIZER_STEPS
        and iteration.get("cumulative_optimizer_steps") == FIRST_UPDATE_OPTIMIZER_STEPS
        and iteration.get("returns_all_finite") is True
        and iteration.get("losses_all_finite") is True
        and iteration.get("storage_reset_after_update") is True,
        "no_escalation": value.get("long_training_authorized") is False,
    }


def validate_training_summary(
    value: Mapping[str, Any],
    *,
    initial_policy_sha256: str,
    initial_checkpoint_sha256: str,
    reward_support_rows_sha256: str,
    first_update_checkpoint_sha256: str,
    first_update_policy_state_sha256: str,
) -> None:
    _require(value.get("schema") == TRAINING_SCHEMA and value.get("status") == "COMPLETE" and value.get("failures") == [], "training incomplete")
    _require(value.get("initial_policy_state_sha256") == initial_policy_sha256 and value.get("initial_checkpoint_sha256") == initial_checkpoint_sha256, "training initial identity changed")
    _require(value.get("reward_support_rows_sha256") == reward_support_rows_sha256 and value.get("reward_support_gate_pass") is True, "training reward-support input changed")
    _exact_int(value.get("training_iterations"), TRAINING_ITERATIONS, "training iterations")
    _exact_int(value.get("rollout_steps"), TOTAL_ROLLOUT_STEPS, "training rollout steps")
    _exact_int(value.get("rollout_samples"), TOTAL_ROLLOUT_SAMPLES, "training samples")
    _exact_int(value.get("ppo_optimizer_steps"), TOTAL_OPTIMIZER_STEPS, "training optimizer steps")
    _require(value.get("optimizer_constructed") is True and value.get("actor_update", {}).get("changed") is True and value.get("critic_update", {}).get("changed") is True, "training update evidence invalid")
    _require(value.get("policy_all_finite") is True and value.get("frozen_low_state_unchanged") is True, "training model integrity failed")
    _require(value.get("low_state_before_sha256") == EXPECTED_LOW_MODULE_SHA256 and value.get("low_state_after_sha256") == EXPECTED_LOW_MODULE_SHA256, "training low identity changed")
    _require(value.get("dense_return_authorizes_behavior") is False and value.get("long_training_authorized") is False, "dense return or V56 authorized escalation")
    first_checkpoint = value.get("first_update_checkpoint", {})
    _require(_checkpoint_receipt_valid(first_checkpoint, "model_first_update.pt"), "training first-update checkpoint invalid")
    _require(
        first_checkpoint.get("sha256") == first_update_checkpoint_sha256
        and first_checkpoint.get("policy_state_sha256") == first_update_policy_state_sha256,
        "training first-update checkpoint bridge changed",
    )
    checkpoint = value.get("final_checkpoint", {})
    _require(isinstance(checkpoint, Mapping) and checkpoint.get("relative_path") == "model_final.pt" and type(checkpoint.get("bytes")) is int and checkpoint["bytes"] > 0, "final checkpoint invalid")
    _sha(checkpoint.get("sha256"), "final checkpoint"); _sha(checkpoint.get("policy_state_sha256"), "final checkpoint state")
    _require(checkpoint.get("policy_state_sha256") == value.get("final_policy_state_sha256"), "final checkpoint state differs from live final policy")
    receipts = value.get("iteration_receipts")
    _require(isinstance(receipts, list) and len(receipts) == TRAINING_ITERATIONS, "iteration receipts changed")
    for iteration, receipt in enumerate(receipts):
        _require(isinstance(receipt, Mapping) and receipt.get("iteration") == iteration, f"iteration {iteration} receipt missing")
        _exact_int(receipt.get("rollout_steps"), FIRST_ROLLOUT_STEPS, f"iteration {iteration} rollout")
        _exact_int(receipt.get("optimizer_steps"), FIRST_UPDATE_OPTIMIZER_STEPS, f"iteration {iteration} optimizer")
        _require(receipt.get("returns_all_finite") is True and receipt.get("losses_all_finite") is True and receipt.get("storage_reset_after_update") is True, f"iteration {iteration} integrity failed")
        hidden = receipt.get("recurrent_hidden_restore", {})
        _require(
            isinstance(hidden, Mapping)
            and all(hidden.get(key) is True for key in ("actor_equal_after_restore", "critic_equal_after_restore", "actor_unchanged_by_update", "critic_unchanged_by_update")),
            f"iteration {iteration} recurrent hidden guard failed",
        )


@dataclass(frozen=True)
class ResultInputs:
    initial: Mapping[str, Any]
    reward_support: Mapping[str, Any]
    first_update: Mapping[str, Any] | None
    training: Mapping[str, Any] | None
    final: Mapping[str, Any] | None


def build_result(contract: Mapping[str, Any], evidence: ResultInputs) -> dict[str, Any]:
    failures: list[str] = []; blockers: list[str] = []
    try:
        validate_evaluation_summary(evidence.initial, phase="initial")
    except ProtocolError as error:
        failures.append(str(error))
    support_checks = reward_support_checks(evidence.reward_support)
    base_names = ("row_count_exact", "complete_unique_grid", "all_finite", "all_formula_reconstructed", "global_nonzero_support", "global_positive_support", "global_negative_support", "positive_support_goal_side_negative", "positive_support_goal_side_positive")
    structural_base_names = ("row_count_exact", "complete_unique_grid", "all_finite", "all_formula_reconstructed")
    signal_names = ("global_nonzero_support", "global_positive_support", "global_negative_support", "positive_support_goal_side_negative", "positive_support_goal_side_positive")
    producer_checks = evidence.reward_support.get("checks", {})
    reconstructed_base_gate = isinstance(producer_checks, Mapping) and all(producer_checks.get(name) is True for name in base_names)
    initial_checkpoint = evidence.initial.get("initial_checkpoint", {})
    initial_checkpoint_sha = initial_checkpoint.get("sha256") if isinstance(initial_checkpoint, Mapping) else None
    support_checks["same_run_initial_identity_bound"] = bool(
        evidence.reward_support.get("initial_policy_state_sha256") == evidence.initial.get("policy_state_sha256")
        and evidence.reward_support.get("initial_checkpoint_sha256") == initial_checkpoint_sha
        and evidence.reward_support.get("frozen_low_state_sha256_before_gate") == evidence.initial.get("low_state_after_sha256")
        and evidence.reward_support.get("frozen_low_state_sha256_before_gate") == EXPECTED_LOW_MODULE_SHA256
    )
    producer_gate = bool(
        support_checks.get("terminal_complete") is True
        and support_checks.get("pre_optimizer_update_gate") is True
        and support_checks.get("transition_rows_receipt_valid") is True
        and support_checks.get("initial_identity_fields_well_formed") is True
        and reconstructed_base_gate
    )
    support_pass = producer_gate and support_checks["same_run_initial_identity_bound"]
    if evidence.reward_support.get("gate_pass") is not producer_gate:
        failures.append("producer reward-support decision disagrees with protocol fields")
    structural_names = (
        "terminal_complete", "pre_optimizer_update_gate",
        "transition_rows_receipt_valid", "initial_identity_fields_well_formed",
        "same_run_initial_identity_bound", *structural_base_names,
    )
    failed_support_integrity = [name for name in structural_names if support_checks.get(name) is not True]
    if failed_support_integrity:
        failures.append("reward-support structural/cross-run integrity failed: " + ",".join(failed_support_integrity))
    if not support_pass:
        blockers.extend(name for name in signal_names if support_checks.get(name) is not True)
        if not failed_support_integrity and support_checks.get("rejected_branch_no_update_integrity") is not True:
            failures.append("rejected reward-support branch lacks no-update integrity evidence")
        if evidence.first_update is not None or evidence.training is not None or evidence.final is not None:
            failures.append("PPO/final evaluation exists after rejected reward-support gate")
    else:
        first_checks: dict[str, bool] = {}
        if evidence.first_update is not None:
            first_checks = first_update_checks(evidence.first_update)
            first_checks["same_run_inputs_bound"] = bool(
                evidence.first_update.get("initial_policy_state_sha256") == evidence.initial.get("policy_state_sha256")
                and evidence.first_update.get("initial_checkpoint_sha256") == initial_checkpoint_sha
                and evidence.first_update.get("reward_support_rows_sha256") == evidence.reward_support.get("transition_rows_sha256")
            )
        if evidence.first_update is None or not all(first_checks.values()):
            failures.append("first PPO update evidence missing or invalid")
        if evidence.training is None or evidence.final is None:
            failures.append("full bounded branch lacks training/final evaluation")
        else:
            try:
                validate_training_summary(
                    evidence.training,
                    initial_policy_sha256=str(evidence.initial.get("policy_state_sha256", "")),
                    initial_checkpoint_sha256=str(initial_checkpoint_sha or ""),
                    reward_support_rows_sha256=str(evidence.reward_support.get("transition_rows_sha256", "")),
                    first_update_checkpoint_sha256=str(evidence.first_update.get("first_update_checkpoint", {}).get("sha256", "")) if evidence.first_update is not None else "",
                    first_update_policy_state_sha256=str(evidence.first_update.get("first_update_checkpoint", {}).get("policy_state_sha256", "")) if evidence.first_update is not None else "",
                )
                validate_evaluation_summary(evidence.final, phase="final")
            except ProtocolError as error:
                failures.append(str(error))
            if evidence.final.get("checkpoint_sha256") != evidence.training.get("final_checkpoint", {}).get("sha256"):
                failures.append("final evaluation checkpoint differs from training output")
            if evidence.final.get("policy_state_sha256") != evidence.training.get("final_checkpoint", {}).get("policy_state_sha256"):
                failures.append("final evaluation policy state differs from training output")
            if evidence.final.get("fixed_case_table_sha256") != evidence.initial.get("fixed_case_table_sha256"):
                failures.append("initial/final fixed case tables differ")
    initial_goals = evidence.initial.get("goal_count")
    final_goals = None if evidence.final is None else evidence.final.get("goal_count")
    delta = int(final_goals) - int(initial_goals) if type(final_goals) is int and type(initial_goals) is int else None
    final_hard = None if evidence.final is None else evidence.final.get("hard_goal_count")
    final_side = {} if evidence.final is None else evidence.final.get("goal_counts_by_goal_side", {})
    final_hard_side = {} if evidence.final is None else evidence.final.get("hard_goal_counts_by_goal_side", {})
    final_unsafe = None if evidence.final is None else evidence.final.get("unsafe_count")
    performance_gate = bool(
        delta is not None and delta > 0 and int(final_goals) >= OVERALL_HALF_GATE
        and type(final_hard) is int and final_hard >= HARD_HALF_GATE
        and all(type(final_side.get(str(side))) is int and final_side[str(side)] >= gate for side, gate in GOAL_SIDE_HALF_GATES.items())
        and all(type(final_hard_side.get(str(side))) is int and final_hard_side[str(side)] >= gate for side, gate in HARD_GOAL_SIDE_HALF_GATES.items())
        and type(final_unsafe) is int and final_unsafe <= MAXIMUM_UNSAFE_HALF_GATE
    )
    if failures:
        status, decision, next_gate = "ERROR", DECISION_INVALID, "INSPECT_PRESERVED_ORIGINAL_FAILURE_BEFORE_MORE_GPU_WORK"
    elif not support_pass:
        status, decision, next_gate = "COMPLETE", DECISION_REWARD_SUPPORT_REJECTED, "STOP_BEFORE_PPO_UPDATE_CHANGE_ONE_CAUSE_ONLY"
    elif performance_gate:
        status, decision, next_gate = "COMPLETE", DECISION_HALF_GATE, "MATCHED_MULTISEED_CONFIRMATION_ONLY"
    elif type(final_goals) is int and final_goals == 0:
        status, decision, next_gate = "COMPLETE", DECISION_NO_BREAKTHROUGH, "NO_ESCALATION_FROM_THIS_BOUNDED_CANARY"
        blockers.append("zero independently verified final fixed goals")
    elif delta is None or delta <= 0:
        status, decision, next_gate = "COMPLETE", DECISION_NONIMPROVING, "NO_LEARNING_ATTRIBUTION_NO_ESCALATION"
        blockers.append("final fixed goals did not exceed initial")
    else:
        status, decision, next_gate = "COMPLETE", DECISION_BREAKTHROUGH, "MATCHED_MULTISEED_CONFIRMATION_ONLY"
    result = {
        "result_schema": RESULT_SCHEMA, "status": status, "decision": decision,
        "engineering_integrity_pass": not failures,
        "reward_support_gate_pass": support_pass, "reward_support_checks": support_checks,
        "optimizer_constructed": evidence.reward_support.get("optimizer_constructed_before_gate") is True,
        "first_update_executed": evidence.first_update is not None,
        "training_executed": evidence.training is not None,
        "fixed_initial_evaluation_executed": True, "fixed_final_evaluation_executed": evidence.final is not None,
        "initial_goal_count": initial_goals, "final_goal_count": final_goals,
        "initial_hard_goal_count": evidence.initial.get("hard_goal_count"), "final_hard_goal_count": final_hard,
        "initial_unsafe_count": evidence.initial.get("unsafe_count"), "final_unsafe_count": final_unsafe,
        "initial_goal_counts_by_goal_side": evidence.initial.get("goal_counts_by_goal_side"),
        "final_goal_counts_by_goal_side": None if evidence.final is None else evidence.final.get("goal_counts_by_goal_side"),
        "initial_hard_goal_counts_by_goal_side": evidence.initial.get("hard_goal_counts_by_goal_side"),
        "final_hard_goal_counts_by_goal_side": None if evidence.final is None else evidence.final.get("hard_goal_counts_by_goal_side"),
        "final_minus_initial_goal_count": delta, "single_seed_half_gate_pass": performance_gate,
        "dense_return_authorizes_behavior": False,
        "ppo_optimizer_steps": 0 if evidence.training is None else evidence.training.get("ppo_optimizer_steps"),
        "initial_evaluation": dict(evidence.initial), "reward_support": dict(evidence.reward_support),
        "first_update": None if evidence.first_update is None else dict(evidence.first_update),
        "training": None if evidence.training is None else dict(evidence.training),
        "final_evaluation": None if evidence.final is None else dict(evidence.final),
        "blockers": sorted(set(blockers)), "failures": failures, "next_gate": next_gate,
        "learned_high_hypothesis_globally_rejected": False, "long_training_authorized": False,
    }
    _require(decision in DECISIONS, "terminal decision changed")
    return result


# ---------------------------------------------------------------------------
# Delivery-only independent filesystem, command-chain, and cross-file checks.
# ---------------------------------------------------------------------------

CONTRACT_FILE_SHA256 = "3b96dbfffcfee02c24f50eba1e8ae1884fdbd3f020dc1b85acb950c0c9379289"
LOW_CHECKPOINT_BYTES = 4_584_447
PARENT_TASK_MANIFEST_SHA256 = EXPECTED_PARENT_V55["task_manifest_sha256"]
SOURCE_RECEIPT_HASHES = {
    "v14": {
        "continuous_reset.py": "1846c595f4b4a10b0ae7b91230eecee68f716926a022b281ac718bd5ab66edab",
        "entity_observation.py": "cf770f74592b7d573a61fba1430b3bf912e90166796810e9b88f8892e2c49ca4",
        "high_command_interface.py": "76688b1d6b344942f1bdaf5fdc813fe0bf8cdb64f1e09250e4980cb75fc8352b",
        "interface_contract.py": "bf9f2be51b636832bb8dd755d795c3e152b1d678c1030e9c0f508b1f8621eed0",
        "sparse_goal_reward.py": "4e6f1a633c31df0cf41c9cef19e255eba833ca27f4ef1ae95c68899bcd52fe84",
        "sparse_high_env_cfg.py": "30acfab2b6bd15680319d82b1c4b46c0105ca18d7cbcb1590a69879727ce1a8c",
    },
    "current": {
        "code06/check_frozen_low_state.py": "4191dd62d59e578ac86a45795adbad71ed5b0da907b917f80587126507b43f10",
        "code06/frozen_low_actor.py": "d4cd9f975fb350b4f597f949db41445063f10dbd4b16ec7e6d9aec6198677750",
        "code06/staging_command_tensor.py": "c252d478563585f56a6ff3caf6156b43aab8762029e7c3b09718ae664f00242c",
        "code06/staging_contact_contract.py": "2eb17a913eb9df85a9829d7ad547af19b470286f294ca2fe1025441c7fb87e89",
        "code06/native_low_executor_v1/native_low_mdp.py": "e84cec4984c09ee3503779be76a5edda697a3f4ba43b5a53c86229e120510bf9",
        "code06/native_low_executor_v1/native_safe_joint_action.py": "0d592389e9d16a56fedf0c0234f8a5eed70013259f8759feb80b7656defc14be",
    },
}
RUNNER_IDENTITY = {
    "policy_class": "ActorCriticRecurrent", "is_recurrent": True,
    "observation_dim": 70, "critic_observation_dim": 70, "action_dim": 3,
    "rnn_type": "gru", "actor_gru_hidden_dim": 256,
    "critic_gru_hidden_dim": 256, "actor_gru_layers": 1,
    "critic_gru_layers": 1,
    "actor_linear_dimensions": [256, 256, 256, 128, 3],
    "critic_linear_dimensions": [256, 256, 256, 128, 1],
    "num_steps_per_env": 200, "schedule": "fixed", "learning_rate": 0.0003,
    "desired_kl": None, "entropy_coef": 0.03, "gamma": 0.997,
    "lambda": 0.98, "num_learning_epochs": 5, "num_mini_batches": 4,
    "actor_observation_normalization": False,
    "critic_observation_normalization": False,
    "correlated_or_ar_exploration": False, "noise_std_type": "scalar",
    "state_dependent_std": False, "clip_param": 0.2,
    "value_loss_coef": 1.0, "use_clipped_value_loss": True,
    "max_grad_norm": 1.0, "normalize_advantage_per_mini_batch": False,
    "rnd_registered": False, "symmetry_registered": False,
}
EPISODE_KEYS = frozenset({
    "schema", "evaluation_assignment", "counts_for_evaluation",
    "hard_goal_facing_ball_behind", "bucket", "goal_side", "sample_serial",
    "item_seed", "reset_robot_xy_m", "reset_robot_yaw_rad", "reset_ball_xy_m",
    "outcome", "goal", "own_goal", "boundary", "fall", "joint_hard_limit",
    "timeout", "unsafe_pre_goal", "requested_goal_line_reached", "contact_seen",
    "first_true_contact_native_tick", "maximum_contact_force_n",
    "terminal_contact_force_n", "terminal_base_contact_force_n",
    "terminal_joint_hard_limit_excess_rad", "native_safe_guard_changed_elements",
    "native_safe_guard_max_correction_rad", "episode_high_ticks",
    "episode_native_ticks", "terminal_robot_xy_m", "terminal_ball_xy_m",
    "initial_l_m", "final_l_m", "cumulative_progress_reward",
    "requested_goal_axis_ball_displacement_m", "abs_lateral_ball_drift_m",
})
TRANSITION_KEYS = frozenset({
    "schema", "global_transition_index", "rollout_iteration", "high_step_index",
    "env_id", "sample_serial", "episode_serial_before_step", "item_seed",
    "episode_high_step_index_before_step", "goal_side", "reset_robot_xy_m",
    "reset_robot_yaw_rad", "reset_ball_xy_m", "pre_robot_xy_m", "pre_ball_xy_m",
    "post_robot_xy_pre_reset_m", "post_ball_xy_pre_reset_m", "goal_xy_m",
    "raw_staging_target_pre_xy_m", "raw_staging_target_post_xy_m",
    "staging_target_pre_xy_m", "staging_target_post_xy_m",
    "staging_projection_pre_applied", "staging_projection_post_applied",
    "l_pre_m", "l_post_m", "normalizer_m", "progress_reward",
    "strict_goal_reward", "total_reward", "raw_action", "applied_command",
    "native_ticks_executed", "terminal_reason", "goal", "own_goal", "boundary",
    "fall", "joint_hard_limit", "timeout", "unsafe_same_tick", "done",
    "reset_after_step", "contact_seen", "maximum_contact_force_n",
    "terminal_contact_force_n", "terminal_base_contact_force_n",
    "terminal_joint_hard_limit_excess_rad", "post_state_source",
})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"{label} missing or unsafe")
    metadata = path.lstat()
    _require(stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1, f"{label} is not a single-link regular file")
    try:
        value = json.loads(
            path.read_bytes(), object_pairs_hook=_reject_duplicate_json_object,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ProtocolError(f"{label} is invalid JSON: {error}") from error
    _require(isinstance(value, dict), f"{label} is not an object")
    return value


def strict_jsonl(path: Path, label: str) -> tuple[bytes, list[dict[str, Any]]]:
    _require(path.is_file() and not path.is_symlink(), f"{label} missing or unsafe")
    metadata = path.lstat()
    _require(stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1, f"{label} is not a single-link regular file")
    payload = path.read_bytes()
    _require(payload.endswith(b"\n"), f"{label} lacks final newline")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(payload.splitlines(keepends=True), 1):
        _require(line not in (b"", b"\n"), f"{label} has empty row {number}")
        try:
            row = json.loads(
                line, object_pairs_hook=_reject_duplicate_json_object,
                parse_constant=_reject_nonfinite_json_constant,
            )
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ProtocolError(f"{label} row {number} invalid: {error}") from error
        _require(isinstance(row, dict), f"{label} row {number} is not an object")
        _require(canonical_json_bytes(row) == line, f"{label} row {number} is not canonical")
        rows.append(row)
    return payload, rows


def real_directory(path: Path, label: str, *, parent: Path | None = None) -> Path:
    metadata = path.lstat()
    _require(stat.S_ISDIR(metadata.st_mode) and not path.is_symlink(), f"{label} is not a real directory")
    resolved = path.resolve(strict=True)
    if parent is not None:
        _require(resolved.is_relative_to(parent), f"{label} escaped its parent")
    return resolved


def read_environment(path: Path) -> dict[str, str]:
    _require(path.is_file() and not path.is_symlink() and path.lstat().st_nlink == 1,
             "runtime environment missing or unsafe")
    result: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        _require("=" in line, f"runtime environment line {number} malformed")
        key, value = line.split("=", 1)
        _require(bool(key) and key not in result, f"runtime environment duplicate key: {key}")
        result[key] = value
    return result


def resolved_absolute(value: Any, label: str) -> Path:
    _require(isinstance(value, str) and value.startswith("/"), f"{label} not absolute")
    try:
        return Path(value).resolve(strict=True)
    except OSError as error:
        raise ProtocolError(f"{label} unresolved: {error}") from error


def _f32_add(left: float, right: float) -> float:
    return _canonical_float32(_canonical_float32(left) + _canonical_float32(right))


def _f32_sub(left: float, right: float) -> float:
    return _canonical_float32(_canonical_float32(left) - _canonical_float32(right))


def _f32_mul(left: float, right: float) -> float:
    return _canonical_float32(_canonical_float32(left) * _canonical_float32(right))


def _f32_div(left: float, right: float) -> float:
    return _canonical_float32(_canonical_float32(left) / _canonical_float32(right))


COMMAND_VERTICES = tuple(tuple(_canonical_float32(x) for x in row) for row in EXPECTED_COMMAND_INTERFACE["vertices"])
COMMAND_HALF_RANGES = tuple(_canonical_float32((hi - lo) / 2.0) for lo, hi in EXPECTED_COMMAND_INTERFACE["axis_ranges"])
COMMAND_RATE_LIMITS = tuple(_canonical_float32(x) for x in EXPECTED_COMMAND_INTERFACE["rate_limits_per_second"])
HIGH_DT = _canonical_float32(0.1)
ACTION_CHAIN_GROSS_TOLERANCE = 2.0e-3
COMMAND_FEASIBILITY_TOLERANCE = 3.0e-6
COMMAND_RATE_TOLERANCE = 5.0e-5


def _sum_products(left: Sequence[float], right: Sequence[float]) -> float:
    total = _canonical_float32(0.0)
    for x, y in zip(left, right, strict=True):
        total = _f32_add(total, _f32_mul(x, y))
    return total


def project_command(desired: Sequence[float]) -> list[float]:
    desired32 = [_canonical_float32(x) for x in desired]
    distances = []
    for vertex in COMMAND_VERTICES:
        delta = [_f32_sub(desired32[i], vertex[i]) for i in range(3)]
        distances.append(_sum_products(delta, delta))
    current = list(COMMAND_VERTICES[min(range(len(distances)), key=distances.__getitem__)])
    tolerance = _canonical_float32(1.0e-9)
    for _ in range(64):
        gradient = [_f32_sub(current[i], desired32[i]) for i in range(3)]
        scores = [_sum_products(gradient, vertex) for vertex in COMMAND_VERTICES]
        vertex = COMMAND_VERTICES[min(range(len(scores)), key=scores.__getitem__)]
        direction = [_f32_sub(vertex[i], current[i]) for i in range(3)]
        gap = _canonical_float32(-_sum_products(gradient, direction))
        denominator = _sum_products(direction, direction)
        if not (gap > tolerance and denominator > tolerance):
            break
        step = min(1.0, max(0.0, _f32_div(gap, denominator)))
        current = [_f32_add(current[i], _f32_mul(step, direction[i])) for i in range(3)]
    return current


def reconstruct_applied_command(raw_action: Sequence[float], previous: Sequence[float]) -> list[float]:
    bounded = [_canonical_float32(math.tanh(_canonical_float32(value))) for value in raw_action]
    physical = [_f32_mul(bounded[index], COMMAND_HALF_RANGES[index]) for index in range(3)]
    target = project_command(physical)
    prior = project_command(previous)
    delta = [_f32_sub(target[index], prior[index]) for index in range(3)]
    permitted = [_f32_mul(COMMAND_RATE_LIMITS[index], HIGH_DT) for index in range(3)]
    ratios = [
        _f32_div(permitted[index], abs(delta[index])) if abs(delta[index]) > 0.0 else _canonical_float32(1.0)
        for index in range(3)
    ]
    scale = _canonical_float32(min(1.0, max(0.0, min(ratios))))
    return [_f32_add(prior[index], _f32_mul(scale, delta[index])) for index in range(3)]


def _command_hull_planes() -> tuple[tuple[tuple[float, float, float], float], ...]:
    """Return every supporting plane induced by a non-collinear vertex triple."""
    planes: list[tuple[tuple[float, float, float], float]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for left, middle, right in itertools.combinations(COMMAND_VERTICES, 3):
        first = tuple(middle[index] - left[index] for index in range(3))
        second = tuple(right[index] - left[index] for index in range(3))
        normal = (
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        )
        length = math.sqrt(sum(item * item for item in normal))
        if length <= 1.0e-12:
            continue
        unit = tuple(item / length for item in normal)
        offset = sum(unit[index] * left[index] for index in range(3))
        distances = [sum(unit[index] * vertex[index] for index in range(3)) - offset for vertex in COMMAND_VERTICES]
        if max(distances) <= 1.0e-9:
            oriented, bound = unit, offset
        elif min(distances) >= -1.0e-9:
            oriented, bound = tuple(-item for item in unit), -offset
        else:
            continue
        key = tuple(round(item * 1.0e9) for item in (*oriented, bound))
        if key not in seen:
            seen.add(key)
            planes.append((oriented, bound))
    _require(bool(planes), "qualified command hull has no supporting planes")
    return tuple(planes)


COMMAND_HULL_PLANES = _command_hull_planes()


def command_in_qualified_hull(command: Sequence[float]) -> bool:
    return all(
        sum(normal[index] * command[index] for index in range(3)) - bound
        <= COMMAND_FEASIBILITY_TOLERANCE
        for normal, bound in COMMAND_HULL_PLANES
    )


def command_respects_rate_limit(command: Sequence[float], previous: Sequence[float]) -> bool:
    projected_previous = project_command(previous)
    return all(
        abs(command[axis] - projected_previous[axis])
        <= float(COMMAND_RATE_LIMITS[axis]) * float(HIGH_DT) + COMMAND_RATE_TOLERANCE
        for axis in range(3)
    )


def validate_episode_file(path: Path, summary: Mapping[str, Any], *, phase: str) -> dict[str, Any]:
    payload, rows = strict_jsonl(path, f"{phase} episode rows")
    for index, row in enumerate(rows):
        _require(set(row) == EPISODE_KEYS, f"{phase} row[{index}] key set changed")
        first_contact = row.get("first_true_contact_native_tick")
        _require(first_contact is None or (type(first_contact) is int and 0 <= first_contact < 1000), f"{phase} row[{index}] first-contact tick invalid")
        _bounded_int(row.get("native_safe_guard_changed_elements"), 0, 12, f"{phase} row[{index}] guard elements")
        _require(_finite(row.get("native_safe_guard_max_correction_rad"), f"{phase} row[{index}] guard correction") >= 0.0, f"{phase} row[{index}] guard correction negative")
    metrics = validate_episode_rows(rows)
    _require(summary.get("episode_rows_sha256") == hashlib.sha256(payload).hexdigest(), f"{phase} row SHA differs")
    exact_keys = (
        "assignment_sha256", "fixed_case_table_sha256", "goal_count",
        "hard_goal_count", "hard_case_count", "unsafe_count", "contact_count",
        "case_counts_by_goal_side", "hard_case_counts_by_goal_side",
        "goal_counts_by_goal_side", "hard_goal_counts_by_goal_side", "outcome_counts",
    )
    for key in exact_keys:
        _require(summary.get(key) == metrics[key], f"{phase} summary differs from rows: {key}")
    for key in (
        "maximum_live_reset_error", "mean_initial_l_m", "mean_final_l_m",
        "mean_cumulative_progress_reward",
        "mean_requested_goal_axis_ball_displacement_m",
        "mean_abs_lateral_ball_drift_m",
    ):
        _require(close_scalar(summary.get(key), float(metrics[key]), 1.0e-9), f"{phase} summary differs from rows: {key}")
    return metrics


def validate_transition_file(path: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    payload, rows = strict_jsonl(path, "first rollout transitions")
    previous_commands: dict[int, list[float]] = {env_id: [0.0, 0.0, 0.0] for env_id in range(128)}
    abs_command_sum = [0.0, 0.0, 0.0]
    terminal_reasons: Counter[str] = Counter()
    contact_transitions = 0
    for index, row in enumerate(rows):
        _require(set(row) == TRANSITION_KEYS, f"transition[{index}] key set changed")
        env_id = row.get("env_id")
        _require(type(env_id) is int and 0 <= env_id < 128, f"transition[{index}] env invalid")
        raw = _vector(row.get("raw_action"), 3, f"transition[{index}].raw_action")
        applied = _vector(row.get("applied_command"), 3, f"transition[{index}].applied_command")
        previous = previous_commands[env_id]
        reconstructed = reconstruct_applied_command(raw, previous)
        # The producer executes the unchanged Frank-Wolfe map on CUDA.  Near
        # equal-score vertices can diverge slightly from this independent CPU
        # replay, so replay is a gross chain guard; feasibility and rate are
        # checked separately with tight, decisive tolerances.
        _require(close_vector(applied, reconstructed, ACTION_CHAIN_GROSS_TOLERANCE), f"transition[{index}] applied command chain grossly differs")
        ranges = EXPECTED_COMMAND_INTERFACE["axis_ranges"]
        _require(all(
            ranges[axis][0] - COMMAND_FEASIBILITY_TOLERANCE <= applied[axis] <= ranges[axis][1] + COMMAND_FEASIBILITY_TOLERANCE
            for axis in range(3)
        ), f"transition[{index}] command outside qualified axis range")
        _require(command_in_qualified_hull(applied), f"transition[{index}] command outside qualified convex hull")
        _require(command_respects_rate_limit(applied, previous),
                 f"transition[{index}] command violates 0.1-second rate limit")
        for axis in range(3):
            abs_command_sum[axis] += abs(applied[axis])
        terminal_reasons[str(row.get("terminal_reason"))] += int(row.get("done") is True)
        contact_transitions += int(row.get("contact_seen") is True)
        previous_commands[env_id] = [0.0, 0.0, 0.0] if row.get("done") is True else applied
    derived = validate_first_rollout_transitions(rows)
    digest = hashlib.sha256(payload).hexdigest()
    _require(summary.get("transition_rows_relative_path") == path.name, "reward support row path changed")
    _require(summary.get("transition_rows_bytes") == len(payload), "reward support row byte count differs")
    _require(summary.get("transition_rows_sha256") == digest, "reward support row SHA differs")
    for key, expected in derived.items():
        if key in {"checks", "gate_pass"}:
            continue
        actual = summary.get(key)
        if isinstance(expected, float):
            _require(close_scalar(actual, expected, 1.0e-12), f"reward support summary differs: {key}")
        else:
            _require(actual == expected, f"reward support summary differs: {key}")
    base_checks = derived["checks"]
    _require(all(summary.get("checks", {}).get(key) is value for key, value in base_checks.items()), "reward support producer checks differ from raw rows")
    signal_names = (
        "global_nonzero_support", "global_positive_support", "global_negative_support",
        "positive_support_goal_side_negative", "positive_support_goal_side_positive",
    )
    expected_gate = all(base_checks[name] for name in signal_names)
    _require(summary.get("gate_pass") is expected_gate, "reward support gate differs from raw rows")
    derived["transition_rows_sha256"] = digest
    derived["mean_abs_applied_command"] = [value / len(rows) for value in abs_command_sum]
    derived["terminal_reason_counts"] = dict(sorted(terminal_reasons.items()))
    derived["contact_transition_count"] = contact_transitions
    return derived


def validate_runner(value: Any, label: str) -> None:
    _require(isinstance(value, Mapping) and set(value) == set(RUNNER_IDENTITY), f"{label} runner keys changed")
    for key, expected in RUNNER_IDENTITY.items():
        actual = value.get(key)
        if isinstance(expected, float):
            _require(close_scalar(actual, expected, 1.0e-12), f"{label} runner differs: {key}")
        else:
            _require(actual == expected, f"{label} runner differs: {key}")


def validate_std(value: Any, *, initial: bool, label: str) -> None:
    _require(isinstance(value, Mapping) and value.get("shape") == [3], f"{label} std schema changed")
    values = _vector(value.get("values"), 3, f"{label}.values")
    _require(value.get("all_finite_positive") is True and all(item > 0.0 for item in values), f"{label} std invalid")
    _require(value.get("unit_initial_required") is initial, f"{label} std initial flag changed")
    if initial:
        _require(value.get("unit_initial_pass") is True and all(close_scalar(item, 1.0, 1.0e-7) for item in values), f"{label} fresh std changed")


def validate_heads(value: Any, label: str) -> None:
    _require(isinstance(value, Mapping) and set(value) == {"actor", "critic"}, f"{label} output heads changed")
    for name, shape in (("actor", [3, 128]), ("critic", [1, 128])):
        item = value[name]
        _require(isinstance(item, Mapping) and item.get("weight_shape") == shape, f"{label}.{name} shape changed")
        _require(_finite(item.get("weight_l2"), f"{label}.{name}.l2") > 0.0, f"{label}.{name} zero")
        _require(_finite(item.get("weight_max_abs"), f"{label}.{name}.max") > 0.0 and item.get("weight_nonzero") is True, f"{label}.{name} zero")


def validate_source_receipts(value: Any, source: Path, label: str) -> None:
    _require(isinstance(value, Mapping) and set(value) == {"v14", "current", "roots"}, f"{label} source receipts changed")
    expected_roots = {
        "v14": (source / "vendor/v14").resolve(strict=True),
        "current": (source / "vendor/current_qualified").resolve(strict=True),
    }
    roots = value["roots"]
    _require(isinstance(roots, Mapping) and set(roots) == {"v14", "current"}, f"{label} source roots changed")
    for group, expected_root in expected_roots.items():
        root = resolved_absolute(roots[group], f"{label}.{group} root")
        _require(root == expected_root, f"{label}.{group} escaped package")
        receipts = value[group]
        _require(isinstance(receipts, Mapping) and set(receipts) == set(SOURCE_RECEIPT_HASHES[group]), f"{label}.{group} receipt set changed")
        for relative, digest in SOURCE_RECEIPT_HASHES[group].items():
            path = (root / relative).resolve(strict=True)
            _require(path.is_relative_to(root) and path.is_file() and not path.is_symlink()
                     and path.lstat().st_nlink == 1, f"{label}.{group} path unsafe")
            _require(receipts[relative] == {"sha256": digest, "bytes": path.stat().st_size}, f"{label}.{group} receipt differs: {relative}")
            _require(file_sha256(path) == digest, f"{label}.{group} bytes differ: {relative}")


def validate_runtime(
    value: Any,
    environment: Mapping[str, str],
    *,
    expected_seed: int,
    label: str,
    expected_num_envs: int = 128,
) -> None:
    _require(isinstance(value, Mapping), f"{label} runtime missing")
    _exact_int(value.get("num_envs"), expected_num_envs, f"{label}.num_envs")
    _exact_int(value.get("reset_seed"), expected_seed, f"{label}.reset_seed")
    sim_dt = _finite(value.get("sim_dt_seconds"), f"{label}.sim_dt")
    decimation = _bounded_int(value.get("decimation"), 1, 100, f"{label}.decimation")
    _require(close_scalar(sim_dt * decimation, 0.02, 1.0e-12), f"{label} low period changed")
    _require(value.get("high_policy_hz") == 10 and value.get("low_policy_hz") == 50 and value.get("low_steps_per_high_action") == 5, f"{label} hierarchy changed")
    _require(value.get("finite_horizon") is True and value.get("time_outs_bootstrap_key_emitted") is False, f"{label} timeout semantics changed")
    _require(value.get("native_auto_terminations_disabled") == ["actual_joint_hard_limit", "base_contact", "time_out"], f"{label} native termination override changed")
    low = value.get("current_low_checkpoint")
    _require(low == {"basename": "model_final.pt", "bytes": LOW_CHECKPOINT_BYTES, "sha256": EXPECTED_LOW_CHECKPOINT_SHA256, "runtime_module_state_sha256": EXPECTED_LOW_MODULE_SHA256}, f"{label} low identity changed")
    _require(value.get("parent_v55_identity") == EXPECTED_PARENT_V55, f"{label} parent identity changed")
    _require(all(environment.get(f"PARENT_V55_{key.upper()}") == str(expected) for key, expected in EXPECTED_PARENT_V55.items()), f"{label} runtime environment parent differs")
    task = value.get("task_package_import_origin", {})
    _require(task.get("status") == "PASS" and resolved_absolute(task.get("distribution_root"), f"{label} task root") == resolved_absolute(environment.get("TASK_PACKAGE_SNAPSHOT"), "task snapshot"), f"{label} task import escaped")
    _bounded_int(task.get("loaded_module_count"), 3, 100_000, f"{label} task modules"); _sha(task.get("module_origins_sha256"), f"{label} task origins")
    isaac = value.get("isaac_import_origins", {})
    expected_env = {"isaaclab": "V56_ISAACLAB_DISTRIBUTION_ROOT", "isaaclab_tasks": "V56_ISAACLAB_TASKS_DISTRIBUTION_ROOT", "isaaclab_assets": "V56_ISAACLAB_ASSETS_DISTRIBUTION_ROOT", "isaaclab_rl": "V56_ISAACLAB_RL_DISTRIBUTION_ROOT"}
    _require(isinstance(isaac, Mapping) and set(isaac) == set(expected_env), f"{label} Isaac receipts changed")
    for package, key in expected_env.items():
        item = isaac[package]
        _require(item.get("status") == "PASS" and resolved_absolute(item.get("distribution_root"), f"{label}.{package}") == resolved_absolute(environment.get(key), key), f"{label}.{package} origin differs")
        _bounded_int(item.get("loaded_module_count"), 1, 100_000, f"{label}.{package} count"); _sha(item.get("module_origins_sha256"), f"{label}.{package} digest")
    rsl = value.get("rsl_rl_import_origin", {})
    _require(rsl.get("status") == "PASS" and rsl.get("version") == "3.1.2", f"{label} RSL receipt changed")
    _require(resolved_absolute(rsl.get("distribution_root"), f"{label} RSL root") == resolved_absolute(environment.get("V56_RSL_RL_DISTRIBUTION_ROOT"), "RSL root"), f"{label} RSL root differs")
    _bounded_int(rsl.get("loaded_module_count"), 1, 100_000, f"{label} RSL modules"); _sha(rsl.get("module_origins_sha256"), f"{label} RSL digest")


def validate_environment_summary(value: Any, *, evaluation: bool, label: str, expected_high_steps: int | None = None) -> None:
    _require(isinstance(value, Mapping) and value.get("status") == "CURRENT_TASK_PROGRESS_HIGH_ENV_COMPLETE", f"{label} environment missing")
    expected = {
        "observation_dim": 70, "action_dim": 3, "high_policy_rate_hz": 10,
        "low_policy_rate_hz": 50, "low_steps_per_high_action": 5,
        "finite_horizon": True, "time_outs_bootstrap_key_emitted": False,
        "terminal_semantics": "V53_STRICT_UNSAFE_SAME_TICK_REJECTS_GOAL_PRE_RESET_SNAPSHOT",
        "actual_joint_hard_limit_is_terminal": True,
        "native_safe_guard_correction_is_terminal": False,
        "curriculum_registered": False, "teacher_registered": False,
        "runtime_router_registered": False, "scripted_phase_transition": False,
        "kick_registered": False, "native_base_reward_may_be_registered": True,
        "native_base_reward_discarded": True,
        "native_base_reward_consumed_by_high_or_ppo": False,
        "high_ppo_reward_kind": "TASK_PROGRESS_PLUS_STRICT_TERMINAL",
        "strict_terminal_reward_values": [-1.0, 0.0, 1.0],
        "dense_return_authorizes_behavior": False, "contact_is_terminal": False,
    }
    _require(all(value.get(key) == expected_value for key, expected_value in expected.items()), f"{label} environment contract changed")
    high_steps = _bounded_int(value.get("high_vector_steps"), 1, 2_400 if not evaluation else 800, f"{label} high steps")
    _exact_int(value.get("low_vector_steps"), 5 * high_steps, f"{label} low steps")
    if expected_high_steps is not None:
        _exact_int(high_steps, expected_high_steps, f"{label} expected high steps")
    _bounded_int(value.get("contact_positive_low_ticks"), 0, 10**12, f"{label} contact-positive ticks")
    _bounded_int(value.get("contact_negative_low_ticks"), 0, 10**12, f"{label} contact-negative ticks")
    _require(isinstance(value.get("outcome_counts"), Mapping) and isinstance(value.get("bucket_outcome_counts"), Mapping), f"{label} outcome telemetry missing")
    reward_terms = value.get("reward_term_means")
    _require(isinstance(reward_terms, Mapping) and set(reward_terms) == {"progress", "strict_goal", "total"}, f"{label} reward telemetry changed")
    _require(all(math.isfinite(float(item)) for item in reward_terms.values()), f"{label} reward telemetry non-finite")
    commands = _vector(value.get("mean_abs_applied_command"), 3, f"{label} mean command")
    _require(all(item >= 0.0 for item in commands), f"{label} mean command negative")
    _bounded_int(value.get("native_safe_guard_changed_elements"), 0, 10**12, f"{label} guard count")
    _require(_finite(value.get("native_safe_guard_max_correction_rad"), f"{label} guard max") >= 0.0, f"{label} guard max negative")
    if evaluation:
        _exact_int(value.get("evaluation_cases_completed"), FIXED_CASE_COUNT, f"{label} evaluation cases")
    else:
        _exact_int(value.get("evaluation_cases_completed"), 0, f"{label} evaluation cases")


def validate_evaluation_extended(
    summary: Mapping[str, Any], rows_path: Path, checkpoint_path: Path,
    *, phase: str, environment: Mapping[str, str], source: Path,
) -> dict[str, Any]:
    validate_evaluation_summary(summary, phase=phase)
    metrics = validate_episode_file(rows_path, summary, phase=phase)
    validate_runner(summary.get("runner_identity"), phase)
    validate_std(summary.get("action_std"), initial=phase == "initial", label=phase)
    validate_heads(summary.get("output_heads"), phase)
    validate_runtime(summary.get("runtime_identity"), environment, expected_seed=FIXED_EVALUATION_SEED, label=phase)
    validate_source_receipts(summary.get("source_receipts"), source, phase)
    validate_environment_summary(summary.get("environment"), evaluation=True, label=phase)
    _require(summary.get("policy_state_unchanged") is True and summary.get("frozen_low_state_unchanged") is True, f"{phase} mutated state")
    if phase == "initial":
        receipt = summary.get("initial_checkpoint", {})
        _require(receipt.get("relative_path") == checkpoint_path.name and receipt.get("bytes") == checkpoint_path.stat().st_size, "initial checkpoint receipt path/bytes differ")
        _require(receipt.get("sha256") == file_sha256(checkpoint_path), "initial checkpoint bytes differ")
    else:
        _require(summary.get("checkpoint_sha256") == file_sha256(checkpoint_path), "final checkpoint bytes differ")
        _require(summary.get("loaded_policy_state_sha256") == summary.get("policy_state_sha256"), "final loaded state differs")
    return metrics


def validate_training_extended(
    value: Mapping[str, Any], first: Mapping[str, Any], support: Mapping[str, Any],
    initial: Mapping[str, Any], environment: Mapping[str, str], source: Path,
    first_checkpoint: Path, final_checkpoint: Path,
) -> None:
    initial_receipt = initial["initial_checkpoint"]
    first_receipt = first.get("first_update_checkpoint", {})
    validate_training_summary(
        value,
        initial_policy_sha256=initial["policy_state_sha256"],
        initial_checkpoint_sha256=initial_receipt["sha256"],
        reward_support_rows_sha256=support["transition_rows_sha256"],
        first_update_checkpoint_sha256=first_receipt.get("sha256", ""),
        first_update_policy_state_sha256=first_receipt.get("policy_state_sha256", ""),
    )
    checks = first_update_checks(first)
    _require(first.get("checks") == checks and all(checks.values()), "first-update checks differ")
    _require(first.get("initial_policy_state_sha256") == initial["policy_state_sha256"], "first update initial state bridge differs")
    _require(first.get("initial_checkpoint_sha256") == initial_receipt["sha256"], "first update initial checkpoint bridge differs")
    _require(first.get("reward_support_rows_sha256") == support["transition_rows_sha256"], "first update support bridge differs")
    _require(first_receipt.get("bytes") == first_checkpoint.stat().st_size and first_receipt.get("sha256") == file_sha256(first_checkpoint), "first checkpoint receipt differs")
    final_receipt = value.get("final_checkpoint", {})
    _require(final_receipt.get("bytes") == final_checkpoint.stat().st_size and final_receipt.get("sha256") == file_sha256(final_checkpoint), "final checkpoint receipt differs")
    validate_runner(first.get("runner_identity"), "first update")
    validate_runner(value.get("runner_identity"), "training")
    validate_runtime(first.get("runtime_identity"), environment, expected_seed=TRAINING_SEED, label="first update")
    validate_runtime(value.get("runtime_identity"), environment, expected_seed=TRAINING_SEED, label="training")
    validate_source_receipts(first.get("source_receipts"), source, "first update")
    validate_source_receipts(value.get("source_receipts"), source, "training")
    validate_std(value.get("initial_action_std"), initial=True, label="training initial")
    validate_std(value.get("final_action_std"), initial=False, label="training final")
    validate_heads(value.get("initial_output_heads"), "training initial")
    validate_heads(value.get("final_output_heads"), "training final")
    validate_environment_summary(value.get("environment"), evaluation=False, label="training", expected_high_steps=TOTAL_ROLLOUT_STEPS)
    _require(value.get("full_distribution_from_iteration_zero") is True and value.get("online_curriculum_registered") is False, "training distribution changed")
    expected_resets = expected_training_initial_resets()
    _require(value.get("training_initial_reset_count") == 128 and value.get("training_initial_resets") == expected_resets, "training initial resets changed")
    normalization = value.get("normalization_lifecycle", {})
    _require(isinstance(normalization, Mapping) and set(normalization) == {
        "installed", "original_method_present", "normalization_enabled", "semantics"
    }, "normalization lifecycle keys changed")
    _require(normalization.get("installed") is True and type(normalization.get("original_method_present")) is bool, "normalization hook changed")
    _require(normalization.get("normalization_enabled") is False and normalization.get("semantics") == "COUNTED_DISABLED_NORMALIZATION_LIFECYCLE", "normalization lifecycle changed")
    _exact_int(value.get("normalization_update_calls"), TOTAL_ROLLOUT_STEPS, "training normalization calls")


def validate_task_snapshot(run: Path, environment: Mapping[str, str]) -> None:
    manifest_path = run / "source/TASK_PACKAGE_RUNTIME.sha256"
    snapshot = run / "source/runtime_project/source/isaac_go2_low_level_rl"
    receipt_path = run / "evidence/task_package_snapshot_receipt.json"
    _require(manifest_path.is_file() and not manifest_path.is_symlink()
             and manifest_path.lstat().st_nlink == 1, "task manifest missing or unsafe")
    _require(snapshot.is_dir() and not snapshot.is_symlink() and snapshot.stat().st_mode & 0o222 == 0, "task snapshot root missing or writable")
    payload = manifest_path.read_bytes()
    _require(payload.endswith(b"\n") and hashlib.sha256(payload).hexdigest() == PARENT_TASK_MANIFEST_SHA256, "task manifest differs from exact V55 OWN path")
    entries: dict[str, str] = {}
    for number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        pieces = line.split("  ", 1)
        _require(len(pieces) == 2, f"task manifest line {number} malformed")
        digest, relative = pieces
        _sha(digest, f"task manifest line {number}")
        parts = Path(relative)
        _require(not parts.is_absolute() and "\\" not in relative and parts.parts and parts.parts[0] == "isaac_go2_low_level_rl" and all(item not in ("", ".", "..") for item in parts.parts), f"task manifest path unsafe: {relative}")
        _require(relative not in entries, f"task manifest duplicate: {relative}")
        entries[relative] = digest
    _require(bool(entries) and list(entries) == sorted(entries), "task manifest empty or unsorted")
    actual: dict[str, str] = {}
    parent = snapshot.parent
    for path in snapshot.rglob("*"):
        info = path.lstat(); relative = path.relative_to(parent).as_posix()
        _require(not stat.S_ISLNK(info.st_mode), f"task snapshot symlink: {relative}")
        _require(path.name not in {"sitecustomize.py", "usercustomize.py"}, f"task startup hook: {relative}")
        if stat.S_ISDIR(info.st_mode):
            _require(path.name not in {"__pycache__", ".pytest_cache", ".git"} and info.st_mode & 0o222 == 0, f"task directory unsafe: {relative}")
            continue
        _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                 and path.suffix not in {".pyc", ".pyo"} and info.st_mode & 0o222 == 0,
                 f"task file unsafe: {relative}")
        actual[relative] = file_sha256(path)
    _require(actual == entries, "task snapshot manifest/file set differs")
    receipt = strict_json(receipt_path, "task snapshot receipt")
    _require(receipt.get("schema") == "sparse_high_task_progress_canary_v1/task_snapshot/v1" and receipt.get("status") == "PASS" and receipt.get("failures") == [], "task snapshot receipt changed")
    _require(receipt.get("manifest_sha256") == PARENT_TASK_MANIFEST_SHA256 and receipt.get("regular_file_count") == len(actual), "task snapshot receipt identity differs")
    _require(receipt.get("symlink_count") == 0 and receipt.get("cache_or_bytecode_count") == 0, "task snapshot receipt reports unsafe members")
    _require(resolved_absolute(environment.get("TASK_PACKAGE_SNAPSHOT"), "task snapshot env") == snapshot.resolve(strict=True), "task snapshot environment differs")
    _require(environment.get("TASK_PACKAGE_MANIFEST_SHA256") == PARENT_TASK_MANIFEST_SHA256, "task manifest environment differs")


def guard(path: Path, label: str) -> None:
    _require(path.is_file() and not path.is_symlink() and path.lstat().st_nlink == 1
             and path.read_bytes() == b"PASS\n", f"{label} guard not PASS")


def child_rc(path: Path, label: str) -> None:
    _require(path.is_file() and not path.is_symlink() and path.lstat().st_nlink == 1
             and path.read_bytes() == b"0\n", f"{label} child RC not zero")


CONTACT_RESET_STATE_KEYS = {
    "selected_ball_cache_zero_after", "selected_base_cache_zero_after",
    "selected_ball_cache_max_abs_after", "selected_base_cache_max_abs_after",
    "selected_ball_metadata_reset", "selected_base_metadata_reset",
    "selected_ball_root_state_shape", "expected_ball_root_state_13d",
    "actual_ball_root_state_13d", "selected_ball_root_state_13d_max_error",
    "selected_ball_root_state_13d_canonical",
}


def _exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping) and set(value) == expected, f"{label} key set changed")
    return value


def _validate_contact_reset_state(
    value: Any,
    *,
    selected_ids: Sequence[int],
    label: str,
    include_pass: bool = True,
) -> None:
    expected_keys = CONTACT_RESET_STATE_KEYS | ({"pass"} if include_pass else set())
    receipt = _exact_keys(value, expected_keys, label)
    _require(
        receipt.get("selected_ball_cache_zero_after") is True
        and receipt.get("selected_base_cache_zero_after") is True
        and receipt.get("selected_ball_metadata_reset") is True
        and receipt.get("selected_base_metadata_reset") is True,
        f"{label} did not clear selected private sensor state",
    )
    _require(
        close_scalar(receipt.get("selected_ball_cache_max_abs_after"), 0.0, 0.0)
        and close_scalar(receipt.get("selected_base_cache_max_abs_after"), 0.0, 0.0),
        f"{label} selected cache is not exactly zero",
    )
    row_count = len(selected_ids)
    _require(
        receipt.get("selected_ball_root_state_shape") == [row_count, 13],
        f"{label} canonical ball state shape changed",
    )
    expected_rows = receipt.get("expected_ball_root_state_13d")
    actual_rows = receipt.get("actual_ball_root_state_13d")
    _require(
        isinstance(expected_rows, list) and isinstance(actual_rows, list)
        and len(expected_rows) == row_count and len(actual_rows) == row_count,
        f"{label} canonical ball state rows changed",
    )
    maximum_error = 0.0
    for row_index, (expected_row, actual_row) in enumerate(
        zip(expected_rows, actual_rows, strict=True)
    ):
        expected_vector = _vector(expected_row, 13, f"{label}.expected[{row_index}]")
        actual_vector = _vector(actual_row, 13, f"{label}.actual[{row_index}]")
        maximum_error = max(
            maximum_error,
            max(abs(left - right) for left, right in zip(expected_vector, actual_vector, strict=True)),
        )
    serialized_error = _finite(
        receipt.get("selected_ball_root_state_13d_max_error"),
        f"{label}.selected_ball_root_state_13d_max_error",
    )
    _require(
        serialized_error >= 0.0
        and close_scalar(serialized_error, maximum_error, 1.0e-12)
        and serialized_error <= 1.0e-6
        and receipt.get("selected_ball_root_state_13d_canonical") is True,
        f"{label} complete 13D canonical ball state did not reconstruct",
    )
    if include_pass:
        _require(receipt.get("pass") is True, f"{label} did not pass")


def _validate_contact_qualification(value: Any) -> tuple[str, int, int, int]:
    keys = {
        "qualification_job", "contact_body_contract_sha256", "accepted_body_names",
        "live_body_names", "base_contact_body_names", "base_body_index_by_accepted_body",
        "body_count",
        "ball_sensor_body_count", "base_sensor_body_count", "filter_count",
        "ball_force_matrix_shape", "base_net_force_shape", "candidate_order",
        "max_raw_physics_ticks_per_candidate", "contact_threshold_n",
        "current_same_tick_same_body_required", "history_used_as_contact_truth",
        "ball_displacement_used_as_contact_truth", "low_actor_used",
        "low_env_step_used", "action_manager_processing_used", "trials",
        "winning_body_name", "winning_body_index", "winning_filter_index",
        "winning_base_body_index", "winning_tick", "pass",
    }
    receipt = _exact_keys(value, keys, "contact qualification")
    _require(
        receipt.get("qualification_job") == CONTACT_RESET_QUALIFICATION_JOB
        and receipt.get("contact_body_contract_sha256") == CONTACT_RESET_BODY_CONTRACT_SHA256,
        "contact qualification provenance changed",
    )
    expected_names = list(CONTACT_RESET_BODY_NAMES)
    _require(
        receipt.get("accepted_body_names") == expected_names
        and receipt.get("live_body_names") == expected_names,
        "accepted/live articulation body order changed",
    )
    base_names = receipt.get("base_contact_body_names")
    _require(
        isinstance(base_names, list)
        and len(base_names) == len(expected_names)
        and all(isinstance(name, str) for name in base_names)
        and len(set(base_names)) == len(base_names)
        and set(base_names) == set(expected_names),
        "base ContactSensor body names are not a unique exact accepted set",
    )
    base_index_by_accepted_body = [base_names.index(name) for name in expected_names]
    _exact_id_list(
        receipt.get("base_body_index_by_accepted_body"),
        base_index_by_accepted_body,
        "accepted-to-base ContactSensor body mapping",
    )
    _exact_int(receipt.get("body_count"), 19, "contact body count")
    _exact_int(receipt.get("ball_sensor_body_count"), 1, "ball sensor body count")
    _exact_int(receipt.get("base_sensor_body_count"), 19, "base sensor body count")
    _exact_int(receipt.get("filter_count"), 19, "ball contact filter count")
    _require(
        receipt.get("ball_force_matrix_shape") == [8, 1, 19, 3]
        and receipt.get("base_net_force_shape") == [8, 19, 3],
        "contact qualification current-force shape changed",
    )
    _require(
        receipt.get("candidate_order") == list(CONTACT_RESET_CALF_NAMES),
        "qualified calf order changed",
    )
    _exact_int(
        receipt.get("max_raw_physics_ticks_per_candidate"), 4,
        "maximum raw ticks per calf",
    )
    threshold = _finite(receipt.get("contact_threshold_n"), "contact threshold")
    _require(close_scalar(threshold, CONTACT_RESET_FORCE_THRESHOLD_N, 0.0), "contact threshold changed")
    _require(
        receipt.get("current_same_tick_same_body_required") is True
        and receipt.get("history_used_as_contact_truth") is False
        and receipt.get("ball_displacement_used_as_contact_truth") is False
        and receipt.get("low_actor_used") is False
        and receipt.get("low_env_step_used") is False
        and receipt.get("action_manager_processing_used") is False,
        "contact qualification evidence boundary changed",
    )

    trials = receipt.get("trials")
    _require(isinstance(trials, list) and 1 <= len(trials) <= 4, "contact trials changed")
    _require(
        [trial.get("body_name") if isinstance(trial, Mapping) else None for trial in trials]
        == list(CONTACT_RESET_CALF_NAMES[: len(trials)]),
        "contact trials are not the ordered calf prefix",
    )
    winning_trials = 0
    last_trial_name = ""
    last_trial_index = -1
    last_trial_filter_index = -1
    last_trial_base_body_index = -1
    for trial_index, trial_value in enumerate(trials):
        label = f"contact trial[{trial_index}]"
        trial = _exact_keys(
            trial_value,
            {
                "body_name", "body_index", "filter_index", "base_body_index",
                "ticks_run", "ticks",
                "per_env", "all_selected_same_tick_hit", "witness_tick",
            },
            label,
        )
        body_name = CONTACT_RESET_CALF_NAMES[trial_index]
        body_index = receipt["live_body_names"].index(body_name)
        filter_index = receipt["accepted_body_names"].index(body_name)
        base_body_index = base_names.index(body_name)
        _exact_int(
            body_index, CONTACT_RESET_CALF_INDICES[trial_index],
            f"{label} reconstructed articulation body index",
        )
        _exact_int(
            filter_index, CONTACT_RESET_CALF_INDICES[trial_index],
            f"{label} reconstructed ball-filter index",
        )
        _exact_int(
            base_index_by_accepted_body[filter_index], base_body_index,
            f"{label} reconstructed base ContactSensor index",
        )
        _require(
            trial.get("body_name") == body_name
            and trial.get("body_index") == body_index
            and trial.get("filter_index") == filter_index
            and trial.get("base_body_index") == base_body_index,
            f"{label} articulation/filter/base body identity changed",
        )
        ticks_run = _bounded_int(trial.get("ticks_run"), 1, 4, f"{label}.ticks_run")
        ticks = trial.get("ticks")
        _require(isinstance(ticks, list) and len(ticks) == ticks_run, f"{label} tick trace changed")
        ball_peaks = {env_id: 0.0 for env_id in CONTACT_RESET_PARTIAL_IDS}
        robot_peaks = {env_id: 0.0 for env_id in CONTACT_RESET_PARTIAL_IDS}
        first_hits = {env_id: -1 for env_id in CONTACT_RESET_PARTIAL_IDS}
        all_selected_witness_ticks: list[int] = []
        for tick_index, tick_value in enumerate(ticks, 1):
            tick = _exact_keys(tick_value, {"tick", "per_env"}, f"{label}.tick[{tick_index}]")
            _exact_int(tick.get("tick"), tick_index, f"{label}.tick index")
            rows = tick.get("per_env")
            _require(isinstance(rows, list) and len(rows) == 3, f"{label} per-tick env rows changed")
            _exact_id_list(
                [row.get("env_id") if isinstance(row, Mapping) else None for row in rows],
                CONTACT_RESET_PARTIAL_IDS,
                f"{label} per-tick selected env order",
            )
            for row in rows:
                row = _exact_keys(
                    row,
                    {"env_id", "ball_current_n", "robot_current_n", "dual_threshold_hit"},
                    f"{label}.tick[{tick_index}].env",
                )
                env_id = int(row["env_id"])
                ball_force = _finite(row.get("ball_current_n"), f"{label}.ball_current_n")
                robot_force = _finite(row.get("robot_current_n"), f"{label}.robot_current_n")
                _require(ball_force >= 0.0 and robot_force >= 0.0, f"{label} force is negative")
                dual_hit = ball_force >= threshold and robot_force >= threshold
                _require(
                    row.get("dual_threshold_hit") is dual_hit,
                    f"{label} same-tick dual-contact Boolean did not reconstruct",
                )
                ball_peaks[env_id] = max(ball_peaks[env_id], ball_force)
                robot_peaks[env_id] = max(robot_peaks[env_id], robot_force)
                if dual_hit and first_hits[env_id] < 0:
                    first_hits[env_id] = tick_index
            if all(row["dual_threshold_hit"] is True for row in rows):
                all_selected_witness_ticks.append(tick_index)
        summaries = trial.get("per_env")
        _require(isinstance(summaries, list) and len(summaries) == 3, f"{label} env summary changed")
        _exact_id_list(
            [row.get("env_id") if isinstance(row, Mapping) else None for row in summaries],
            CONTACT_RESET_PARTIAL_IDS,
            f"{label} summary selected env order",
        )
        for summary in summaries:
            summary = _exact_keys(
                summary,
                {
                    "env_id", "ball_current_peak_n", "robot_current_peak_n",
                    "first_dual_hit_tick", "ever_same_tick_dual_hit",
                },
                f"{label}.summary",
            )
            env_id = int(summary["env_id"])
            _require(
                close_scalar(summary.get("ball_current_peak_n"), ball_peaks[env_id], 1.0e-12)
                and close_scalar(summary.get("robot_current_peak_n"), robot_peaks[env_id], 1.0e-12),
                f"{label} current-force peaks did not reconstruct",
            )
            _exact_int(
                summary.get("first_dual_hit_tick"), first_hits[env_id],
                f"{label}.first_dual_hit_tick",
            )
            _require(
                summary.get("ever_same_tick_dual_hit") is (first_hits[env_id] > 0),
                f"{label} dual-contact summary did not reconstruct",
            )
        all_selected_hit = bool(all_selected_witness_ticks)
        witness_tick = all_selected_witness_ticks[0] if all_selected_witness_ticks else None
        _require(
            not all_selected_hit
            or (len(all_selected_witness_ticks) == 1 and witness_tick == ticks_run),
            f"{label} continued after its same-global-tick witness",
        )
        _require(
            trial.get("all_selected_same_tick_hit") is all_selected_hit,
            f"{label} same-global-tick selected-slot contact did not reconstruct",
        )
        if all_selected_hit:
            _exact_int(trial.get("witness_tick"), int(witness_tick), f"{label}.witness_tick")
            winning_trials += 1
            _require(trial_index == len(trials) - 1, "probe continued after a qualifying calf")
        else:
            _require(trial.get("witness_tick") is None, f"{label} fabricated a witness tick")
        last_trial_name = body_name
        last_trial_index = body_index
        last_trial_filter_index = filter_index
        last_trial_base_body_index = base_body_index
    _require(winning_trials == 1, "exactly one terminal qualifying calf trial is required")
    _require(
        receipt.get("winning_body_name") == last_trial_name
        and receipt.get("winning_body_index") == last_trial_index
        and receipt.get("winning_filter_index") == last_trial_filter_index
        and receipt.get("winning_base_body_index") == last_trial_base_body_index
        and receipt.get("pass") is True,
        "winning calf/articulation/filter/base receipt changed",
    )
    _exact_int(
        receipt.get("winning_tick"), int(trials[-1]["witness_tick"]),
        "contact qualification winning_tick",
    )
    return (
        last_trial_name,
        last_trial_index,
        last_trial_filter_index,
        last_trial_base_body_index,
    )


def _validate_post_reset_raw_query(
    value: Any, *, body_name: str, base_body_index: int
) -> None:
    keys = {
        "boundary", "physics_dt_s", "ball_raw_native_shape", "ball_raw_reshaped_shape",
        "base_raw_native_shape", "base_raw_reshaped_shape", "ball_sensor_body_count",
        "ball_filter_count", "base_sensor_body_count", "base_body_name",
        "base_body_index", "ball_raw_selected_max_n",
        "base_raw_selected_body_max_n", "raw_values_finite",
        "selected_private_digest_before_sha256", "selected_private_digest_after_sha256",
        "unselected_private_digest_before_sha256", "unselected_private_digest_after_sha256",
        "selected_private_max_delta", "unselected_private_max_delta",
        "private_cache_unchanged_selected", "private_cache_unchanged_unselected",
        "raw_force_magnitude_used_as_pass_criterion", "public_contact_sensor_data_reads", "pass",
    }
    receipt = _exact_keys(value, keys, "post-reset raw query")
    _require(
        receipt.get("boundary") == "AFTER_RESET_BEFORE_PUBLIC_SENSOR_DATA_OR_PHYSICS"
        and close_scalar(receipt.get("physics_dt_s"), CONTACT_RESET_PHYSICS_DT_S, 1.0e-12),
        "post-reset raw query boundary changed",
    )
    for key in ("ball_raw_native_shape", "base_raw_native_shape"):
        shape = receipt.get(key)
        _require(
            isinstance(shape, list) and bool(shape)
            and all(type(item) is int and item > 0 for item in shape),
            f"post-reset {key} is invalid",
        )
    _require(
        math.prod(receipt["ball_raw_native_shape"]) == 8 * 1 * 19 * 3
        and math.prod(receipt["base_raw_native_shape"]) == 8 * 19 * 3
        and receipt.get("ball_raw_reshaped_shape") == [8, 1, 19, 3]
        and receipt.get("base_raw_reshaped_shape") == [8, 19, 3],
        "post-reset raw PhysX shape changed",
    )
    _exact_int(receipt.get("ball_sensor_body_count"), 1, "raw ball body count")
    _exact_int(receipt.get("ball_filter_count"), 19, "raw ball filter count")
    _exact_int(receipt.get("base_sensor_body_count"), 19, "raw base body count")
    _require(receipt.get("base_body_name") == body_name, "raw base body name changed")
    _exact_int(receipt.get("base_body_index"), base_body_index, "raw base body index")
    _require(
        receipt.get("raw_values_finite") is True
        and _finite(receipt.get("ball_raw_selected_max_n"), "raw ball maximum") >= 0.0
        and _finite(receipt.get("base_raw_selected_body_max_n"), "raw base maximum") >= 0.0,
        "post-reset raw values are invalid",
    )
    selected_before = _sha(
        receipt.get("selected_private_digest_before_sha256"), "selected private digest before"
    )
    selected_after = _sha(
        receipt.get("selected_private_digest_after_sha256"), "selected private digest after"
    )
    unselected_before = _sha(
        receipt.get("unselected_private_digest_before_sha256"), "unselected private digest before"
    )
    unselected_after = _sha(
        receipt.get("unselected_private_digest_after_sha256"), "unselected private digest after"
    )
    _require(
        selected_before == selected_after and unselected_before == unselected_after
        and close_scalar(receipt.get("selected_private_max_delta"), 0.0, 0.0)
        and close_scalar(receipt.get("unselected_private_max_delta"), 0.0, 0.0)
        and receipt.get("private_cache_unchanged_selected") is True
        and receipt.get("private_cache_unchanged_unselected") is True,
        "raw PhysX query mutated the ContactSensor private cache",
    )
    _exact_int(receipt.get("public_contact_sensor_data_reads"), 0, "pre-physics public sensor reads")
    _require(
        receipt.get("raw_force_magnitude_used_as_pass_criterion") is False
        and receipt.get("pass") is True,
        "raw PhysX diagnostic was promoted into a scientific gate",
    )


def _timestamp_vector(value: Any, expected: float, label: str) -> None:
    timestamps = _vector(value, len(CONTACT_RESET_PARTIAL_IDS), label)
    _require(
        all(math.isclose(item, expected, rel_tol=0.0, abs_tol=1.0e-8) for item in timestamps),
        f"{label} does not equal the exact physics-tick boundary",
    )


def _validate_first_post_reset_tick(
    value: Any,
    *,
    body_name: str,
    body_index: int,
    filter_index: int,
    base_body_index: int,
) -> None:
    keys = {
        "boundary", "raw_physics_ticks", "physics_dt_s",
        "public_contact_sensor_data_reads_before_tick",
        "public_contact_sensor_data_reads_after_tick", "contact_body_name",
        "contact_body_index", "base_body_name", "base_body_index",
        "ball_force_matrix_shape", "base_net_force_shape",
        "ball_history_shape", "base_history_shape", "ball_history_tail_max_abs",
        "base_history_tail_max_abs", "ball_index_zero_selected_max_n",
        "base_index_zero_selected_body_max_n", "index_zero_allowed_any_contact",
        "ball_timestamp_s", "ball_timestamp_last_update_s", "base_timestamp_s",
        "base_timestamp_last_update_s", "ball_outdated_after_tick",
        "base_outdated_after_tick", "finite", "pass",
    }
    receipt = _exact_keys(value, keys, "first post-reset physics tick")
    _require(
        receipt.get("boundary") == "EXACTLY_ONE_RAW_PHYSICS_TICK_AFTER_PHYSICAL_PARTIAL_RESET"
        and receipt.get("contact_body_name") == body_name
        and receipt.get("base_body_name") == body_name,
        "first post-reset physics boundary/body name changed",
    )
    _exact_int(body_index, filter_index, "winner articulation/filter index agreement")
    _exact_int(receipt.get("contact_body_index"), filter_index, "first refresh filter body index")
    _exact_int(receipt.get("base_body_index"), base_body_index, "first refresh base body index")
    _exact_int(receipt.get("raw_physics_ticks"), 1, "first refresh physics ticks")
    _require(close_scalar(receipt.get("physics_dt_s"), 0.005, 1.0e-12), "first refresh dt changed")
    _exact_int(receipt.get("public_contact_sensor_data_reads_before_tick"), 0, "pre-tick public reads")
    _exact_int(receipt.get("public_contact_sensor_data_reads_after_tick"), 2, "post-tick public reads")
    _require(
        receipt.get("ball_force_matrix_shape") == [8, 1, 19, 3]
        and receipt.get("base_net_force_shape") == [8, 19, 3]
        and receipt.get("ball_history_shape") == [8, 4, 1, 19, 3]
        and receipt.get("base_history_shape") == [8, 3, 19, 3],
        "first refresh ContactSensor shape changed",
    )
    _require(
        close_scalar(receipt.get("ball_history_tail_max_abs"), 0.0, 0.0)
        and close_scalar(receipt.get("base_history_tail_max_abs"), 0.0, 0.0),
        "pre-reset contact history survived the first physics tick",
    )
    _require(
        _finite(receipt.get("ball_index_zero_selected_max_n"), "first-tick ball force") >= 0.0
        and _finite(receipt.get("base_index_zero_selected_body_max_n"), "first-tick base force") >= 0.0
        and receipt.get("index_zero_allowed_any_contact") is True,
        "first-tick index-zero contact policy changed",
    )
    for key in (
        "ball_timestamp_s", "ball_timestamp_last_update_s",
        "base_timestamp_s", "base_timestamp_last_update_s",
    ):
        _timestamp_vector(receipt.get(key), 0.005, f"first tick {key}")
    _require(
        receipt.get("ball_outdated_after_tick") is False
        and receipt.get("base_outdated_after_tick") is False
        and receipt.get("finite") is True
        and receipt.get("pass") is True,
        "first post-reset physics tick did not pass",
    )


def _validate_production_low_step(value: Any) -> None:
    keys = {
        "boundary", "fresh_reset", "configured_decimation", "normal_low_steps",
        "raw_physics_ticks", "physics_dt_s", "low_action_max_abs", "low_action_min",
        "low_action_max", "ball_history_shape", "base_history_shape", "ball_timestamp_s",
        "ball_timestamp_last_update_s", "base_timestamp_s", "base_timestamp_last_update_s",
        "ball_outdated_after_step", "base_outdated_after_step", "finite", "pass",
    }
    receipt = _exact_keys(value, keys, "production low step")
    _require(
        receipt.get("boundary") == "INDEPENDENT_FRESH_RESET_THEN_ONE_NORMAL_LOW_ENV_STEP",
        "production low-step boundary changed",
    )
    _validate_contact_reset_state(
        receipt.get("fresh_reset"),
        selected_ids=CONTACT_RESET_PARTIAL_IDS,
        label="production low-step fresh reset",
    )
    _exact_int(receipt.get("configured_decimation"), 4, "production decimation")
    _exact_int(receipt.get("normal_low_steps"), 1, "production low steps")
    _exact_int(receipt.get("raw_physics_ticks"), 4, "production raw physics ticks")
    _require(close_scalar(receipt.get("physics_dt_s"), 0.005, 1.0e-12), "production dt changed")
    action_abs = _finite(receipt.get("low_action_max_abs"), "production low action max abs")
    action_min = _finite(receipt.get("low_action_min"), "production low action minimum")
    action_max = _finite(receipt.get("low_action_max"), "production low action maximum")
    _require(
        action_min <= action_max
        and action_abs >= 0.0
        and close_scalar(action_abs, max(abs(action_min), abs(action_max)), 1.0e-6),
        "production low-action range is inconsistent",
    )
    _require(
        receipt.get("ball_history_shape") == [8, 4, 1, 19, 3]
        and receipt.get("base_history_shape") == [8, 3, 19, 3],
        "production low-step sensor shape changed",
    )
    for key in (
        "ball_timestamp_s", "ball_timestamp_last_update_s",
        "base_timestamp_s", "base_timestamp_last_update_s",
    ):
        _timestamp_vector(receipt.get(key), 0.02, f"production low step {key}")
    _require(
        receipt.get("ball_outdated_after_step") is False
        and receipt.get("base_outdated_after_step") is False
        and receipt.get("finite") is True
        and receipt.get("pass") is True,
        "independent four-tick production low step failed",
    )


def validate_contact_reset_probe(
    value: Any,
    *,
    environment: Mapping[str, str],
    source: Path,
) -> None:
    top_keys = {
        "schema", "status", "num_envs", "partial_reset_ids", "unselected_ids",
        "zero_tolerance", "state_tolerance", "contact_threshold_n",
        "contact_qualification", "physical_partial_reset", "post_reset_raw_query",
        "first_post_reset_physics_tick", "production_low_step", "sentinel_partial",
        "sentinel_full", "checks", "ppo_constructed", "optimizer_constructed",
        "runtime_identity", "source_receipts", "failures", "long_training_authorized",
    }
    receipt = _exact_keys(value, top_keys, "contact reset probe")
    _require(
        receipt.get("schema") == CONTACT_RESET_PROBE_SCHEMA and receipt.get("status") == "PASS",
        "contact reset probe did not pass",
    )
    _exact_int(receipt.get("num_envs"), CONTACT_RESET_NUM_ENVS, "contact reset probe num_envs")
    _exact_id_list(
        receipt.get("partial_reset_ids"), CONTACT_RESET_PARTIAL_IDS,
        "contact reset probe partial_reset_ids",
    )
    _exact_id_list(
        receipt.get("unselected_ids"), CONTACT_RESET_UNSELECTED_IDS,
        "contact reset probe unselected_ids",
    )
    _require(
        close_scalar(receipt.get("zero_tolerance"), 0.0, 0.0)
        and close_scalar(receipt.get("state_tolerance"), 1.0e-6, 0.0)
        and close_scalar(receipt.get("contact_threshold_n"), 1.0, 0.0),
        "contact reset probe tolerances changed",
    )
    _require(
        receipt.get("ppo_constructed") is False
        and receipt.get("optimizer_constructed") is False
        and receipt.get("failures") == []
        and receipt.get("long_training_authorized") is False,
        "contact reset probe execution scope changed",
    )

    winner_name, winner_index, winner_filter_index, winner_base_index = _validate_contact_qualification(
        receipt.get("contact_qualification")
    )
    physical_keys = CONTACT_RESET_STATE_KEYS | {
        "selected_ids", "reset_called_immediately_after_witness",
        "selected_physical_caches_overwritten_before_reset",
        "unselected_ball_sensor_unchanged", "unselected_base_sensor_unchanged",
        "unselected_ball_state_unchanged", "pass",
    }
    physical = _exact_keys(receipt.get("physical_partial_reset"), physical_keys, "physical partial reset")
    _exact_id_list(
        physical.get("selected_ids"), CONTACT_RESET_PARTIAL_IDS,
        "physical partial reset selected_ids",
    )
    _require(
        physical.get("reset_called_immediately_after_witness") is True
        and physical.get("selected_physical_caches_overwritten_before_reset") is False,
        "physical reset was not the immediate mutation after the contact witness",
    )
    _validate_contact_reset_state(
        {key: physical[key] for key in CONTACT_RESET_STATE_KEYS | {"pass"}},
        selected_ids=CONTACT_RESET_PARTIAL_IDS,
        label="physical partial reset",
    )
    _require(
        physical.get("unselected_ball_sensor_unchanged") is True
        and physical.get("unselected_base_sensor_unchanged") is True
        and physical.get("unselected_ball_state_unchanged") is True,
        "physical partial reset changed an unselected slot",
    )

    _validate_post_reset_raw_query(
        receipt.get("post_reset_raw_query"),
        body_name=winner_name,
        base_body_index=winner_base_index,
    )
    _validate_first_post_reset_tick(
        receipt.get("first_post_reset_physics_tick"),
        body_name=winner_name,
        body_index=winner_index,
        filter_index=winner_filter_index,
        base_body_index=winner_base_index,
    )
    _validate_production_low_step(receipt.get("production_low_step"))

    sentinel_partial_keys = CONTACT_RESET_STATE_KEYS | {
        "selected_ids", "sentinel_scheme", "selected_sensor_cache_dirty_before",
        "unselected_ball_sensor_unchanged", "unselected_base_sensor_unchanged",
        "unselected_ball_state_unchanged", "pass",
    }
    sentinel_partial = _exact_keys(
        receipt.get("sentinel_partial"), sentinel_partial_keys, "partial sentinel lane"
    )
    _exact_id_list(
        sentinel_partial.get("selected_ids"), CONTACT_RESET_PARTIAL_IDS,
        "partial sentinel selected_ids",
    )
    _require(
        sentinel_partial.get("sentinel_scheme") == "DISTINCT_BASE_PLUS_FIELD_PLUS_ENV_SLOT"
        and sentinel_partial.get("selected_sensor_cache_dirty_before") is True,
        "partial sentinel precondition changed",
    )
    _validate_contact_reset_state(
        {key: sentinel_partial[key] for key in CONTACT_RESET_STATE_KEYS | {"pass"}},
        selected_ids=CONTACT_RESET_PARTIAL_IDS,
        label="partial sentinel lane",
    )
    _require(
        sentinel_partial.get("unselected_ball_sensor_unchanged") is True
        and sentinel_partial.get("unselected_base_sensor_unchanged") is True
        and sentinel_partial.get("unselected_ball_state_unchanged") is True,
        "partial sentinel reset changed an unselected slot",
    )

    sentinel_full_keys = CONTACT_RESET_STATE_KEYS | {
        "selected_ids", "sentinel_scheme", "selected_sensor_cache_dirty_before", "pass",
    }
    sentinel_full = _exact_keys(
        receipt.get("sentinel_full"), sentinel_full_keys, "full sentinel lane"
    )
    _exact_id_list(
        sentinel_full.get("selected_ids"), range(CONTACT_RESET_NUM_ENVS),
        "full sentinel selected_ids",
    )
    _require(
        sentinel_full.get("sentinel_scheme") == "DISTINCT_BASE_PLUS_FIELD_PLUS_ENV_SLOT"
        and sentinel_full.get("selected_sensor_cache_dirty_before") is True,
        "full sentinel precondition changed",
    )
    _validate_contact_reset_state(
        {key: sentinel_full[key] for key in CONTACT_RESET_STATE_KEYS | {"pass"}},
        selected_ids=tuple(range(CONTACT_RESET_NUM_ENVS)),
        label="full sentinel lane",
    )

    expected_checks = {
        "contact_qualification", "physical_partial_reset", "post_reset_raw_query",
        "first_post_reset_physics_tick", "production_low_step", "sentinel_partial",
        "sentinel_full",
    }
    checks = _exact_keys(receipt.get("checks"), expected_checks, "contact reset checks")
    _require(
        all(checks.get(key) is True for key in expected_checks)
        and checks["contact_qualification"] is receipt["contact_qualification"]["pass"]
        and checks["physical_partial_reset"] is receipt["physical_partial_reset"]["pass"]
        and checks["post_reset_raw_query"] is receipt["post_reset_raw_query"]["pass"]
        and checks["first_post_reset_physics_tick"] is receipt["first_post_reset_physics_tick"]["pass"]
        and checks["production_low_step"] is receipt["production_low_step"]["pass"]
        and checks["sentinel_partial"] is receipt["sentinel_partial"]["pass"]
        and checks["sentinel_full"] is receipt["sentinel_full"]["pass"],
        "contact reset check summary changed or failed",
    )
    validate_runtime(
        receipt.get("runtime_identity"), environment,
        expected_seed=TRAINING_SEED, expected_num_envs=CONTACT_RESET_NUM_ENVS,
        label="contact reset probe",
    )
    validate_source_receipts(receipt.get("source_receipts"), source, "contact reset probe")


def require_absent(paths: Sequence[Path], label: str) -> None:
    for path in paths:
        _require(not path.exists() and not path.is_symlink(), f"{label} contains forbidden artifact: {path.name}")


def exact_directory(path: Path, expected: set[str], label: str) -> None:
    directory = real_directory(path, label)
    actual = {item.name for item in directory.iterdir()}
    _require(actual == expected, f"{label} members differ: actual={sorted(actual)} expected={sorted(expected)}")
    for item in directory.iterdir():
        _require(item.is_file() and not item.is_symlink() and stat.S_ISREG(item.lstat().st_mode)
                 and item.lstat().st_nlink == 1, f"{label} member unsafe: {item.name}")


def validate_phase_events(path: Path, *, support_pass: bool, in_job: bool) -> None:
    _require(path.is_file() and not path.is_symlink() and path.lstat().st_nlink == 1,
             "phase event log missing or unsafe")
    lines = path.read_text(encoding="utf-8").splitlines()
    _require(lines and lines[0] == "timestamp\tphase\tstate\trc", "phase header changed")
    actual: list[tuple[str, str, str]] = []
    for number, line in enumerate(lines[1:], 2):
        fields = line.split("\t")
        _require(len(fields) == 4 and fields[0], f"phase line {number} malformed")
        actual.append((fields[1], fields[2], fields[3]))
    expected = [
        ("wrapper_static_preflight", "START", ""),
        ("wrapper_static_preflight", "COMPLETE", "0"),
        ("task_package_snapshot", "START", ""),
        ("task_package_snapshot", "COMPLETE", "0"),
        ("contact_reset_lifecycle_probe", "START", ""),
        ("contact_reset_lifecycle_probe", "COMPLETE", "0"),
        ("initial_fixed_evaluation", "START", ""),
        ("initial_fixed_evaluation", "COMPLETE", "0"),
        ("bounded_task_progress_ppo", "START", ""),
        ("bounded_task_progress_ppo", "COMPLETE" if support_pass else "REJECT", "0"),
    ]
    if support_pass:
        expected += [
            ("final_fixed_evaluation", "START", ""),
            ("final_fixed_evaluation", "COMPLETE", "0"),
        ]
    expected += [
        ("terminal_aggregate", "START", ""),
        ("terminal_aggregate", "COMPLETE", "0"),
        ("independent_result_check", "START", ""),
    ]
    if not in_job:
        expected.append(("independent_result_check", "COMPLETE", "0"))
    _require(actual == expected, "phase event sequence differs from legal branch")


def validate_transaction(run: Path, result: Mapping[str, Any], *, support_pass: bool) -> None:
    evidence = run / "evidence"
    process_rc = evidence / "process_rc.txt"
    _require(process_rc.is_file() and not process_rc.is_symlink() and process_rc.lstat().st_nlink == 1
             and process_rc.read_bytes() == b"0\n", "process RC is missing, unsafe, or not zero")
    guard(evidence / "SPARSE_HIGH_TASK_PROGRESS_CANARY_TRANSACTION_FINAL_GUARD", "outer transaction")
    transaction = strict_json(evidence / "transaction_status.json", "transaction status")
    expected = {
        "schema": "sparse_high_task_progress_canary_v1/transaction_status/v1",
        "status": "COMPLETE", "process_rc": 0, "failed_phase": None, "failure": None,
        "parent_v55_run_label": EXPECTED_PARENT_V55["run_label"],
        "decision": result["decision"],
        "initial_evaluation_executed": True,
        "reward_support_executed": True,
        "reward_support_gate_pass": support_pass,
        "first_update_executed": support_pass,
        "training_executed": support_pass,
        "final_evaluation_executed": support_pass,
        "ppo_optimizer_steps": TOTAL_OPTIMIZER_STEPS if support_pass else 0,
        "long_training_authorized": False,
    }
    _require(transaction == expected, "transaction status differs from evidence")


def validate_source_binding(source: Path, environment: Mapping[str, str]) -> str:
    manifest = source / "SPARSE_HIGH_TASK_PROGRESS_CANARY_V1_SOURCE.sha256"
    _require(manifest.is_file() and not manifest.is_symlink() and manifest.lstat().st_nlink == 1,
             "source manifest missing or unsafe")
    digest = file_sha256(manifest)
    receipt = source.parent / f"{source.name}.source-manifest.sha256"
    _require(receipt.is_file() and not receipt.is_symlink() and receipt.lstat().st_nlink == 1,
             "external source-manifest receipt missing or unsafe")
    _require(receipt.read_text(encoding="ascii") == f"{digest}  {manifest.name}\n", "external source-manifest receipt differs")
    _require(environment.get("BOUND_SOURCE_MANIFEST_SHA256") == digest, "runtime source manifest binding differs")
    _require(resolved_absolute(environment.get("SOURCE"), "SOURCE") == source, "runtime SOURCE differs from checker package")
    return digest


def validate_result_exact(
    result: Mapping[str, Any], contract: Mapping[str, Any],
    initial: Mapping[str, Any], support: Mapping[str, Any],
    first: Mapping[str, Any] | None, training: Mapping[str, Any] | None,
    final: Mapping[str, Any] | None, hashes: Mapping[str, str],
) -> None:
    expected = build_result(contract, ResultInputs(initial, support, first, training, final))
    expected["input_summary_sha256"] = {
        key: hashes[key]
        for key, value in (
            ("initial_evaluation", initial), ("reward_support", support),
            ("first_update", first), ("training", training), ("final_evaluation", final),
        )
        if value is not None
    }
    _require(result == expected, "terminal result differs from independent reconstruction")


def validate_component_update(value: Any, label: str) -> None:
    _require(isinstance(value, Mapping) and type(value.get("changed")) is bool, f"{label} malformed")
    names = value.get("parameter_names")
    _require(isinstance(names, list) and names and all(isinstance(item, str) and item for item in names), f"{label} parameter list invalid")
    _bounded_int(value.get("parameter_count"), 1, 10**9, f"{label} parameter count")
    maximum = _finite(value.get("maximum_absolute_delta"), f"{label} max delta")
    l2 = _finite(value.get("l2_delta"), f"{label} l2 delta")
    _require(maximum >= 0.0 and l2 >= 0.0 and value["changed"] is (maximum > 0.0 or l2 > 0.0), f"{label} delta receipt inconsistent")


def diagnose_failed_transaction(run: Path, evidence: Path) -> int:
    transaction = strict_json(evidence / "transaction_status.json", "failed transaction status")
    _require(transaction.get("schema") == "sparse_high_task_progress_canary_v1/transaction_status/v1",
             "failed transaction schema changed")
    _require(transaction.get("status") == "ERROR", "result is missing without ERROR transaction status")
    process_rc = transaction.get("process_rc")
    _require(type(process_rc) is int and process_rc != 0, "failed transaction process RC is invalid")
    failed_phase = transaction.get("failed_phase")
    failure = transaction.get("failure")
    ppo_optimizer_steps = transaction.get("ppo_optimizer_steps")
    _require(isinstance(failed_phase, str) and failed_phase
             and isinstance(failure, str) and failure,
             "failed transaction attribution is incomplete")
    _require(type(ppo_optimizer_steps) is int and 0 <= ppo_optimizer_steps <= TOTAL_OPTIMIZER_STEPS,
             "failed transaction PPO optimizer-step receipt is invalid")
    progress_path = evidence / "training/iteration_progress.json"
    progress_sha = "ABSENT"
    if progress_path.exists():
        progress = strict_json(progress_path, "failed transaction iteration progress")
        _require(set(progress) == {
            "status", "completed_iterations", "rollout_steps", "optimizer_steps", "last_iteration",
        }, "failed transaction iteration progress keys changed")
        completed_iterations = progress.get("completed_iterations")
        _require(type(completed_iterations) is int and 1 <= completed_iterations <= TRAINING_ITERATIONS,
                 "failed transaction completed iteration count is invalid")
        expected_rollout_steps = completed_iterations * FIRST_ROLLOUT_STEPS
        expected_optimizer_steps = completed_iterations * FIRST_UPDATE_OPTIMIZER_STEPS
        expected_progress_status = "COMPLETE" if completed_iterations == TRAINING_ITERATIONS else "ACTIVE"
        last_iteration = progress.get("last_iteration")
        _require(progress.get("status") == expected_progress_status
                 and progress.get("rollout_steps") == expected_rollout_steps
                 and progress.get("optimizer_steps") == expected_optimizer_steps
                 and isinstance(last_iteration, Mapping)
                 and last_iteration.get("iteration") == completed_iterations - 1
                 and last_iteration.get("rollout_steps") == FIRST_ROLLOUT_STEPS
                 and last_iteration.get("optimizer_steps") == FIRST_UPDATE_OPTIMIZER_STEPS
                 and last_iteration.get("cumulative_optimizer_steps") == expected_optimizer_steps,
                 "failed transaction iteration progress is internally inconsistent")
        _require(ppo_optimizer_steps == expected_optimizer_steps,
                 "failed transaction optimizer steps differ from iteration progress")
        progress_sha = file_sha256(progress_path)
    else:
        _require(ppo_optimizer_steps == 0,
                 "failed transaction reports optimizer steps without iteration progress")
    process_path = evidence / "process_rc.txt"
    _require(process_path.is_file() and not process_path.is_symlink()
             and process_path.lstat().st_nlink == 1
             and process_path.read_text(encoding="ascii") == f"{process_rc}\n",
             "failed transaction process RC receipt differs")
    outer_path = evidence / "SPARSE_HIGH_TASK_PROGRESS_CANARY_TRANSACTION_FINAL_GUARD"
    expected_outer = f"FAIL phase={failed_phase} process_rc={process_rc}\n"
    _require(outer_path.is_file() and not outer_path.is_symlink()
             and outer_path.lstat().st_nlink == 1
             and outer_path.read_text(encoding="utf-8") == expected_outer,
             "failed transaction outer guard differs")

    phase_path = evidence / "phase_events.tsv"
    _require(phase_path.is_file() and not phase_path.is_symlink()
             and phase_path.lstat().st_nlink == 1,
             "failed transaction phase log missing or unsafe")
    phase_lines = phase_path.read_text(encoding="utf-8").splitlines()
    _require(phase_lines and phase_lines[0] == "timestamp\tphase\tstate\trc",
             "failed transaction phase header changed")
    events: list[tuple[str, str, str]] = []
    for number, line in enumerate(phase_lines[1:], 2):
        fields = line.split("\t")
        _require(len(fields) == 4 and fields[0], f"failed phase line {number} malformed")
        events.append((fields[1], fields[2], fields[3]))
    _require(events and events[-1][0] == failed_phase and events[-1][1] == "FAIL",
             "failed transaction last phase does not attribute the failure")
    phase_rc_text = events[-1][2]
    _require(phase_rc_text.isdigit() and int(phase_rc_text) != 0,
             "failed transaction phase RC is missing from phase log")
    phase_rc = int(phase_rc_text)

    child_receipt_by_phase = {
        "contact_reset_lifecycle_probe": evidence / "contact_reset_probe_child_rc.txt",
        "initial_fixed_evaluation": evidence / "initial_child_rc.txt",
        "bounded_task_progress_ppo": evidence / "training_child_rc.txt",
        "final_fixed_evaluation": evidence / "final_child_rc.txt",
    }
    child_receipt = child_receipt_by_phase.get(failed_phase)
    child_rc: int | None = None
    if child_receipt is not None and child_receipt.exists():
        _require(child_receipt.is_file() and not child_receipt.is_symlink()
                 and child_receipt.lstat().st_nlink == 1,
                 "failed child RC receipt is unsafe")
        child_rc_text = child_receipt.read_text(encoding="ascii").strip()
        _require(child_rc_text.isdigit(), "failed child RC receipt is malformed")
        child_rc = int(child_rc_text)
        if child_rc != 0:
            _require(child_rc == phase_rc, "failed child RC receipt differs from phase log")
        else:
            _require(phase_rc == process_rc,
                     "wrapper/gate phase RC differs from transaction process RC")
    else:
        _require(phase_rc == process_rc,
                 "failure without child receipt differs from transaction process RC")
    failure_kind = "CHILD_PROCESS" if child_rc not in (None, 0) else "WRAPPER_OR_GATE"

    probe_failure_path = evidence / "contact_reset_probe/contact_reset_lifecycle_probe_failure.json"
    probe_failure_sha = "ABSENT"
    probe_error = "ABSENT"
    probe_failure_classification = "ABSENT"
    probe_traceback_sha = "ABSENT"
    if failed_phase == "contact_reset_lifecycle_probe":
        require_absent([
            evidence / "initial_child_rc.txt",
            evidence / "training_child_rc.txt",
            evidence / "final_child_rc.txt",
            evidence / "result/task_progress_canary_result.json",
        ], "failed pre-training contact reset probe")
    if failed_phase == "contact_reset_lifecycle_probe" and probe_failure_path.exists():
        probe_failure = strict_json(probe_failure_path, "contact reset probe failure")
        probe_failure_classification, probe_traceback_sha = (
            validate_contact_reset_probe_failure(probe_failure)
        )
        probe_failure_sha = file_sha256(probe_failure_path)
        probe_error = f"{probe_failure['exception_type']}: {probe_failure['message']}"
    log_by_phase = {
        "contact_reset_lifecycle_probe": run / "logs/contact_reset_lifecycle_probe.log",
        "initial_fixed_evaluation": run / "logs/initial_evaluation.log",
        "bounded_task_progress_ppo": run / "logs/training.log",
        "final_fixed_evaluation": run / "logs/final_evaluation.log",
        "terminal_aggregate": run / "logs/finalization.log",
        "independent_result_check": run / "logs/in_job_checker.log",
    }
    probe_log_path = log_by_phase.get(failed_phase, run / "logs" / f"{failed_phase}.log")
    probe_log_sha = "ABSENT"
    if probe_log_path.exists():
        _require(probe_log_path.is_file() and not probe_log_path.is_symlink()
                 and probe_log_path.lstat().st_nlink == 1,
                 "contact reset probe log is unsafe")
        probe_log_sha = file_sha256(probe_log_path)

    training_failure_path = evidence / "training/training_failure.json"
    training_failure_sha = "ABSENT"
    if training_failure_path.exists():
        _require(training_failure_path.is_file() and not training_failure_path.is_symlink()
                 and training_failure_path.lstat().st_nlink == 1,
                 "training failure receipt is unsafe")
        training_failure_sha = file_sha256(training_failure_path)

    print("V56_CHECK=FAIL")
    print("ENGINEERING_INTEGRITY=FAIL")
    print(f"PROCESS_RC={process_rc}")
    print(f"FAILED_PHASE={failed_phase}")
    print(f"ERROR={failure}")
    print(f"FAILURE_KIND={failure_kind}")
    print(f"FAILED_PHASE_RC={phase_rc}")
    print(f"FAILED_CHILD_RC={child_rc if child_rc is not None else 'ABSENT'}")
    print(f"PPO_OPTIMIZER_STEPS={ppo_optimizer_steps}")
    print(f"ITERATION_PROGRESS_SHA256={progress_sha}")
    print(f"OUTER_TRANSACTION_GUARD={expected_outer.strip()}")
    print(f"PROBE_ERROR={probe_error}")
    print(f"PROBE_FAILURE_CLASSIFICATION={probe_failure_classification}")
    print(f"PROBE_TRACEBACK_SHA256={probe_traceback_sha}")
    print(f"PROBE_FAILURE_RECEIPT={probe_failure_path if probe_failure_path.exists() else 'ABSENT'}")
    print(f"PROBE_FAILURE_SHA256={probe_failure_sha}")
    print(f"PROBE_LOG={probe_log_path}")
    print(f"PROBE_LOG_SHA256={probe_log_sha}")
    print(f"FAILED_PHASE_LOG={probe_log_path}")
    print(f"FAILED_PHASE_LOG_SHA256={probe_log_sha}")
    print(f"TRAINING_FAILURE_RECEIPT={training_failure_path if training_failure_path.exists() else 'ABSENT'}")
    print(f"TRAINING_FAILURE_SHA256={training_failure_sha}")
    print(f"SPARSE_HIGH_TASK_PROGRESS_RUN={run}")
    print("LONG_TRAINING_AUTHORIZED=NO")
    return 2


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--in-job", action="store_true")
    args = parser.parse_args()
    try:
        _require(args.run.is_absolute(), "run path must be absolute")
        run = real_directory(args.run, "run")
        _require(run.name.startswith("sparse_high_task_progress_canary_v1_"), "run label prefix changed")
        source = Path(__file__).resolve(strict=True).parent
        contract_path = source / "SPARSE_HIGH_TASK_PROGRESS_CANARY_V1_CONTRACT.json"
        _require(file_sha256(contract_path) == CONTRACT_FILE_SHA256, "release contract bytes changed")
        contract = load_contract(contract_path)
        evidence = real_directory(run / "evidence", "evidence", parent=run)
        transaction_path = evidence / "transaction_status.json"
        if transaction_path.exists():
            transaction_state = strict_json(transaction_path, "transaction dispatch status")
            if transaction_state.get("status") == "ERROR":
                return diagnose_failed_transaction(run, evidence)
        initial_dir = real_directory(evidence / "initial", "initial", parent=evidence)
        training_dir = real_directory(evidence / "training", "training", parent=evidence)
        final_dir = real_directory(evidence / "final", "final", parent=evidence)
        result_dir = real_directory(evidence / "result", "result", parent=evidence)
        environment = read_environment(evidence / "runtime_environment.txt")
        _require(
            environment.get("V56_SUBMITTER_FULL_STATIC_SUITE") == "PASS"
            and environment.get("V56_SUBMITTER_RSL_RL_SMOKE") == "PASS"
            and environment.get("RSL_RL_NO_SIM_RECURRENT_LIFECYCLE_SMOKE") == "PASS",
            "submitter/job preflight evidence bridge is incomplete",
        )
        source_manifest_sha = validate_source_binding(source, environment)
        validate_task_snapshot(run, environment)

        child_rc(evidence / "contact_reset_probe_child_rc.txt", "contact reset lifecycle probe")
        contact_reset_dir = real_directory(
            evidence / "contact_reset_probe", "contact reset probe", parent=evidence
        )
        exact_directory(contact_reset_dir, {
            "contact_reset_lifecycle_probe.json",
            "CONTACT_RESET_LIFECYCLE_PROBE_FINAL_GUARD",
        }, "contact reset probe")
        guard(
            contact_reset_dir / "CONTACT_RESET_LIFECYCLE_PROBE_FINAL_GUARD",
            "contact reset lifecycle probe",
        )
        contact_reset_path = contact_reset_dir / "contact_reset_lifecycle_probe.json"
        contact_reset_probe = strict_json(contact_reset_path, "contact reset lifecycle probe")
        validate_contact_reset_probe(contact_reset_probe, environment=environment, source=source)

        child_rc(evidence / "initial_child_rc.txt", "initial evaluation")
        guard(initial_dir / "INITIAL_EVALUATION_FINAL_GUARD", "initial evaluation")
        initial_path = initial_dir / "initial_evaluation_summary.json"
        initial_rows = initial_dir / "initial_episode_results.jsonl"
        initial_checkpoint = initial_dir / "model_initial.pt"
        exact_directory(initial_dir, {"initial_evaluation_summary.json", "initial_episode_results.jsonl", "model_initial.pt", "INITIAL_EVALUATION_FINAL_GUARD"}, "initial")
        initial = strict_json(initial_path, "initial summary")
        initial_metrics = validate_evaluation_extended(initial, initial_rows, initial_checkpoint, phase="initial", environment=environment, source=source)

        child_rc(evidence / "training_child_rc.txt", "training")
        guard(training_dir / "FIRST_ROLLOUT_REWARD_SUPPORT_FINAL_GUARD", "reward support")
        support_path = training_dir / "first_rollout_reward_support_summary.json"
        transitions_path = training_dir / "first_rollout_transitions.jsonl"
        support = strict_json(support_path, "reward support summary")
        transition_metrics = validate_transition_file(transitions_path, support)
        _require(support.get("initial_policy_state_sha256") == initial.get("policy_state_sha256"), "support initial policy bridge differs")
        _require(support.get("initial_checkpoint_sha256") == initial.get("initial_checkpoint", {}).get("sha256"), "support initial checkpoint bridge differs")
        _require(support.get("frozen_low_state_sha256_before_gate") == initial.get("low_state_after_sha256") == EXPECTED_LOW_MODULE_SHA256, "support low-state bridge differs")
        support_pass = bool(support.get("gate_pass") is True)

        first: dict[str, Any] | None = None
        training: dict[str, Any] | None = None
        final: dict[str, Any] | None = None
        final_metrics: dict[str, Any] | None = None
        if not support_pass:
            guard(training_dir / "REWARD_SUPPORT_REJECTED_FINAL_GUARD", "reward support rejection")
            exact_directory(training_dir, {
                "first_rollout_transitions.jsonl", "first_rollout_reward_support_summary.json",
                "FIRST_ROLLOUT_REWARD_SUPPORT_FINAL_GUARD", "REWARD_SUPPORT_REJECTED_FINAL_GUARD",
            }, "training reject branch")
            rejection_checks = reward_support_checks(support)
            _require(support.get("checks") == rejection_checks and rejection_checks.get("rejected_branch_no_update_integrity") is True, "reward support rejection integrity differs")
            require_absent([
                evidence / "final_child_rc.txt", training_dir / "model_first_update.pt",
                training_dir / "first_update_summary.json", training_dir / "FIRST_UPDATE_FINAL_GUARD",
                training_dir / "iteration_progress.json", training_dir / "model_final.pt",
                training_dir / "training_summary.json", training_dir / "TRAINING_FINAL_GUARD",
            ], "reward support reject branch")
            exact_directory(final_dir, set(), "final reject branch")
        else:
            exact_directory(training_dir, {
                "first_rollout_transitions.jsonl", "first_rollout_reward_support_summary.json",
                "FIRST_ROLLOUT_REWARD_SUPPORT_FINAL_GUARD", "model_first_update.pt",
                "first_update_summary.json", "FIRST_UPDATE_FINAL_GUARD", "iteration_progress.json",
                "model_final.pt", "training_summary.json", "TRAINING_FINAL_GUARD",
            }, "training pass branch")
            guard(training_dir / "FIRST_UPDATE_FINAL_GUARD", "first update")
            guard(training_dir / "TRAINING_FINAL_GUARD", "training")
            first_path = training_dir / "first_update_summary.json"
            training_path = training_dir / "training_summary.json"
            first = strict_json(first_path, "first-update summary")
            training = strict_json(training_path, "training summary")
            validate_component_update(first.get("actor_update"), "first actor update")
            validate_component_update(first.get("critic_update"), "first critic update")
            validate_component_update(training.get("actor_update"), "training actor update")
            validate_component_update(training.get("critic_update"), "training critic update")
            validate_training_extended(training, first, support, initial, environment, source, training_dir / "model_first_update.pt", training_dir / "model_final.pt")
            initial_policy_sha = initial.get("initial_checkpoint", {}).get("policy_state_sha256")
            first_policy_sha = first.get("first_update_checkpoint", {}).get("policy_state_sha256")
            final_policy_sha = training.get("final_checkpoint", {}).get("policy_state_sha256")
            _require(
                all(_is_sha(value) for value in (initial_policy_sha, first_policy_sha, final_policy_sha)),
                "checkpoint policy-state identities are malformed",
            )
            _require(initial_policy_sha != first_policy_sha, "first PPO update did not change actual checkpoint state")
            _require(initial_policy_sha != final_policy_sha, "bounded PPO did not change actual checkpoint state")
            _require(first_policy_sha != final_policy_sha, "bounded continuation did not change actual checkpoint state")
            progress = strict_json(training_dir / "iteration_progress.json", "iteration progress")
            _require(progress.get("status") == "COMPLETE" and progress.get("completed_iterations") == TRAINING_ITERATIONS and progress.get("rollout_steps") == TOTAL_ROLLOUT_STEPS and progress.get("optimizer_steps") == TOTAL_OPTIMIZER_STEPS, "iteration progress incomplete")
            child_rc(evidence / "final_child_rc.txt", "final evaluation")
            guard(final_dir / "FINAL_EVALUATION_FINAL_GUARD", "final evaluation")
            exact_directory(final_dir, {"final_evaluation_summary.json", "final_episode_results.jsonl", "FINAL_EVALUATION_FINAL_GUARD"}, "final")
            final_path = final_dir / "final_evaluation_summary.json"
            final = strict_json(final_path, "final summary")
            final_metrics = validate_evaluation_extended(final, final_dir / "final_episode_results.jsonl", training_dir / "model_final.pt", phase="final", environment=environment, source=source)
            _require(final.get("checkpoint_sha256") == training.get("final_checkpoint", {}).get("sha256"), "final evaluation checkpoint bridge differs")
            _require(final.get("policy_state_sha256") == training.get("final_checkpoint", {}).get("policy_state_sha256"), "final evaluation state bridge differs")

        result_path = result_dir / "task_progress_canary_result.json"
        exact_directory(result_dir, {"task_progress_canary_result.json", "V56_RESULT_FINAL_GUARD"}, "result")
        guard(result_dir / "V56_RESULT_FINAL_GUARD", "inner result")
        result = strict_json(result_path, "terminal result")
        hashes = {
            "initial_evaluation": file_sha256(initial_path),
            "reward_support": file_sha256(support_path),
        }
        if support_pass:
            hashes.update({
                "first_update": file_sha256(training_dir / "first_update_summary.json"),
                "training": file_sha256(training_dir / "training_summary.json"),
                "final_evaluation": file_sha256(final_dir / "final_evaluation_summary.json"),
            })
        validate_result_exact(result, contract, initial, support, first, training, final, hashes)
        validate_transaction(run, result, support_pass=support_pass)
        validate_phase_events(evidence / "phase_events.tsv", support_pass=support_pass, in_job=args.in_job)
    except (OSError, UnicodeError, ValueError, TypeError, ProtocolError) as error:
        print(f"V56_CHECK=FAIL {error}", file=sys.stderr)
        return 2

    print("V56_CHECK=PASS")
    print("ENGINEERING_INTEGRITY=PASS")
    print("PROCESS_RC=0")
    print("INNER_RESULT_GUARD=PASS")
    print("OUTER_TRANSACTION_GUARD=PASS")
    print(f"PHASES={json.dumps({'contact_reset_lifecycle_probe': True, 'initial_fixed_evaluation': True, 'reward_support': True, 'first_update': support_pass, 'bounded_training': support_pass, 'final_fixed_evaluation': support_pass}, sort_keys=True, separators=(',', ':'))}")
    print(f"RESULT_SHA256={file_sha256(result_path)}")
    print(f"CONTACT_RESET_LIFECYCLE_PROBE=PASS")
    print(f"CONTACT_RESET_LIFECYCLE_PROBE_SHA256={file_sha256(contact_reset_path)}")
    print(f"SOURCE_MANIFEST_SHA256={source_manifest_sha}")
    print(f"INPUT_SUMMARY_SHA256={json.dumps(hashes, sort_keys=True, separators=(',', ':'))}")
    print(f"TASK_PACKAGE_MANIFEST_SHA256={PARENT_TASK_MANIFEST_SHA256}")
    print("SUBMITTER_FULL_STATIC_SUITE=PASS")
    print("SUBMITTER_RSL_RL_SMOKE=PASS")
    print("JOB_RSL_RL_NO_SIM_RECURRENT_LIFECYCLE_SMOKE=PASS")
    print(f"FROZEN_LOW_CHECKPOINT_SHA256={EXPECTED_LOW_CHECKPOINT_SHA256}")
    print(f"FROZEN_LOW_RUNTIME_STATE_SHA256={EXPECTED_LOW_MODULE_SHA256}")
    print(f"REWARD_SUPPORT_GATE_PASS={'YES' if support_pass else 'NO'}")
    print(f"FIRST_ROLLOUT_TRANSITIONS={support['transition_row_count']}/25600")
    print(f"PROGRESS_SUPPORT_NONZERO={support['nonzero_progress_count']}")
    print(f"PROGRESS_SUPPORT_POSITIVE={support['positive_progress_count']}")
    print(f"PROGRESS_SUPPORT_NEGATIVE={support['negative_progress_count']}")
    print("PROGRESS_SUPPORT_POSITIVE_BY_GOAL_SIDE=" + json.dumps(support["positive_progress_count_by_goal_side"], sort_keys=True, separators=(",", ":")))
    print(f"FIRST_ROLLOUT_PROGRESS_REWARD_MIN={support['progress_reward_min']}")
    print(f"FIRST_ROLLOUT_PROGRESS_REWARD_MAX={support['progress_reward_max']}")
    print(f"FIRST_ROLLOUT_PROGRESS_REWARD_MEAN={support['progress_reward_mean']}")
    print(f"FIRST_ROLLOUT_TOTAL_REWARD_MEAN={support['total_reward_mean']}")
    print(f"FIRST_ROLLOUT_TERMINAL_TRANSITIONS={support['terminal_transition_count']}")
    print("FIRST_ROLLOUT_TERMINAL_REASON_COUNTS=" + json.dumps(transition_metrics["terminal_reason_counts"], sort_keys=True, separators=(",", ":")))
    print(f"FIRST_ROLLOUT_CONTACT_TRANSITIONS={transition_metrics['contact_transition_count']}/25600")
    print("FIRST_ROLLOUT_MEAN_ABS_APPLIED_COMMAND=" + json.dumps(transition_metrics["mean_abs_applied_command"], separators=(",", ":")))
    print(f"INITIAL_FIXED_GOALS={initial['goal_count']}/288")
    print(f"INITIAL_HARD_GOALS={initial['hard_goal_count']}/36")
    print(f"INITIAL_CONTACT_CASES={initial['contact_count']}/288")
    print(f"INITIAL_UNSAFE_CASES={initial['unsafe_count']}/288")
    print("INITIAL_OUTCOME_COUNTS=" + json.dumps(initial["outcome_counts"], sort_keys=True, separators=(",", ":")))
    print(f"INITIAL_GOAL_SIDE_NEGATIVE_GOALS={initial['goal_counts_by_goal_side']['-1']}/132")
    print(f"INITIAL_GOAL_SIDE_POSITIVE_GOALS={initial['goal_counts_by_goal_side']['1']}/156")
    if support_pass:
        assert training is not None and final is not None and final_metrics is not None
        print("FIRST_UPDATE_EXECUTED=YES")
        print("FIRST_UPDATE_PPO_OPTIMIZER_STEPS=20")
        print("BOUNDED_CONTINUATION_EXECUTED=YES")
        print("TOTAL_PPO_OPTIMIZER_STEPS=240")
        print("TRAINING_OUTCOME_COUNTS=" + json.dumps(training["environment"]["outcome_counts"], sort_keys=True, separators=(",", ":")))
        print(f"TRAINING_CONTACT_POSITIVE_LOW_TICKS={training['environment']['contact_positive_low_ticks']}")
        print(f"TRAINING_CONTACT_NEGATIVE_LOW_TICKS={training['environment']['contact_negative_low_ticks']}")
        print("TRAINING_MEAN_ABS_APPLIED_COMMAND=" + json.dumps(training["environment"]["mean_abs_applied_command"], separators=(",", ":")))
        print("FINAL_FIXED_EVALUATION_EXECUTED=YES")
        print(f"FINAL_FIXED_GOALS={final['goal_count']}/288")
        print(f"FINAL_HARD_GOALS={final['hard_goal_count']}/36")
        print(f"FINAL_CONTACT_CASES={final['contact_count']}/288")
        print(f"FINAL_UNSAFE_CASES={final['unsafe_count']}/288")
        print("FINAL_OUTCOME_COUNTS=" + json.dumps(final["outcome_counts"], sort_keys=True, separators=(",", ":")))
        print(f"FINAL_GOAL_SIDE_NEGATIVE_GOALS={final['goal_counts_by_goal_side']['-1']}/132")
        print(f"FINAL_GOAL_SIDE_POSITIVE_GOALS={final['goal_counts_by_goal_side']['1']}/156")
        print(f"FINAL_MINUS_INITIAL_GOALS={result['final_minus_initial_goal_count']}")
    else:
        print("FIRST_UPDATE_EXECUTED=NO")
        print("FIRST_UPDATE_PPO_OPTIMIZER_STEPS=0")
        print("BOUNDED_CONTINUATION_EXECUTED=NO")
        print("TOTAL_PPO_OPTIMIZER_STEPS=0")
        print("FINAL_FIXED_EVALUATION_EXECUTED=NO")
    print(f"DECISION={result['decision']}")
    print(f"NEXT_GATE={result['next_gate']}")
    print("BLOCKERS=" + json.dumps(result["blockers"], sort_keys=True, separators=(",", ":")))
    print("DENSE_RETURN_AUTHORIZES_BEHAVIOR=NO")
    print("LONG_TRAINING_AUTHORIZED=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
