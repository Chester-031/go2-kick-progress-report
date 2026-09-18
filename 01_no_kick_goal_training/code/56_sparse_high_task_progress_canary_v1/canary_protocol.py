"""Pure, simulator-independent V56 evidence protocol.

The producer and independent checker both use these frozen schemas and
decision rules.  This module deliberately has no Isaac Lab dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from task_progress_reward import (
    BALL_SIDE_BOUNDARY_M,
    BASE_FALL_FORCE_THRESHOLD_N,
    CONTACT_FORCE_THRESHOLD_N,
    CONTINUITY_ABS_TOLERANCE,
    GOAL_CENTER_X_M,
    PATH_ABS_TOLERANCE,
    PATH_NORMALIZER_M,
    REWARD_ABS_TOLERANCE,
    ROBOT_FIELD_HALF_LENGTH_M,
    ROBOT_FIELD_HALF_WIDTH_M,
    STAGING_OFFSET_M,
    STAGING_SAFE_MAX_X_M,
    STAGING_SAFE_MAX_Y_M,
    STAGING_SAFE_MIN_X_M,
    STAGING_SAFE_MIN_Y_M,
    SUPPORT_EPSILON,
    TERMINAL_GOAL_HALF_WIDTH_M,
    JOINT_HARD_LIMIT_TOLERANCE_RAD,
    close_scalar,
    close_vector,
    reconstruct_transition,
    scalar_task_geometry,
)


CONTRACT_SCHEMA = "sparse_high_task_progress_canary_v1/contract/v1"
EVALUATION_SCHEMA = "sparse_high_task_progress_canary_v1/fixed_evaluation/v1"
EPISODE_ROW_SCHEMA = "sparse_high_task_progress_canary_v1/episode_row/v1"
TRANSITION_ROW_SCHEMA = "sparse_high_task_progress_canary_v1/first_rollout_transition/v1"
REWARD_SUPPORT_SCHEMA = "sparse_high_task_progress_canary_v1/reward_support/v1"
FIRST_UPDATE_SCHEMA = "sparse_high_task_progress_canary_v1/first_update/v1"
TRAINING_SCHEMA = "sparse_high_task_progress_canary_v1/bounded_training/v1"
RESULT_SCHEMA = "sparse_high_task_progress_canary_v1/result/v1"

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
