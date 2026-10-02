"""Robot-ball ContactSensor body mapping used by the soccer task."""

from __future__ import annotations

from typing import Sequence


ROBOT_BODY_NAMES = (
    "base",
    "FL_hip",
    "FR_hip",
    "Head_upper",
    "RL_hip",
    "RR_hip",
    "FL_thigh",
    "FR_thigh",
    "Head_lower",
    "RL_thigh",
    "RR_thigh",
    "FL_calf",
    "FR_calf",
    "RL_calf",
    "RR_calf",
    "FL_foot",
    "FR_foot",
    "RL_foot",
    "RR_foot",
)


def exact_robot_filter_paths(
    body_names: Sequence[str] = ROBOT_BODY_NAMES,
) -> tuple[str, ...]:
    names = tuple(str(name) for name in body_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("Robot body names must be non-empty and unique.")
    if any(not name.replace("_", "").isalnum() for name in names):
        raise ValueError("Robot body name contains an unsafe character.")
    return tuple(f"{{ENV_REGEX_NS}}/Robot/{name}" for name in names)


def validate_configured_robot_filter_paths(filter_paths: Sequence[str]) -> tuple[str, ...]:
    """Fail closed on the unexpanded ContactSensor configuration contract.

    Isaac Lab expands ``{ENV_REGEX_NS}`` in-place while constructing an
    ``InteractiveScene``.  The exact source contract must therefore be checked
    before ``gym.make`` rather than by comparing the already-mutated live
    sensor cfg with the original placeholder strings.
    """

    received = tuple(str(path) for path in filter_paths)
    expected = exact_robot_filter_paths()
    if received != expected:
        raise RuntimeError(
            "Configured robot-ball ContactSensor filters differ from the "
            f"accepted contract: expected={expected} received={received}"
        )
    return received


def validate_live_robot_body_names(body_names: Sequence[str]) -> None:
    received = tuple(str(name) for name in body_names)
    if received != ROBOT_BODY_NAMES:
        raise RuntimeError(
            "Live robot body order differs from ContactSensor qualification: "
            f"expected={ROBOT_BODY_NAMES} received={received}"
        )
