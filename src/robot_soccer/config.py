"""Behavioral constants for the no-Kick robot-soccer training task.

The values in this module are the executable settings used by the reported
training path. Deployment metadata and filesystem-specific checkpoint
locations deliberately do not belong here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


LOW_OBSERVATION_DIM = 48
ENTITY_FEATURE_DIM = 22
POLICY_OBSERVATION_DIM = LOW_OBSERVATION_DIM + ENTITY_FEATURE_DIM
POLICY_ACTION_DIM = 3

ENTITY_FEATURE_FIELDS = (
    "ball_body_x",
    "ball_body_y",
    "goal_body_x",
    "goal_body_y",
    "ball_velocity_body_x",
    "ball_velocity_body_y",
    "goal_direction_body_x",
    "goal_direction_body_y",
    "boundary_ray_front",
    "boundary_ray_left",
    "boundary_ray_back",
    "boundary_ray_right",
    "opponent_body_x",
    "opponent_body_y",
    "opponent_velocity_body_x",
    "opponent_velocity_body_y",
    "opponent_present",
    "selected_goal_side",
    "time_remaining",
    "contact_seen",
    "robot_from_ball_goal_axis",
    "robot_from_ball_goal_left",
)


@dataclass(frozen=True)
class FieldGeometry:
    """Geometry shared by reset, observation, reward, and termination code."""

    field_half_length: float = 4.0
    field_half_width: float = 2.5
    goal_half_width: float = 0.60
    ball_radius: float = 0.11
    robot_margin: float = 0.35
    reset_ball_margin: float = 0.35

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value) and value > 0.0 for value in vars(self).values()
        ):
            raise ValueError("Field geometry values must be positive and finite.")
        if self.goal_center_lateral_limit <= 0.0:
            raise ValueError("The ball must fit through the goal opening.")

    @property
    def strict_goal_center_x(self) -> float:
        return self.field_half_length + self.ball_radius

    @property
    def goal_center_lateral_limit(self) -> float:
        return self.goal_half_width - self.ball_radius


# High/low control and finite-horizon task settings.
LOW_POLICY_HZ = 50
HIGH_POLICY_HZ = 10
LOW_STEPS_PER_HIGH_ACTION = 5
EPISODE_SECONDS = 20.0
MAX_EPISODE_HIGH_STEPS = 200

# Training run settings shared by every control.
SEED = 83_001
NUM_ENVS = 4_096
ACTIVATION = "elu"
NUM_LEARNING_EPOCHS = 5
NUM_MINI_BATCHES = 4
CLIP_PARAM = 0.2
VALUE_LOSS_COEF = 1.0
MAX_GRAD_NORM = 1.0
INITIAL_NOISE_STD = 1.0
# The recurrent policy of the held-command hierarchy.
GRU_HIDDEN_DIM = 256
GRU_LAYERS = 1


@dataclass(frozen=True)
class TrainingSettings:
    """Policy, PPO and run-length settings of one control.

    ``gamma`` is also the shaping discount (potential-based shaping keeps
    the optimal policy only when the two match).  ``rnn_hidden_dim`` 0
    means a feed-forward policy.
    """

    gamma: float
    lam: float
    learning_rate: float
    schedule: str
    desired_kl: float | None
    entropy_coef: float
    steps_per_env: int
    iterations: int
    checkpoint_interval: int
    actor_hidden_dims: tuple[int, ...]
    critic_hidden_dims: tuple[int, ...]
    rnn_hidden_dim: int


# Policy straight on the joints at 50 Hz: Isaac Lab's stock Go2 walking
# PPO, with a longer discount (0.995: ~4 s) for a 20 s scoring task.
DIRECT_JOINT_TRAINING = TrainingSettings(
    gamma=0.995,
    lam=0.95,
    learning_rate=1.0e-3,
    schedule="adaptive",
    desired_kl=0.01,
    entropy_coef=0.01,
    steps_per_env=24,
    iterations=3_000,
    checkpoint_interval=100,
    actor_hidden_dims=(512, 256, 128),
    critic_hidden_dims=(512, 256, 128),
    rnn_hidden_dim=0,
)
# The Stage B hierarchy: recurrent 10 Hz command policy.
HELD_COMMAND_TRAINING = TrainingSettings(
    gamma=0.997,
    lam=0.98,
    learning_rate=3.0e-4,
    schedule="fixed",
    desired_kl=None,
    entropy_coef=0.03,
    steps_per_env=200,
    iterations=300,
    checkpoint_interval=25,
    actor_hidden_dims=(256, 256, 128),
    critic_hidden_dims=(256, 256, 128),
    rnn_hidden_dim=GRU_HIDDEN_DIM,
)
TRAINING_BY_CONTROL = {
    "position": DIRECT_JOINT_TRAINING,
    "velocity": DIRECT_JOINT_TRAINING,
    "hierarchical": HELD_COMMAND_TRAINING,
}

# Qualified continuous command surface: (vx, vy, wz).
COMMAND_AXIS_RANGES = (
    (-0.45, 0.45),
    (-0.35, 0.35),
    (-0.60, 0.60),
)
COMMAND_RATE_LIMITS_PER_SECOND = (1.0, 0.8, 1.5)
COMMAND_VERTICES = (
    (0.0, 0.0, 0.0),
    (0.4, 0.0, 0.0),
    (-0.4, 0.0, 0.0),
    (0.0, 0.3, 0.0),
    (0.0, -0.3, 0.0),
    (0.0, 0.0, 0.55),
    (0.0, 0.0, -0.55),
    (0.3, 0.0, 0.35),
    (-0.3, 0.0, -0.35),
    (0.0, 0.25, 0.35),
    (0.0, -0.25, -0.35),
)


if len(ENTITY_FEATURE_FIELDS) != ENTITY_FEATURE_DIM:
    raise AssertionError("Entity feature names and dimension differ.")
if POLICY_OBSERVATION_DIM != 70 or POLICY_ACTION_DIM != 3:
    raise AssertionError("Published policy tensor schema changed.")
if LOW_POLICY_HZ != HIGH_POLICY_HZ * LOW_STEPS_PER_HIGH_ACTION:
    raise AssertionError("High/low control rates are inconsistent.")
if MAX_EPISODE_HIGH_STEPS != round(EPISODE_SECONDS * HIGH_POLICY_HZ):
    raise AssertionError("Finite-horizon settings are inconsistent.")
