"""EXPLANATORY PROJECTION — NOT THE PRODUCTION TRAINER.

Exact Stage C scientific constants, without runtime or artifact handling.
"""

from __future__ import annotations

from dataclasses import dataclass


# Continuation boundary.
PARENT_ITERATION = 220
FORMAL_START_ITERATION = 221
FORMAL_END_ITERATION = 300
FORMAL_ITERATIONS = 80
MILESTONE_ITERATIONS = (240, 260, 280, 300)
PARENT_CUMULATIVE_OPTIMIZER_STEPS = 4_400
FINAL_CUMULATIVE_OPTIMIZER_STEPS = 6_000

# Rollout and hierarchical control.
TRAINING_SEED = 83_001
NUM_ENVS = 4_096
ROLLOUT_STEPS_PER_ENV = 200
TRANSITIONS_PER_ITERATION = 819_200
FORMAL_TRANSITIONS = 65_536_000
EPISODE_SECONDS = 20.0
MAX_HIGH_STEPS_PER_EPISODE = 200
LOW_POLICY_HZ = 50
HIGH_POLICY_HZ = 10
LOW_TICKS_PER_HIGH_ACTION = 5

# Recurrent high policy.
LOW_OBSERVATION_DIM = 48
ENTITY_FEATURE_DIM = 22
POLICY_OBSERVATION_DIM = 70
POLICY_ACTION_DIM = 3
GRU_HIDDEN_DIM = 256
GRU_LAYERS = 1
ACTOR_HIDDEN_DIMS = (256, 256, 128)
CRITIC_HIDDEN_DIMS = (256, 256, 128)
ACTIVATION = "elu"
INITIAL_ACTION_NOISE_STD = 1.0

# PPO. This is num_mini_batches=4, not batch_size=4. Five epochs times four
# minibatches gives 20 Adam steps per iteration.
LEARNING_RATE = 3.0e-4
GAMMA = 0.997
GAE_LAMBDA = 0.98
ENTROPY_COEFFICIENT = 0.03
NUM_LEARNING_EPOCHS = 5
NUM_MINI_BATCHES = 4
NOMINAL_TRANSITIONS_PER_MINIBATCH = TRANSITIONS_PER_ITERATION // NUM_MINI_BATCHES
OPTIMIZER_STEPS_PER_ITERATION = 20
FORMAL_OPTIMIZER_STEPS = 1_600
CLIP_PARAMETER = 0.2
VALUE_LOSS_COEFFICIENT = 1.0
MAX_GRADIENT_NORM = 1.0
USE_CLIPPED_VALUE_LOSS = True
NORMALIZE_ADVANTAGE_PER_MINIBATCH = False
LEARNING_RATE_SCHEDULE = "fixed"

# The three primary reset/analysis axes.
ORIENTATION_SUPPORT_DEG = (0.0, 45.0, 90.0, 180.0)
ROBOT_BALL_DISTANCE_SUPPORT_M = (0.8, 1.6)
BALL_GOAL_DISTANCE_SUPPORT_M = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0)
BALL_GOAL_DISTANCE_WEIGHT_UNITS = (2, 4, 3, 3, 5, 5, 5, 5)
BALL_GOAL_DISTANCE_ALLOCATION_SLOTS = 32
LONG_DISTANCE_ALLOCATION_SLOTS = 20
LONG_DISTANCE_ALLOCATION_FRACTION = 0.625
BALANCED_SIGN_CELLS = 16
COMPLETE_RESET_ALLOCATION_PERIOD = 4_096

# Continuous command surface: forward, lateral, yaw-rate.
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

# Reward and terminal geometry.
GOAL_CENTER_ABS_X_M = 4.11
GOAL_HALF_WIDTH_FOR_BALL_CENTER_M = 0.49
STAGING_OFFSET_M = 0.55
STAGING_SAFE_X_M = (-3.65, 3.65)
STAGING_SAFE_Y_M = (-2.15, 2.15)
FIELD_HALF_LENGTH_M = 4.0
FIELD_HALF_WIDTH_M = 2.5
BALL_SIDE_BOUNDARY_M = 2.61
TRUE_CONTACT_FORCE_THRESHOLD_N = 1.0
BASE_FALL_FORCE_THRESHOLD_N = 1.0
JOINT_HARD_LIMIT_TOLERANCE_RAD = 1.0e-5


@dataclass(frozen=True)
class PendingScreenArm:
    """One declared A0019 arm; declaration is not an observed result."""

    name: str
    learning_rate: float
    minibatches: int
    optimizer_steps_per_iteration: int


A0019_STATUS = "PENDING_NOT_A_RESULT"
A0019_SCOPE = "DIRECTIONAL_STAR_SCREEN_NOT_FULL_FACTORIAL"
A0019_PENDING_ARMS = (
    PendingScreenArm("lr_1e5_mb4", 1.0e-5, 4, 20),
    PendingScreenArm("lr_1e4_mb4", 1.0e-4, 4, 20),
    PendingScreenArm("control_lr3e4_mb4", 3.0e-4, 4, 20),
    PendingScreenArm("lr_1e3_mb4", 1.0e-3, 4, 20),
    PendingScreenArm("lr_1e2_mb4", 1.0e-2, 4, 20),
    PendingScreenArm("lr3e4_mb2", 3.0e-4, 2, 10),
    PendingScreenArm("lr3e4_mb8", 3.0e-4, 8, 40),
    PendingScreenArm("bridge_lr1p5e4_mb8", 1.5e-4, 8, 40),
)


def optimizer_steps_per_iteration(epochs: int, minibatches: int) -> int:
    """Return Adam calls implied by one PPO iteration."""

    if epochs <= 0 or minibatches <= 0:
        raise ValueError("epochs and minibatches must be positive")
    return epochs * minibatches


def rollout_transitions(iterations: int) -> int:
    """Return transitions for the fixed 4,096 × 200 rollout batch."""

    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    return iterations * NUM_ENVS * ROLLOUT_STEPS_PER_ENV
