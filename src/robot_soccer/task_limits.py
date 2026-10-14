"""Where the field ends and what ends an episode.

Field-frame constants of the soccer task, shared by the environment's
termination checks, ADR's start placement and the metrics.  Each value is
rounded to float32 once here so that comparisons against simulator
(float32) states give the same answer everywhere.  The reward is not
here: it is the sparse goal/own-goal reward in ``environment`` plus the
potential-based shaping in ``shaping``.
"""

from __future__ import annotations

import struct


def _canonical_float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


# The ball's centre at this |x| inside the goal mouth is a goal (the whole
# ball past the 4.0 m goal line); the centre of the attacked goal.
GOAL_CENTER_X_M = _canonical_float32(4.11)
# Half the width of the goal mouth the ball's centre must pass through.
TERMINAL_GOAL_HALF_WIDTH_M = _canonical_float32(0.49)
# The robot's base out past these ends the episode.
ROBOT_FIELD_HALF_LENGTH_M = _canonical_float32(4.0)
ROBOT_FIELD_HALF_WIDTH_M = _canonical_float32(2.5)
# The ball out past this |y| ends the episode.
BALL_SIDE_BOUNDARY_M = _canonical_float32(2.61)
# Robot-ball contact force that counts as touching the ball.
CONTACT_FORCE_THRESHOLD_N = _canonical_float32(1.0)
# Any contact force on the base (the trunk touching the ground) is a fall.
BASE_FALL_FORCE_THRESHOLD_N = _canonical_float32(1.0)
# How far past a joint's hard limit ends the episode.  Hard kicks push the
# simulated joints a little past their stops; ending every such episode
# (the old 1e-5 rad) ended about a fifth of all episodes.
JOINT_HARD_LIMIT_TOLERANCE_RAD = _canonical_float32(0.05)
# The margin inside each hard limit the joint-limit shaping potential sees.
JOINT_LIMIT_MARGIN_RAD = _canonical_float32(0.1)
