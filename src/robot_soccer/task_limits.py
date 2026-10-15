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
# The robot's base past these has left the walled arena (over a side wall,
# or behind a goal's back wall at 4.6 m), which ends the episode.  Inside
# the arena, goal boxes included, the robot may go anywhere.
ROBOT_ARENA_HALF_LENGTH_M = _canonical_float32(4.7)
ROBOT_ARENA_HALF_WIDTH_M = _canonical_float32(2.6)
# The ball out past this |y| ends the episode.
BALL_SIDE_BOUNDARY_M = _canonical_float32(2.61)
# Robot-ball contact force that counts as touching the ball.
CONTACT_FORCE_THRESHOLD_N = _canonical_float32(1.0)
# Any contact force on the base (the trunk on the ground) counts as being
# down.  Being down ends nothing; episodes count falls and time down.
BASE_FALL_FORCE_THRESHOLD_N = _canonical_float32(1.0)
