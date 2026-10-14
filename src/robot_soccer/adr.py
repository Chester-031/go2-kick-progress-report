"""Automatic domain randomization (ADR) over where episodes start.

OpenAI, "Solving Rubik's Cube with a Robot Hand" (2019), Section 5.  Each
knob has its own range [low, high].  Training episodes draw every knob
uniformly from its current range, so what was already learned stays in
the mix.  Probe episodes pin one randomly chosen knob at one of its two
edges and record whether the robot scored; once an edge has
``buffer_size`` results, an average at or above ``high_threshold`` moves
that edge one step outward and one at or below ``low_threshold`` moves it
one step inward.  Ranges therefore grow where the policy copes and shrink
where it does not, each knob on its own.

The three knobs are the task axes of the reset: the ball's distance from
the goal it is to be scored in, the robot's distance from the ball, and
the robot's facing error to the ball (0: looking at the ball, 180: looking
straight away).  Every range starts as a single point, an easy start
close to the goal, and is clamped to the knob's limits, which keep every
combination on the field.  Everything else about a start is drawn the
same way at every difficulty (``reset_geometry``).

A small share of episodes are fixed-difficulty episodes instead
(``difficulty_ranges``): difficulty d spans each knob from its start
value d of the way out to its limits.  Those ranges depend on d alone, so
their goal rate compares across runs and over training, unlike the goal
rate of training episodes, whose task keeps changing.

``AutomaticDomainRandomization`` is the environment's reset sampler: every
start it hands out carries tags (``adr_kind``, ``adr_parameter``,
``adr_edge``, ``adr_difficulty``, ``adr_values``) that come back in the
finished-episode records, and ``record`` reads the probe results from
those records.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch

from .resets import ResetBatch
from .rewards import GOAL_CENTER_X_M

LOWER, UPPER = 0, 1
EPISODE_KINDS = ("train", "probe", "evaluation")
TRAIN, PROBE, EVALUATION = range(len(EPISODE_KINDS))


@dataclass(frozen=True)
class ADRParameter:
    """One knob: its valid limits, the starting value and the step per
    edge move."""

    name: str
    minimum: float
    maximum: float
    start: float
    step: float


PARAMETERS = (
    # Ball to the centre of the goal it is to be scored in.
    ADRParameter("ball_goal_distance_m", 0.5, 6.0, 1.0, 0.25),
    # Robot base to ball.
    ADRParameter("robot_ball_distance_m", 0.5, 2.3, 0.8, 0.1),
    # Robot heading versus the bearing to the ball (either side).
    ADRParameter("facing_error_deg", 0.0, 180.0, 0.0, 10.0),
)
PARAMETER_INDEX = {parameter.name: index for index, parameter in enumerate(PARAMETERS)}

# The free choices of a start, the same at every difficulty: the ball sits
# within a degree of the goal's centre line, and the robot sits beside and
# behind it, 122-136 degrees around the ball from the ball-to-goal
# direction, on either side.  With the knob limits above this keeps the
# robot and the ball inside |x| <= 3.65, |y| <= 2.15.
BALL_LATERAL_DEG = (0.1, 1.0)
ROBOT_ORBIT_DEG = (122.0, 136.0)
# Widths below this count as this for the entropy, so a point is finite.
_ENTROPY_FLOOR = 1.0e-3


def reset_geometry(values: torch.Tensor, generator: torch.Generator) -> ResetBatch:
    """Episode starts realizing ``values`` (``[n,3]`` in PARAMETERS order).

    The goal side, the ball's small lateral angle around the goal centre,
    the robot's position around the ball and which way it turns away from
    the ball are drawn here, uniformly, from ``generator``.
    """

    count = values.shape[0]
    values = values.double()
    ball_goal = values[:, PARAMETER_INDEX["ball_goal_distance_m"]]
    robot_ball = values[:, PARAMETER_INDEX["robot_ball_distance_m"]]
    facing = torch.deg2rad(values[:, PARAMETER_INDEX["facing_error_deg"]])

    def uniform(bounds: tuple[float, float]) -> torch.Tensor:
        low, high = (math.radians(bound) for bound in bounds)
        return low + (high - low) * torch.rand(count, generator=generator, dtype=torch.float64)

    def sign() -> torch.Tensor:
        return torch.randint(0, 2, (count,), generator=generator, dtype=torch.float64) * 2.0 - 1.0

    goal_side = sign()
    lateral = sign() * uniform(BALL_LATERAL_DEG)
    orbit = sign() * uniform(ROBOT_ORBIT_DEG)
    facing_sign = sign()
    goal_x = goal_side * GOAL_CENTER_X_M
    ball_x = goal_x - goal_side * ball_goal * torch.cos(lateral)
    ball_y = ball_goal * torch.sin(lateral)
    bearing = torch.atan2(-ball_y, goal_x - ball_x) + orbit
    robot_x = ball_x + robot_ball * torch.cos(bearing)
    robot_y = ball_y + robot_ball * torch.sin(bearing)
    to_ball = torch.atan2(ball_y - robot_y, ball_x - robot_x)
    yaw = to_ball + facing_sign * facing
    return ResetBatch(
        robot_xy=torch.stack((robot_x, robot_y), dim=1).float(),
        robot_yaw=torch.atan2(torch.sin(yaw), torch.cos(yaw)).float(),
        ball_xy=torch.stack((ball_x, ball_y), dim=1).float(),
        goal_side=goal_side.long(),
    )


class AutomaticDomainRandomization:
    """The ADR ranges, how episode starts sample them, and how probe
    results move them.  Call it with a count to get a ``ResetBatch``."""

    def __init__(
        self,
        *,
        buffer_size: int = 256,
        low_threshold: float = 0.3,
        high_threshold: float = 0.6,
        probe_fraction: float = 0.2,
        evaluation_fraction: float = 0.05,
        evaluation_difficulties: Sequence[float] = (0.25, 0.5, 1.0),
        seed: int = 0,
    ) -> None:
        if not 0.0 <= low_threshold < high_threshold <= 1.0:
            raise ValueError("need 0 <= low_threshold < high_threshold <= 1")
        if buffer_size < 1:
            raise ValueError("buffer_size must be positive")
        if probe_fraction < 0.0 or evaluation_fraction < 0.0 or probe_fraction + evaluation_fraction > 1.0:
            raise ValueError("probe and evaluation fractions must be non-negative and sum to at most 1")
        if not evaluation_difficulties or any(not 0.0 <= value <= 1.0 for value in evaluation_difficulties):
            raise ValueError("evaluation difficulties must be in [0, 1]")
        self.buffer_size = int(buffer_size)
        self.low_threshold = float(low_threshold)
        self.high_threshold = float(high_threshold)
        self.probe_fraction = float(probe_fraction)
        self.evaluation_fraction = float(evaluation_fraction)
        self.evaluation_difficulties = tuple(float(value) for value in evaluation_difficulties)
        self.generator = torch.Generator().manual_seed(int(seed))
        self._ranges = torch.tensor(
            [[parameter.start, parameter.start] for parameter in PARAMETERS], dtype=torch.float64
        )
        # Probe results not yet decided on: [parameter, edge] -> (count, successes).
        self._buffer_count = torch.zeros(len(PARAMETERS), 2, dtype=torch.long)
        self._buffer_successes = torch.zeros(len(PARAMETERS), 2, dtype=torch.long)
        # Since the last ``take_period_metrics``: results recorded and edge
        # moves per [parameter, edge] (+1 outward, -1 inward).
        self._period_results = torch.zeros(len(PARAMETERS), 2, dtype=torch.long)
        self._period_successes = torch.zeros(len(PARAMETERS), 2, dtype=torch.long)
        self._period_moves = torch.zeros(len(PARAMETERS), 2, dtype=torch.long)

    def ranges(self) -> torch.Tensor:
        """A copy of the current ``[low, high]`` per knob, in PARAMETERS order."""

        return self._ranges.clone()

    def set_range(self, name: str, low: float, high: float) -> None:
        """Set one range directly (tests and restarts), clamped to its limits."""

        parameter = PARAMETERS[PARAMETER_INDEX[name]]
        low = max(parameter.minimum, float(low))
        high = min(parameter.maximum, float(high))
        if low > high:
            raise ValueError("low must not exceed high")
        self._ranges[PARAMETER_INDEX[name]] = torch.tensor([low, high], dtype=torch.float64)

    @staticmethod
    def difficulty_ranges(difficulty: float) -> torch.Tensor:
        """The fixed ranges of evaluation difficulty ``difficulty``: each
        knob from its start value ``difficulty`` of the way to each limit
        (0: the start point, 1: the full limits)."""

        if not 0.0 <= difficulty <= 1.0:
            raise ValueError("difficulty must be in [0, 1]")
        return torch.tensor(
            [
                [
                    parameter.start - difficulty * (parameter.start - parameter.minimum),
                    parameter.start + difficulty * (parameter.maximum - parameter.start),
                ]
                for parameter in PARAMETERS
            ],
            dtype=torch.float64,
        )

    def __call__(self, count: int) -> ResetBatch:
        """Starts for ``count`` new episodes, each a training, probe or
        fixed-difficulty episode, tagged for ``record`` and the metrics."""

        generator = self.generator
        draw = torch.rand(count, generator=generator, dtype=torch.float64)
        kind = torch.full((count,), TRAIN, dtype=torch.long)
        kind[draw < self.probe_fraction + self.evaluation_fraction] = EVALUATION
        kind[draw < self.probe_fraction] = PROBE

        low = self._ranges[:, 0].expand(count, -1).clone()
        high = self._ranges[:, 1].expand(count, -1).clone()
        difficulty = torch.full((count,), -1.0, dtype=torch.float64)
        evaluation = kind == EVALUATION
        if bool(evaluation.any()):
            choice = torch.randint(
                0, len(self.evaluation_difficulties), (int(evaluation.sum()),), generator=generator
            )
            difficulty[evaluation] = torch.tensor(self.evaluation_difficulties, dtype=torch.float64)[choice]
            fixed = torch.stack([self.difficulty_ranges(float(value)) for value in difficulty[evaluation]])
            low[evaluation] = fixed[:, :, 0]
            high[evaluation] = fixed[:, :, 1]
        values = low + (high - low) * torch.rand(count, len(PARAMETERS), generator=generator, dtype=torch.float64)

        parameter = torch.randint(0, len(PARAMETERS), (count,), generator=generator)
        edge = torch.randint(0, 2, (count,), generator=generator)
        probe = kind == PROBE
        rows = torch.nonzero(probe).flatten()
        values[rows, parameter[rows]] = self._ranges[parameter[rows], edge[rows]]
        parameter[~probe] = -1
        edge[~probe] = -1

        batch = reset_geometry(values, generator)
        batch.tags.update(
            {
                "adr_kind": kind,
                "adr_parameter": parameter,
                "adr_edge": edge,
                "adr_difficulty": difficulty.float(),
                "adr_values": values.float(),
            }
        )
        return batch

    def record(self, episodes: Mapping[str, torch.Tensor]) -> int:
        """Add the probe results among finished ``episodes`` (success = a
        goal) and move every edge whose buffer fills.  Returns the number
        of edges that moved."""

        if not episodes:
            return 0
        probe = episodes["adr_kind"] == PROBE
        if not bool(probe.any()):
            return 0
        moves = 0
        for index, edge, success in zip(
            episodes["adr_parameter"][probe].tolist(),
            episodes["adr_edge"][probe].tolist(),
            episodes["goal"][probe].tolist(),
        ):
            self._period_results[index, edge] += 1
            self._period_successes[index, edge] += int(success)
            self._buffer_count[index, edge] += 1
            self._buffer_successes[index, edge] += int(success)
            if int(self._buffer_count[index, edge]) < self.buffer_size:
                continue
            average = int(self._buffer_successes[index, edge]) / int(self._buffer_count[index, edge])
            self._buffer_count[index, edge] = 0
            self._buffer_successes[index, edge] = 0
            if average >= self.high_threshold:
                direction = 1
            elif average <= self.low_threshold:
                direction = -1
            else:
                continue
            if self._move(index, edge, direction):
                self._period_moves[index, edge] += direction
                moves += 1
        return moves

    def _move(self, index: int, edge: int, direction: int) -> bool:
        parameter = PARAMETERS[index]
        low, high = self._ranges[index].tolist()
        if edge == UPPER:
            high = (
                min(parameter.maximum, high + parameter.step)
                if direction > 0
                else max(low, high - parameter.step)
            )
        else:
            low = (
                max(parameter.minimum, low - parameter.step)
                if direction > 0
                else min(high, low + parameter.step)
            )
        moved = [low, high] != self._ranges[index].tolist()
        self._ranges[index] = torch.tensor([low, high], dtype=torch.float64)
        return moved

    def entropy(self) -> float:
        """ADR entropy: mean log range width (nats per knob)."""

        widths = (self._ranges[:, 1] - self._ranges[:, 0]).clamp_min(_ENTROPY_FLOOR)
        return float(torch.log(widths).mean())

    def take_period_metrics(self) -> dict[str, Any]:
        """The ranges now, plus probe results and edge moves since the last
        call (one call per training iteration), then reset the period.

        Per knob: ``low``/``high``, and per edge the number of probe
        results, their success rate and the net edge moves.
        """

        knobs = {}
        for index, parameter in enumerate(PARAMETERS):
            low, high = self._ranges[index].tolist()
            knob: dict[str, Any] = {"low": low, "high": high}
            for edge, edge_name in ((LOWER, "low_edge"), (UPPER, "high_edge")):
                results = int(self._period_results[index, edge])
                knob[edge_name] = {
                    "probe_results": results,
                    "probe_success_rate": (
                        int(self._period_successes[index, edge]) / results if results else None
                    ),
                    "moves": int(self._period_moves[index, edge]),
                    "buffered": int(self._buffer_count[index, edge]),
                }
            knobs[parameter.name] = knob
        self._period_results.zero_()
        self._period_successes.zero_()
        self._period_moves.zero_()
        return {"entropy": self.entropy(), "knobs": knobs}

    def state_dict(self) -> dict[str, Any]:
        """Everything needed to continue ADR from a checkpoint."""

        return {
            "ranges": self._ranges.tolist(),
            "buffer_count": self._buffer_count.tolist(),
            "buffer_successes": self._buffer_successes.tolist(),
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._ranges = torch.tensor(state["ranges"], dtype=torch.float64)
        self._buffer_count = torch.tensor(state["buffer_count"], dtype=torch.long)
        self._buffer_successes = torch.tensor(state["buffer_successes"], dtype=torch.long)
        self.generator.set_state(state["generator_state"])
