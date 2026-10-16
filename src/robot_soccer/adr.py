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

The five knobs are the task axes of a start: the ball's distance from the
centre of the goal it is to be scored in and its angle off the goal's
centre line, the robot's distance from the ball, where the robot stands
around the ball (0: straight behind it, in line with the goal; 180:
between the ball and the goal) and the robot's facing error to the ball
(0: looking at it; 180: looking straight away).  At their limits they
reach every corner of the field.
Every range starts as a single point at the easiest start: the ball a
hand's width from the goal, the robot right behind it looking at it and
at the goal, so that even a policy that only flails sometimes pushes the
ball in.  Steps grow with distance from the start (``ADRParameter.step_at``):
while the robot is still learning to stand and walk, the task moves by a
few centimetres or degrees at a time; once it can reach and push the
ball, the ranges open in large steps.

Some knob combinations cannot fit on the field (the robot 2 m on the goal
side of a ball 0.3 m from the goal).  Such starts are redrawn: every knob
except a probe's pinned one, and the free choices.  A probe whose pinned
value admits no start on the field with the current ranges becomes a
training episode instead.

A lead knob (``lead_parameter``) can be widened first: until its upper
edge reaches ``lead_until``, it is the only knob probed and every other
knob stays at its start value in training and probe episodes.  The robot's
distance from the ball leads by default in training, so the policy learns
to cover distance to the ball before the other knobs make the approach
awkward.  Once the gate opens it stays open.

A small share of episodes are fixed-difficulty episodes
(``difficulty_ranges``): difficulty d spans each knob from its start value
d of the way out to its limits.  Those ranges depend on d alone, so their
goal rate compares across runs and over training, unlike the goal rate
of training episodes, whose task keeps changing.

``AutomaticDomainRandomization`` is the environment's reset sampler: every
start it hands out carries tags (``adr_kind``, ``adr_parameter``,
``adr_edge``, ``adr_difficulty``, ``adr_values``) that come back in the
finished-episode records, and ``record`` reads the probe results from
those records.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .resets import ResetBatch
from .task_limits import GOAL_CENTER_X_M

LOWER, UPPER = 0, 1
EPISODE_KINDS = ("train", "probe", "evaluation")
TRAIN, PROBE, EVALUATION = range(len(EPISODE_KINDS))


@dataclass(frozen=True)
class ADRParameter:
    """One knob: its valid limits, its starting value, and the step of an
    edge move, which grows linearly from ``min_step`` at the start value to
    ``max_step`` at the limit on that side."""

    name: str
    minimum: float
    maximum: float
    start: float
    min_step: float
    max_step: float

    def step_at(self, edge: float) -> float:
        """How far an edge currently at ``edge`` moves in one step."""

        limit = self.maximum if edge >= self.start else self.minimum
        span = abs(limit - self.start)
        progress = min(1.0, abs(edge - self.start) / span) if span > 0.0 else 0.0
        return self.min_step + (self.max_step - self.min_step) * progress


PARAMETERS = (
    # Ball to the centre of the goal it is to be scored in (8.3 m: the far
    # corners of the 8 x 5 m field).
    ADRParameter("ball_goal_distance_m", 0.15, 8.3, 0.25, 0.02, 0.5),
    # Robot base to ball (0.45 m: the robot's nose at the ball; 8 m: across
    # the field).
    ADRParameter("robot_ball_distance_m", 0.4, 8.0, 0.45, 0.02, 0.5),
    # Robot heading versus the bearing to the ball (either side).
    ADRParameter("facing_error_deg", 0.0, 180.0, 0.0, 2.0, 20.0),
    # Where the robot stands around the ball, from straight behind it.
    ADRParameter("robot_bearing_deg", 0.0, 180.0, 0.0, 2.0, 20.0),
    # The ball's angle off the goal's centre line, seen from the goal
    # centre (either side).
    ADRParameter("ball_angle_deg", 0.0, 80.0, 0.5, 1.0, 10.0),
)
PARAMETER_INDEX = {parameter.name: index for index, parameter in enumerate(PARAMETERS)}

# A start is on the field when |x| and |y| are within these (metres).
ROBOT_FIELD_LIMIT_XY = (3.65, 2.15)
BALL_FIELD_LIMIT_XY = (3.97, 2.15)
# Redraw rounds before an unplaceable start falls back.
_PLACEMENT_ROUNDS = 64
# Widths below this count as this for the entropy, so a point is finite.
_ENTROPY_FLOOR = 1.0e-3


def reset_geometry(values: torch.Tensor, generator: torch.Generator) -> tuple[ResetBatch, torch.Tensor]:
    """Episode starts realizing ``values`` (``[n,4]`` in PARAMETERS order)
    and whether each lies on the field.

    The goal side and which side of the goal's centre line the ball is on,
    which side of the ball the robot stands and which way it turns are
    drawn here, uniformly, from ``generator``.  The geometry is exact even when a
    start does not fit; the caller decides what to do with those.
    """

    count = values.shape[0]
    values = values.double()
    ball_goal = values[:, PARAMETER_INDEX["ball_goal_distance_m"]]
    robot_ball = values[:, PARAMETER_INDEX["robot_ball_distance_m"]]
    facing = torch.deg2rad(values[:, PARAMETER_INDEX["facing_error_deg"]])
    bearing = torch.deg2rad(values[:, PARAMETER_INDEX["robot_bearing_deg"]])
    ball_angle = torch.deg2rad(values[:, PARAMETER_INDEX["ball_angle_deg"]])

    def sign() -> torch.Tensor:
        return torch.randint(0, 2, (count,), generator=generator, dtype=torch.float64) * 2.0 - 1.0

    goal_side = sign()
    lateral = sign() * ball_angle
    goal_x = goal_side * GOAL_CENTER_X_M
    ball_x = goal_x - goal_side * ball_goal * torch.cos(lateral)
    ball_y = ball_goal * torch.sin(lateral)
    # Straight behind the ball is the direction from the goal centre through it.
    behind = torch.atan2(ball_y, ball_x - goal_x)
    around = behind + sign() * bearing
    robot_x = ball_x + robot_ball * torch.cos(around)
    robot_y = ball_y + robot_ball * torch.sin(around)
    to_ball = torch.atan2(ball_y - robot_y, ball_x - robot_x)
    yaw = to_ball + sign() * facing
    legal = (
        (robot_x.abs() <= ROBOT_FIELD_LIMIT_XY[0])
        & (robot_y.abs() <= ROBOT_FIELD_LIMIT_XY[1])
        & (ball_x.abs() <= BALL_FIELD_LIMIT_XY[0])
        & (ball_y.abs() <= BALL_FIELD_LIMIT_XY[1])
    )
    batch = ResetBatch(
        robot_xy=torch.stack((robot_x, robot_y), dim=1).float(),
        robot_yaw=torch.atan2(torch.sin(yaw), torch.cos(yaw)).float(),
        ball_xy=torch.stack((ball_x, ball_y), dim=1).float(),
        goal_side=goal_side.long(),
    )
    return batch, legal


def _merge(into: ResetBatch, rows: torch.Tensor, other: ResetBatch) -> None:
    into.robot_xy[rows] = other.robot_xy
    into.robot_yaw[rows] = other.robot_yaw
    into.ball_xy[rows] = other.ball_xy
    into.goal_side[rows] = other.goal_side


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
        lead_parameter: str | None = None,
        lead_until: float = 0.0,
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
        if lead_parameter is not None and lead_parameter not in PARAMETER_INDEX:
            raise ValueError(f"unknown lead knob {lead_parameter}")
        self.lead_index = None if lead_parameter is None else PARAMETER_INDEX[lead_parameter]
        self.lead_until = float(lead_until)
        self.lead_done = lead_parameter is None
        self.generator = torch.Generator().manual_seed(int(seed))
        self._ranges = torch.tensor(
            [[parameter.start, parameter.start] for parameter in PARAMETERS], dtype=torch.float64
        )
        # Probe results not yet decided on: [parameter, edge] -> (count, successes).
        self._buffer_count = torch.zeros(len(PARAMETERS), 2, dtype=torch.long)
        self._buffer_successes = torch.zeros(len(PARAMETERS), 2, dtype=torch.long)
        # Since the last ``take_period_metrics``: results recorded, edge moves
        # per [parameter, edge] (+1 outward, -1 inward) and probes that could
        # not be placed on the field.
        self._period_results = torch.zeros(len(PARAMETERS), 2, dtype=torch.long)
        self._period_successes = torch.zeros(len(PARAMETERS), 2, dtype=torch.long)
        self._period_moves = torch.zeros(len(PARAMETERS), 2, dtype=torch.long)
        self._period_unplaced_probes = torch.zeros(len(PARAMETERS), 2, dtype=torch.long)

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

    def _open_gate_if_due(self) -> None:
        if not self.lead_done and float(self._ranges[self.lead_index, UPPER]) >= self.lead_until:
            self.lead_done = True

    def __call__(self, count: int) -> ResetBatch:
        """Starts for ``count`` new episodes, each a training, probe or
        fixed-difficulty episode, tagged for ``record`` and the metrics."""

        self._open_gate_if_due()
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
        parameter = torch.randint(0, len(PARAMETERS), (count,), generator=generator)
        edge = torch.randint(0, 2, (count,), generator=generator)
        probe = kind == PROBE
        if not self.lead_done:
            # Only the lead knob is probed; the others stay at their starts.
            parameter[:] = self.lead_index
            starts = torch.tensor([p.start for p in PARAMETERS], dtype=torch.float64)
            others = torch.ones(len(PARAMETERS), dtype=torch.bool)
            others[self.lead_index] = False
            playing = ~evaluation
            low[playing.unsqueeze(1) & others] = starts.expand(count, -1)[playing.unsqueeze(1) & others]
            high[playing.unsqueeze(1) & others] = starts.expand(count, -1)[playing.unsqueeze(1) & others]
        rows = torch.nonzero(probe).flatten()
        # A probe's range for its pinned knob is the edge itself.
        pinned = self._ranges[parameter[rows], edge[rows]]
        low[rows, parameter[rows]] = pinned
        high[rows, parameter[rows]] = pinned

        values = low + (high - low) * torch.rand(count, len(PARAMETERS), generator=generator, dtype=torch.float64)
        batch, legal = reset_geometry(values, generator)
        for _ in range(_PLACEMENT_ROUNDS):
            if bool(legal.all()):
                break
            redo = torch.nonzero(~legal).flatten()
            values[redo] = low[redo] + (high[redo] - low[redo]) * torch.rand(
                redo.numel(), len(PARAMETERS), generator=generator, dtype=torch.float64
            )
            redone, redone_legal = reset_geometry(values[redo], generator)
            _merge(batch, redo, redone)
            legal[redo] = redone_legal
        if not bool(legal.all()):
            # Unplaceable probes become training episodes; anything still off
            # the field starts at the start values, which always fit.
            stuck = ~legal
            for index, side in zip(parameter[stuck & probe].tolist(), edge[stuck & probe].tolist()):
                self._period_unplaced_probes[index, side] += 1
            kind[stuck & probe] = TRAIN
            redo = torch.nonzero(stuck).flatten()
            values[redo] = torch.tensor([p.start for p in PARAMETERS], dtype=torch.float64)
            redone, _ = reset_geometry(values[redo], generator)
            _merge(batch, redo, redone)
        probe = kind == PROBE
        parameter[~probe] = -1
        edge[~probe] = -1
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
            if not self.lead_done and index != self.lead_index:
                continue
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
                self._open_gate_if_due()
        return moves

    def _move(self, index: int, edge: int, direction: int) -> bool:
        parameter = PARAMETERS[index]
        low, high = self._ranges[index].tolist()
        if edge == UPPER:
            step = parameter.step_at(high)
            high = min(parameter.maximum, high + step) if direction > 0 else max(low, high - step)
        else:
            step = parameter.step_at(low)
            low = max(parameter.minimum, low - step) if direction > 0 else min(high, low + step)
        moved = [low, high] != self._ranges[index].tolist()
        self._ranges[index] = torch.tensor([low, high], dtype=torch.float64)
        return moved

    def entropy(self) -> float:
        """ADR entropy: mean log range width (nats per knob)."""

        widths = (self._ranges[:, 1] - self._ranges[:, 0]).clamp_min(_ENTROPY_FLOOR)
        return float(torch.log(widths).mean())

    def take_period_metrics(self) -> dict[str, Any]:
        """The ranges now, plus probe results, edge moves and unplaceable
        probes since the last call (one call per training iteration), then
        reset the period.
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
                    "unplaced_probes": int(self._period_unplaced_probes[index, edge]),
                }
            knobs[parameter.name] = knob
        self._period_results.zero_()
        self._period_successes.zero_()
        self._period_moves.zero_()
        self._period_unplaced_probes.zero_()
        return {"entropy": self.entropy(), "lead_done": self.lead_done, "knobs": knobs}

    def state_dict(self) -> dict[str, Any]:
        """Everything needed to continue ADR from a checkpoint."""

        return {
            "parameter_names": [parameter.name for parameter in PARAMETERS],
            "lead_done": self.lead_done,
            "ranges": self._ranges.tolist(),
            "buffer_count": self._buffer_count.tolist(),
            "buffer_successes": self._buffer_successes.tolist(),
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore ranges, buffers and the generator.  Knobs the checkpoint
        does not know (added since; older checkpoints list the first four
        knobs, unnamed) start at their start values, clamped to today's
        limits."""

        names = state.get("parameter_names") or [parameter.name for parameter in PARAMETERS][: len(state["ranges"])]
        saved = dict(zip(names, zip(state["ranges"], state["buffer_count"], state["buffer_successes"])))
        for index, parameter in enumerate(PARAMETERS):
            if parameter.name in saved:
                (low, high), count, successes = saved[parameter.name]
                self._ranges[index] = torch.tensor(
                    [max(parameter.minimum, low), min(parameter.maximum, high)], dtype=torch.float64
                )
                self._buffer_count[index] = torch.tensor(count, dtype=torch.long)
                self._buffer_successes[index] = torch.tensor(successes, dtype=torch.long)
            else:
                self._ranges[index] = parameter.start
                self._buffer_count[index] = 0
                self._buffer_successes[index] = 0
        self.lead_done = self.lead_index is None or bool(state.get("lead_done", False))
        self._open_gate_if_due()
        self.generator.set_state(state["generator_state"])
