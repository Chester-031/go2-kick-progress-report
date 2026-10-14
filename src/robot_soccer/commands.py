"""Map a 3-D latent action into the qualified low-policy command surface."""

from __future__ import annotations

from typing import Sequence

import torch


INSIDE_TOLERANCE = 1.0e-6


def hull_facets(vertices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The supporting planes of the convex hull of 3-D ``vertices``.

    Returns unit outward normals ``[F,3]`` and offsets ``[F]`` such that a
    point ``p`` is in the hull exactly when ``normals @ p <= offsets``.
    Every plane through three affinely independent vertices that has all
    vertices on one side is a facet plane; coplanar triples give the same
    plane and are kept once.  This runs once per command surface, so a
    plain loop over the vertex triples is fine.
    """

    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] < 4:
        raise ValueError("vertices must have shape [V,3] with V >= 4.")
    points = vertices.detach().double().cpu()
    planes: dict[tuple[float, ...], tuple[torch.Tensor, float]] = {}
    count = points.shape[0]
    for first in range(count):
        for second in range(first + 1, count):
            for third in range(second + 1, count):
                normal = torch.linalg.cross(points[second] - points[first], points[third] - points[first])
                length = float(torch.linalg.vector_norm(normal))
                if length < 1.0e-12:
                    continue
                normal = normal / length
                offset = float(normal @ points[first])
                side = points @ normal - offset
                if bool((side <= 1.0e-9).all()):
                    pass
                elif bool((side >= -1.0e-9).all()):
                    normal, offset = -normal, -offset
                else:
                    continue
                key = tuple(round(value, 9) for value in (*normal.tolist(), offset))
                planes[key] = (normal, offset)
    if len(planes) < 4:
        raise ValueError("vertices do not span a 3-D hull.")
    normals = torch.stack([normal for normal, _ in planes.values()])
    offsets = torch.tensor([offset for _, offset in planes.values()], dtype=torch.float64)
    return normals.to(vertices), offsets.to(vertices)


def project_onto_hull(
    desired: torch.Tensor,
    vertices: torch.Tensor,
    normals: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Exact Euclidean projection of each row of ``desired`` onto the hull.

    The nearest hull point is the point itself when it is inside, otherwise
    it lies in a facet or on a segment between two vertices.  Every
    candidate is computed at once — the point, its projection on each facet
    plane when that lands inside the hull, and its nearest point on every
    vertex-to-vertex segment — and the nearest admissible one wins.  There
    is no iteration and no host synchronization.
    """

    inside = (desired @ normals.T - offsets).amax(dim=1) <= INSIDE_TOLERANCE
    plane_distance = desired @ normals.T - offsets
    on_planes = desired[:, None, :] - plane_distance[:, :, None] * normals[None, :, :]
    planes_admissible = (
        torch.einsum("nfk,gk->nfg", on_planes, normals) - offsets
    ).amax(dim=2) <= INSIDE_TOLERANCE
    first, second = torch.triu_indices(vertices.shape[0], vertices.shape[0], offset=1, device=vertices.device)
    start = vertices[first]
    span = vertices[second] - start
    fraction = (((desired[:, None, :] - start[None]) * span[None]).sum(dim=2) / span.square().sum(dim=1)).clamp(0.0, 1.0)
    on_segments = start[None] + fraction[:, :, None] * span[None]
    candidates = torch.cat((desired[:, None, :], on_planes, on_segments), dim=1)
    admissible = torch.cat(
        (
            inside[:, None],
            planes_admissible,
            torch.ones_like(fraction, dtype=torch.bool),
        ),
        dim=1,
    )
    distance = (candidates - desired[:, None, :]).square().sum(dim=2)
    distance = torch.where(admissible, distance, torch.full_like(distance, torch.inf))
    best = distance.argmin(dim=1)
    return candidates[torch.arange(desired.shape[0], device=desired.device), best]


def project_to_convex_hull(desired: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """Exact projection onto the hull of ``vertices``, building the facets
    on every call.  Use ``CommandSurface`` in loops, which builds them once."""

    if desired.ndim != 2 or desired.shape[1] != 3:
        raise ValueError("desired must have shape [N,3].")
    normals, offsets = hull_facets(vertices)
    return project_onto_hull(desired, vertices, normals, offsets)



class CommandSurface:
    """The (vx, vy, wz) commands the frozen low policy is qualified for, and
    how a 3-D latent policy action becomes one.

    A latent action is squashed with ``tanh``, scaled into the per-axis
    ranges, projected exactly onto the convex hull of the qualified command
    vertices, then rate-limited from the previous command: the whole step
    from the previous command is scaled down until no axis changes faster
    than its limit.  The hull facets are built once here, and every check
    on the constants happens at construction, so ``limit`` launches a
    fixed set of kernels and never synchronizes with the host.
    """

    def __init__(
        self,
        *,
        vertices: Sequence[Sequence[float]],
        axis_ranges: Sequence[Sequence[float]],
        rate_limits_per_second: Sequence[float],
        high_control_dt: float,
        device: torch.device | str,
    ) -> None:
        self.vertices = torch.as_tensor(vertices, dtype=torch.float32, device=device)
        self.axis_ranges = torch.as_tensor(axis_ranges, dtype=torch.float32, device=device)
        rate_limits = torch.as_tensor(rate_limits_per_second, dtype=torch.float32, device=device)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices must have shape [V,3].")
        if self.axis_ranges.shape != (3, 2) or not bool(torch.isfinite(self.axis_ranges).all()):
            raise ValueError("axis_ranges must be a finite [3,2] tensor.")
        if bool((self.axis_ranges[:, 0] > self.axis_ranges[:, 1]).any()):
            raise ValueError("An axis range has minimum greater than maximum.")
        if rate_limits.shape != (3,) or bool((rate_limits <= 0.0).any()):
            raise ValueError("rate_limits_per_second must be a positive [3] tensor.")
        if high_control_dt <= 0.0:
            raise ValueError("high_control_dt must be positive.")
        self.normals, self.offsets = hull_facets(self.vertices)
        self.center = self.axis_ranges.mean(dim=1)
        self.half_width = (self.axis_ranges[:, 1] - self.axis_ranges[:, 0]) / 2.0
        self.permitted_step = rate_limits * float(high_control_dt)

    def project(self, command: torch.Tensor) -> torch.Tensor:
        """The nearest qualified command to each row of ``command``."""

        return project_onto_hull(command, self.vertices, self.normals, self.offsets)

    def limit(
        self, latent_action: torch.Tensor, previous_command: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map ``[N,3]`` latent actions to applied commands.

        ``previous_command`` must be a command this surface produced (or
        zero).  Returns the applied commands and the ``tanh``-bounded
        actions.  The caller checks that ``latent_action`` is finite.
        """

        bounded = torch.tanh(latent_action)
        target = self.project(self.center + bounded * self.half_width)
        # The previous command needs no projection: it is zero after a reset
        # (a vertex) or a step between two hull points, so it is in the hull.
        delta = target - previous_command
        ratios = torch.where(
            delta.abs() > 0.0,
            self.permitted_step / delta.abs(),
            torch.ones_like(delta),
        )
        scale = ratios.amin(dim=1).clamp(0.0, 1.0)
        return previous_command + scale.unsqueeze(1) * delta, bounded
