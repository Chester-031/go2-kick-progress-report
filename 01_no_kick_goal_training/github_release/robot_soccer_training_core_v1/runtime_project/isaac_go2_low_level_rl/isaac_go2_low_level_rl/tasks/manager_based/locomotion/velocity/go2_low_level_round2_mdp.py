"""Round-2 command generator and reward terms for Go2 locomotion."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.envs.mdp.commands.commands_cfg import (
    UniformVelocityCommandCfg,
)
from isaaclab.envs.mdp.commands.velocity_command import (
    UniformVelocityCommand,
)
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


class AxisSeparatedVelocityCommand(UniformVelocityCommand):
    """Sample stand, pure-x, pure-y, or pure-yaw commands.

    Only one non-zero command component is active for each environment.
    This keeps one shared actor while making each locomotion primitive
    identifiable during the second curriculum round.
    """

    cfg: "AxisSeparatedVelocityCommandCfg"

    def __init__(
        self,
        cfg: "AxisSeparatedVelocityCommandCfg",
        env: ManagerBasedEnv,
    ):
        super().__init__(cfg, env)

        probabilities = torch.tensor(
            cfg.mode_probabilities,
            dtype=torch.float,
            device=self.device,
        )

        if probabilities.shape != (4,):
            raise ValueError(
                "mode_probabilities must contain four entries: "
                "(stand, x, y, yaw)."
            )

        if torch.any(probabilities < 0) or probabilities.sum() <= 0:
            raise ValueError(
                "mode_probabilities must be non-negative and have "
                "a positive sum."
            )

        self._mode_probabilities = (
            probabilities / probabilities.sum()
        )

        self._validate_nonzero_range(
            cfg.ranges.lin_vel_x,
            cfg.min_abs_velocity[0],
            "lin_vel_x",
        )
        self._validate_nonzero_range(
            cfg.ranges.lin_vel_y,
            cfg.min_abs_velocity[1],
            "lin_vel_y",
        )
        self._validate_nonzero_range(
            cfg.ranges.ang_vel_z,
            cfg.min_abs_velocity[2],
            "ang_vel_z",
        )

    @staticmethod
    def _validate_nonzero_range(
        limits: tuple[float, float],
        minimum_absolute_value: float,
        name: str,
    ) -> None:
        if minimum_absolute_value <= 0:
            raise ValueError(
                f"{name} minimum absolute value must be positive."
            )

        if (
            limits[0] > -minimum_absolute_value
            or limits[1] < minimum_absolute_value
        ):
            raise ValueError(
                f"{name} range {limits} must support both signs with "
                f"|command| >= {minimum_absolute_value}."
            )

    def _sample_signed_nonzero(
        self,
        count: int,
        limits: tuple[float, float],
        minimum_absolute_value: float,
    ) -> torch.Tensor:
        if count == 0:
            return torch.empty(0, device=self.device)

        positive = (
            torch.rand(count, device=self.device) >= 0.5
        )

        positive_limit = torch.full(
            (count,),
            float(limits[1]),
            device=self.device,
        )
        negative_limit = torch.full(
            (count,),
            abs(float(limits[0])),
            device=self.device,
        )

        maximum_magnitude = torch.where(
            positive,
            positive_limit,
            negative_limit,
        )

        magnitude = minimum_absolute_value + (
            torch.rand(count, device=self.device)
            * (maximum_magnitude - minimum_absolute_value)
        )

        return torch.where(positive, magnitude, -magnitude)

    def _resample_command(
        self,
        env_ids: Sequence[int],
    ) -> None:
        ids = torch.as_tensor(
            env_ids,
            dtype=torch.long,
            device=self.device,
        ).flatten()

        if ids.numel() == 0:
            return

        modes = torch.multinomial(
            self._mode_probabilities,
            num_samples=ids.numel(),
            replacement=True,
        )

        # Mode map:
        #   0 = stand
        #   1 = pure linear-x
        #   2 = pure linear-y
        #   3 = pure yaw
        self.vel_command_b[ids, :] = 0.0
        self.heading_target[ids] = 0.0
        self.is_heading_env[ids] = False
        self.is_standing_env[ids] = modes == 0

        x_ids = ids[modes == 1]
        y_ids = ids[modes == 2]
        yaw_ids = ids[modes == 3]

        self.vel_command_b[x_ids, 0] = (
            self._sample_signed_nonzero(
                x_ids.numel(),
                self.cfg.ranges.lin_vel_x,
                self.cfg.min_abs_velocity[0],
            )
        )

        self.vel_command_b[y_ids, 1] = (
            self._sample_signed_nonzero(
                y_ids.numel(),
                self.cfg.ranges.lin_vel_y,
                self.cfg.min_abs_velocity[1],
            )
        )

        self.vel_command_b[yaw_ids, 2] = (
            self._sample_signed_nonzero(
                yaw_ids.numel(),
                self.cfg.ranges.ang_vel_z,
                self.cfg.min_abs_velocity[2],
            )
        )


@configclass
class AxisSeparatedVelocityCommandCfg(
    UniformVelocityCommandCfg
):
    """Configuration for axis-separated velocity sampling."""

    class_type: type = AxisSeparatedVelocityCommand

    # stand, pure-x, pure-y, pure-yaw
    mode_probabilities: tuple[
        float,
        float,
        float,
        float,
    ] = (0.15, 0.35, 0.25, 0.25)

    # vx [m/s], vy [m/s], yaw rate [rad/s]
    min_abs_velocity: tuple[
        float,
        float,
        float,
    ] = (0.20, 0.15, 0.25)


def base_height_from_default_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize deviation from the robot's default root height."""

    asset: Articulation = env.scene[asset_cfg.name]

    height_error = (
        asset.data.root_pos_w[:, 2]
        - asset.data.default_root_state[:, 2]
    )

    return torch.square(height_error)
