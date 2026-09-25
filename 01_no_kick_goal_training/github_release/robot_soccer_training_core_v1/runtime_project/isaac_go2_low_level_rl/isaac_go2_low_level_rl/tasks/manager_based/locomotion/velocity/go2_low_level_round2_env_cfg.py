"""Round-2 isolated-command curriculum for Go2 locomotion."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import go2_low_level_round2_mdp as round2_mdp
from .go2_low_level_env_cfg import (
    CommandsCfg,
    Go2LowLevelEnvCfg,
    RewardsCfg,
)


@configclass
class Round2CommandsCfg(CommandsCfg):
    """Balanced isolated commands for one shared locomotion actor."""

    base_velocity = (
        round2_mdp.AxisSeparatedVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 6.0),
            rel_standing_envs=0.0,
            rel_heading_envs=0.0,
            heading_command=False,
            debug_vis=False,
            mode_probabilities=(0.15, 0.35, 0.25, 0.25),
            min_abs_velocity=(0.20, 0.15, 0.25),
            ranges=(
                round2_mdp.AxisSeparatedVelocityCommandCfg.Ranges(
                    lin_vel_x=(-0.65, 0.65),
                    lin_vel_y=(-0.45, 0.45),
                    ang_vel_z=(-0.85, 0.85),
                    heading=(-3.141592653589793, 3.141592653589793),
                )
            ),
        )
    )


@configclass
class Round2RewardsCfg(RewardsCfg):
    """Round-2 reward additions."""

    base_height_l2 = RewTerm(
        func=round2_mdp.base_height_from_default_l2,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class Go2LowLevelRound2EnvCfg(Go2LowLevelEnvCfg):
    """Continue the same actor with isolated velocity commands."""

    commands: Round2CommandsCfg = Round2CommandsCfg()
    rewards: Round2RewardsCfg = Round2RewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # Round 2 first learns clear command-conditioned motion.
        # Disturbance recovery returns after command tracking passes.
        self.events.push_robot = None

        # Increase command discrimination and discourage crouched,
        # abrupt solutions without prescribing a gait.
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.flat_orientation_l2.weight = -3.0
        self.rewards.action_rate_l2.weight = -0.02
