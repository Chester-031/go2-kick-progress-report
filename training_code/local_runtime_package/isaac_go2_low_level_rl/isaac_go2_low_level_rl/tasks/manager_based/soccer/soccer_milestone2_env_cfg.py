"""Deterministic Go2 soccer environment for Milestone-2 visualization."""

from __future__ import annotations

from copy import deepcopy

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from ..locomotion.velocity.go2_low_level_env_cfg import (
    Go2LowLevelSceneCfg,
)
from ..locomotion.velocity.go2_low_level_round2_env_cfg import (
    Go2LowLevelRound2EnvCfg,
)
from .soccer_scene_cfg import SoccerPhysicsSceneCfg




# Isaac Lab configclass fields are instance attributes.
# Use one template instance and deepcopy every asset config into the
# combined Go2 + soccer scene.
_SOCCER_SCENE_TEMPLATE = SoccerPhysicsSceneCfg(
    num_envs=1,
    env_spacing=12.0,
    replicate_physics=False,
)


@configclass
class Go2SoccerMilestone2SceneCfg(Go2LowLevelSceneCfg):
    """Frozen Go2 locomotion scene with the validated soccer field."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.8,
            dynamic_friction=0.6,
            restitution=0.05,
        ),
        debug_vis=False,
    )

    ball: RigidObjectCfg = deepcopy(_SOCCER_SCENE_TEMPLATE.ball)

    side_wall_left: AssetBaseCfg = (
        deepcopy(_SOCCER_SCENE_TEMPLATE.side_wall_left)
    )
    side_wall_right: AssetBaseCfg = (
        deepcopy(_SOCCER_SCENE_TEMPLATE.side_wall_right)
    )

    positive_end_wall_upper: AssetBaseCfg = (
        deepcopy(_SOCCER_SCENE_TEMPLATE.positive_end_wall_upper)
    )
    positive_end_wall_lower: AssetBaseCfg = (
        deepcopy(_SOCCER_SCENE_TEMPLATE.positive_end_wall_lower)
    )
    negative_end_wall_upper: AssetBaseCfg = (
        deepcopy(_SOCCER_SCENE_TEMPLATE.negative_end_wall_upper)
    )
    negative_end_wall_lower: AssetBaseCfg = (
        deepcopy(_SOCCER_SCENE_TEMPLATE.negative_end_wall_lower)
    )

    positive_goal_upper: AssetBaseCfg = (
        deepcopy(_SOCCER_SCENE_TEMPLATE.positive_goal_upper)
    )
    positive_goal_lower: AssetBaseCfg = (
        deepcopy(_SOCCER_SCENE_TEMPLATE.positive_goal_lower)
    )
    positive_goal_back: AssetBaseCfg = (
        deepcopy(_SOCCER_SCENE_TEMPLATE.positive_goal_back)
    )

    negative_goal_upper: AssetBaseCfg = (
        deepcopy(_SOCCER_SCENE_TEMPLATE.negative_goal_upper)
    )
    negative_goal_lower: AssetBaseCfg = (
        deepcopy(_SOCCER_SCENE_TEMPLATE.negative_goal_lower)
    )
    negative_goal_back: AssetBaseCfg = (
        deepcopy(_SOCCER_SCENE_TEMPLATE.negative_goal_back)
    )


@configclass
class Go2SoccerMilestone2EnvCfg(Go2LowLevelRound2EnvCfg):
    """Single-Go2 deterministic soccer visualization environment."""

    scene: Go2SoccerMilestone2SceneCfg = (
        Go2SoccerMilestone2SceneCfg(
            num_envs=1,
            env_spacing=12.0,
        )
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        # M2_BALL_RESISTANCE_V2
        # Coulomb friction alone does not reliably dissipate energy
        # during near-pure rolling. Add moderate linear/angular
        # damping to represent rolling and aerodynamic resistance.
        ball_spawn = self.scene.ball.spawn

        if ball_spawn.rigid_props is None:
            raise RuntimeError(
                "Soccer ball rigid_props must be configured."
            )

        if ball_spawn.physics_material is None:
            raise RuntimeError(
                "Soccer ball physics_material must be configured."
            )

        ball_spawn.rigid_props.linear_damping = 0.35
        ball_spawn.rigid_props.angular_damping = 0.45

        ball_spawn.physics_material.static_friction = 0.90
        ball_spawn.physics_material.dynamic_friction = 0.70
        ball_spawn.physics_material.restitution = 0.02


        self.scene.num_envs = 1
        self.scene.env_spacing = 12.0

        # Deterministic evaluation: no training-time randomization.
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.push_robot = None

        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_robot_joints.params[
            "position_range"
        ] = (1.0, 1.0)

        self.observations.policy.enable_corruption = False

        # Enough time for turn, approach, push, and a short goal hold.
        self.episode_length_s = 30.0

        # Retain validated collision behavior for the soccer ball.
        self.sim.physx.enable_ccd = True
        self.sim.render_interval = self.decimation
