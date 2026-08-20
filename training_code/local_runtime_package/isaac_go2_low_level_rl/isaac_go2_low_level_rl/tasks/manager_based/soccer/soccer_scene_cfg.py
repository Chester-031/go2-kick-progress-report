"""Single-environment enclosed soccer field for Phase-2 physics tests."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass


FIELD_LENGTH = 8.0
FIELD_WIDTH = 5.0

FIELD_HALF_LENGTH = FIELD_LENGTH / 2.0
FIELD_HALF_WIDTH = FIELD_WIDTH / 2.0

GOAL_WIDTH = 1.2
GOAL_HALF_WIDTH = GOAL_WIDTH / 2.0
GOAL_DEPTH = 0.6

WALL_HEIGHT = 0.35
WALL_THICKNESS = 0.10
WALL_Z = WALL_HEIGHT / 2.0

BALL_RADIUS = 0.11
BALL_MASS = 0.43

GOAL_BACK_X = FIELD_HALF_LENGTH + GOAL_DEPTH

END_WALL_SEGMENT_LENGTH = (
    FIELD_WIDTH - GOAL_WIDTH
) / 2.0

END_WALL_SEGMENT_CENTER_Y = (
    GOAL_HALF_WIDTH
    + END_WALL_SEGMENT_LENGTH / 2.0
)


def _wall_cfg(
    prim_name: str,
    *,
    size: tuple[float, float, float],
    pos: tuple[float, float, float],
) -> AssetBaseCfg:
    """Create one fixed collision wall."""

    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="average",
                restitution_combine_mode="average",
                static_friction=0.8,
                dynamic_friction=0.6,
                restitution=0.20,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.15, 0.25, 0.55),
                roughness=0.8,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


@configclass
class SoccerPhysicsSceneCfg(InteractiveSceneCfg):
    """Frictional field, one ball, enclosing walls, and two goals."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(12.0, 8.0),
            color=(0.08, 0.28, 0.10),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=0.8,
                dynamic_friction=0.6,
                restitution=0.05,
            ),
        ),
    )

    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SoccerBall",
        spawn=sim_utils.SphereCfg(
            radius=BALL_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                linear_damping=0.12,
                angular_damping=0.05,
                max_linear_velocity=20.0,
                max_depenetration_velocity=5.0,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=BALL_MASS,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=0.5,
                dynamic_friction=0.4,
                restitution=0.25,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.92, 0.92, 0.92),
                roughness=0.65,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, BALL_RADIUS + 0.005),
        ),
    )

    # Complete side walls.
    side_wall_left = _wall_cfg(
        "SideWallLeft",
        size=(
            FIELD_LENGTH + 2.0 * WALL_THICKNESS,
            WALL_THICKNESS,
            WALL_HEIGHT,
        ),
        pos=(0.0, FIELD_HALF_WIDTH, WALL_Z),
    )

    side_wall_right = _wall_cfg(
        "SideWallRight",
        size=(
            FIELD_LENGTH + 2.0 * WALL_THICKNESS,
            WALL_THICKNESS,
            WALL_HEIGHT,
        ),
        pos=(0.0, -FIELD_HALF_WIDTH, WALL_Z),
    )

    # Positive-x end wall, split around the goal opening.
    positive_end_wall_upper = _wall_cfg(
        "PositiveEndWallUpper",
        size=(
            WALL_THICKNESS,
            END_WALL_SEGMENT_LENGTH,
            WALL_HEIGHT,
        ),
        pos=(
            FIELD_HALF_LENGTH,
            END_WALL_SEGMENT_CENTER_Y,
            WALL_Z,
        ),
    )

    positive_end_wall_lower = _wall_cfg(
        "PositiveEndWallLower",
        size=(
            WALL_THICKNESS,
            END_WALL_SEGMENT_LENGTH,
            WALL_HEIGHT,
        ),
        pos=(
            FIELD_HALF_LENGTH,
            -END_WALL_SEGMENT_CENTER_Y,
            WALL_Z,
        ),
    )

    # Negative-x end wall, split around the goal opening.
    negative_end_wall_upper = _wall_cfg(
        "NegativeEndWallUpper",
        size=(
            WALL_THICKNESS,
            END_WALL_SEGMENT_LENGTH,
            WALL_HEIGHT,
        ),
        pos=(
            -FIELD_HALF_LENGTH,
            END_WALL_SEGMENT_CENTER_Y,
            WALL_Z,
        ),
    )

    negative_end_wall_lower = _wall_cfg(
        "NegativeEndWallLower",
        size=(
            WALL_THICKNESS,
            END_WALL_SEGMENT_LENGTH,
            WALL_HEIGHT,
        ),
        pos=(
            -FIELD_HALF_LENGTH,
            -END_WALL_SEGMENT_CENTER_Y,
            WALL_Z,
        ),
    )

    # Positive-x goal enclosure.
    positive_goal_upper = _wall_cfg(
        "PositiveGoalUpper",
        size=(GOAL_DEPTH, WALL_THICKNESS, WALL_HEIGHT),
        pos=(
            FIELD_HALF_LENGTH + GOAL_DEPTH / 2.0,
            GOAL_HALF_WIDTH,
            WALL_Z,
        ),
    )

    positive_goal_lower = _wall_cfg(
        "PositiveGoalLower",
        size=(GOAL_DEPTH, WALL_THICKNESS, WALL_HEIGHT),
        pos=(
            FIELD_HALF_LENGTH + GOAL_DEPTH / 2.0,
            -GOAL_HALF_WIDTH,
            WALL_Z,
        ),
    )

    positive_goal_back = _wall_cfg(
        "PositiveGoalBack",
        size=(
            WALL_THICKNESS,
            GOAL_WIDTH + WALL_THICKNESS,
            WALL_HEIGHT,
        ),
        pos=(GOAL_BACK_X, 0.0, WALL_Z),
    )

    # Negative-x goal enclosure.
    negative_goal_upper = _wall_cfg(
        "NegativeGoalUpper",
        size=(GOAL_DEPTH, WALL_THICKNESS, WALL_HEIGHT),
        pos=(
            -FIELD_HALF_LENGTH - GOAL_DEPTH / 2.0,
            GOAL_HALF_WIDTH,
            WALL_Z,
        ),
    )

    negative_goal_lower = _wall_cfg(
        "NegativeGoalLower",
        size=(GOAL_DEPTH, WALL_THICKNESS, WALL_HEIGHT),
        pos=(
            -FIELD_HALF_LENGTH - GOAL_DEPTH / 2.0,
            -GOAL_HALF_WIDTH,
            WALL_Z,
        ),
    )

    negative_goal_back = _wall_cfg(
        "NegativeGoalBack",
        size=(
            WALL_THICKNESS,
            GOAL_WIDTH + WALL_THICKNESS,
            WALL_HEIGHT,
        ),
        pos=(-GOAL_BACK_X, 0.0, WALL_Z),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            color=(0.75, 0.75, 0.75),
        ),
    )
