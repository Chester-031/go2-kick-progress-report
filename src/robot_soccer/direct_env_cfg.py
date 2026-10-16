"""Isaac Lab scene for training soccer straight from the Go2's joints.

Built on Isaac Lab's stock flat-ground Go2 locomotion config (it exists
unchanged from Isaac Lab 2.0 on, so this runs on the cluster's install):
the same robot asset, actuator model, sensors and 48-D proprioceptive
observation the stock walking policies use.  On top of it this config

- adds the soccer field of ``soccer_scene_cfg.py`` (side walls, two goals,
  the 0.43 kg ball with rolling resistance) and a contact sensor on the
  ball filtered by every robot body;
- keeps only the stock motion regularizers as the scene's own reward
  (joint torque, joint acceleration and action rate, at the stock Go2
  weights): they are not potential-based, and deliberately bias the policy
  toward smooth, low-effort motion;
- removes every stock termination, command, push and timeout: the soccer
  environment (``environment.py``) owns episodes, rewards and resets;
- sets what the policy's 12 outputs mean (``control``):

  * ``"position"``: joint position targets around the standing pose
    (scale 0.25 rad), tracked by the stock Go2 PD drive (Kp 25, Kd 0.5);
  * ``"velocity"``: joint velocity targets (scale ``VELOCITY_ACTION_SCALE``
    rad/s) tracked with damping ``VELOCITY_DRIVE_DAMPING``, plus the stock
    spring (stiffness ``VELOCITY_DRIVE_STIFFNESS``, Kp 25) toward the standing
    pose so that, as in position control, zero action means "stand".  A
    damping-only drive (no spring) lets a fresh policy collapse into the
    joint limits within half a second, and Kp 10 sags it into a 0.17 m
    crouch where random commands hit the joint limits.  Gains are a first
    guess.

Importing this module needs Isaac Lab (launch the app first).
"""

from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

from isaaclab.sensors import ContactSensorCfg
from isaaclab.envs import mdp
from isaaclab.envs.mdp.actions.joint_actions import JointVelocityAction
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg

from .contact import exact_robot_filter_paths

CONTROLS = ("position", "velocity")
POSITION_ACTION_SCALE = 0.25
VELOCITY_ACTION_SCALE = 6.0
VELOCITY_DRIVE_DAMPING = 1.0
VELOCITY_DRIVE_STIFFNESS = 25.0
# The stock reward terms kept as motion regularizers.
REGULARIZERS = ("dof_torques_l2", "dof_acc_l2", "action_rate_l2")
FIELD_ASSETS = (
    "side_wall_left",
    "side_wall_right",
    "positive_end_wall_upper",
    "positive_end_wall_lower",
    "negative_end_wall_upper",
    "negative_end_wall_lower",
    "positive_goal_upper",
    "positive_goal_lower",
    "positive_goal_back",
    "negative_goal_upper",
    "negative_goal_lower",
    "negative_goal_back",
)


class StandingPoseVelocityAction(JointVelocityAction):
    """Joint velocity targets from the policy, with every joint's position
    target held at the standing pose (Isaac Lab starts position targets at
    0 rad, which a nonzero drive stiffness would pull the legs toward)."""

    def apply_actions(self) -> None:
        super().apply_actions()
        self._asset.set_joint_position_target(
            self._asset.data.default_joint_pos[:, self._joint_ids], joint_ids=self._joint_ids
        )


@configclass
class StandingPoseVelocityActionCfg(mdp.JointVelocityActionCfg):
    class_type: type = StandingPoseVelocityAction


def _field_template():
    """The soccer field scene of the runtime project, loaded from its file
    so that the package's task registrations (written for a newer Isaac
    Lab) are not imported."""

    path = (
        Path(__file__).resolve().parents[2]
        / "runtime_project/isaac_go2_low_level_rl/isaac_go2_low_level_rl/tasks/manager_based/soccer/soccer_scene_cfg.py"
    )
    spec = importlib.util.spec_from_file_location("robot_soccer_field_scene_cfg", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the soccer field config from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SoccerPhysicsSceneCfg(num_envs=1, env_spacing=12.0, replicate_physics=False)


@configclass
class DirectSoccerEnvCfg(UnitreeGo2FlatEnvCfg):
    """Stock Go2 + soccer field; set ``control`` before ``__post_init__``
    runs (use ``direct_soccer_env_cfg``)."""

    control: str = "position"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.control not in CONTROLS:
            raise ValueError(f"control must be one of {CONTROLS}")

        field = _field_template()
        self.scene.env_spacing = 12.0
        self.scene.ball = deepcopy(field.ball)
        ball = self.scene.ball.spawn
        # The robot-ball contact sensor lives on the ball.
        ball.activate_contact_sensors = True
        # Rolling and air resistance (Coulomb friction alone barely slows a
        # rolling ball), as in the validated Stage B field.
        ball.rigid_props.linear_damping = 0.35
        ball.rigid_props.angular_damping = 0.45
        ball.physics_material.static_friction = 0.90
        ball.physics_material.dynamic_friction = 0.70
        ball.physics_material.restitution = 0.02
        for name in FIELD_ASSETS:
            setattr(self.scene, name, deepcopy(getattr(field, name)))
        self.scene.robot_ball_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/SoccerBall",
            history_length=self.decimation,
            filter_prim_paths_expr=list(exact_robot_filter_paths()),
        )
        self.sim.physx.enable_ccd = True
        # When rendered (videos), look at env 0's whole field from the side.
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0
        self.viewer.eye = (0.0, -7.5, 6.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.viewer.resolution = (1280, 720)

        if self.control == "position":
            self.actions.joint_pos.scale = POSITION_ACTION_SCALE
            self.actions.joint_pos.use_default_offset = True
        else:
            self.actions.joint_pos = None
            self.actions.joint_vel = StandingPoseVelocityActionCfg(
                asset_name="robot",
                joint_names=[".*"],
                scale=VELOCITY_ACTION_SCALE,
                use_default_offset=False,
            )
            for actuator in self.scene.robot.actuators.values():
                actuator.stiffness = VELOCITY_DRIVE_STIFFNESS
                actuator.damping = VELOCITY_DRIVE_DAMPING

        # The soccer environment owns rewards, terminations and resets.
        for name in self.rewards.__dataclass_fields__:
            if name not in REGULARIZERS:
                setattr(self.rewards, name, None)
        for name in self.terminations.__dataclass_fields__:
            setattr(self.terminations, name, None)
        self.episode_length_s = 1000.0
        self.curriculum.terrain_levels = None
        command = self.commands.base_velocity
        command.heading_command = False
        command.rel_standing_envs = 1.0
        command.resampling_time_range = (1.0e9, 1.0e9)
        command.ranges.lin_vel_x = (0.0, 0.0)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (0.0, 0.0)
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        # Resets start from the standing pose; the soccer environment then
        # writes the ADR start.
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)


def direct_soccer_env_cfg(*, control: str, num_envs: int, seed: int, device: str) -> DirectSoccerEnvCfg:
    """The scene config for one training run."""

    cfg = DirectSoccerEnvCfg(control=control)
    cfg.scene.num_envs = int(num_envs)
    cfg.seed = int(seed)
    cfg.sim.device = device
    return cfg
