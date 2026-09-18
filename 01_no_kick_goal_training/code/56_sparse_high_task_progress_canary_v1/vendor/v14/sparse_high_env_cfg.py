"""Isaac soccer configuration for a frozen 50 Hz low and 10 Hz high actor."""

from __future__ import annotations

import math

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaac_go2_low_level_rl.tasks.manager_based.locomotion.velocity.go2_low_level_env_cfg import (
    TerminationsCfg,
)
from isaac_go2_low_level_rl.tasks.manager_based.soccer.soccer_milestone2_env_cfg import (
    Go2SoccerMilestone2EnvCfg,
    Go2SoccerMilestone2SceneCfg,
)

import native_low_mdp
from native_safe_joint_action import NativeSafeJointPositionAction
from staging_contact_contract import exact_robot_filter_paths, validate_configured_robot_filter_paths


@configclass
class SparseHighSceneCfg(Go2SoccerMilestone2SceneCfg):
    robot_ball_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/SoccerBall",
        filter_prim_paths_expr=list(exact_robot_filter_paths()),
        history_length=4,
        track_air_time=False,
        force_threshold=1.0,
    )


@configclass
class SparseHighTerminationsCfg(TerminationsCfg):
    actual_joint_hard_limit = DoneTerm(
        func=native_low_mdp.actual_joint_hard_limit,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "tolerance_rad": 1.0e-5,
        },
    )


@configclass
class SparseHighEnvCfg(Go2SoccerMilestone2EnvCfg):
    scene: SparseHighSceneCfg = SparseHighSceneCfg(num_envs=1, env_spacing=12.0)
    terminations: SparseHighTerminationsCfg = SparseHighTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.joint_pos.class_type = NativeSafeJointPositionAction
        self.scene.ball.spawn.activate_contact_sensors = True
        self.scene.robot_ball_contact.update_period = self.sim.dt
        self.events.push_robot = None
        self.episode_length_s = 120.0
        if self.decimation != 4 or not math.isclose(self.sim.dt, 0.005):
            raise RuntimeError("Frozen-low control must remain at exactly 50 Hz.")
        if self.scene.robot_ball_contact.history_length != 4:
            raise RuntimeError("Contact history must cover one complete low action.")
        validate_configured_robot_filter_paths(
            self.scene.robot_ball_contact.filter_prim_paths_expr
        )

