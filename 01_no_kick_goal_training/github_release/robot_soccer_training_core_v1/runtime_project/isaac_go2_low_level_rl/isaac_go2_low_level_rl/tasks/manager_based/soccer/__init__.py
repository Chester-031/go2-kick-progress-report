"""Register Phase-2 Go2 soccer environments."""

import gymnasium as gym

from ..locomotion.velocity import agents


TASK_ID = "Isaac-Go2-Soccer-Milestone2-v0"

if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{__name__}.soccer_milestone2_env_cfg:"
                "Go2SoccerMilestone2EnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                f"{agents.__name__}:"
                "Go2LowLevelPPORunnerCfg"
            ),
        },
    )
