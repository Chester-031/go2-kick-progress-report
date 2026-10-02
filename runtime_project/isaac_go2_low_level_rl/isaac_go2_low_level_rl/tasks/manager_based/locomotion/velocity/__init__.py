"""Register the custom Go2 low-level locomotion environment."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Go2-Low-Level-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.go2_low_level_env_cfg:Go2LowLevelEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}:Go2LowLevelPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Go2-Low-Level-Round2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.go2_low_level_round2_env_cfg:"
            "Go2LowLevelRound2EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}:Go2LowLevelPPORunnerCfg"
        ),
    },
)
