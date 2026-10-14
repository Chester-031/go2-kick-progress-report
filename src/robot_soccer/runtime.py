"""Build the Isaac Lab environment and recurrent RSL-RL policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from .config import (
    ACTIVATION,
    ACTOR_HIDDEN_DIMS,
    CHECKPOINT_ITERATIONS,
    CLIP_PARAM,
    COMMAND_AXIS_RANGES,
    COMMAND_RATE_LIMITS_PER_SECOND,
    COMMAND_VERTICES,
    CRITIC_HIDDEN_DIMS,
    ENTROPY_COEF,
    GAMMA,
    GRU_HIDDEN_DIM,
    GRU_LAYERS,
    INITIAL_NOISE_STD,
    LAMBDA,
    LEARNING_RATE,
    LOW_STEPS_PER_HIGH_ACTION,
    MAX_GRAD_NORM,
    NUM_LEARNING_EPOCHS,
    NUM_MINI_BATCHES,
    POLICY_ACTION_DIM,
    POLICY_OBSERVATION_DIM,
    ROLLOUT_STEPS,
    SEED,
    TRAINING_ITERATIONS,
    VALUE_LOSS_COEF,
)
from .low_policy import file_sha256, load_frozen_low_actor
TASK_ID = "Isaac-Go2-Soccer-Milestone2-v0"
LOW_ACTOR_DIMS = (48, 512, 256, 128, 12)


def prepare_output_directory(path: str | Path) -> Path:
    """Create or accept an empty output directory and refuse mixed runs."""

    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise ValueError("output directory must not be a symbolic link")
    destination = requested.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError(f"output directory is not empty: {destination}")
    return destination


def runner_config(
    *,
    device: str,
    num_steps: int = ROLLOUT_STEPS,
    seed: int = SEED,
) -> dict[str, Any]:
    """Return the PPO configuration used by the reported training run."""

    from isaac_go2_low_level_rl.tasks.manager_based.locomotion.velocity.agents.rsl_rl_ppo_cfg import (
        Go2LowLevelPPORunnerCfg,
    )

    cfg = Go2LowLevelPPORunnerCfg().to_dict()
    cfg.update(
        {
            "seed": int(seed),
            "device": device,
            "num_steps_per_env": int(num_steps),
            "max_iterations": TRAINING_ITERATIONS,
            "save_interval": TRAINING_ITERATIONS + 1,
            "experiment_name": "robot_soccer_stage_b",
            "run_name": f"seed_{int(seed)}",
            "obs_groups": {"policy": ["policy"], "critic": ["policy"]},
            "clip_actions": None,
        }
    )
    cfg["policy"].update(
        {
            "class_name": "ActorCriticRecurrent",
            "rnn_type": "gru",
            "rnn_hidden_dim": GRU_HIDDEN_DIM,
            "rnn_num_layers": GRU_LAYERS,
            "actor_hidden_dims": list(ACTOR_HIDDEN_DIMS),
            "critic_hidden_dims": list(CRITIC_HIDDEN_DIMS),
            "activation": ACTIVATION,
            "init_noise_std": INITIAL_NOISE_STD,
            "actor_obs_normalization": False,
            "critic_obs_normalization": False,
            "noise_std_type": "scalar",
            "state_dependent_std": False,
        }
    )
    cfg["algorithm"].update(
        {
            "schedule": "fixed",
            "learning_rate": LEARNING_RATE,
            "desired_kl": None,
            "entropy_coef": ENTROPY_COEF,
            "gamma": GAMMA,
            "lam": LAMBDA,
            "num_learning_epochs": NUM_LEARNING_EPOCHS,
            "num_mini_batches": NUM_MINI_BATCHES,
            "clip_param": CLIP_PARAM,
            "value_loss_coef": VALUE_LOSS_COEF,
            "use_clipped_value_loss": True,
            "max_grad_norm": MAX_GRAD_NORM,
            "normalize_advantage_per_mini_batch": False,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        }
    )
    return cfg


def make_environment(
    *,
    low_checkpoint: str | Path,
    low_checkpoint_sha256: str,
    num_envs: int,
    seed: int,
    device: str,
    task: str = TASK_ID,
    render_mode: str | None = None,
):
    """Construct the frozen-low soccer environment.

    Isaac Lab must already be launched before this function is called.
    """

    import gymnasium as gym
    import isaac_go2_low_level_rl.tasks  # noqa: F401
    import isaac_go2_low_level_rl.tasks.manager_based.soccer  # noqa: F401

    from .env_cfg import SparseHighEnvCfg
    from .resets import StageBResetSampler
    from .environment import (
        CurrentTaskProgressHighGoalEnv,
        CurrentTaskProgressHighRslAdapter,
        FrozenLowInferenceEnv,
    )

    actor = load_frozen_low_actor(
        checkpoint=low_checkpoint,
        expected_checkpoint_sha256=low_checkpoint_sha256,
        expected_layer_dimensions=LOW_ACTOR_DIMS,
        activation="elu",
        device=device,
    )
    cfg = SparseHighEnvCfg()
    cfg.scene.num_envs = int(num_envs)
    cfg.seed = int(seed)
    cfg.sim.device = device
    cfg.observations.policy.enable_corruption = False
    cfg.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.events.push_robot = None
    for name in ("time_out", "base_contact", "actual_joint_hard_limit"):
        setattr(cfg.terminations, name, None)
    # Native timeouts are disabled; the high-level wrapper owns the 20 s end.
    cfg.episode_length_s = 1000.0

    gym_env = gym.make(task, cfg=cfg, render_mode=render_mode)
    env = CurrentTaskProgressHighGoalEnv(
        low_env=FrozenLowInferenceEnv(gym_env),
        frozen_low_actor=actor,
        reset_sampler=StageBResetSampler(int(seed)),
        command_vertices=COMMAND_VERTICES,
        command_axis_ranges=COMMAND_AXIS_RANGES,
        command_rate_limits=COMMAND_RATE_LIMITS_PER_SECOND,
        low_steps_per_high_action=LOW_STEPS_PER_HIGH_ACTION,
    )
    adapter = CurrentTaskProgressHighRslAdapter(env)
    return env, adapter, actor


def policy_module(runner: Any) -> torch.nn.Module:
    policy = getattr(runner.alg, "policy", None)
    if policy is None:
        policy = getattr(runner.alg, "actor_critic", None)
    if policy is None:
        raise RuntimeError("RSL-RL runner does not expose its policy")
    return policy


def clone_module_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
        if torch.is_tensor(value)
    }


def module_state_equal(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> bool:
    return set(left) == set(right) and all(torch.equal(left[name], right[name]) for name in left)


def load_high_policy(
    policy: torch.nn.Module,
    checkpoint: str | Path,
    *,
    expected_checkpoint_sha256: str,
    require_pre_action_initial: bool = False,
) -> int:
    """Verify and strictly load high-policy weights."""

    path = Path(checkpoint).expanduser().resolve(strict=True)
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_checkpoint_sha256:
        raise RuntimeError(
            "High checkpoint SHA256 mismatch: "
            f"expected={expected_checkpoint_sha256} actual={actual_sha256}"
        )
    payload = torch.load(
        path,
        map_location=next(policy.parameters()).device,
        weights_only=True,
    )
    if not isinstance(payload, Mapping) or "model_state_dict" not in payload:
        raise ValueError("checkpoint has no model_state_dict")
    if require_pre_action_initial:
        infos = payload.get("infos", {})
        if not isinstance(infos, Mapping) or (
            infos.get("architecture")
            != "fresh_recurrent_high_v56_task_progress_70d_3d"
            or infos.get("source") != "RUN_LOCAL_INITIAL_EVALUATION"
            or infos.get("kind") != "initial_before_any_act"
            or infos.get("v4d_weights_loaded") is not False
            or infos.get("teacher_loaded") is not False
            or int(payload.get("iter", -1)) != 0
            or payload.get("optimizer_state_dict") is not None
        ):
            raise RuntimeError("initial high checkpoint is not the pre-action parent")
    policy.load_state_dict(payload["model_state_dict"], strict=True)
    return int(payload.get("iter", 0))


def save_high_policy(
    path: str | Path,
    *,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
) -> None:
    """Atomically save a resume-capable high-policy checkpoint."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iter": int(iteration),
            "infos": {
                "observation_dim": POLICY_OBSERVATION_DIM,
                "action_dim": POLICY_ACTION_DIM,
                "policy_class": "ActorCriticRecurrent",
                "checkpoint_iterations": list(CHECKPOINT_ITERATIONS),
            },
        },
        temporary,
    )
    temporary.replace(destination)
