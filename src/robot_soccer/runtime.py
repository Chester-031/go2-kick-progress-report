"""Build the Isaac Lab soccer environment and the RSL-RL configuration.

Three controls (``config.TRAINING_BY_CONTROL``): ``"position"`` and
``"velocity"`` train one policy straight on the 12 joints (the scene of
``direct_env_cfg``); ``"hierarchical"`` trains a 10 Hz command policy over a
frozen low-level policy (the Stage B scene, which needs Isaac Lab 2.3 and
the qualified low checkpoint).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from .config import (
    ACTIVATION,
    CLIP_PARAM,
    COMMAND_AXIS_RANGES,
    COMMAND_RATE_LIMITS_PER_SECOND,
    COMMAND_VERTICES,
    GRU_LAYERS,
    INITIAL_NOISE_STD,
    MAX_GRAD_NORM,
    NUM_LEARNING_EPOCHS,
    NUM_MINI_BATCHES,
    POLICY_OBSERVATION_DIM,
    SEED,
    TRAINING_BY_CONTROL,
    VALUE_LOSS_COEF,
    TrainingSettings,
)
from .low_policy import file_sha256, load_frozen_low_actor
from .resets import ResetSampler

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
    settings: TrainingSettings,
    device: str,
    iterations: int,
    num_steps: int,
    seed: int = SEED,
) -> dict[str, Any]:
    """The full RSL-RL 3.1 runner configuration for one control's
    ``settings``: a feed-forward or GRU actor-critic trained by PPO.

    Written out in full rather than derived from the Isaac Lab runner
    config class: every policy and algorithm field is set here, so the
    training configuration does not depend on Isaac Lab's defaults and can
    be built without Isaac Lab.
    """

    if settings.rnn_hidden_dim:
        policy: dict[str, Any] = {
            "class_name": "ActorCriticRecurrent",
            "rnn_type": "gru",
            "rnn_hidden_dim": settings.rnn_hidden_dim,
            "rnn_num_layers": GRU_LAYERS,
        }
    else:
        policy = {"class_name": "ActorCritic"}
    policy.update(
        {
            "actor_hidden_dims": list(settings.actor_hidden_dims),
            "critic_hidden_dims": list(settings.critic_hidden_dims),
            "activation": ACTIVATION,
            "init_noise_std": INITIAL_NOISE_STD,
            "actor_obs_normalization": False,
            "critic_obs_normalization": False,
            "noise_std_type": "scalar",
            "state_dependent_std": False,
        }
    )
    return {
        "seed": int(seed),
        "device": device,
        "num_steps_per_env": int(num_steps),
        "max_iterations": int(iterations),
        "save_interval": int(iterations) + 1,
        "experiment_name": "robot_soccer_adr",
        "run_name": f"seed_{int(seed)}",
        "logger": "tensorboard",
        "obs_groups": {"policy": ["policy"], "critic": ["policy"]},
        "clip_actions": None,
        "policy": policy,
        "algorithm": {
            "class_name": "PPO",
            "schedule": settings.schedule,
            "learning_rate": settings.learning_rate,
            "desired_kl": settings.desired_kl,
            "entropy_coef": settings.entropy_coef,
            "gamma": settings.gamma,
            "lam": settings.lam,
            "num_learning_epochs": NUM_LEARNING_EPOCHS,
            "num_mini_batches": NUM_MINI_BATCHES,
            "clip_param": CLIP_PARAM,
            "value_loss_coef": VALUE_LOSS_COEF,
            "use_clipped_value_loss": True,
            "max_grad_norm": MAX_GRAD_NORM,
            "normalize_advantage_per_mini_batch": False,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
    }


def make_environment(
    *,
    control: str,
    num_envs: int,
    seed: int,
    device: str,
    reset_sampler: ResetSampler,
    low_checkpoint: str | Path | None = None,
    low_checkpoint_sha256: str | None = None,
    render_mode: str | None = None,
    gamma: float | None = None,
):
    """Construct the soccer environment for ``control``, its RSL-RL adapter
    and the frozen low actor (``None`` for the direct-joint controls).
    Episode starts come from ``reset_sampler``; the shaping discount is
    ``gamma``, which must be the PPO discount (default: the control's).

    Isaac Lab must already be launched before this function is called.
    """

    from .controls import DirectJointControl, HeldCommandControl
    from .environment import (
        CurrentTaskProgressHighGoalEnv,
        CurrentTaskProgressHighRslAdapter,
        FrozenLowInferenceEnv,
        writable_command,
    )
    from .shaping import PotentialShaping, ShapingWeights

    if control not in TRAINING_BY_CONTROL:
        raise ValueError(f"control must be one of {tuple(TRAINING_BY_CONTROL)}")
    shaping = PotentialShaping(
        ShapingWeights(), gamma=TRAINING_BY_CONTROL[control].gamma if gamma is None else gamma
    )

    if control != "hierarchical":
        from isaaclab.envs import ManagerBasedRLEnv

        from .direct_env_cfg import direct_soccer_env_cfg

        scene = ManagerBasedRLEnv(
            cfg=direct_soccer_env_cfg(control=control, num_envs=num_envs, seed=seed, device=device),
            render_mode=render_mode,
        )
        env = CurrentTaskProgressHighGoalEnv(
            low_env=FrozenLowInferenceEnv(scene),
            control=DirectJointControl(num_envs=num_envs, device=torch.device(device)),
            reset_sampler=reset_sampler,
            shaping=shaping,
            # The scene's reward is its motion regularizers.
            native_reward_scale=1.0,
        )
        return env, CurrentTaskProgressHighRslAdapter(env), None

    import gymnasium as gym
    import isaac_go2_low_level_rl.tasks  # noqa: F401
    import isaac_go2_low_level_rl.tasks.manager_based.soccer  # noqa: F401

    from .commands import CommandSurface
    from .env_cfg import SparseHighEnvCfg

    if low_checkpoint is None or low_checkpoint_sha256 is None:
        raise ValueError("the hierarchical control needs --low-checkpoint and --low-checkpoint-sha256")
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
    # Native timeouts are disabled; the soccer environment owns the 20 s end.
    cfg.episode_length_s = 1000.0

    gym_env = gym.make(TASK_ID, cfg=cfg, render_mode=render_mode)
    low_env = FrozenLowInferenceEnv(gym_env)
    command_control = HeldCommandControl(
        low_actor=actor,
        command_surface=CommandSurface(
            vertices=COMMAND_VERTICES,
            axis_ranges=COMMAND_AXIS_RANGES,
            rate_limits_per_second=COMMAND_RATE_LIMITS_PER_SECOND,
            high_control_dt=0.1,
            device=device,
        ),
        command=writable_command(low_env.unwrapped),
        num_envs=num_envs,
        device=torch.device(device),
    )
    env = CurrentTaskProgressHighGoalEnv(
        low_env=low_env,
        control=command_control,
        reset_sampler=reset_sampler,
        shaping=shaping,
    )
    return env, CurrentTaskProgressHighRslAdapter(env), actor


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
    expected_checkpoint_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Strictly load high-policy weights from ``checkpoint`` and return the
    whole checkpoint payload (a resume also needs its optimizer state,
    iteration and ADR state).  When ``expected_checkpoint_sha256`` is given
    the file must match it."""

    path = Path(checkpoint).expanduser().resolve(strict=True)
    if expected_checkpoint_sha256 is not None:
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
    policy.load_state_dict(payload["model_state_dict"], strict=True)
    return payload


def save_high_policy(
    path: str | Path,
    *,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    adr_state: Mapping[str, Any],
) -> None:
    """Atomically save a resume-capable checkpoint: policy, optimizer,
    iteration and the ADR ranges and probe buffers."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iter": int(iteration),
            "adr_state": dict(adr_state),
            "infos": {"observation_dim": POLICY_OBSERVATION_DIM},
        },
        temporary,
    )
    temporary.replace(destination)
