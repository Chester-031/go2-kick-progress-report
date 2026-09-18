"""Shared V56 runtime construction, state identity and durable IO."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
from types import MethodType
from typing import Any, Mapping

import gymnasium as gym
import torch
from isaac_go2_low_level_rl.tasks.manager_based.locomotion.velocity.agents.rsl_rl_ppo_cfg import (
    Go2LowLevelPPORunnerCfg,
)

from canary_protocol import canonical_json_bytes, semantic_sha256
from check_frozen_low_state import module_state_snapshot
from current_task_progress_env import (
    CurrentTaskProgressHighGoalEnv,
    CurrentTaskProgressHighRslAdapter,
    FrozenLowInferenceEnv,
)
from frozen_low_actor import load_frozen_low_actor
from sparse_high_env_cfg import SparseHighEnvCfg


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n",
    )


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_bytes(path, b"".join(canonical_json_bytes(row) for row in rows))


def _module_origins(prefix: str, root: Path) -> dict[str, list[str]]:
    origins: dict[str, list[str]] = {}
    for name, module in sorted(tuple(sys.modules.items())):
        if name != prefix and not name.startswith(prefix + "."):
            continue
        if module is None:
            raise RuntimeError(f"runtime module is unresolved: {name}")
        candidates: list[Path] = []
        file_name = getattr(module, "__file__", None)
        if file_name:
            candidates.append(Path(str(file_name)).resolve(strict=True))
        package_paths = getattr(module, "__path__", None)
        if package_paths is not None:
            candidates.extend(Path(str(item)).resolve(strict=True) for item in package_paths)
        resolved = sorted(set(candidates), key=str)
        if not resolved:
            raise RuntimeError(f"runtime module has no auditable origin: {name}")
        for path in resolved:
            if not path.is_relative_to(root):
                raise RuntimeError(f"runtime import escaped frozen root: {name} -> {path}")
        origins[name] = [str(path.relative_to(root)) for path in resolved]
    if prefix not in origins:
        raise RuntimeError(f"top-level package was not imported: {prefix}")
    return origins


def runtime_provenance(contract: Mapping[str, Any]) -> dict[str, Any]:
    parent = contract["parent_v55"]
    environment_parent = {
        "run_label": os.environ.get("V56_PARENT_V55_RUN_LABEL", ""),
        "result_sha256": os.environ.get("V56_PARENT_V55_RESULT_SHA256", ""),
        "teacher_fixed_summary_sha256": os.environ.get("V56_PARENT_V55_TEACHER_FIXED_SUMMARY_SHA256", ""),
        "teacher_fixed_rows_sha256": os.environ.get("V56_PARENT_V55_TEACHER_FIXED_ROWS_SHA256", ""),
        "transaction_status_sha256": os.environ.get("V56_PARENT_V55_TRANSACTION_STATUS_SHA256", ""),
        "task_manifest_sha256": os.environ.get("V56_PARENT_V55_TASK_MANIFEST_SHA256", ""),
    }
    for key, actual in environment_parent.items():
        expected = parent.get(key)
        if not isinstance(expected, str) or not expected or actual != expected:
            raise RuntimeError(f"exact parent V55 own identity changed: {key}")
    task_root_raw = os.environ.get("V56_TASK_PACKAGE_SNAPSHOT")
    if not task_root_raw:
        raise RuntimeError("V56_TASK_PACKAGE_SNAPSHOT is not set")
    task_root = Path(task_root_raw).resolve(strict=True)
    task_origins = _module_origins("isaac_go2_low_level_rl", task_root)
    required_task = {
        "isaac_go2_low_level_rl",
        "isaac_go2_low_level_rl.tasks",
        "isaac_go2_low_level_rl.tasks.manager_based.soccer",
    }
    if not required_task <= set(task_origins):
        raise RuntimeError("required task modules were not imported from snapshot")
    task_receipt = {
        "status": "PASS",
        "distribution_root": str(task_root),
        "loaded_module_count": len(task_origins),
        "module_origins_sha256": semantic_sha256(task_origins),
    }
    isaac_receipts: dict[str, Any] = {}
    for package, environment_name in {
        "isaaclab": "V56_ISAACLAB_DISTRIBUTION_ROOT",
        "isaaclab_tasks": "V56_ISAACLAB_TASKS_DISTRIBUTION_ROOT",
        "isaaclab_assets": "V56_ISAACLAB_ASSETS_DISTRIBUTION_ROOT",
        "isaaclab_rl": "V56_ISAACLAB_RL_DISTRIBUTION_ROOT",
    }.items():
        raw = os.environ.get(environment_name)
        if not raw:
            raise RuntimeError(f"{environment_name} is not set")
        root = Path(raw).resolve(strict=True)
        importlib.import_module(package)
        origins = _module_origins(package, root)
        isaac_receipts[package] = {
            "status": "PASS",
            "distribution_root": str(root),
            "loaded_module_count": len(origins),
            "module_origins_sha256": semantic_sha256(origins),
        }
    rsl_root_raw = os.environ.get("V56_RSL_RL_DISTRIBUTION_ROOT")
    if not rsl_root_raw:
        raise RuntimeError("V56_RSL_RL_DISTRIBUTION_ROOT is not set")
    rsl_root = Path(rsl_root_raw).resolve(strict=True)
    importlib.import_module("rsl_rl")
    rsl_origins = _module_origins("rsl_rl", rsl_root)
    distributions = importlib.metadata.packages_distributions().get("rsl_rl", [])
    versions = sorted({importlib.metadata.version(name) for name in distributions})
    if versions != ["3.1.2"]:
        raise RuntimeError(f"rsl_rl distribution version changed: {versions}")
    rsl_receipt = {
        "status": "PASS",
        "distribution_root": str(rsl_root),
        "loaded_module_count": len(rsl_origins),
        "module_origins_sha256": semantic_sha256(rsl_origins),
        "version": versions[0],
    }
    return {
        "parent_v55_identity": environment_parent,
        "task_package_import_origin": task_receipt,
        "isaac_import_origins": isaac_receipts,
        "rsl_rl_import_origin": rsl_receipt,
    }


def module_bit_state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    state = module.state_dict()
    for name in sorted(state):
        value = state[name]
        if not torch.is_tensor(value):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def action_std_report(policy: torch.nn.Module, *, require_initial_unit: bool) -> dict[str, Any]:
    value = getattr(policy, "std", None)
    if not torch.is_tensor(value):
        raise RuntimeError("policy does not expose its marginal action std")
    flat = value.detach().reshape(-1)
    if flat.numel() != 3 or not bool(torch.isfinite(flat).all()) or not bool((flat > 0.0).all()):
        raise RuntimeError("policy marginal action std is not finite positive 3-D")
    if require_initial_unit and not bool(torch.allclose(flat, torch.ones_like(flat), rtol=0.0, atol=1.0e-7)):
        raise RuntimeError(f"fresh policy marginal action std changed: {flat.tolist()}")
    return {
        "values": flat.cpu().tolist(),
        "shape": [3],
        "all_finite_positive": True,
        "unit_initial_required": bool(require_initial_unit),
        "unit_initial_pass": bool(torch.allclose(flat, torch.ones_like(flat), rtol=0.0, atol=1.0e-7)),
    }


def output_head_report(policy: torch.nn.Module) -> dict[str, Any]:
    def report(module: torch.nn.Module, label: str, expected_out: int) -> dict[str, Any]:
        linears = [item for item in module.modules() if isinstance(item, torch.nn.Linear)]
        if not linears or int(linears[-1].out_features) != expected_out:
            raise RuntimeError(f"{label} output head shape changed")
        weight = linears[-1].weight.detach()
        if not bool(torch.isfinite(weight).all()):
            raise RuntimeError(f"{label} output head is non-finite")
        l2 = float(torch.linalg.vector_norm(weight.float()).item())
        maximum = float(weight.abs().max().item())
        return {
            "weight_shape": list(weight.shape),
            "weight_l2": l2,
            "weight_max_abs": maximum,
            "weight_nonzero": bool(torch.count_nonzero(weight).item()),
        }

    actor = report(policy.actor, "actor", 3)
    critic = report(policy.critic, "critic", 1)
    if not actor["weight_nonzero"] or actor["weight_l2"] <= 0.0:
        raise RuntimeError("fresh actor output head is zero initialized")
    return {"actor": actor, "critic": critic}


def clone_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
        if torch.is_tensor(value)
    }


def state_finite(state: Mapping[str, torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in state.values())


def component_update(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    prefixes: tuple[str, ...],
) -> dict[str, Any]:
    names = sorted(name for name in before if name.startswith(prefixes))
    if not names or any(name not in after for name in names):
        raise RuntimeError(f"policy component schema changed: {prefixes}")
    deltas = [(after[name] - before[name]).float() for name in names]
    return {
        "parameter_names": names,
        "parameter_count": sum(value.numel() for value in deltas),
        "changed": any(bool(torch.count_nonzero(value).item()) for value in deltas),
        "maximum_absolute_delta": max(float(value.abs().max().item()) for value in deltas),
        "l2_delta": float(torch.sqrt(sum(torch.sum(value * value) for value in deltas)).item()),
    }


def policy_module(runner: Any) -> torch.nn.Module:
    value = getattr(runner.alg, "policy", None)
    if value is None:
        value = getattr(runner.alg, "actor_critic", None)
    if value is None:
        raise RuntimeError("could not locate recurrent high policy")
    return value


def install_counted_normalization(policy: torch.nn.Module) -> dict[str, Any]:
    """Count the RSL lifecycle call while keeping disabled normalization inert."""

    if bool(getattr(policy, "actor_obs_normalization", True)) or bool(
        getattr(policy, "critic_obs_normalization", True)
    ):
        raise RuntimeError("counted no-op normalization requires both normalizers disabled")
    original = getattr(policy, "update_normalization", None)
    original_callable = callable(original)
    policy.normalization_update_calls = 0

    def counted(self: torch.nn.Module, observation: Any) -> None:
        tensor = observation
        if isinstance(observation, Mapping):
            tensor = observation.get("policy")
        if hasattr(observation, "keys") and not torch.is_tensor(tensor):
            tensor = observation["policy"]
        if not torch.is_tensor(tensor) or tensor.ndim != 2 or tuple(tensor.shape)[1:] != (70,):
            raise RuntimeError("normalization lifecycle observation shape changed")
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError("normalization lifecycle observation is non-finite")
        self.normalization_update_calls += 1
        if original_callable:
            original(observation)

    policy.update_normalization = MethodType(counted, policy)
    return {
        "installed": True,
        "original_method_present": original_callable,
        "normalization_enabled": False,
        "semantics": "COUNTED_DISABLED_NORMALIZATION_LIFECYCLE",
    }


def runner_cfg(*, contract: Mapping[str, Any], device: str) -> dict[str, Any]:
    run = contract["bounded_canary"]
    architecture = contract["architecture"]
    ppo = contract["ppo"]
    cfg = Go2LowLevelPPORunnerCfg().to_dict()
    cfg.update(
        {
            "seed": int(run["policy_initialization_seed"]),
            "device": device,
            "num_steps_per_env": int(run["num_steps_per_env"]),
            "max_iterations": int(run["learning_iterations"]),
            "save_interval": int(run["learning_iterations"]) + 1,
            "experiment_name": "go2_sparse_high_task_progress_canary_v1",
            "run_name": f"fresh_task_progress_high_seed{run['policy_initialization_seed']}",
            "obs_groups": {"policy": ["policy"], "critic": ["policy"]},
            "clip_actions": None,
        }
    )
    cfg["policy"].update(
        {
            "class_name": "ActorCriticRecurrent",
            "rnn_type": "gru",
            "rnn_hidden_dim": int(architecture["gru_hidden_dim"]),
            "rnn_num_layers": int(architecture["gru_layers"]),
            "actor_hidden_dims": list(architecture["actor_hidden_dims"]),
            "critic_hidden_dims": list(architecture["critic_hidden_dims"]),
            "activation": str(architecture["activation"]),
            "init_noise_std": float(ppo["initial_noise_std"]),
            "actor_obs_normalization": False,
            "critic_obs_normalization": False,
            "noise_std_type": "scalar",
            "state_dependent_std": False,
        }
    )
    cfg["algorithm"].update(
        {
            "schedule": str(ppo["schedule"]),
            "learning_rate": float(ppo["learning_rate"]),
            "desired_kl": ppo["desired_kl"],
            "entropy_coef": float(ppo["entropy_coef"]),
            "gamma": float(ppo["gamma"]),
            "lam": float(ppo["lambda"]),
            "num_learning_epochs": int(ppo["num_learning_epochs"]),
            "num_mini_batches": int(ppo["num_mini_batches"]),
            "clip_param": float(ppo["clip_param"]),
            "value_loss_coef": float(ppo["value_loss_coef"]),
            "use_clipped_value_loss": bool(ppo["use_clipped_value_loss"]),
            "max_grad_norm": float(ppo["max_grad_norm"]),
            "normalize_advantage_per_mini_batch": bool(ppo["normalize_advantage_per_mini_batch"]),
            "rnd_cfg": None,
            "symmetry_cfg": None,
        }
    )
    return cfg


def runner_identity(runner: Any, policy: torch.nn.Module, contract: Mapping[str, Any]) -> dict[str, Any]:
    algorithm = runner.alg
    actor_memory = getattr(getattr(policy, "memory_a", None), "rnn", None)
    critic_memory = getattr(getattr(policy, "memory_c", None), "rnn", None)
    if not isinstance(actor_memory, torch.nn.GRU) or not isinstance(critic_memory, torch.nn.GRU):
        raise RuntimeError("runtime recurrent modules are not GRU")

    def linear_dimensions(module: torch.nn.Module) -> list[int]:
        linears = [item for item in module.modules() if isinstance(item, torch.nn.Linear)]
        if not linears:
            raise RuntimeError("runtime MLP has no Linear layers")
        dimensions = [int(linears[0].in_features)] + [int(item.out_features) for item in linears]
        for left, right in zip(linears, linears[1:], strict=False):
            if int(left.out_features) != int(right.in_features):
                raise RuntimeError("runtime MLP linear chain changed")
        return dimensions

    actor_dimensions = linear_dimensions(policy.actor)
    critic_dimensions = linear_dimensions(policy.critic)
    report = {
        "policy_class": type(policy).__name__,
        "is_recurrent": bool(getattr(policy, "is_recurrent", False)),
        "observation_dim": int(actor_memory.input_size),
        "critic_observation_dim": int(critic_memory.input_size),
        "action_dim": int(getattr(runner.env, "num_actions", -1)),
        "rnn_type": "gru",
        "actor_gru_hidden_dim": int(actor_memory.hidden_size),
        "critic_gru_hidden_dim": int(critic_memory.hidden_size),
        "actor_gru_layers": int(actor_memory.num_layers),
        "critic_gru_layers": int(critic_memory.num_layers),
        "actor_linear_dimensions": actor_dimensions,
        "critic_linear_dimensions": critic_dimensions,
        "num_steps_per_env": int(runner.num_steps_per_env),
        "schedule": str(algorithm.schedule),
        "learning_rate": float(algorithm.learning_rate),
        "desired_kl": algorithm.desired_kl,
        "entropy_coef": float(algorithm.entropy_coef),
        "gamma": float(algorithm.gamma),
        "lambda": float(algorithm.lam),
        "num_learning_epochs": int(algorithm.num_learning_epochs),
        "num_mini_batches": int(algorithm.num_mini_batches),
        "actor_observation_normalization": bool(policy.actor_obs_normalization),
        "critic_observation_normalization": bool(policy.critic_obs_normalization),
        "correlated_or_ar_exploration": False,
        "noise_std_type": str(policy.noise_std_type),
        "state_dependent_std": bool(policy.state_dependent_std),
        "clip_param": float(algorithm.clip_param),
        "value_loss_coef": float(algorithm.value_loss_coef),
        "use_clipped_value_loss": bool(algorithm.use_clipped_value_loss),
        "max_grad_norm": float(algorithm.max_grad_norm),
        "normalize_advantage_per_mini_batch": bool(algorithm.normalize_advantage_per_mini_batch),
        "rnd_registered": getattr(algorithm, "rnd", None) is not None,
        "symmetry_registered": getattr(algorithm, "symmetry", None) is not None,
    }
    expected = {
        "policy_class": "ActorCriticRecurrent",
        "is_recurrent": True,
        "observation_dim": 70,
        "critic_observation_dim": 70,
        "action_dim": 3,
        "rnn_type": "gru",
        "actor_gru_hidden_dim": 256,
        "critic_gru_hidden_dim": 256,
        "actor_gru_layers": 1,
        "critic_gru_layers": 1,
        "actor_linear_dimensions": [256, 256, 256, 128, 3],
        "critic_linear_dimensions": [256, 256, 256, 128, 1],
        "num_steps_per_env": 200,
        "schedule": "fixed",
        "learning_rate": 0.0003,
        "desired_kl": None,
        "entropy_coef": 0.03,
        "gamma": 0.997,
        "lambda": 0.98,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "actor_observation_normalization": False,
        "critic_observation_normalization": False,
        "correlated_or_ar_exploration": False,
        "noise_std_type": "scalar",
        "state_dependent_std": False,
        "clip_param": 0.2,
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "max_grad_norm": 1.0,
        "normalize_advantage_per_mini_batch": False,
        "rnd_registered": False,
        "symmetry_registered": False,
    }
    for key, required in expected.items():
        actual = report[key]
        if isinstance(required, float):
            if not math.isclose(float(actual), required, rel_tol=0.0, abs_tol=1.0e-12):
                raise RuntimeError(f"runner contract changed: {key}={actual}")
        elif actual != required:
            raise RuntimeError(f"runner contract changed: {key}={actual}")
    if int(contract["bounded_canary"]["policy_initialization_seed"]) != 83001:
        raise RuntimeError("policy initialization seed changed")
    return report


def make_runtime(
    *,
    contract: Mapping[str, Any],
    num_envs: int,
    reset_seed: int,
    device: str,
    task: str,
    evaluation_case_count: int | None = None,
    capture_transition_rows: bool = False,
) -> tuple[CurrentTaskProgressHighGoalEnv, CurrentTaskProgressHighRslAdapter, torch.nn.Module, dict[str, Any], dict[str, Any]]:
    low = contract["qualified_low"]
    checkpoint = Path(str(low["checkpoint_cluster_path"]))
    if checkpoint.is_symlink():
        raise RuntimeError("qualified low checkpoint is a symlink")
    checkpoint = checkpoint.resolve(strict=True)
    before_stat = checkpoint.stat()
    if before_stat.st_size != int(low["checkpoint_bytes"]):
        raise RuntimeError("qualified low checkpoint byte count changed")
    if file_sha256(checkpoint) != low["checkpoint_sha256"]:
        raise RuntimeError("qualified low checkpoint SHA256 changed")
    actor = load_frozen_low_actor(
        checkpoint=checkpoint,
        expected_checkpoint_sha256=low["checkpoint_sha256"],
        expected_layer_dimensions=low["actor_dimensions"],
        activation=low["actor_activation"],
        device=device,
    )
    low_before = module_state_snapshot(actor)
    if low_before["sha256"] != low["runtime_module_state_sha256"]:
        raise RuntimeError("qualified low runtime module state changed")
    cfg = SparseHighEnvCfg()
    cfg.scene.num_envs = int(num_envs)
    cfg.seed = int(reset_seed)
    cfg.sim.device = device
    if not hasattr(cfg.observations.policy, "enable_corruption"):
        raise RuntimeError("policy corruption flag is unavailable")
    cfg.observations.policy.enable_corruption = False
    cfg.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.events.push_robot = None
    for name in ("time_out", "base_contact", "actual_joint_hard_limit"):
        if not hasattr(cfg.terminations, name):
            raise RuntimeError(f"native termination term missing: {name}")
        setattr(cfg.terminations, name, None)
    cfg.episode_length_s = 1000.0
    gym_env = gym.make(task, cfg=cfg, render_mode=None)
    base = gym_env.unwrapped
    manager = base.termination_manager
    active_terms = getattr(manager, "active_terms", None)
    if active_terms is None:
        terms = getattr(manager, "_terms", {})
        active_terms = list(terms) if isinstance(terms, dict) else []
    forbidden = {"time_out", "base_contact", "actual_joint_hard_limit"}
    if forbidden.intersection(str(item) for item in active_terms):
        raise RuntimeError("native auto-termination remained active")
    run = contract["bounded_canary"]
    env = CurrentTaskProgressHighGoalEnv(
        low_env=FrozenLowInferenceEnv(gym_env),
        frozen_low_actor=actor,
        seed=int(reset_seed),
        command_vertices=contract["command_interface"]["vertices"],
        command_axis_ranges=contract["command_interface"]["axis_ranges"],
        command_rate_limits=contract["command_interface"]["rate_limits_per_second"],
        episode_seconds=float(contract["task"]["episode_seconds"]),
        low_steps_per_high_action=int(contract["architecture"]["low_steps_per_high_action"]),
        evaluation_case_count=evaluation_case_count,
        evaluation_hard_case_count=(int(run["hard_case_count"]) if evaluation_case_count is not None else 0),
        capture_transition_rows=capture_transition_rows,
    )
    checkpoint_identity = {
        "basename": checkpoint.name,
        "bytes": before_stat.st_size,
        "sha256": low["checkpoint_sha256"],
        "runtime_module_state_sha256": low_before["sha256"],
    }
    runtime_identity = {
        "num_envs": int(num_envs),
        "reset_seed": int(reset_seed),
        "sim_dt_seconds": float(base.cfg.sim.dt),
        "decimation": int(base.cfg.decimation),
        "high_policy_hz": 10,
        "low_policy_hz": 50,
        "low_steps_per_high_action": 5,
        "finite_horizon": True,
        "time_outs_bootstrap_key_emitted": False,
        "native_auto_terminations_disabled": sorted(forbidden),
        "current_low_checkpoint": checkpoint_identity,
    }
    return env, CurrentTaskProgressHighRslAdapter(env), actor, low_before, runtime_identity


def save_checkpoint(
    path: Path,
    *,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    iteration: int,
    kind: str,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "iter": int(iteration),
        "infos": {
            "architecture": "fresh_recurrent_high_v56_task_progress_70d_3d",
            "source": "RUN_LOCAL_INITIAL_EVALUATION",
            "kind": str(kind),
            "v4d_weights_loaded": False,
            "teacher_loaded": False,
        },
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)
    return {
        "relative_path": path.name,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "policy_state_sha256": module_bit_state_sha256(policy),
    }


def load_same_run_initial_policy(
    *,
    policy: torch.nn.Module,
    checkpoint: Path,
    expected_file_sha256: str,
    expected_state_sha256: str,
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve(strict=True)
    if file_sha256(checkpoint) != expected_file_sha256:
        raise RuntimeError("same-run initial checkpoint file identity changed")
    value = torch.load(checkpoint, map_location=next(policy.parameters()).device, weights_only=True)
    infos = value.get("infos", {})
    if (
        infos.get("architecture") != "fresh_recurrent_high_v56_task_progress_70d_3d"
        or infos.get("source") != "RUN_LOCAL_INITIAL_EVALUATION"
        or infos.get("kind") != "initial_before_any_act"
        or infos.get("v4d_weights_loaded") is not False
        or infos.get("teacher_loaded") is not False
    ):
        raise RuntimeError("initial checkpoint provenance changed")
    if value.get("optimizer_state_dict") is not None:
        raise RuntimeError("initial checkpoint unexpectedly contains optimizer state")
    policy.load_state_dict(value["model_state_dict"], strict=True)
    actual_state = module_bit_state_sha256(policy)
    if actual_state != expected_state_sha256:
        raise RuntimeError("loaded initial policy state differs from initial evaluation")
    return {
        "relative_path": checkpoint.name,
        "bytes": checkpoint.stat().st_size,
        "sha256": expected_file_sha256,
        "policy_state_sha256": actual_state,
        "optimizer_loaded": False,
    }


def load_same_run_final_policy(
    *,
    policy: torch.nn.Module,
    checkpoint: Path,
    expected_file_sha256: str,
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve(strict=True)
    actual_file_sha256 = file_sha256(checkpoint)
    if actual_file_sha256 != expected_file_sha256:
        raise RuntimeError("same-run final checkpoint file identity changed")
    value = torch.load(checkpoint, map_location=next(policy.parameters()).device, weights_only=True)
    infos = value.get("infos", {})
    if (
        infos.get("architecture") != "fresh_recurrent_high_v56_task_progress_70d_3d"
        or infos.get("source") != "RUN_LOCAL_INITIAL_EVALUATION"
        or infos.get("kind") != "bounded_training_final"
        or infos.get("v4d_weights_loaded") is not False
        or infos.get("teacher_loaded") is not False
    ):
        raise RuntimeError("final checkpoint provenance changed")
    policy.load_state_dict(value["model_state_dict"], strict=True)
    return {
        "relative_path": checkpoint.name,
        "bytes": checkpoint.stat().st_size,
        "sha256": actual_file_sha256,
        "policy_state_sha256": module_bit_state_sha256(policy),
        "training_iteration": int(value.get("iter", -1)),
    }
