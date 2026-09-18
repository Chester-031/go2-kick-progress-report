#!/usr/bin/env python3
"""Independently inspect a V56 torch checkpoint without constructing Isaac or RSL."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import stat
from typing import Any

import torch


HEX = frozenset("0123456789abcdef")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(value: Any) -> str:
    require(isinstance(value, dict) and value, "model_state_dict is missing or empty")
    digest = hashlib.sha256()
    tensor_count = 0
    for name in sorted(value):
        tensor = value[name]
        require(isinstance(name, str), "state-dict key is not text")
        require(torch.is_tensor(tensor), f"state-dict value is not a tensor: {name}")
        require(not tensor.is_sparse, f"sparse tensor is forbidden: {name}")
        item = tensor.detach().cpu().contiguous()
        require(bool(torch.isfinite(item).all()), f"checkpoint tensor is non-finite: {name}")
        digest.update(name.encode("utf-8"))
        digest.update(str(item.dtype).encode("ascii"))
        digest.update(str(tuple(item.shape)).encode("ascii"))
        digest.update(item.numpy().tobytes())
        tensor_count += 1
    require(tensor_count > 0, "checkpoint contains no tensors")
    return digest.hexdigest()


def validate(
    checkpoint: Path,
    *,
    expected_kind: str,
    expected_iteration: int,
    expected_sha256: str,
    expected_policy_state_sha256: str,
    optimizer: str,
) -> dict[str, Any]:
    metadata = checkpoint.lstat()
    require(stat.S_ISREG(metadata.st_mode) and not checkpoint.is_symlink(), "checkpoint is not a safe regular file")
    require(metadata.st_nlink == 1, "checkpoint must have exactly one hard link")
    require(metadata.st_size > 0, "checkpoint is empty")
    for label, value in (("file", expected_sha256), ("policy state", expected_policy_state_sha256)):
        require(isinstance(value, str) and len(value) == 64 and set(value) <= HEX, f"expected {label} SHA256 is invalid")
    require(sha256_file(checkpoint) == expected_sha256, "checkpoint file SHA256 differs")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(isinstance(payload, dict), "checkpoint root is not a dictionary")
    require(set(payload) == {"model_state_dict", "optimizer_state_dict", "iter", "infos"}, "checkpoint top-level keys changed")
    require(type(payload.get("iter")) is int and payload["iter"] == expected_iteration, "checkpoint iteration changed")
    expected_layout = {
        "initial_before_any_act": (0, "absent"),
        "first_update": (1, "present"),
        "bounded_training_final": (12, "present"),
    }
    require(expected_layout[expected_kind] == (expected_iteration, optimizer), "checkpoint kind/iteration/optimizer request is inconsistent")
    infos = payload.get("infos")
    require(isinstance(infos, dict), "checkpoint infos are missing")
    expected_infos = {
        "architecture": "fresh_recurrent_high_v56_task_progress_70d_3d",
        "source": "RUN_LOCAL_INITIAL_EVALUATION",
        "kind": expected_kind,
        "v4d_weights_loaded": False,
        "teacher_loaded": False,
    }
    require(infos == expected_infos, "checkpoint provenance changed")
    optimizer_state = payload.get("optimizer_state_dict")
    if optimizer == "absent":
        require(optimizer_state is None, "checkpoint unexpectedly contains optimizer state")
    else:
        require(isinstance(optimizer_state, dict) and optimizer_state, "checkpoint optimizer state is missing")
        groups = optimizer_state.get("param_groups")
        states = optimizer_state.get("state")
        require(isinstance(groups, list) and groups, "checkpoint optimizer parameter groups are missing")
        require(len(groups) == 1, "checkpoint optimizer must contain exactly one parameter group")
        require(isinstance(states, dict) and states, "checkpoint optimizer state mapping is missing or empty")
        parameter_ids: list[int] = []
        for group in groups:
            require(isinstance(group, dict) and isinstance(group.get("params"), list) and group["params"],
                    "optimizer parameter group is malformed or empty")
            require(all(type(item) is int and item >= 0 for item in group["params"]),
                    "optimizer parameter identifier is invalid")
            parameter_ids.extend(group["params"])
            require(float(group.get("lr", -1.0)) == 0.0003, "optimizer learning rate changed")
            require(tuple(group.get("betas", ())) == (0.9, 0.999), "optimizer Adam betas changed")
            require(float(group.get("eps", -1.0)) == 1.0e-8, "optimizer Adam epsilon changed")
            require(float(group.get("weight_decay", -1.0)) == 0.0, "optimizer weight decay changed")
            require(group.get("amsgrad") is False and group.get("maximize") is False, "optimizer Adam mode changed")
        require(len(parameter_ids) == len(set(parameter_ids)), "optimizer parameter identifiers are repeated")
        require(set(states) == set(parameter_ids), "optimizer state does not cover its parameter groups exactly")
        model_state = payload.get("model_state_dict")
        require(isinstance(model_state, dict) and model_state, "checkpoint model state is missing")
        ordered_model_tensors = list(model_state.values())
        require(set(parameter_ids) == set(range(len(ordered_model_tensors))),
                "optimizer parameter groups do not cover model tensors exactly")
        expected_step = 20 if expected_kind == "first_update" else 240
        for parameter_id, state in states.items():
            require(isinstance(state, dict) and set(state) == {"step", "exp_avg", "exp_avg_sq"},
                    f"optimizer parameter state is malformed: {parameter_id}")
            step = state["step"]
            if torch.is_tensor(step):
                require(step.numel() == 1 and bool(torch.isfinite(step).all()),
                        f"optimizer step is non-finite or non-scalar: {parameter_id}")
                step_value = float(step.detach().cpu().item())
            else:
                require(isinstance(step, (int, float)) and not isinstance(step, bool) and math.isfinite(float(step)),
                        f"optimizer step is invalid: {parameter_id}")
                step_value = float(step)
            require(step_value == float(expected_step), f"optimizer step differs from {expected_step}: {parameter_id}")
            for name in ("exp_avg", "exp_avg_sq"):
                item = state[name]
                require(torch.is_tensor(item) and item.numel() > 0 and bool(torch.isfinite(item).all()),
                        f"optimizer {name} is missing, empty, or non-finite: {parameter_id}")
            require(tuple(state["exp_avg"].shape) == tuple(state["exp_avg_sq"].shape),
                    f"optimizer moment shapes differ: {parameter_id}")
            require(torch.is_tensor(ordered_model_tensors[parameter_id]) and
                    tuple(state["exp_avg"].shape) == tuple(ordered_model_tensors[parameter_id].shape),
                    f"optimizer moments do not match model tensor: {parameter_id}")
            for item in state.values():
                if torch.is_tensor(item):
                    require(bool(torch.isfinite(item).all()), "optimizer checkpoint tensor is non-finite")
                elif isinstance(item, (int, float)) and not isinstance(item, bool):
                    require(math.isfinite(float(item)), "optimizer checkpoint scalar is non-finite")
    actual_state = state_sha256(payload.get("model_state_dict"))
    require(actual_state == expected_policy_state_sha256, "checkpoint policy state SHA256 differs")
    return {
        "bytes": metadata.st_size,
        "sha256": expected_sha256,
        "policy_state_sha256": actual_state,
        "kind": expected_kind,
        "iteration": expected_iteration,
        "optimizer": optimizer,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--kind", choices=("initial_before_any_act", "first_update", "bounded_training_final"), required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--policy-state-sha256", required=True)
    parser.add_argument("--optimizer", choices=("absent", "present"), required=True)
    args = parser.parse_args()
    try:
        report = validate(
            args.checkpoint,
            expected_kind=args.kind,
            expected_iteration=args.iteration,
            expected_sha256=args.sha256,
            expected_policy_state_sha256=args.policy_state_sha256,
            optimizer=args.optimizer,
        )
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        print(f"V56_CHECKPOINT_PAYLOAD_GATE=FAIL {error}")
        return 2
    print("V56_CHECKPOINT_PAYLOAD_GATE=PASS")
    print(f"V56_CHECKPOINT_KIND={report['kind']}")
    print(f"V56_CHECKPOINT_ITERATION={report['iteration']}")
    print(f"V56_CHECKPOINT_SHA256={report['sha256']}")
    print(f"V56_CHECKPOINT_POLICY_STATE_SHA256={report['policy_state_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
