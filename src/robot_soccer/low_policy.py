"""Load only the deterministic low actor from an RSL-RL checkpoint.

The high-level trainer never constructs an optimizer for this module.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Mapping, Sequence

import torch
from torch import nn


_ACTOR_PARAMETER = re.compile(r"^(?P<prefix>.*(?:^|\.)actor)\.(?P<index>[0-9]+)\.(?P<kind>weight|bias)$")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_mapping(payload: object) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        for key in (
            "model_state_dict",
            "state_dict",
            "actor_state_dict",
            "policy_state_dict",
        ):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping) and any(
                torch.is_tensor(value) for value in candidate.values()
            ):
                return candidate  # type: ignore[return-value]
        if any(torch.is_tensor(value) for value in payload.values()):
            return payload  # type: ignore[return-value]
    raise ValueError("Checkpoint contains no recognized tensor state mapping.")


def _extract_actor_layers(
    state: Mapping[str, torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor], str]:
    candidates: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    for name, value in state.items():
        match = _ACTOR_PARAMETER.fullmatch(str(name))
        if match is None or not torch.is_tensor(value):
            continue
        prefix = match.group("prefix")
        index = int(match.group("index"))
        candidates.setdefault(prefix, {}).setdefault(index, {})[
            match.group("kind")
        ] = value
    complete = {
        prefix: layers
        for prefix, layers in candidates.items()
        if layers and all(set(values) == {"weight", "bias"} for values in layers.values())
    }
    if len(complete) != 1:
        raise ValueError(
            "Checkpoint must expose exactly one complete sequential actor; "
            f"found={sorted(complete)}"
        )
    prefix, indexed = next(iter(complete.items()))
    weights: list[torch.Tensor] = []
    biases: list[torch.Tensor] = []
    for index in sorted(indexed):
        weight = indexed[index]["weight"]
        bias = indexed[index]["bias"]
        if weight.ndim != 2 or bias.ndim != 1 or weight.shape[0] != bias.shape[0]:
            raise ValueError(f"Malformed actor linear layer at index {index}.")
        weights.append(weight)
        biases.append(bias)
    return weights, biases, prefix


def actor_layer_dimensions(
    state: Mapping[str, torch.Tensor],
) -> tuple[int, ...]:
    weights, _, _ = _extract_actor_layers(state)
    dimensions = [int(weights[0].shape[1])]
    for previous, weight in zip(weights, weights[1:]):
        if int(previous.shape[0]) != int(weight.shape[1]):
            raise ValueError("Actor linear layers are not composable.")
        dimensions.append(int(previous.shape[0]))
    dimensions.append(int(weights[-1].shape[0]))
    return tuple(dimensions)


class FrozenLowActor(nn.Module):
    """A deterministic, optimizer-free low actor."""

    def __init__(self, layers: Sequence[nn.Linear], activation: str) -> None:
        super().__init__()
        if activation != "elu":
            raise ValueError("Only the checked low activation 'elu' is supported.")
        modules: list[nn.Module] = []
        for index, layer in enumerate(layers):
            modules.append(layer)
            if index + 1 < len(layers):
                modules.append(nn.ELU())
        self.actor = nn.Sequential(*modules)
        self.activation = activation
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor(observation)


def load_frozen_low_actor(
    *,
    checkpoint: str | Path,
    expected_checkpoint_sha256: str,
    expected_layer_dimensions: Sequence[int] = (48, 512, 256, 128, 12),
    activation: str = "elu",
    device: str | torch.device = "cpu",
) -> FrozenLowActor:
    """Validate the actor schema and construct an eval-only low actor.

    The returned module contains no optimizer and no observation normalizer.
    A changed schema is an error; the loader never pads or truncates tensors.
    """

    path = Path(checkpoint).resolve(strict=True)
    actual_sha = file_sha256(path)
    if actual_sha != expected_checkpoint_sha256:
        raise RuntimeError(
            "Low checkpoint SHA256 mismatch: "
            f"expected={expected_checkpoint_sha256} actual={actual_sha}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = _tensor_mapping(payload)
    normalizer_keys = sorted(
        str(name)
        for name in state
        if any(
            token in str(name).lower()
            for token in ("normalizer", "running_mean", "running_var")
        )
    )
    if normalizer_keys:
        raise RuntimeError(
            "Low checkpoint unexpectedly contains normalization state: "
            + ",".join(normalizer_keys)
        )
    weights, biases, _ = _extract_actor_layers(state)
    dimensions = actor_layer_dimensions(state)
    expected = tuple(int(value) for value in expected_layer_dimensions)
    if dimensions != expected:
        raise RuntimeError(
            f"Low actor schema mismatch: expected={expected} actual={dimensions}"
        )
    layers: list[nn.Linear] = []
    for weight, bias in zip(weights, biases, strict=True):
        layer = nn.Linear(int(weight.shape[1]), int(weight.shape[0]), bias=True)
        with torch.no_grad():
            layer.weight.copy_(weight)
            layer.bias.copy_(bias)
        layers.append(layer)
    actor = FrozenLowActor(layers, activation=activation).to(device)
    actor.eval()
    if actor.training:
        raise AssertionError("Frozen low actor did not remain in eval mode.")
    if any(parameter.requires_grad for parameter in actor.parameters()):
        raise AssertionError("Frozen low actor has trainable parameters.")
    return actor
