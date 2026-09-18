#!/usr/bin/env python3
"""Canonical SHA256 snapshots for frozen low parameters and buffers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Mapping


def _update_text(digest: "hashlib._Hash", value: str) -> None:
    payload = value.encode("utf-8")
    digest.update(struct.pack(">Q", len(payload)))
    digest.update(payload)


def _tensor_bytes(tensor: Any) -> bytes:
    import torch

    contiguous = tensor.detach().cpu().contiguous()
    if contiguous.numel() == 0:
        return b""
    return contiguous.view(torch.uint8).numpy().tobytes(order="C")


def canonical_mapping_sha256(mapping: Mapping[str, Any]) -> str:
    """Hash tensor names, dtypes, shapes, and bytes in deterministic order."""

    import torch

    digest = hashlib.sha256()
    tensor_count = 0
    for name in sorted(mapping):
        value = mapping[name]
        if not torch.is_tensor(value):
            continue
        tensor_count += 1
        _update_text(digest, name)
        _update_text(digest, str(value.dtype))
        _update_text(digest, json.dumps(list(value.shape), separators=(",", ":")))
        payload = _tensor_bytes(value)
        digest.update(struct.pack(">Q", len(payload)))
        digest.update(payload)
    if tensor_count == 0:
        raise ValueError("State mapping contains no tensors.")
    return digest.hexdigest()


def module_state_snapshot(module: Any) -> dict[str, Any]:
    """Snapshot all module parameters and persistent buffers."""

    state = module.state_dict()
    return {
        "sha256": canonical_mapping_sha256(state),
        "tensor_count": sum(1 for value in state.values() if hasattr(value, "shape")),
        "parameter_count": sum(1 for _ in module.named_parameters()),
        "buffer_count": sum(1 for _ in module.named_buffers()),
        "training": bool(module.training),
        "requires_grad_parameters": sorted(
            name for name, parameter in module.named_parameters() if parameter.requires_grad
        ),
    }


def assert_unchanged(before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
    if before.get("sha256") != after.get("sha256"):
        raise RuntimeError(
            "Frozen low state changed: "
            f"before={before.get('sha256')} after={after.get('sha256')}"
        )


def write_snapshot(module: Any, path: Path) -> dict[str, Any]:
    snapshot = module_state_snapshot(module)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return snapshot


def _extract_tensor_mapping(payload: Any) -> Mapping[str, Any]:
    import torch

    if isinstance(payload, Mapping):
        for key in (
            "model_state_dict",
            "state_dict",
            "actor_state_dict",
            "policy_state_dict",
        ):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping) and any(torch.is_tensor(value) for value in candidate.values()):
                return candidate
        if any(torch.is_tensor(value) for value in payload.values()):
            return payload
    raise ValueError("Could not find a tensor state mapping in checkpoint.")


def checkpoint_state_snapshot(checkpoint: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    mapping = _extract_tensor_mapping(payload)
    normalizer_keys = sorted(
        key
        for key in mapping
        if any(token in key.lower() for token in ("normalizer", "running_mean", "running_var"))
    )
    return {
        "checkpoint": str(checkpoint.resolve()),
        "state_sha256": canonical_mapping_sha256(mapping),
        "tensor_count": sum(1 for value in mapping.values() if hasattr(value, "shape")),
        "normalizer_keys": normalizer_keys,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--after", type=Path)
    args = parser.parse_args()

    if args.checkpoint:
        result = checkpoint_state_snapshot(args.checkpoint)
        encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
        else:
            print(encoded, end="")
        return 0

    if args.before and args.after:
        before = json.loads(args.before.read_text(encoding="utf-8"))
        after = json.loads(args.after.read_text(encoding="utf-8"))
        assert_unchanged(before, after)
        print("FROZEN_LOW_STATE_UNCHANGED")
        return 0

    parser.error("Use --checkpoint or both --before and --after.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
