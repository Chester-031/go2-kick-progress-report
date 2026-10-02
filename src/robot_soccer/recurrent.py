"""Pure torch guard for RSL recurrent bootstrap hidden-state side effects."""

from __future__ import annotations

from typing import Any

import torch


HiddenState = torch.Tensor | tuple[torch.Tensor, ...] | None


def clone_hidden_state(value: Any) -> HiddenState:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, tuple) and value and all(torch.is_tensor(item) for item in value):
        return tuple(item.detach().clone() for item in value)
    raise RuntimeError("unsupported recurrent hidden-state representation")


def hidden_state_equal(left: HiddenState, right: HiddenState) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if torch.is_tensor(left) and torch.is_tensor(right):
        return torch.equal(left, right)
    if isinstance(left, tuple) and isinstance(right, tuple) and len(left) == len(right):
        return all(torch.equal(a, b) for a, b in zip(left, right, strict=True))
    return False


def restore_hidden_state(memory: Any, snapshot: HiddenState) -> None:
    """Restore an exact detached clone through the RSL Memory reset API."""

    if not hasattr(memory, "reset") or not callable(memory.reset):
        raise RuntimeError("recurrent memory lacks reset API")
    cloned = clone_hidden_state(snapshot)
    if cloned is None:
        memory.reset()
    else:
        memory.reset(hidden_state=cloned)
    if not hidden_state_equal(memory.hidden_state, snapshot):
        raise RuntimeError("recurrent hidden state was not restored exactly")
