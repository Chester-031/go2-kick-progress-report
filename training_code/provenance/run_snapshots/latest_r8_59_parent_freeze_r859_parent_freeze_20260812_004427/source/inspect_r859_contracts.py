#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch


def state_dict(payload):
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint is not a dict: {type(payload)}")
    for key in ("model_state_dict", "state_dict", "policy_state_dict", "actor_critic_state_dict"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    if payload and all(torch.is_tensor(v) for v in payload.values()):
        return payload
    raise RuntimeError(f"state dict not found; keys={sorted(payload)[:30]}")


def tensor_shapes(sd):
    return {k: list(v.shape) for k, v in sd.items() if torch.is_tensor(v)}


def first_actor_input(shapes):
    candidates = [(k, v) for k, v in shapes.items() if "actor" in k.lower() and k.endswith("weight") and len(v) == 2]
    if not candidates:
        raise RuntimeError("actor input layer not found")
    candidates.sort(key=lambda x: (x[1][1], x[0]))
    return {"key": candidates[0][0], "shape": candidates[0][1]}


def last_actor_output(shapes):
    candidates = [(k, v) for k, v in shapes.items() if "actor" in k.lower() and k.endswith("weight") and len(v) == 2]
    if not candidates:
        raise RuntimeError("actor output layer not found")
    twelve = [x for x in candidates if x[1][0] == 12]
    chosen = sorted(twelve or candidates, key=lambda x: x[0])[-1]
    return {"key": chosen[0], "shape": chosen[1]}


def inspect_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    sd = state_dict(payload)
    shapes = tensor_shapes(sd)
    return {
        "file": str(path),
        "bytes": path.stat().st_size,
        "payload_keys": sorted(payload.keys()) if isinstance(payload, dict) else [],
        "tensor_count": len(shapes),
        "actor_input": first_actor_input(shapes),
        "actor_output": last_actor_output(shapes),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    args = p.parse_args()
    root = Path(args.root).resolve()
    v4d_contract = json.loads((root / "v4d/v4d_joint_contract.json").read_text(encoding="utf-8"))
    v4d_summary = json.loads((root / "v4d/v4d_joint_summary.json").read_text(encoding="utf-8"))
    r851_summary = json.loads((root / "kick/r851_summary.json").read_text(encoding="utf-8"))
    r855_summary = json.loads((root / "kick/r855_summary.json").read_text(encoding="utf-8"))
    r858_summary = json.loads((root / "kick/r858_summary.json").read_text(encoding="utf-8"))
    winner = json.loads((root / "kick/r858_hold_winner.json").read_text(encoding="utf-8"))

    if v4d_contract["high_level"]["input"] != "17_dim_soccer_state":
        raise RuntimeError("V4D high observation contract changed")
    if v4d_contract["high_level"]["action"] != "3_dim_continuous_intent":
        raise RuntimeError("V4D high action contract changed")
    if v4d_contract["low_level"]["input"] != "teacher_free_62_dim_direct_delta_pose_context":
        raise RuntimeError("V4D low observation contract changed")
    if v4d_contract["low_level"]["action"] != "12_joint_position_residuals":
        raise RuntimeError("V4D low action contract changed")
    if v4d_summary.get("kick_included") is not False:
        raise RuntimeError("V4D parent unexpectedly includes kick")
    # R8.51 proves filtered true contact and the 64D strike interface.
    # Its standalone router verdict was negative; downstream strength
    # monotonicity and full-chain acceptance are asserted using R8.58.
    if (
        r851_summary.get("actual_contact") is not True
        or r851_summary.get("obs_contract") != "64D=70D[0:63]+70D[69]"
        or r851_summary.get("frozen_load_lift_swing_recovery") is not True
    ):
        raise RuntimeError("R8.51 true-contact strike interface changed")
    if r855_summary.get("actual_contact") is not True or r855_summary.get("kick_frozen") is not True:
        raise RuntimeError("R8.55 recovery parent contract changed")
    if r858_summary.get("accepted") is not True or r858_summary.get("verdict") != "READY_FOR_VIDEO_REVIEW":
        raise RuntimeError("R8.58 is not accepted")
    proven = set(r858_summary.get("reused_proven_chain", []))
    required = {"true_contact", "kick_strength_monotonic", "high_strength_hold"}
    if not required.issubset(proven):
        raise RuntimeError(
            f"R8.58 proven-chain contract changed: {sorted(proven)}"
        )
    if set(winner["by_strength"]) != {"0.75", "1.0", "1.25"}:
        raise RuntimeError("R8.58 strength anchors changed")

    checkpoints = {
        "v4d_high": inspect_checkpoint(root / "v4d/high_model_final.pt"),
        "v4d_low": inspect_checkpoint(root / "v4d/low_model_final.pt"),
        "r851_kick64": inspect_checkpoint(root / "kick/kick64_final.pt"),
        "r855_recovery64": inspect_checkpoint(root / "kick/recovery64_final.pt"),
    }
    expected = {
        "v4d_high": (17, 3),
        "v4d_low": (62, 12),
        "r851_kick64": (64, 12),
        "r855_recovery64": (64, 12),
    }
    for name, (obs, act) in expected.items():
        found = checkpoints[name]
        if found["actor_input"]["shape"][1] != obs:
            raise RuntimeError(f"{name} actor obs mismatch: {found['actor_input']}")
        if found["actor_output"]["shape"][0] != act:
            raise RuntimeError(f"{name} actor action mismatch: {found['actor_output']}")

    contract = {
        "version": "r8_59_parent_freeze_before_integration",
        "mutation": "none_to_parent_runs",
        "training": False,
        "execution_parents": {
            "high": "v4d/high_model_final.pt",
            "delta_pose_low": "v4d/low_model_final.pt",
            "kick_residual": "kick/kick64_final.pt",
            "post_contact_recovery": "kick/recovery64_final.pt",
            "kick_controller": "kick/r858_hold_winner.json",
            "ballfree_components": "kick/ballfree_artifacts/",
        },
        "ancestry_only": {"v3_strict_zero_sum": "ancestry/v3_model_final.pt"},
        "interfaces": {
            "high": {"obs": 17, "action": 3, "reward": {"goal": 1, "own_goal": -1, "other": 0}},
            "delta_pose_low": {"obs": 62, "action": 12, "teacher_free": True, "legacy_command_slots_zero": True},
            "kick_internal": {"isolated_obs": 70, "kick_obs": 64, "recovery_obs": 64, "action": 12},
            "router_boundary": "exclusive 12D action ownership immediately before low_env.step",
            "forbidden": ["adding V4D and R8.58 actions", "loading kick64 into V4D 62D actor", "modifying parent source or checkpoints"],
        },
        "strength_anchors": winner["by_strength"],
        "checkpoint_inspection": checkpoints,
    }
    out = root / "integration_parent_contract.json"
    out.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("R859_CONTRACT_INSPECTION_OK")
    print(f"CONTRACT={out}")


if __name__ == "__main__":
    main()
