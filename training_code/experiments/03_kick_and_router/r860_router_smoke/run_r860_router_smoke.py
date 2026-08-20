#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from r860_router import KICK_CONTROLLERS, Phase, R860ActionRouter, Skill


def main() -> None:
    v4d = tuple(float(i) for i in range(12))
    kick = tuple(100.0 + i for i in range(12))
    recovery = tuple(200.0 + i for i in range(12))

    router = R860ActionRouter(strike_steps=3, recovery_steps=2)
    records = []

    # Normal ownership: output must be exactly the V4D tensor.
    action, owner = router.step(v4d, kick, recovery)
    assert owner == "v4d_locomotion" and action == v4d
    records.append({"case": "locomotion", "owner": owner})

    # Safety gate: a non-ready kick is rejected without changing ownership.
    assert not router.request(Skill.KICK, strength=1.25, ready=False)
    action, owner = router.step(v4d, kick, recovery)
    assert owner == "v4d_locomotion" and action == v4d
    records.append({"case": "kick_not_ready", "owner": owner})

    # Kick ownership is persistent and cannot be interrupted by a new request.
    assert router.request(Skill.KICK, strength=1.25, ready=True)
    for i in range(3):
        # Contact is observed on the final strike tick; recovery owns the
        # following tick, not the tick that produced the contact event.
        action, owner = router.step(v4d, kick, recovery, contact=(i == 2))
        assert owner == "r858_kick" and action == kick
        records.append({"case": f"kick_{i}", "owner": owner, "phase": router.state.phase.value})
        assert not router.request(Skill.PUSH, ready=True)

    # Recovery owns the next steps; then V4D regains ownership.
    action, owner = router.step(v4d, kick, recovery, recovered=False)
    assert owner == "r855_recovery" and action == recovery
    records.append({"case": "recovery_0", "owner": owner})
    action, owner = router.step(v4d, kick, recovery, recovered=True)
    assert owner == "r855_recovery" and action == recovery
    records.append({"case": "recovery_1", "owner": owner})
    action, owner = router.step(v4d, kick, recovery)
    assert owner == "v4d_locomotion" and action == v4d
    records.append({"case": "return_to_v4d", "owner": owner})

    assert router.low_env_step_count == len(records)
    assert set(KICK_CONTROLLERS) == {0.75, 1.0, 1.25}
    output = {
        "version": "r8_60_isolated_router_smoke",
        "training": False,
        "parent_mutation": False,
        "reward_contract": {"goal": 1, "own_goal": -1, "other": 0},
        "exclusive_action_owner": True,
        "single_low_env_step_per_tick": True,
        "records": records,
        "summary": router.summary(),
    }
    out = Path(__file__).with_name("r860_smoke_summary.json")
    out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("R860_ROUTER_SMOKE_OK")
    print(f"SUMMARY={out}")
    print(json.dumps(output["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
