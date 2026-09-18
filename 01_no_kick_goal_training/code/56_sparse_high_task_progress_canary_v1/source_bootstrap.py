"""Resolve and verify the frozen V14/current runtime support roots."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
from typing import Any


SOURCE_DIR = Path(__file__).resolve().parent

V14_FILES = {
    "continuous_reset.py": "1846c595f4b4a10b0ae7b91230eecee68f716926a022b281ac718bd5ab66edab",
    "entity_observation.py": "cf770f74592b7d573a61fba1430b3bf912e90166796810e9b88f8892e2c49ca4",
    "high_command_interface.py": "76688b1d6b344942f1bdaf5fdc813fe0bf8cdb64f1e09250e4980cb75fc8352b",
    "interface_contract.py": "bf9f2be51b636832bb8dd755d795c3e152b1d678c1030e9c0f508b1f8621eed0",
    "sparse_goal_reward.py": "4e6f1a633c31df0cf41c9cef19e255eba833ca27f4ef1ae95c68899bcd52fe84",
    "sparse_high_env_cfg.py": "30acfab2b6bd15680319d82b1c4b46c0105ca18d7cbcb1590a69879727ce1a8c",
}
CURRENT_FILES = {
    "code06/check_frozen_low_state.py": "4191dd62d59e578ac86a45795adbad71ed5b0da907b917f80587126507b43f10",
    "code06/frozen_low_actor.py": "d4cd9f975fb350b4f597f949db41445063f10dbd4b16ec7e6d9aec6198677750",
    "code06/staging_command_tensor.py": "c252d478563585f56a6ff3caf6156b43aab8762029e7c3b09718ae664f00242c",
    "code06/staging_contact_contract.py": "2eb17a913eb9df85a9829d7ad547af19b470286f294ca2fe1025441c7fb87e89",
    "code06/native_low_executor_v1/native_low_mdp.py": "e84cec4984c09ee3503779be76a5edda697a3f4ba43b5a53c86229e120510bf9",
    "code06/native_low_executor_v1/native_safe_joint_action.py": "0d592389e9d16a56fedf0c0234f8a5eed70013259f8759feb80b7656defc14be",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _root(environment_name: str, packaged: Path) -> Path:
    expected = packaged.resolve(strict=True)
    raw = os.environ.get(environment_name)
    candidate = expected if raw is None else Path(raw).resolve(strict=True)
    if candidate != expected or not candidate.is_dir():
        raise RuntimeError(
            f"{environment_name} must resolve to the package-local frozen support root"
        )
    return candidate


def configure_source_roots() -> dict[str, Any]:
    v14_root = _root(
        "V56_V14_SOURCE_ROOT",
        SOURCE_DIR / "vendor/v14",
    )
    current_root = _root(
        "V56_CURRENT_SUPPORT_ROOT",
        SOURCE_DIR / "vendor/current_qualified",
    )
    receipts: dict[str, Any] = {"v14": {}, "current": {}}
    for relative, expected in V14_FILES.items():
        path = (v14_root / relative).resolve(strict=True)
        if not path.is_relative_to(v14_root) or file_sha256(path) != expected:
            raise RuntimeError(f"frozen V14 support changed: {relative}")
        receipts["v14"][relative] = {"sha256": expected, "bytes": path.stat().st_size}
    for relative, expected in CURRENT_FILES.items():
        path = (current_root / relative).resolve(strict=True)
        if not path.is_relative_to(current_root) or file_sha256(path) != expected:
            raise RuntimeError(f"current runtime support changed: {relative}")
        receipts["current"][relative] = {"sha256": expected, "bytes": path.stat().st_size}
    roots = (
        v14_root,
        current_root / "code06",
        current_root / "code06/native_low_executor_v1",
    )
    for root in reversed(roots):
        rendered = str(root)
        if rendered not in sys.path:
            sys.path.insert(0, rendered)
    receipts["roots"] = {
        "v14": str(v14_root),
        "current": str(current_root),
    }
    return receipts
