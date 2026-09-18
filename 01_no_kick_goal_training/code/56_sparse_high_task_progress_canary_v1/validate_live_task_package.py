#!/usr/bin/env python3
"""Rebuild the canonical live task-package manifest without mutation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
from typing import Any


SCHEMA = "sparse_high_task_progress_canary_v1/live_task_manifest_gate/v1"
EXCLUDED_DIRECTORIES = {"__pycache__", ".pytest_cache", ".git"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
FORBIDDEN_STARTUP_HOOKS = {"sitecustomize.py", "usercustomize.py"}
HEX = frozenset("0123456789abcdef")


class TaskPackageError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TaskPackageError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_manifest(distribution: Path) -> tuple[bytes, int]:
    try:
        root_info = distribution.lstat()
    except OSError as error:
        raise TaskPackageError(f"task distribution is unavailable: {error}") from error
    require(stat.S_ISDIR(root_info.st_mode) and not stat.S_ISLNK(root_info.st_mode),
            "task distribution is not a real directory")
    distribution = distribution.resolve(strict=True)
    require(distribution.name == "isaac_go2_low_level_rl", "task distribution basename changed")
    require((distribution / "setup.py").is_file(), "task setup.py is missing")
    require((distribution / "isaac_go2_low_level_rl/__init__.py").is_file(),
            "task package __init__.py is missing")
    parent = distribution.parent
    rows: list[tuple[str, str]] = []
    for path in distribution.rglob("*"):
        relative_parts = path.relative_to(distribution).parts
        if any(part in EXCLUDED_DIRECTORIES for part in relative_parts) or path.suffix in EXCLUDED_SUFFIXES:
            continue
        info = path.lstat()
        relative = path.relative_to(parent).as_posix()
        require(not stat.S_ISLNK(info.st_mode), f"task package contains a symlink: {relative}")
        require(path.name not in FORBIDDEN_STARTUP_HOOKS,
                f"task package contains a startup hook: {relative}")
        if stat.S_ISDIR(info.st_mode):
            continue
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
                f"task package contains a special or multiply-linked file: {relative}")
        rows.append((relative, file_sha256(path)))
    rows.sort()
    require(bool(rows), "task package manifest is empty")
    require(len(rows) == len({relative for relative, _ in rows}), "task manifest has duplicate paths")
    return "".join(f"{digest}  {relative}\n" for relative, digest in rows).encode("ascii"), len(rows)


def validate(distribution: Path, expected_sha256: str, expected_manifest: Path | None = None) -> dict[str, Any]:
    require(len(expected_sha256) == 64 and set(expected_sha256) <= HEX,
            "expected task manifest SHA-256 is invalid")
    payload, count = canonical_manifest(distribution)
    digest = hashlib.sha256(payload).hexdigest()
    require(digest == expected_sha256, "live task package differs from exact V55 OWN manifest")
    if expected_manifest is not None:
        info = expected_manifest.lstat()
        require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode) and info.st_nlink == 1,
                "stored task manifest is unsafe")
        require(expected_manifest.read_bytes() == payload, "stored task manifest differs from canonical walk")
    return {
        "schema": SCHEMA,
        "status": "PASS",
        "manifest_sha256": digest,
        "regular_file_count": count,
        "symlink_count": 0,
        "cache_or_bytecode_count": 0,
        "startup_hook_count": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distribution", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-manifest", type=Path)
    args = parser.parse_args()
    try:
        report = validate(args.distribution, args.expected_sha256, args.expected_manifest)
    except (OSError, UnicodeError, TaskPackageError) as error:
        print(f"V56_LIVE_TASK_PACKAGE_GATE=FAIL {error}")
        return 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
