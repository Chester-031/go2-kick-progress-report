#!/usr/bin/env python3
"""Validate exact membership and bytes of an installed V56 source tree."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any


SCHEMA = "sparse_high_task_progress_canary_v1/source_manifest_gate/v1"
MANIFEST_NAME = "SPARSE_HIGH_TASK_PROGRESS_CANARY_V1_SOURCE.sha256"
LINE = re.compile(r"^([0-9a-f]{64})  ([^\r\n]+)$")
FORBIDDEN_DIRECTORIES = {"__pycache__", ".pytest_cache", ".git"}
FORBIDDEN_STARTUP_HOOKS = {"sitecustomize.py", "usercustomize.py"}


class SourceManifestError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceManifestError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_relative(raw: str) -> str:
    require("\\" not in raw, f"manifest path uses a backslash: {raw!r}")
    path = PurePosixPath(raw)
    require(not path.is_absolute(), f"manifest path is absolute: {raw!r}")
    require(bool(path.parts) and all(part not in ("", ".", "..") for part in path.parts),
            f"manifest path is not canonical: {raw!r}")
    require(path.as_posix() == raw, f"manifest path spelling is not canonical: {raw!r}")
    require(raw != MANIFEST_NAME, "source manifest must not hash itself")
    return raw


def parse_manifest(path: Path) -> list[tuple[str, str]]:
    payload = path.read_text(encoding="utf-8")
    require(payload.endswith("\n"), "source manifest lacks its terminal newline")
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    for number, line in enumerate(payload.splitlines(), 1):
        match = LINE.fullmatch(line)
        require(match is not None, f"malformed source manifest line {number}")
        relative = canonical_relative(match.group(2))
        require(relative not in seen, f"duplicate source manifest path: {relative}")
        seen.add(relative)
        entries.append((relative, match.group(1)))
    require(bool(entries), "source manifest is empty")
    require([name for name, _ in entries] == sorted(name for name, _ in entries),
            "source manifest paths are not in canonical order")
    return entries


def actual_regular_files(root: Path, manifest: Path) -> set[str]:
    result: set[str] = set()
    for path in root.rglob("*"):
        info = path.lstat()
        relative = path.relative_to(root).as_posix()
        require(not stat.S_ISLNK(info.st_mode), f"source tree contains a symlink: {relative}")
        require(path.name not in FORBIDDEN_STARTUP_HOOKS,
                f"source tree contains a forbidden Python startup hook: {relative}")
        if stat.S_ISDIR(info.st_mode):
            require(path.name not in FORBIDDEN_DIRECTORIES,
                    f"source tree contains a cache/control directory: {relative}")
            continue
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
                f"source tree contains a special or multiply-linked file: {relative}")
        if path == manifest:
            continue
        require(path.suffix not in {".pyc", ".pyo"}, f"source tree contains bytecode: {relative}")
        result.add(relative)
    return result


def validate(root: Path, manifest: Path, receipt: Path) -> dict[str, Any]:
    root_info, manifest_info, receipt_info = root.lstat(), manifest.lstat(), receipt.lstat()
    require(stat.S_ISDIR(root_info.st_mode) and not stat.S_ISLNK(root_info.st_mode),
            "source root is not a real directory")
    require(stat.S_ISREG(manifest_info.st_mode) and not stat.S_ISLNK(manifest_info.st_mode)
            and manifest_info.st_nlink == 1,
            "source manifest is not a regular file")
    require(stat.S_ISREG(receipt_info.st_mode) and not stat.S_ISLNK(receipt_info.st_mode)
            and receipt_info.st_nlink == 1,
            "source manifest receipt is not a regular file")
    root, manifest, receipt = root.resolve(strict=True), manifest.resolve(strict=True), receipt.resolve(strict=True)
    require(manifest.parent == root and manifest.name == MANIFEST_NAME, "source manifest path changed")
    require(receipt == root.parent / f"{root.name}.source-manifest.sha256", "source manifest receipt path changed")
    entries = parse_manifest(manifest)
    declared = {name for name, _ in entries}
    actual = actual_regular_files(root, manifest)
    require(declared == actual, "source manifest file set is incomplete or has extras")
    for relative, expected in entries:
        require(file_sha256(root / relative) == expected, f"source hash changed: {relative}")
    manifest_sha256 = file_sha256(manifest)
    require(receipt.read_text(encoding="ascii") == f"{manifest_sha256}  {MANIFEST_NAME}\n",
            "source manifest receipt differs")
    return {"schema": SCHEMA, "status": "PASS", "manifest_sha256": manifest_sha256,
            "regular_file_count": len(actual), "symlink_count": 0,
            "cache_or_bytecode_count": 0, "startup_hook_count": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = validate(args.root, args.manifest, args.receipt)
    except (OSError, UnicodeError, SourceManifestError) as error:
        print(f"V56_SOURCE_MANIFEST_GATE=FAIL {error}")
        return 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
