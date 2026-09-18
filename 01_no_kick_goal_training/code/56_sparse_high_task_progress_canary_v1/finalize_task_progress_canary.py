#!/usr/bin/env python3
"""Hash-bound, simulator-free V56 result aggregation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any

from canary_protocol import ResultInputs, build_result, load_contract


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _duplicate_guard(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise RuntimeError(f"non-finite JSON constant is forbidden: {value}")


def _read_json(root: Path, path: Path, expected_sha256: str) -> dict[str, Any]:
    if path.is_symlink():
        raise RuntimeError(f"evidence path is a symlink: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise RuntimeError(f"evidence path escaped root: {path}")
    metadata = resolved.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise RuntimeError(f"evidence path is not a single-link regular file: {path}")
    if _sha256(resolved) != expected_sha256:
        raise RuntimeError(f"evidence SHA256 changed: {path.name}")
    with resolved.open("r", encoding="utf-8") as stream:
        value = json.load(stream, parse_constant=_nonfinite, object_pairs_hook=_duplicate_guard)
    if not isinstance(value, dict):
        raise RuntimeError(f"evidence is not an object: {path.name}")
    return value


def _atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _optional(parser: argparse.ArgumentParser, name: str) -> None:
    flag = name.replace("_", "-")
    parser.add_argument(f"--{flag}", type=Path)
    parser.add_argument(f"--{flag}-sha256")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-evaluation", type=Path, required=True)
    parser.add_argument("--initial-evaluation-sha256", required=True)
    parser.add_argument("--reward-support", type=Path, required=True)
    parser.add_argument("--reward-support-sha256", required=True)
    _optional(parser, "first_update")
    _optional(parser, "training")
    _optional(parser, "final_evaluation")
    args = parser.parse_args()

    root = args.evidence_root.resolve(strict=True)
    output_requested = args.output_dir
    if output_requested.name != "result" or output_requested.parent.resolve(strict=True) != root:
        raise RuntimeError("output-dir must be the evidence-root/result child")
    output_requested.mkdir(mode=0o700, exist_ok=True)
    if output_requested.is_symlink() or not output_requested.is_dir():
        raise RuntimeError("result output is not a real directory")
    output = output_requested.resolve(strict=True)
    if output.parent != root:
        raise RuntimeError("result output escaped evidence root")

    optional: dict[str, dict[str, Any] | None] = {}
    for name in ("first_update", "training", "final_evaluation"):
        path = getattr(args, name)
        digest = getattr(args, f"{name}_sha256")
        if (path is None) != (digest is None):
            raise RuntimeError(f"{name} path/SHA must be both present or both absent")
        optional[name] = None if path is None else _read_json(root, path, digest)

    contract = load_contract(args.contract)
    result = build_result(
        contract,
        ResultInputs(
            initial=_read_json(root, args.initial_evaluation, args.initial_evaluation_sha256),
            reward_support=_read_json(root, args.reward_support, args.reward_support_sha256),
            first_update=optional["first_update"],
            training=optional["training"],
            final=optional["final_evaluation"],
        ),
    )
    input_summary_sha256 = {
        "initial_evaluation": args.initial_evaluation_sha256,
        "reward_support": args.reward_support_sha256,
    }
    for name in ("first_update", "training", "final_evaluation"):
        digest = getattr(args, f"{name}_sha256")
        if digest is not None:
            input_summary_sha256[name] = digest
    result["input_summary_sha256"] = input_summary_sha256
    payload = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    result_path = output / "task_progress_canary_result.json"
    _atomic_bytes(result_path, payload)
    guard = output / "V56_RESULT_FINAL_GUARD"
    if result["status"] == "ERROR":
        _atomic_bytes(guard, ("FAIL " + "; ".join(result["failures"]) + "\n").encode())
        print("V56_INNER_RESULT=ERROR")
        return 2
    _atomic_bytes(guard, b"PASS\n")
    print(f"V56_INNER_DECISION={result['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
