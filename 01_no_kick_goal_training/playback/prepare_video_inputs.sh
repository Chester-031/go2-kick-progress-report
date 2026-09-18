#!/usr/bin/env bash

set -Eeuo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 RUN OUTPUT_TAR" >&2
  exit 64
fi

RUN=$1
OUTPUT_TAR=$2
LOW=/scratch/lj2634/runs/go2_native_low_lateral_deadzone_repair_v1/native_low_lateral_deadzone_repair_16543750_1_symmetric_lateral_10_20260829_012702/evidence/training/model_final.pt
EXPECTED_HIGH=2cec710ac921b220fe69a7819483dc0fc841da9a9ee35fda6e85927ebd33a919
EXPECTED_HIGH_STATE=ed34eac1694ea8ecce5592de8727e45a5bbdbdb996d402a728e8a72c3e8b02ce
EXPECTED_LOW=f46d3b760c43664350aac8099ae5bbd32bc1207f97c8814f5d21c9b531c3d96c
EXPECTED_LOW_BYTES=4584447

[[ "$RUN" = /* && -d "$RUN" && ! -L "$RUN" ]] || { echo "RUN must be an absolute real directory" >&2; exit 65; }
[[ "$OUTPUT_TAR" = /* ]] || { echo "OUTPUT_TAR must be absolute" >&2; exit 65; }
[[ ! -e "$OUTPUT_TAR" && ! -L "$OUTPUT_TAR" ]] || { echo "OUTPUT_TAR already exists" >&2; exit 65; }
[[ ! -e "${OUTPUT_TAR}.sha256" && ! -L "${OUTPUT_TAR}.sha256" ]] || { echo "OUTPUT_TAR sidecar already exists" >&2; exit 65; }
mkdir -p "$(dirname "$OUTPUT_TAR")"

HIGH=$RUN/evidence/training/model_final.pt
TRAINING=$RUN/evidence/training/training_summary.json
FINAL_SUMMARY=$RUN/evidence/final/final_evaluation_summary.json
FINAL_ROWS=$RUN/evidence/final/final_episode_results.jsonl
RESULT=$RUN/evidence/result/task_progress_canary_result.json

python3 - "$RUN" "$HIGH" "$TRAINING" "$FINAL_SUMMARY" "$FINAL_ROWS" "$RESULT" "$LOW" "$EXPECTED_HIGH" "$EXPECTED_HIGH_STATE" "$EXPECTED_LOW" "$EXPECTED_LOW_BYTES" <<'PY'
import hashlib
import json
from pathlib import Path
import stat
import sys


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise RuntimeError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def nonfinite(value):
    raise RuntimeError(f"non-finite JSON constant: {value}")


def checked_file(path, root=None):
    if path.is_symlink():
        raise RuntimeError(f"symlink is forbidden: {path}")
    resolved = path.resolve(strict=True)
    if root is not None and not resolved.is_relative_to(root):
        raise RuntimeError(f"path escaped run: {path}")
    if not stat.S_ISREG(resolved.stat().st_mode):
        raise RuntimeError(f"not a regular file: {path}")
    return resolved


def read_json(path):
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=unique_object, parse_constant=nonfinite)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


run = Path(sys.argv[1]).resolve(strict=True)
high, training_path, final_path, rows_path, result_path = [
    checked_file(Path(value), run) for value in sys.argv[2:7]
]
low = checked_file(Path(sys.argv[7]))
expected_high, expected_high_state, expected_low = sys.argv[8:11]
expected_low_bytes = int(sys.argv[11])
actual_high = sha256(high)
actual_low = sha256(low)
if actual_high != expected_high:
    raise RuntimeError(f"high checkpoint SHA256 changed: {actual_high}")
if actual_low != expected_low:
    raise RuntimeError(f"low checkpoint SHA256 changed: {actual_low}")
if low.stat().st_size != expected_low_bytes:
    raise RuntimeError(f"low checkpoint byte count changed: {low.stat().st_size}")
training = read_json(training_path)
final = read_json(final_path)
result = read_json(result_path)
training_sha = sha256(training_path)
final_sha = sha256(final_path)
rows_sha = sha256(rows_path)
checkpoint = training.get("final_checkpoint")
if not isinstance(checkpoint, dict) or checkpoint.get("sha256") != expected_high:
    raise RuntimeError("training summary does not bind the expected high checkpoint")
if checkpoint.get("policy_state_sha256") != expected_high_state:
    raise RuntimeError("training summary does not bind the expected high policy state")
if training.get("final_policy_state_sha256") != expected_high_state:
    raise RuntimeError("training final policy state changed")
if final.get("checkpoint_sha256") != expected_high:
    raise RuntimeError("final evaluation does not bind the expected high checkpoint")
if final.get("policy_state_sha256") != expected_high_state or final.get("loaded_policy_state_sha256") != expected_high_state:
    raise RuntimeError("final evaluation high policy state changed")
if final.get("episode_rows_sha256") != rows_sha:
    raise RuntimeError("final evaluation does not bind the episode rows")
if result.get("status") != "COMPLETE" or result.get("engineering_integrity_pass") is not True:
    raise RuntimeError("result is not an engineering-complete run")
if result.get("ppo_optimizer_steps") != 240 or result.get("final_goal_count") != 0:
    raise RuntimeError("result does not match the completed bounded run")
if result.get("training") != training or result.get("final_evaluation") != final:
    raise RuntimeError("result embedded summaries differ from source evidence")
input_sha = result.get("input_summary_sha256")
if not isinstance(input_sha, dict):
    raise RuntimeError("result input hashes are missing")
if input_sha.get("training") != training_sha or input_sha.get("final_evaluation") != final_sha:
    raise RuntimeError("result input hashes differ")
print(f"HIGH_SHA256={actual_high}")
print(f"HIGH_POLICY_STATE_SHA256={expected_high_state}")
print(f"LOW_SHA256={actual_low}")
print(f"FINAL_ROWS_SHA256={rows_sha}")
PY

STAGE=$(mktemp -d "$(dirname "$OUTPUT_TAR")/.video-inputs.XXXXXX")
cleanup() {
  rm -rf -- "$STAGE"
}
trap cleanup EXIT

mkdir "$STAGE/current_policy_inputs"
cp -- "$HIGH" "$STAGE/current_policy_inputs/high_policy.pt"
cp -- "$LOW" "$STAGE/current_policy_inputs/low_policy.pt"
cp -- "$TRAINING" "$STAGE/current_policy_inputs/training_summary.json"
cp -- "$FINAL_SUMMARY" "$STAGE/current_policy_inputs/final_evaluation_summary.json"
cp -- "$FINAL_ROWS" "$STAGE/current_policy_inputs/final_episode_results.jsonl"
cp -- "$RESULT" "$STAGE/current_policy_inputs/result.json"
(
  cd "$STAGE/current_policy_inputs"
  sha256sum high_policy.pt low_policy.pt training_summary.json final_evaluation_summary.json final_episode_results.jsonl result.json > SHA256SUMS
)
(
  cd "$STAGE"
  tar --sort=name --owner=0 --group=0 --numeric-owner --mtime='UTC 1970-01-01' -czf "$OUTPUT_TAR" current_policy_inputs
)

tar -tzf "$OUTPUT_TAR"
(
  cd "$(dirname "$OUTPUT_TAR")"
  sha256sum "$(basename "$OUTPUT_TAR")" | tee "$(basename "$OUTPUT_TAR").sha256"
)
