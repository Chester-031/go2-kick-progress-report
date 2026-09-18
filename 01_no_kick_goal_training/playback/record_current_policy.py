#!/usr/bin/env python3

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Sequence


WORKSPACE = Path(__file__).resolve().parents[1]
RUNTIME_SOURCE = WORKSPACE / "code/56_sparse_high_task_progress_canary_v1"
DEFAULT_CONTRACT = RUNTIME_SOURCE / "SPARSE_HIGH_TASK_PROGRESS_CANARY_V1_CONTRACT.json"
DEFAULT_INPUT_DIR = Path("/home/ubuntu/下载/go2_soccer_phaseA_results/current_policy_inputs")
DEFAULT_OUTPUT_DIR = Path("/home/ubuntu/下载/go2_soccer_phaseA_results/r898_video_bundle")
DEFAULT_TASK_SOURCE = Path(
    "/home/ubuntu/go2_r898_video_16044032_gate045/project_source_current/isaac_go2_low_level_rl"
)
HIGH_CHECKPOINT_SHA256 = "2cec710ac921b220fe69a7819483dc0fc841da9a9ee35fda6e85927ebd33a919"
HIGH_POLICY_STATE_SHA256 = "ed34eac1694ea8ecce5592de8727e45a5bbdbdb996d402a728e8a72c3e8b02ce"
LOW_CHECKPOINT_SHA256 = "f46d3b760c43664350aac8099ae5bbd32bc1207f97c8814f5d21c9b531c3d96c"
TASK_MANIFEST_SHA256 = "b3af08bf243fa793257ebf625012ef05a43eb38df8e533293be68dfda9b36e3f"
FIXED_CASE_COUNT = 288


class NativeStepFrameCapture:
    def __init__(self, step: Any, capture: Any) -> None:
        self.step = step
        self.capture = capture
        self.count = 0

    def __call__(self, action: Any) -> Any:
        result = self.step(action)
        self.capture()
        self.count += 1
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record the current no-kick soccer policy on one fixed evaluation case."
    )
    parser.add_argument("--assignment", type=int, required=True)
    parser.add_argument(
        "--high-checkpoint",
        type=Path,
        default=DEFAULT_INPUT_DIR / "high_policy.pt",
    )
    parser.add_argument(
        "--low-checkpoint",
        type=Path,
        default=DEFAULT_INPUT_DIR / "low_policy.pt",
    )
    parser.add_argument(
        "--evidence-rows",
        type=Path,
        default=DEFAULT_INPUT_DIR / "final_episode_results.jsonl",
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--task-source", type=Path, default=DEFAULT_TASK_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--task", default="Isaac-Go2-Soccer-Milestone2-v0")
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_regular_file(
    path: Path,
    expected_sha256: str,
    *,
    expected_bytes: int | None = None,
) -> Path:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise RuntimeError(f"input is not a regular non-symlink file: {path}")
    resolved = path.resolve(strict=True)
    if expected_bytes is not None and info.st_size != expected_bytes:
        raise RuntimeError(f"input byte count differs: {path}")
    if file_sha256(resolved) != expected_sha256:
        raise RuntimeError(f"input SHA256 differs: {path}")
    return resolved


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        tensor = value.detach().cpu()
        return tensor.item() if tensor.ndim == 0 else tensor.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _frame_array(render_env: Any, numpy: Any) -> Any:
    frame = render_env.render()
    if isinstance(frame, (list, tuple)):
        if not frame:
            raise RuntimeError("render returned an empty frame list")
        frame = frame[0]
    array = numpy.asarray(frame)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise RuntimeError(f"render returned an unsupported frame shape: {array.shape}")
    if array.shape[2] == 4:
        array = array[:, :, :3]
    if array.dtype != numpy.uint8:
        if not numpy.isfinite(array).all():
            raise RuntimeError("render returned a non-finite frame")
        if float(array.max()) <= 1.0:
            array = array * 255.0
        array = numpy.clip(array, 0.0, 255.0).astype(numpy.uint8)
    return numpy.ascontiguousarray(array)


def _render_environment(value: Any) -> Any:
    current = value
    seen: set[int] = set()
    while not callable(getattr(current, "render", None)):
        identity = id(current)
        if identity in seen:
            raise RuntimeError("render environment unwrap cycle")
        seen.add(identity)
        next_value = getattr(current, "env", None)
        if next_value is None or next_value is current:
            next_value = getattr(current, "unwrapped", None)
        if next_value is None or next_value is current:
            raise RuntimeError("no render-capable environment")
        current = next_value
    return current


def _output_paths(output_dir: Path, assignment: int) -> dict[str, Path]:
    stem = f"assignment_{assignment:03d}"
    return {
        "video": output_dir / f"{stem}.mp4",
        "trace": output_dir / f"{stem}.jsonl",
        "summary": output_dir / f"{stem}.json",
    }


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def verify_replay(expected: dict[str, Any], observed: dict[str, Any], assignment: int) -> None:
    fields = (
        "evaluation_assignment",
        "outcome",
        "contact_seen",
        "hard_goal_facing_ball_behind",
        "goal_side",
    )
    if expected.get("evaluation_assignment") != assignment:
        raise RuntimeError("evidence assignment differs")
    for field in fields:
        if observed.get(field) != expected.get(field):
            raise RuntimeError(f"replay differs from evidence: {field}")


def _validate_early(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path, dict[str, Any], str]:
    if not 0 <= args.assignment < FIXED_CASE_COUNT:
        raise RuntimeError(f"assignment must be in [0, {FIXED_CASE_COUNT - 1}]")
    if args.fps <= 0:
        raise RuntimeError("fps must be positive")
    high = verify_regular_file(args.high_checkpoint, HIGH_CHECKPOINT_SHA256)
    low = verify_regular_file(
        args.low_checkpoint,
        LOW_CHECKPOINT_SHA256,
        expected_bytes=4_584_447,
    )
    contract = args.contract.resolve(strict=True)
    task_source = args.task_source.resolve(strict=True)
    if str(RUNTIME_SOURCE) not in sys.path:
        sys.path.insert(0, str(RUNTIME_SOURCE))
    from canary_protocol import load_contract
    from validate_live_task_package import validate
    from select_behavior_cases import load_rows

    loaded_contract = load_contract(contract)
    if loaded_contract["qualified_low"]["checkpoint_sha256"] != LOW_CHECKPOINT_SHA256:
        raise RuntimeError("contract low checkpoint identity differs")
    validate(task_source, TASK_MANIFEST_SHA256)
    rows, rows_sha256 = load_rows(args.evidence_rows)
    expected_row = next(
        row for row in rows if row["evaluation_assignment"] == args.assignment
    )
    paths = _output_paths(args.output_dir.resolve(), args.assignment)
    if not args.overwrite:
        existing = [path for path in paths.values() if path.exists()]
        if existing:
            raise RuntimeError(f"output already exists: {existing[0]}")
    return high, low, contract, task_source, expected_row, rows_sha256


def record(
    args: argparse.Namespace,
    validated_inputs: tuple[Path, Path, Path, Path, dict[str, Any], str],
) -> int:
    high_checkpoint, low_checkpoint, contract_path, task_source, expected_row, rows_sha256 = validated_inputs
    if str(task_source) not in sys.path:
        sys.path.insert(0, str(task_source))
    if str(RUNTIME_SOURCE) not in sys.path:
        sys.path.insert(0, str(RUNTIME_SOURCE))

    from source_bootstrap import configure_source_roots

    configure_source_roots()

    import imageio.v2 as imageio
    import numpy as np
    import torch
    from rsl_rl.runners import OnPolicyRunner
    import isaac_go2_low_level_rl
    import isaac_go2_low_level_rl.tasks
    import isaac_go2_low_level_rl.tasks.manager_based.soccer
    from canary_protocol import FIXED_CASE_COUNT as runtime_case_count
    from canary_protocol import load_contract
    import runtime_common
    from runtime_common import (
        file_sha256 as runtime_file_sha256,
        load_same_run_final_policy,
        make_runtime,
        policy_module,
        runner_cfg,
        runner_identity,
    )

    if runtime_case_count != FIXED_CASE_COUNT:
        raise RuntimeError("fixed evaluation case count differs")
    task_origin = Path(isaac_go2_low_level_rl.__file__).resolve(strict=True)
    if not task_origin.is_relative_to(task_source):
        raise RuntimeError("task package import escaped the validated local source")
    contract = deepcopy(load_contract(contract_path))
    contract["qualified_low"]["checkpoint_cluster_path"] = str(low_checkpoint)
    original_gym_make = runtime_common.gym.make

    def rgb_array_make(task: str, *values: Any, **kwargs: Any) -> Any:
        if kwargs.get("render_mode") not in (None, "rgb_array"):
            raise RuntimeError("unexpected render mode")
        kwargs["render_mode"] = "rgb_array"
        return original_gym_make(task, *values, **kwargs)

    runtime_common.gym.make = rgb_array_make
    env = adapter = None
    try:
        env, adapter, _, _, _ = make_runtime(
            contract=contract,
            num_envs=1,
            reset_seed=int(contract["bounded_canary"]["fixed_evaluation_seed"]),
            device=args.device,
            task=args.task,
            evaluation_case_count=FIXED_CASE_COUNT,
        )
    finally:
        runtime_common.gym.make = original_gym_make

    output_paths = _output_paths(args.output_dir.resolve(), args.assignment)
    output_paths["video"].parent.mkdir(parents=True, exist_ok=True)
    temporary_video = output_paths["video"].with_name(
        f".{output_paths['video'].stem}.{os.getpid()}.tmp.mp4"
    )
    writer = None
    native_step_original = None
    completed = False
    try:
        policy_seed = int(contract["bounded_canary"]["policy_initialization_seed"])
        torch.manual_seed(policy_seed)
        runner = OnPolicyRunner(
            adapter,
            runner_cfg(contract=contract, device=args.device),
            log_dir=None,
            device=args.device,
        )
        policy = policy_module(runner)
        identity = runner_identity(runner, policy, contract)
        loaded = load_same_run_final_policy(
            policy=policy,
            checkpoint=high_checkpoint,
            expected_file_sha256=HIGH_CHECKPOINT_SHA256,
        )
        if loaded["policy_state_sha256"] != HIGH_POLICY_STATE_SHA256:
            raise RuntimeError("high policy state identity differs")
        if loaded["training_iteration"] != int(contract["bounded_canary"]["learning_iterations"]):
            raise RuntimeError("high checkpoint iteration differs")
        policy.eval()
        policy.reset()
        initial_actor_hidden, initial_critic_hidden = policy.get_hidden_states()
        if initial_actor_hidden is not None or initial_critic_hidden is not None:
            raise RuntimeError("recurrent hidden state did not reset before first inference")

        env.evaluation_cursor = args.assignment
        selected_id = torch.zeros(1, dtype=torch.long, device=env.device)
        env._reset_idx(selected_id)
        if int(env.assignment[0].item()) != args.assignment:
            raise RuntimeError("fixed evaluation assignment was not installed")
        if not bool(env.counts_for_evaluation[0].item()):
            raise RuntimeError("selected fixed evaluation assignment is not counted")
        selected_sample = env.reset_samples[0]
        if selected_sample is None:
            raise RuntimeError("fixed evaluation sample is unavailable")
        selected_case = {
            "robot_xy_m": list(selected_sample.robot_xy),
            "robot_yaw_rad": selected_sample.robot_yaw,
            "ball_xy_m": list(selected_sample.ball_xy),
            "goal_side": selected_sample.goal_side,
            "hard": bool(env.hard_case[0].item()),
            "bucket": env.bucket[0],
        }

        observation = adapter.get_observations().to(args.device)
        if tuple(observation["policy"].shape) != (1, 70):
            raise RuntimeError("high policy observation shape differs")
        render_env = _render_environment(env.low_env)
        writer = imageio.get_writer(
            temporary_video,
            fps=args.fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
        )
        writer.append_data(_frame_array(render_env, np))
        native_step_original = env.low_env.step
        native_capture = NativeStepFrameCapture(
            native_step_original,
            lambda: writer.append_data(_frame_array(render_env, np)),
        )
        env.low_env.step = native_capture
        trace: list[dict[str, Any]] = []
        terminal_events: list[dict[str, Any]] = []
        high_steps = 0
        hidden_reset_verified = False
        with torch.inference_mode():
            for high_step in range(env.max_episode_length):
                action = policy.act_inference(observation)
                observation, reward, done, extras = adapter.step(action.to(adapter.device))
                policy.reset(done.to(args.device))
                if bool(done[0].item()):
                    for hidden in policy.get_hidden_states():
                        tensors = hidden if isinstance(hidden, tuple) else (hidden,)
                        for tensor in tensors:
                            if tensor is not None and bool(torch.count_nonzero(tensor[..., done, :]).item()):
                                raise RuntimeError("recurrent hidden state did not reset on terminal step")
                    hidden_reset_verified = True
                observation = observation.to(args.device)
                if tuple(observation["policy"].shape) != (1, 70):
                    raise RuntimeError("high policy observation shape differs after step")
                receipt = extras["v56_pre_reset_step_receipt"]
                trace.append(
                    {
                        "assignment": args.assignment,
                        "high_step": high_step,
                        "action": action[0].detach().cpu().tolist(),
                        "applied_command": receipt["applied_command"][0].detach().cpu().tolist(),
                        "progress_reward": float(receipt["progress_reward"][0].item()),
                        "total_reward": float(reward[0].item()),
                        "contact": bool(receipt["contact_this_high"][0].item()),
                        "done": bool(done[0].item()),
                    }
                )
                terminal_events.extend(dict(event) for event in extras.get("terminal_events", []))
                high_steps = high_step + 1
                if bool(done[0].item()):
                    break

        if len(terminal_events) != 1:
            raise RuntimeError("recording did not end with exactly one terminal event")
        terminal_event = terminal_events[0]
        verify_replay(expected_row, terminal_event, args.assignment)
        native_frame_count = native_capture.count
        expected_native_frames = high_steps * int(contract["architecture"]["low_steps_per_high_action"])
        if native_frame_count != expected_native_frames:
            raise RuntimeError("native frame count differs from executed native steps")
        if native_frame_count <= 0:
            raise RuntimeError("recording contains no native-step frames")

        writer.close()
        writer = None
        os.replace(temporary_video, output_paths["video"])
        trace_payload = b"".join(
            (json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
            for row in trace
        )
        _write_atomic(output_paths["trace"], trace_payload)
        summary = {
            "status": "COMPLETE",
            "assignment": args.assignment,
            "fixed_case": selected_case,
            "outcome": terminal_event["outcome"],
            "terminal_event": _json_ready(terminal_event),
            "high_steps": high_steps,
            "native_step_frames": native_frame_count,
            "frame_count": native_frame_count + 1,
            "fps": args.fps,
            "observation_dim": identity["observation_dim"],
            "action_dim": identity["action_dim"],
            "recurrent_policy": identity["is_recurrent"],
            "hidden_state_reset_on_done": hidden_reset_verified,
            "replay_matches_evidence": True,
            "evidence_episode_rows_sha256": rows_sha256,
            "evidence_case": expected_row,
            "high_checkpoint_sha256": HIGH_CHECKPOINT_SHA256,
            "high_policy_state_sha256": loaded["policy_state_sha256"],
            "low_checkpoint_sha256": LOW_CHECKPOINT_SHA256,
            "task_manifest_sha256": TASK_MANIFEST_SHA256,
            "video": {
                "path": str(output_paths["video"]),
                "bytes": output_paths["video"].stat().st_size,
                "sha256": runtime_file_sha256(output_paths["video"]),
            },
            "trace": {
                "path": str(output_paths["trace"]),
                "rows": len(trace),
                "sha256": runtime_file_sha256(output_paths["trace"]),
            },
        }
        if summary["video"]["bytes"] <= 0:
            raise RuntimeError("video output is empty")
        _write_atomic(
            output_paths["summary"],
            (json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"),
        )
        completed = True
        print(f"VIDEO={output_paths['video']}")
        print(f"SUMMARY={output_paths['summary']}")
        print(f"OUTCOME={summary['outcome']}")
        return 0
    finally:
        if native_step_original is not None and env is not None:
            env.low_env.step = native_step_original
        if writer is not None:
            writer.close()
        if temporary_video.exists() and not completed:
            temporary_video.unlink()
        if env is not None:
            env.close()


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if "--help" in values or "-h" in values:
        build_parser().parse_args(values)
        return 0
    early_args, _ = build_parser().parse_known_args(values)
    validated_inputs = _validate_early(early_args)
    from isaaclab.app import AppLauncher

    parser = build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(values)
    args.enable_cameras = True
    simulation_app = AppLauncher(args).app
    try:
        return record(args, validated_inputs)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
