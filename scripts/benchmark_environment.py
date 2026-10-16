"""Time the soccer wrapper's own overhead on the fake Isaac scene.

The fake scene's physics costs almost nothing, so this measures what the
wrapper adds per high-level step: reset sampling and writes, terminal
bookkeeping, observation building and host/device synchronization.  Run it
on a GPU node with ``--device cuda:0`` to include the cost of the
device-to-host syncs that dominate there.

    PYTHONPATH=src:. python scripts/benchmark_environment.py --num-envs 4096
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import time

import torch

from tests.fake_isaac import (
    FakeGymEnv,
    FakeSoccerBaseEnv,
    frozen_low_actor,
    install_isaaclab_stub,
    scripted_latent_action,
)
from tests.test_environment_characterization import make_wrapper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--profile", action="store_true", help="print the top cProfile entries")
    args = parser.parse_args()

    install_isaaclab_stub()
    from robot_soccer import environment

    base = FakeSoccerBaseEnv(args.num_envs, device=args.device)
    started = time.perf_counter()
    env = make_wrapper(environment.FrozenLowInferenceEnv(FakeGymEnv(base)), frozen_low_actor(args.device))
    construction = time.perf_counter() - started
    observation = env.get_observations()
    for _ in range(args.warmup):
        observation, _, _, _ = env.step(scripted_latent_action(observation))

    profiler = cProfile.Profile() if args.profile else None
    episodes = 0
    if profiler is not None:
        profiler.enable()
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(args.steps):
        observation, _, done, _ = env.step(scripted_latent_action(observation))
        episodes += int(done.sum())
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if profiler is not None:
        profiler.disable()

    print(
        f"num_envs={args.num_envs} device={args.device} construction_s={construction:.3f} "
        f"step_ms={1000.0 * elapsed / args.steps:.2f} "
        f"high_steps_per_s={args.num_envs * args.steps / elapsed:,.0f} "
        f"episodes={episodes}",
        flush=True,
    )
    if profiler is not None:
        pstats.Stats(profiler).sort_stats("cumulative").print_stats(25)


if __name__ == "__main__":
    main()
