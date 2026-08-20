#!/usr/bin/env python3
"""Add deterministic-versus-stochastic low-policy attribution to R8.79."""

from __future__ import annotations

import argparse
from pathlib import Path


def once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"R880_PATCH_FATAL={label}_count:{count}")
    return text.replace(old, new, 1)


DIAGNOSTIC = r'''
def run_r880_mean_noise_attribution(
    *,
    high_runner: Any,
    low_runner: Any,
    env: HighLevelSoccerVecEnv,
    low_adapter: PoseLowLevelEnvAdapter,
    log_dir: Path,
) -> None:
    high_before = snapshot_policy(high_runner)
    low_before = snapshot_policy(low_runner)
    module = policy_module(low_runner)
    std_stats: dict[str, float] = {}
    for name in ("std", "_std", "log_std", "_log_std"):
        value = getattr(module, name, None)
        if not torch.is_tensor(value):
            continue
        actual = value.detach().exp() if "log_std" in name else value.detach()
        std_stats = {
            "mean": float(actual.mean()),
            "min": float(actual.min()),
            "max": float(actual.max()),
        }
        break

    torch.manual_seed(88001)
    deterministic = _r879_sustained_evaluation(
        low_runner=low_runner, env=env, low_adapter=low_adapter,
        stochastic=False,
    )
    torch.manual_seed(88001)
    stochastic = _r879_sustained_evaluation(
        low_runner=low_runner, env=env, low_adapter=low_adapter,
        stochastic=True,
    )
    high_delta = parameter_delta(high_before, high_runner)
    low_delta = parameter_delta(low_before, low_runner)
    target_cases = ("backward", "turn_left", "turn_right")
    response_ratios: dict[str, float] = {}
    materially_helped = []
    for name in target_cases:
        det = float(deterministic["cases"][name]["late_response"])
        sto = float(stochastic["cases"][name]["late_response"])
        ratio = sto / max(abs(det), 1.0e-5)
        response_ratios[name] = ratio
        if stochastic["cases"][name]["passed"] and not deterministic["cases"][name]["passed"]:
            materially_helped.append(name)
        elif sto > det + 0.04 and ratio > 1.5:
            materially_helped.append(name)
    noise_dependent = bool(materially_helped)
    result = {
        "version": "r8_80_mean_noise_attribution_v1",
        "training": False,
        "std": std_stats,
        "deterministic": deterministic,
        "stochastic": stochastic,
        "target_response_ratios": response_ratios,
        "materially_helped_cases": materially_helped,
        "noise_dependent": noise_dependent,
        "high_parameter_delta": high_delta,
        "low_parameter_delta": low_delta,
        "verdict": (
            "R880_MEAN_POLICY_COLLAPSE_CONFIRMED"
            if noise_dependent else "R880_MECHANICAL_OR_OBJECTIVE_PERSISTENCE_FAILURE"
        ),
    }
    path = log_dir / "r880_mean_noise_attribution.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[R880_DETERMINISTIC] " + json.dumps(deterministic, sort_keys=True), flush=True)
    print("[R880_STOCHASTIC] " + json.dumps(stochastic, sort_keys=True), flush=True)
    print("[R880_STD] " + json.dumps(std_stats, sort_keys=True), flush=True)
    print(f"R880_SUMMARY={path}", flush=True)
    print(f"R880_VERDICT={result['verdict']}", flush=True)
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = args.source.read_text(encoding="utf-8")

    text = once(
        text,
        '''def _r879_sustained_evaluation(
    *,
    low_runner: Any,
    env: HighLevelSoccerVecEnv,
    low_adapter: PoseLowLevelEnvAdapter,
) -> dict[str, Any]:
''',
        '''def _r879_sustained_evaluation(
    *,
    low_runner: Any,
    env: HighLevelSoccerVecEnv,
    low_adapter: PoseLowLevelEnvAdapter,
    stochastic: bool = False,
) -> dict[str, Any]:
''',
        "evaluation_signature",
    )
    text = once(
        text,
        '''                low_actions = policy_module(low_runner).act_inference(low_obs)
                low_obs, _, low_dones, _, diag = env.low_step(
''',
        '''                if stochastic:
                    low_actions = policy_module(low_runner).act(low_obs)
                else:
                    low_actions = policy_module(low_runner).act_inference(low_obs)
                low_obs, _, low_dones, _, diag = env.low_step(
''',
        "evaluation_action",
    )
    text = once(
        text,
        '''    result: dict[str, Any] = {"cases": {}}
''',
        '''    result: dict[str, Any] = {
        "cases": {},
        "mode": "stochastic" if stochastic else "deterministic",
    }
''',
        "evaluation_mode",
    )
    text = once(text, "\ndef run_v4d_training(\n", DIAGNOSTIC + "\ndef run_v4d_training(\n", "diagnostic_insertion")
    text = once(
        text,
        '''        if os.environ.get("R879_SUSTAINED_LOW_REPAIR") == "1":
            run_r879_low_repair(
''',
        '''        if os.environ.get("R880_MEAN_NOISE_ATTRIBUTION") == "1":
            run_r880_mean_noise_attribution(
                high_runner=high_runner,
                low_runner=low_runner,
                env=env,
                low_adapter=low_adapter,
                log_dir=log_dir,
            )
        elif os.environ.get("R879_SUSTAINED_LOW_REPAIR") == "1":
            run_r879_low_repair(
''',
        "main_dispatch",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print("R880_MEAN_NOISE_EVAL_PATCH_OK")


if __name__ == "__main__":
    main()
