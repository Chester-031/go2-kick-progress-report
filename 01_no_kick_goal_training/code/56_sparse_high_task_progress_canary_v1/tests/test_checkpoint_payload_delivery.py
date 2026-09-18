from __future__ import annotations

import copy
import contextlib
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validate_checkpoint_payload import state_sha256, validate


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _infos(kind: str) -> dict[str, object]:
    return {
        "architecture": "fresh_recurrent_high_v56_task_progress_70d_3d",
        "source": "RUN_LOCAL_INITIAL_EVALUATION",
        "kind": kind,
        "v4d_weights_loaded": False,
        "teacher_loaded": False,
    }


def _optimizer(step: int) -> dict[str, object]:
    return {
        "state": {
            0: {
                "step": torch.tensor(float(step)),
                "exp_avg": torch.tensor([0.1, -0.2]),
                "exp_avg_sq": torch.tensor([0.01, 0.04]),
            }
        },
        "param_groups": [{
            "lr": 0.0003,
            "betas": (0.9, 0.999),
            "eps": 1.0e-8,
            "weight_decay": 0.0,
            "amsgrad": False,
            "maximize": False,
            "params": [0],
        }],
    }


class CheckpointPayloadDeliveryTests(unittest.TestCase):
    def _write(self, directory: Path, kind: str, iteration: int, state_value: float) -> tuple[Path, str]:
        model = {"weight": torch.tensor([state_value, state_value + 1.0])}
        optimizer = None
        if kind == "first_update":
            optimizer = _optimizer(20)
        elif kind == "bounded_training_final":
            optimizer = _optimizer(240)
        path = directory / f"{kind}.pt"
        torch.save({
            "model_state_dict": model,
            "optimizer_state_dict": optimizer,
            "iter": iteration,
            "infos": _infos(kind),
        }, path)
        return path, state_sha256(model)

    def test_exact_initial_first_and_final_payloads_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = (
                ("initial_before_any_act", 0, "absent", 0.0),
                ("first_update", 1, "present", 1.0),
                ("bounded_training_final", 12, "present", 2.0),
            )
            states: list[str] = []
            for kind, iteration, optimizer, value in cases:
                path, state = self._write(root, kind, iteration, value)
                report = validate(
                    path,
                    expected_kind=kind,
                    expected_iteration=iteration,
                    expected_sha256=_sha(path),
                    expected_policy_state_sha256=state,
                    optimizer=optimizer,
                )
                self.assertEqual(report["policy_state_sha256"], state)
                states.append(state)
            self.assertEqual(len(set(states)), 3)

    def test_stale_v55_checkpoint_provenance_and_wrong_step_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, state = self._write(root, "first_update", 1, 1.0)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload["infos"] = copy.deepcopy(payload["infos"])
            payload["infos"]["source"] = "RUN_LOCAL_V56_TRANSACTION"
            torch.save(payload, path)
            with self.assertRaises(RuntimeError):
                validate(path, expected_kind="first_update", expected_iteration=1,
                         expected_sha256=_sha(path), expected_policy_state_sha256=state,
                         optimizer="present")

            path, state = self._write(root, "first_update", 1, 1.0)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload["optimizer_state_dict"]["state"][0]["step"] = torch.tensor(19.0)
            torch.save(payload, path)
            with self.assertRaises(RuntimeError):
                validate(path, expected_kind="first_update", expected_iteration=1,
                         expected_sha256=_sha(path), expected_policy_state_sha256=state,
                         optimizer="present")

    def test_actual_rsl_v56_model_tensor_order_matches_optimizer_ids(self) -> None:
        try:
            from rsl_rl.algorithms import PPO
            from rsl_rl.modules import ActorCriticRecurrent
            from tensordict import TensorDict
        except ImportError:
            self.skipTest("RSL is supplied by the controlled Cluster interpreter")
        observation = TensorDict({"policy": torch.zeros(4, 70)}, batch_size=[4])
        with contextlib.redirect_stdout(io.StringIO()):
            policy = ActorCriticRecurrent(
                obs=observation,
                obs_groups={"policy": ["policy"], "critic": ["policy"]},
                num_actions=3,
                actor_obs_normalization=False,
                critic_obs_normalization=False,
                actor_hidden_dims=[256, 256, 128],
                critic_hidden_dims=[256, 256, 128],
                activation="elu",
                init_noise_std=1.0,
                noise_std_type="scalar",
                state_dependent_std=False,
                rnn_type="gru",
                rnn_hidden_dim=256,
                rnn_num_layers=1,
            )
        algorithm = PPO(
            policy,
            num_learning_epochs=5,
            num_mini_batches=4,
            clip_param=0.2,
            gamma=0.997,
            lam=0.98,
            value_loss_coef=1.0,
            entropy_coef=0.03,
            learning_rate=0.0003,
            max_grad_norm=1.0,
            use_clipped_value_loss=True,
            schedule="fixed",
            desired_kl=None,
            device="cpu",
            normalize_advantage_per_mini_batch=False,
            rnd_cfg=None,
            symmetry_cfg=None,
        )
        state_items = list(policy.state_dict().items())
        named_parameters = list(policy.named_parameters())
        optimizer_ids = algorithm.optimizer.state_dict()["param_groups"][0]["params"]
        self.assertEqual(len(state_items), 25)
        self.assertEqual([name for name, _ in state_items], [name for name, _ in named_parameters])
        self.assertEqual(optimizer_ids, list(range(25)))
        self.assertTrue(all(left.shape == right.shape for (_, left), (_, right) in zip(state_items, named_parameters, strict=True)))

if __name__ == "__main__":
    unittest.main()
