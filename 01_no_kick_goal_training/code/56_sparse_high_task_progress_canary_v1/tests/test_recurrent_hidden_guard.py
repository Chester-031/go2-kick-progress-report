from __future__ import annotations

import importlib.metadata
import contextlib
import io
import math
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recurrent_hidden_guard import clone_hidden_state, hidden_state_equal, restore_hidden_state


class DummyMemory:
    def __init__(self, value: torch.Tensor) -> None:
        self.hidden_state = value

    def reset(self, dones=None, hidden_state=None) -> None:
        del dones
        self.hidden_state = hidden_state


class RecurrentHiddenGuardTests(unittest.TestCase):
    def test_bootstrap_advance_is_restored_bit_exact(self) -> None:
        original = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
        memory = DummyMemory(original.clone())
        saved = clone_hidden_state(memory.hidden_state)
        memory.hidden_state = memory.hidden_state + 7.0
        self.assertFalse(hidden_state_equal(memory.hidden_state, saved))
        restore_hidden_state(memory, saved)
        self.assertTrue(hidden_state_equal(memory.hidden_state, saved))
        self.assertIsNot(memory.hidden_state, saved)

    def test_done_zeroed_components_are_preserved_by_restore(self) -> None:
        value = torch.ones(1, 4, 3)
        value[:, [1, 3], :] = 0.0
        memory = DummyMemory(value.clone())
        saved = clone_hidden_state(memory.hidden_state)
        memory.hidden_state.uniform_(-1.0, 1.0)
        restore_hidden_state(memory, saved)
        self.assertTrue(torch.equal(memory.hidden_state[:, [1, 3], :], torch.zeros(1, 2, 3)))
        self.assertTrue(hidden_state_equal(memory.hidden_state, saved))

    def test_gru_and_lstm_representations_are_supported(self) -> None:
        tensor = torch.randn(1, 2, 4)
        tuple_state = (tensor, tensor + 1.0)
        clone = clone_hidden_state(tuple_state)
        self.assertTrue(hidden_state_equal(tuple_state, clone))
        tuple_state[0].add_(3.0)
        self.assertFalse(hidden_state_equal(tuple_state, clone))

    def test_actual_rsl_3_1_2_rollout_bootstrap_and_recurrent_update_semantics(self) -> None:
        try:
            from rsl_rl.algorithms import PPO
            from rsl_rl.modules import ActorCriticRecurrent
            from tensordict import TensorDict
        except ImportError:
            self.skipTest("RSL is supplied by the controlled Cluster interpreter")
        distributions = importlib.metadata.packages_distributions().get("rsl_rl", [])
        versions = {importlib.metadata.version(name) for name in distributions}
        self.assertEqual(versions, {"3.1.2"})
        torch.manual_seed(56)
        num_envs, observation_dim, action_dim, rollout_steps = 2, 4, 2, 4
        observation = TensorDict(
            {"policy": torch.zeros(num_envs, observation_dim)},
            batch_size=[num_envs],
        )
        with contextlib.redirect_stdout(io.StringIO()):
            policy = ActorCriticRecurrent(
                obs=observation,
                obs_groups={"policy": ["policy"], "critic": ["policy"]},
                num_actions=action_dim,
                actor_obs_normalization=False,
                critic_obs_normalization=False,
                actor_hidden_dims=[8],
                critic_hidden_dims=[8],
                activation="elu",
                init_noise_std=1.0,
                noise_std_type="scalar",
                state_dependent_std=False,
                rnn_type="gru",
                rnn_hidden_dim=6,
                rnn_num_layers=1,
            )
        algorithm = PPO(
            policy,
            num_learning_epochs=1,
            num_mini_batches=1,
            clip_param=0.2,
            gamma=0.99,
            lam=0.95,
            value_loss_coef=1.0,
            entropy_coef=0.0,
            learning_rate=0.001,
            max_grad_norm=1.0,
            use_clipped_value_loss=True,
            schedule="fixed",
            desired_kl=None,
            device="cpu",
            normalize_advantage_per_mini_batch=False,
            rnd_cfg=None,
            symmetry_cfg=None,
        )
        algorithm.init_storage("rl", num_envs, rollout_steps, observation, [action_dim])
        for step in range(rollout_steps):
            action = algorithm.act(observation)
            self.assertTrue(bool(torch.isfinite(action).all()))
            next_observation = TensorDict(
                {"policy": torch.randn(num_envs, observation_dim)},
                batch_size=[num_envs],
            )
            algorithm.process_env_step(
                next_observation,
                torch.tensor([0.1, -0.2]),
                torch.tensor([False, step == rollout_steps - 1]),
                {},
            )
            observation = next_observation

        actor_saved = clone_hidden_state(policy.memory_a.hidden_state)
        critic_saved = clone_hidden_state(policy.memory_c.hidden_state)
        algorithm.compute_returns(observation)
        self.assertFalse(hidden_state_equal(policy.memory_c.hidden_state, critic_saved))
        restore_hidden_state(policy.memory_a, actor_saved)
        restore_hidden_state(policy.memory_c, critic_saved)
        self.assertTrue(hidden_state_equal(policy.memory_a.hidden_state, actor_saved))
        self.assertTrue(hidden_state_equal(policy.memory_c.hidden_state, critic_saved))
        losses = algorithm.update()
        self.assertEqual(set(losses), {"value_function", "surrogate", "entropy"})
        self.assertTrue(all(math.isfinite(float(value)) for value in losses.values()))
        self.assertEqual(int(algorithm.storage.step), 0)
        self.assertTrue(hidden_state_equal(policy.memory_a.hidden_state, actor_saved))
        self.assertTrue(hidden_state_equal(policy.memory_c.hidden_state, critic_saved))


if __name__ == "__main__":
    unittest.main()
