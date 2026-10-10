"""EXPLANATORY PROJECTION — NOT THE PRODUCTION TRAINER.

Immutable outline of checkpoint-220 restoration and formal PPO training.
"""

from __future__ import annotations

from dataclasses import dataclass

from parameters import (
    FINAL_CUMULATIVE_OPTIMIZER_STEPS,
    FORMAL_END_ITERATION,
    FORMAL_ITERATIONS,
    FORMAL_OPTIMIZER_STEPS,
    FORMAL_START_ITERATION,
    FORMAL_TRANSITIONS,
    MILESTONE_ITERATIONS,
    NUM_LEARNING_EPOCHS,
    NUM_MINI_BATCHES,
    OPTIMIZER_STEPS_PER_ITERATION,
    PARENT_CUMULATIVE_OPTIMIZER_STEPS,
    PARENT_ITERATION,
    ROLLOUT_STEPS_PER_ENV,
)


@dataclass(frozen=True)
class OutlineStep:
    name: str
    purpose: str
    invariant: str


@dataclass(frozen=True)
class ContinuationTotals:
    parent_iteration: int
    first_formal_iteration: int
    final_formal_iteration: int
    formal_iterations: int
    formal_transitions: int
    added_optimizer_steps: int
    final_cumulative_optimizer_steps: int


DISPOSABLE_CANARY_OUTLINE = (
    OutlineStep(
        "restore_parent_for_canary",
        "Restore both policy and Adam state from iteration 220.",
        "The canary starts at cumulative Adam step 4,400.",
    ),
    OutlineStep(
        "one_iteration_canary",
        "Exercise one complete rollout, return computation, and PPO update.",
        "Its 20 optimizer steps are disposable and never become formal ancestry.",
    ),
    OutlineStep(
        "discard_canary_state",
        "End the canary process and retain no learned state for formal training.",
        "Formal training must restore iteration 220 again in a fresh process.",
    ),
)


FORMAL_TRAINING_OUTLINE = (
    OutlineStep(
        "construct_hierarchy",
        "Create the 10 Hz recurrent high policy above the frozen 50 Hz low policy.",
        "Only high-policy and critic parameters are trainable.",
    ),
    OutlineStep(
        "restore_model_and_adam",
        "Restore the exact iteration-220 model and Adam state and synchronize the PPO cached learning rate.",
        "This is continuation, not a fresh optimizer or weights-only warm start.",
    ),
    OutlineStep(
        "bootstrap_stage_c_resets",
        "Start the fixed long-distance-weighted Stage C reset distribution at iteration 221.",
        "No episode, recurrent state, or simulator state is inherited from the parent process.",
    ),
    OutlineStep(
        "collect_rollout",
        f"Collect {ROLLOUT_STEPS_PER_ENV} high-policy steps per environment.",
        "Each high action is held for five native low-policy ticks.",
    ),
    OutlineStep(
        "compute_returns",
        "Snapshot recurrent actor and critic hidden states, compute returns, then restore those hidden states.",
        "Return computation must not introduce a recurrent bootstrap side effect.",
    ),
    OutlineStep(
        "ppo_update",
        f"Run {NUM_LEARNING_EPOCHS} epochs × {NUM_MINI_BATCHES} minibatches.",
        f"Exactly {OPTIMIZER_STEPS_PER_ITERATION} Adam steps occur per formal iteration.",
    ),
    OutlineStep(
        "advance_fixed_schedule",
        "Advance from iteration 221 through 300 without changing the reset mixture.",
        "Reward, observation, action, termination, seed, and frozen low policy remain unchanged.",
    ),
    OutlineStep(
        "milestone_state",
        f"Define continuation milestones at {MILESTONE_ITERATIONS}.",
        "This projection describes the milestones but performs no serialization.",
    ),
)


def formal_iteration_numbers() -> tuple[int, ...]:
    return tuple(range(FORMAL_START_ITERATION, FORMAL_END_ITERATION + 1))


def continuation_totals() -> ContinuationTotals:
    return ContinuationTotals(
        parent_iteration=PARENT_ITERATION,
        first_formal_iteration=FORMAL_START_ITERATION,
        final_formal_iteration=FORMAL_END_ITERATION,
        formal_iterations=FORMAL_ITERATIONS,
        formal_transitions=FORMAL_TRANSITIONS,
        added_optimizer_steps=FORMAL_OPTIMIZER_STEPS,
        final_cumulative_optimizer_steps=FINAL_CUMULATIVE_OPTIMIZER_STEPS,
    )


if PARENT_CUMULATIVE_OPTIMIZER_STEPS + FORMAL_OPTIMIZER_STEPS != FINAL_CUMULATIVE_OPTIMIZER_STEPS:
    raise AssertionError("formal continuation optimizer accounting changed")
if len(formal_iteration_numbers()) != FORMAL_ITERATIONS:
    raise AssertionError("formal continuation iteration count changed")
