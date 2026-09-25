"""Core no-Kick robot-soccer policy, reset, reward, and command math."""

from .config import FieldGeometry, POLICY_ACTION_DIM, POLICY_OBSERVATION_DIM
from .resets import (
    FULL,
    REFERENCE,
    ResetConfig,
    ResetSample,
    balanced_reference_samples,
    fixed_evaluation_samples,
    sample_full_reset,
    sample_reference_reset,
    sample_training_reset,
    stationary_training_regime,
)

__all__ = [
    "FULL",
    "REFERENCE",
    "FieldGeometry",
    "POLICY_ACTION_DIM",
    "POLICY_OBSERVATION_DIM",
    "ResetConfig",
    "ResetSample",
    "balanced_reference_samples",
    "fixed_evaluation_samples",
    "sample_full_reset",
    "sample_reference_reset",
    "sample_training_reset",
    "stationary_training_regime",
]
