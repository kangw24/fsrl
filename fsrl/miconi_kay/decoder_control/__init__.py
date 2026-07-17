"""Frozen-core decoder-capacity control for the M&K source model."""

from .protocol import (
    DECODER_CONTROL_BEHAVIOR_SEEDS,
    DECODER_CONTROL_CAUSAL_SEEDS,
    DECODER_CONTROL_REPRESENTATION_SEEDS,
    decoder_control_evaluation_plan,
    decoder_control_training_plan,
)
from .training import (
    DecoderControlFineTuneConfig,
    decoder_control_tree_sha256,
    train_decoder_control,
)

__all__ = [
    "DECODER_CONTROL_BEHAVIOR_SEEDS",
    "DECODER_CONTROL_CAUSAL_SEEDS",
    "DECODER_CONTROL_REPRESENTATION_SEEDS",
    "DecoderControlFineTuneConfig",
    "decoder_control_evaluation_plan",
    "decoder_control_training_plan",
    "decoder_control_tree_sha256",
    "train_decoder_control",
]
