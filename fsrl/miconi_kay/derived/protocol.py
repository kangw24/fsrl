"""Prospective local protocol for the reward-only linked-source checkpoint.

This registration is deliberately kept in code and copied into the run
directory before the first optimizer step.  It is an audit aid, not an
externally timestamped preregistration.
"""

from __future__ import annotations


LINKED_SOURCE_PROTOCOL_VERSION = 1
LINKED_SOURCE_HELD_OUT_SEEDS = (51, 52)
LINKED_SOURCE_TRAINING_CONFIGURATION = {
    "episodes": 1000,
    "learning_rate": 1e-5,
    "seed": 102,
    "batch_size": 8,
    "component_trials": 10,
    "bridge_trials": 4,
    "query_loss_multiplier": 3.0,
}
LINKED_SOURCE_EVALUATION_CONFIGURATION = {
    "batch_size": 512,
    "bootstrap_samples": 2000,
}
LINKED_SOURCE_EVALUATION_THRESHOLDS = {
    "minimum_batch_size": 512,
    "active_minimum_cross_pair_lower_bound": 0.55,
    "minimum_active_true_over_sham_probability": 0.15,
    "minimum_passive_reliably_failed_cross_pairs": 2,
}


def linked_source_evaluation_plan() -> dict[str, object]:
    """Return a fresh JSON-serializable copy of the locked evaluation plan."""

    return {
        "protocol_version": LINKED_SOURCE_PROTOCOL_VERSION,
        "held_out_seeds": list(LINKED_SOURCE_HELD_OUT_SEEDS),
        "configuration": dict(LINKED_SOURCE_EVALUATION_CONFIGURATION),
        "thresholds": dict(LINKED_SOURCE_EVALUATION_THRESHOLDS),
        "adjudication_rule": (
            "both held-out seeds must pass every unchanged exhaustive behavior "
            "gate; no intermediate checkpoint may replace the fixed final "
            "checkpoint"
        ),
    }


def linked_source_training_plan() -> dict[str, object]:
    """Return the single locked reward-only training design."""

    return {
        "protocol_version": LINKED_SOURCE_PROTOCOL_VERSION,
        "configuration": dict(LINKED_SOURCE_TRAINING_CONFIGURATION),
        "development_history": (
            "A 200-episode seed-101 reward-only pilot failed the exhaustive "
            "weakest-pair gate. Before this formal run, the sole amendment was "
            "to extend the fixed budget to 1000 episodes. A proposed direct "
            "correct-action cross-entropy shortcut was rejected before use."
        ),
        "adjudication_rule": (
            "train exactly once, save only the final episode, and do not tune "
            "after either held-out seed is inspected"
        ),
    }
