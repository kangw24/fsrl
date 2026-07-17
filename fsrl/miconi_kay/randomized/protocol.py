"""Locked local protocol for randomized linked-source-v2."""

from __future__ import annotations


RANDOMIZED_SOURCE_PROTOCOL_VERSION = 1
RANDOMIZED_SOURCE_TRAINING_CONFIGURATION = {
    "episodes": 1000,
    "learning_rate": 1e-5,
    "seed": 103,
    "batch_size": 8,
    "hidden_size": 200,
    "cue_size": 15,
    "source_min_items": 4,
    "source_max_items": 8,
    "trial_steps": 4,
    "response_step": 1,
    "item_counts": [4, 6],
    "component_trials": 10,
    "minimum_bridge_trials": 1,
    "maximum_bridge_trials": 3,
    "query_branches": 8,
    "query_loss_multiplier": 3.0,
}
RANDOMIZED_SOURCE_BEHAVIOR_SEEDS = (81, 82)
RANDOMIZED_SOURCE_REPRESENTATION_SEEDS = (91, 92)
RANDOMIZED_SOURCE_CAUSAL_SEEDS = (101, 102)
RANDOMIZED_SOURCE_EVALUATION_CONFIGURATION = {
    "batch_size": 512,
    "bootstrap_samples": 2000,
}
RANDOMIZED_SOURCE_EVALUATION_THRESHOLDS = {
    "minimum_batch_size": 512,
    "active_minimum_cross_pair_lower_bound": 0.55,
    "minimum_active_true_over_sham_probability": 0.15,
}


def randomized_source_training_plan() -> dict[str, object]:
    return {
        "protocol_version": RANDOMIZED_SOURCE_PROTOCOL_VERSION,
        "configuration": dict(RANDOMIZED_SOURCE_TRAINING_CONFIGURATION),
        "development_history": (
            "linked-source-v1 trained directly on all sixteen 8-item challenge "
            "pairs and repaired behavior but failed stable causal specificity. "
            "This new candidate changes the training distribution, not a Liu "
            "metric: 8-item/four-bridge episodes are absent from training. A "
            "five-episode compute smoke used no held-out challenge; before any "
            "8-item evaluation, the budget was set to 1000 optimizer steps to "
            "match linked-source-v1 rather than doubling training exposure."
        ),
        "selection_rule": (
            "one fixed final checkpoint; no intermediate source challenge or "
            "Liu evaluation may select an episode"
        ),
    }


def randomized_source_evaluation_plan() -> dict[str, object]:
    return {
        "protocol_version": RANDOMIZED_SOURCE_PROTOCOL_VERSION,
        "behavior_seeds": list(RANDOMIZED_SOURCE_BEHAVIOR_SEEDS),
        "representation_seeds": list(RANDOMIZED_SOURCE_REPRESENTATION_SEEDS),
        "causal_seeds": list(RANDOMIZED_SOURCE_CAUSAL_SEEDS),
        "configuration": dict(RANDOMIZED_SOURCE_EVALUATION_CONFIGURATION),
        "thresholds": dict(RANDOMIZED_SOURCE_EVALUATION_THRESHOLDS),
        "passive_control_rule": (
            "the unchanged public passive checkpoint must fail both all-pair "
            "coherence criteria evaluated at the same 0.55 threshold; this "
            "replaces the logically inappropriate requirement for two reliably "
            "below-chance pairs"
        ),
        "amendment_timing": (
            "fixed after linked-source-v1 adjudication and before randomized "
            "training; old passive-gate failures remain immutable"
        ),
        "adjudication_rule": (
            "both seeds in every gate family must pass; public matched controls "
            "are diagnostic and cannot override a candidate failure"
        ),
    }
