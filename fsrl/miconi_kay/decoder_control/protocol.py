"""Locked local protocol for the frozen-core decoder-capacity control.

This is a capacity diagnostic, not a new cognitive mechanism.  The public
active recurrent/plastic core is held bitwise fixed while only the actor and
critic readouts are optimized on the randomized reward-only source family.
The exact 8-item, four-bridge exhaustive challenge remains held out.
"""

from __future__ import annotations


DECODER_CONTROL_PROTOCOL_VERSION = 1
DECODER_CONTROL_TRAINABLE_PARAMETERS = (
    "h2o.bias",
    "h2o.weight",
    "h2v.bias",
    "h2v.weight",
)
DECODER_CONTROL_TRAINING_CONFIGURATION = {
    "episodes": 1000,
    "learning_rate": 1e-5,
    "seed": 104,
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
    "trainable_parameters": list(DECODER_CONTROL_TRAINABLE_PARAMETERS),
}
DECODER_CONTROL_BEHAVIOR_SEEDS = (111, 112)
DECODER_CONTROL_REPRESENTATION_SEEDS = (121, 122)
DECODER_CONTROL_CAUSAL_SEEDS = (131, 132)
DECODER_CONTROL_EVALUATION_CONFIGURATION = {
    "batch_size": 512,
    "bootstrap_samples": 2000,
}
DECODER_CONTROL_EVALUATION_THRESHOLDS = {
    "minimum_batch_size": 512,
    "active_minimum_cross_pair_lower_bound": 0.55,
    "minimum_active_true_over_sham_probability": 0.15,
}


def decoder_control_training_plan() -> dict[str, object]:
    return {
        "protocol_version": DECODER_CONTROL_PROTOCOL_VERSION,
        "configuration": dict(DECODER_CONTROL_TRAINING_CONFIGURATION),
        "rationale": (
            "linked-source-v1 and randomized-source-v2 changed the full "
            "recurrent/plastic model and both weakened registered temporal "
            "specificity. This control asks only whether the unchanged public "
            "active state is insufficiently decoded."
        ),
        "selection_rule": (
            "one fixed final checkpoint; no intermediate source challenge or "
            "Liu evaluation may select an episode"
        ),
        "interpretation_limit": (
            "capacity diagnostic only; success does not by itself establish a "
            "new human mechanism or unlock Liu transfer"
        ),
    }


def decoder_control_evaluation_plan() -> dict[str, object]:
    return {
        "protocol_version": DECODER_CONTROL_PROTOCOL_VERSION,
        "behavior_seeds": list(DECODER_CONTROL_BEHAVIOR_SEEDS),
        "representation_seeds": list(DECODER_CONTROL_REPRESENTATION_SEEDS),
        "causal_seeds": list(DECODER_CONTROL_CAUSAL_SEEDS),
        "configuration": dict(DECODER_CONTROL_EVALUATION_CONFIGURATION),
        "thresholds": dict(DECODER_CONTROL_EVALUATION_THRESHOLDS),
        "passive_control_rule": (
            "the unchanged public passive checkpoint must fail both all-pair "
            "coherence criteria evaluated at the same 0.55 threshold"
        ),
        "stage_rule": (
            "run both held-out behavior seeds first; if either fails active "
            "all-pair coherence, stop and do not spend representation/causal "
            "seeds on a behaviorally nonviable capacity control"
        ),
        "adjudication_rule": (
            "if behavior passes both seeds, both representation and both "
            "causal seeds must also pass; successes cannot offset failures"
        ),
    }
