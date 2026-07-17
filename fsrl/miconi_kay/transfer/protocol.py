"""Locked protocol for the first latent-process M&K-to-Liu candidate test.

The protocol admits unobserved cognitive events as hypotheses.  It constrains
them by requiring a human-choice-free generative population, prospective
within-person prediction, matched controls, and secondary prior-predictive
behavioral signatures.  This is a local code registration, not an independent
or externally timestamped preregistration.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


COVERT_FEEDBACK_PROTOCOL_VERSION = 1
COVERT_FEEDBACK_SIMULATION_SEEDS = (161, 162, 163, 164)
COVERT_FEEDBACK_RANDOM_INIT_SEEDS = (171, 172, 173, 174)
COVERT_FEEDBACK_METRIC_SEEDS = (181, 182)
COVERT_FEEDBACK_METRIC_SUBJECTS = 77
COVERT_FEEDBACK_PARTICLES_PER_SEED = 512
COVERT_FEEDBACK_SUPPORT_REPETITIONS = 4
COVERT_FEEDBACK_FIT_BLOCKS = (1, 2, 3, 4, 5)
COVERT_FEEDBACK_TEST_BLOCKS = (6, 7, 8, 9, 10)
COVERT_FEEDBACK_BASELINE_LAPSE = 0.05
COVERT_FEEDBACK_LIKELIHOOD_CLIP = 1e-6

LIU2026_COHORTS = {
    "preregistered": {
        "filename": "preregistered_experiment_data.csv",
        "sha256": "6dcae48511018a85765e3ab7ceed6f358f5185f5ce399ce47543dbc7aad0c227",
        "subjects": 40,
    },
    "replication": {
        "filename": "replication_experiment_data.csv",
        "sha256": "c322cedd587d8e119442f873679d5fd5315e31736f4bf4266dbb89849c068249",
        "subjects": 37,
    },
}

COVERT_FEEDBACK_CANDIDATES = (
    "public_active_covert_feedback_delayed_write",
    "public_passive_covert_feedback_delayed_write",
    "random_init_covert_feedback_delayed_write",
    "uniform_global_permutation_lapse05",
    "independent_binary_preferences_lapse05",
    "chance",
)

COVERT_FEEDBACK_CONTINUATION_GATES = {
    "active_over_passive_subject_delta_lpd_ci_lower": 0.0,
    "active_over_random_subject_delta_lpd_ci_lower": 0.0,
    "active_over_uniform_permutation_subject_delta_lpd_ci_lower": 0.0,
    "minimum_median_posterior_particle_ess": 5.0,
    "maximum_overall_accuracy_absolute_error": 0.05,
    "maximum_learned_accuracy_absolute_error": 0.07,
    "maximum_unlearned_accuracy_absolute_error": 0.07,
    "maximum_response_consistency_absolute_error": 0.05,
    "maximum_bimodal_pair_count_absolute_error": 5,
    "require_positive_distance_slope": True,
}


def transfer_tree_sha256() -> str:
    """Hash only the files defining this frozen transfer candidate."""

    directory = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in (directory / "protocol.py", directory / "covert_feedback.py"):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def covert_feedback_validation_plan() -> dict[str, object]:
    return {
        "protocol_version": COVERT_FEEDBACK_PROTOCOL_VERSION,
        "epistemic_rule": (
            "unobserved cognitive states and parameters are admissible when "
            "explicit, shared across people, causally consequential, and "
            "exposed to held-out prediction and candidate comparison"
        ),
        "latent_microprocess": [
            "encode_pair_before_relational_outcome_is_available",
            "sample_a_covert_relational_choice_from_the_frozen_policy",
            "compare_choice_with_the_observed_signed_relation",
            "encode_internal_signed_prediction_error_at_feedback_step",
            "block_fast_write_at_that_feedback_step",
            "allow_the_following_delay_step_to_write_fast_weights",
        ],
        "observed_task_boundary": (
            "Liu reports pair and signed relation without a required learning "
            "response; the finer microsequence above is a model hypothesis"
        ),
        "support_repetitions": COVERT_FEEDBACK_SUPPORT_REPETITIONS,
        "simulation_seeds": list(COVERT_FEEDBACK_SIMULATION_SEEDS),
        "random_init_seeds": list(COVERT_FEEDBACK_RANDOM_INIT_SEEDS),
        "particles_per_seed": COVERT_FEEDBACK_PARTICLES_PER_SEED,
        "metric_sampling_seeds": list(COVERT_FEEDBACK_METRIC_SEEDS),
        "metric_virtual_subjects_per_seed": COVERT_FEEDBACK_METRIC_SUBJECTS,
        "human_data": {
            "cohorts": LIU2026_COHORTS,
            "conditioning_blocks": list(COVERT_FEEDBACK_FIT_BLOCKS),
            "strictly_scored_blocks": list(COVERT_FEEDBACK_TEST_BLOCKS),
            "primary_observation": "one participant x block x pair choice",
            "primary_score": (
                "conditional joint log predictive density of blocks 6-10 "
                "after posterior updating on blocks 1-5, reported per choice"
            ),
            "status": (
                "locked retrospective reanalysis of previously accessed public "
                "data; not an independently blind confirmation"
            ),
        },
        "candidates": list(COVERT_FEEDBACK_CANDIDATES),
        "baseline_lapse": COVERT_FEEDBACK_BASELINE_LAPSE,
        "likelihood_probability_clip": COVERT_FEEDBACK_LIKELIHOOD_CLIP,
        "continuation_gates": dict(COVERT_FEEDBACK_CONTINUATION_GATES),
        "adjudication": (
            "both Liu cohorts and both metric sampling seeds must pass every "
            "applicable gate; no seed, block, pair, metric, or participant selection"
        ),
        "forbidden_flexibility": [
            "participant_id_parameters_or_checkpoint_lookup",
            "per_participant_preloaded_rank_or_error_template",
            "per_pair_free_logits_fitted_on_scored_blocks",
            "using_Liu_choices_or_summary_metrics_to_train_or_select_the_core",
            "changing_the_microsequence_after_viewing_candidate_results",
        ],
        "claim_ceiling": (
            "a pass supports M&K covert-feedback as a comparatively predictive "
            "candidate for available endpoint choices; it does not identify a "
            "unique human learning microsequence"
        ),
    }
