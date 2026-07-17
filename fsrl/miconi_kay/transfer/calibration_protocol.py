"""Locked v2 calibration protocol after the zero-parameter v1 failure."""

from __future__ import annotations

import hashlib
from pathlib import Path


CALIBRATION_PROTOCOL_VERSION = "2.1"
PARENT_PROBABILITY_FILE_SHA256 = (
    "1c70d5b64f835c82f8c8d30b1d3076326846ac09b8871c5a54601331e432857d"
)
PARENT_REPORT_SHA256 = (
    "e8dcff02031ed51ecda3db1d09ef6fe2e8e59c633dc162751aaff7c6a2951f09"
)
PARENT_POPULATION_SHA256 = {
    "active": "3ce5d31b0846570c32227020710b632e3732b5e4c8f135020fafc541ea5860aa",
    "passive": "ab48769b08c3acf9a513509056720053b85582aa8de05c5ba92a0aeca1e4089e",
    "random_init": "363256ce599b3399f8ef3cf7485a9e892956722791f889fe1bf323557c9a4889",
}
CALIBRATION_ROLES = ("active", "passive", "random_init", "uniform_permutation")
CALIBRATION_TEMPERATURE_BOUNDS = (0.25, 16.0)
CALIBRATION_LOG_GRID_POINTS = 65
CALIBRATION_DEVELOPMENT_COHORT = "preregistered"
CALIBRATION_BLOCKS = (1, 2, 3, 4, 5)
CALIBRATION_SCORED_BLOCKS = (6, 7, 8, 9, 10)
CALIBRATION_METRIC_SEEDS = (183, 184)
CALIBRATION_METRIC_SUBJECTS = 77
CALIBRATION_BOOTSTRAP_SAMPLES = 2000
CALIBRATION_GATES = {
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


def calibration_tree_sha256() -> str:
    directory = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for label, path in (
        ("transfer/calibration_protocol.py", directory / "calibration_protocol.py"),
        (
            "cognitive_validation/heldout.py",
            directory.parents[1] / "cognitive_validation" / "heldout.py",
        ),
    ):
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def covert_feedback_calibration_plan() -> dict[str, object]:
    return {
        "protocol_version": CALIBRATION_PROTOCOL_VERSION,
        "parent_v1": {
            "probability_file_sha256": PARENT_PROBABILITY_FILE_SHA256,
            "report_sha256": PARENT_REPORT_SHA256,
            "population_sha256": dict(PARENT_POPULATION_SHA256),
            "status": "fail_preserved",
            "observed_failure": (
                "active prior metrics were close except learned accuracy, but "
                "extreme probabilities caused poor raw held-out log score and "
                "near-single-particle posterior collapse"
            ),
        },
        "new_hypothesis": (
            "one shared decision temperature is an admissible nuisance/cognitive "
            "parameter separating latent relational state from repeated-choice noise"
        ),
        "roles": list(CALIBRATION_ROLES),
        "equal_parameter_budget": (
            "one role-specific scalar temperature for active, passive, random-init, "
            "and uniform-permutation candidates"
        ),
        "temperature_bounds": list(CALIBRATION_TEMPERATURE_BOUNDS),
        "optimization": {
            "method": "fixed_log_spaced_grid_global_argmax",
            "grid_points_including_boundaries": CALIBRATION_LOG_GRID_POINTS,
            "objective": (
                "maximize marginal log likelihood of preregistered cohort blocks 1-5"
            ),
            "amendment": (
                "v2 bounded-optimizer smoke stopped before held-out scoring because "
                "the particle-mixture objective was non-unimodal and its solution "
                "was worse than a boundary; v2.1 fixes a global grid without changing "
                "candidates, data split, bounds, metrics, or gates"
            ),
        },
        "conditioning_blocks": list(CALIBRATION_BLOCKS),
        "strictly_scored_blocks": list(CALIBRATION_SCORED_BLOCKS),
        "metric_seeds": list(CALIBRATION_METRIC_SEEDS),
        "metric_subjects": CALIBRATION_METRIC_SUBJECTS,
        "bootstrap_samples": CALIBRATION_BOOTSTRAP_SAMPLES,
        "gates": dict(CALIBRATION_GATES),
        "adjudication": (
            "both cohorts and both metric seeds must pass; v1 is not overwritten, "
            "and no participant-specific temperature or scored-block fitting is allowed"
        ),
        "claim_ceiling": (
            "a pass would justify recovery analysis for a calibrated frozen candidate; "
            "it would not prove the latent microsequence or authorize core fine-tuning"
        ),
    }
