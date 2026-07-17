"""Locked local protocol for the post-bridge recurrence diagnostic."""

from __future__ import annotations


POSTBRIDGE_PROTOCOL_VERSION = 1
POSTBRIDGE_FORMAL_SEEDS = (141, 142)
POSTBRIDGE_BLANK_COUNTS = (0, 1, 2, 4, 8)
POSTBRIDGE_PRIMARY_BLANK_COUNT = 4
POSTBRIDGE_CONFIGURATION = {
    "batch_size": 512,
    "bootstrap_samples": 2000,
    "blank_counts": list(POSTBRIDGE_BLANK_COUNTS),
    "primary_blank_count": POSTBRIDGE_PRIMARY_BLANK_COUNT,
    "full_recurrence_at_all_blank_counts": True,
    "freeze_fast_write_control_at_primary_count": True,
    "query_time_channel_held_at_original_value": True,
    "query_hidden_and_eligibility_reset_as_in_source": True,
}
POSTBRIDGE_THRESHOLDS = {
    "minimum_batch_size": 512,
    "minimum_active_primary_pair_probability_lower_bound": 0.55,
    "minimum_active_primary_pair_sampled_lower_bound": 0.55,
    "minimum_active_true_recurrence_gain_lower_bound": 0.03,
    "minimum_true_over_sham_gain_specificity_lower_bound": 0.02,
    "minimum_active_over_passive_gain_specificity_lower_bound": 0.02,
    "minimum_fast_write_dependence_lower_bound": 0.02,
}


def postbridge_recurrence_plan() -> dict[str, object]:
    return {
        "protocol_version": POSTBRIDGE_PROTOCOL_VERSION,
        "formal_seeds": list(POSTBRIDGE_FORMAL_SEEDS),
        "configuration": dict(POSTBRIDGE_CONFIGURATION),
        "thresholds": dict(POSTBRIDGE_THRESHOLDS),
        "mechanism_hypothesis": (
            "time-localized post-bridge recurrence can consolidate the public "
            "active model's carried eligibility into episode fast weights, "
            "improving all-pair list linking without changing slow/core weights"
        ),
        "primary_contrast": (
            "four full-recurrence zero-input steps versus zero extra steps; "
            "four steps equal one source-trial duration and are fixed before data"
        ),
        "mechanism_control": (
            "repeat the four zero-input steps while blocking every fast-weight "
            "write; hidden and eligibility dynamics remain enabled"
        ),
        "secondary_dose_curve": (
            "0/1/2/4/8 full-recurrence steps are descriptive only; no best-dose "
            "selection or substitution for the registered four-step contrast"
        ),
        "adjudication_rule": (
            "both formal seeds must pass every gate; success at another blank "
            "count cannot rescue a four-step failure"
        ),
        "claim_ceiling": (
            "parameter-free source mechanism diagnostic only; no Liu or human-"
            "mechanism claim and no transfer authorization"
        ),
    }
