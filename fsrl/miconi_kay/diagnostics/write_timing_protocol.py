"""Locked protocol for exhaustive bridge write-timing evaluation."""

from __future__ import annotations


WRITE_TIMING_PROTOCOL_VERSION = 1
WRITE_TIMING_FORMAL_SEEDS = (151, 152)
WRITE_TIMING_MODES = ("baseline", "block_reward_step", "block_delay_step")
WRITE_TIMING_CONFIGURATION = {
    "batch_size": 512,
    "bootstrap_samples": 2000,
    "modes": list(WRITE_TIMING_MODES),
    "all_sixteen_cross_pairs_from_cloned_state": True,
}
WRITE_TIMING_THRESHOLDS = {
    "minimum_reward_block_pair_probability_lower_bound": 0.55,
    "minimum_reward_block_pair_sampled_lower_bound": 0.55,
    "minimum_reward_block_gain_lower_bound": 0.05,
    "minimum_reward_over_delay_advantage_lower_bound": 0.10,
    "minimum_active_over_passive_gain_lower_bound": 0.05,
    "minimum_reward_block_true_over_sham_lower_bound": 0.15,
    "minimum_delay_step_necessity_lower_bound": 0.15,
}


def write_timing_plan() -> dict[str, object]:
    return {
        "protocol_version": WRITE_TIMING_PROTOCOL_VERSION,
        "formal_seeds": list(WRITE_TIMING_FORMAL_SEEDS),
        "configuration": dict(WRITE_TIMING_CONFIGURATION),
        "thresholds": dict(WRITE_TIMING_THRESHOLDS),
        "known_before_protocol": (
            "sparse-query macro evaluations already showed that blocking the "
            "bridge reward-step update improves public active behavior while "
            "blocking the following delay-step update harms it; those macro "
            "directions are replication checks, not novel held-out evidence"
        ),
        "held_out_question": (
            "whether reward-step blocking repairs probability and sampled "
            "coherence for every one of the sixteen cloned cross-list pairs"
        ),
        "adjudication_rule": (
            "both formal seeds must pass every gate; no mode, pair, or seed "
            "selection is allowed"
        ),
        "claim_ceiling": (
            "parameter-free source write-timing diagnostic only; a hard lesion "
            "is not itself a human mechanism and cannot authorize Liu transfer"
        ),
    }
