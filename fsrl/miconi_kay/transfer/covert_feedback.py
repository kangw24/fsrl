"""Frozen M&K candidate with an explicit covert-feedback microprocess."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import torch

from fsrl.miconi_kay.source.config import MiconiKaySourceConfig
from fsrl.miconi_kay.source.evaluation import (
    _EvaluationState,
    _clone_evaluation_state,
    _run_segment,
    _tensor_sha256,
)
from fsrl.miconi_kay.source.model import MiconiKayRetroModulRNN
from fsrl.miconi_kay.source.reinstatement import tensor_sha256
from fsrl.miconi_kay.source.task import build_step_inputs, generate_cue_data


@dataclass(frozen=True)
class CovertFeedbackPopulation:
    canonical_pairs: np.ndarray
    correct_choice_probability: np.ndarray
    protocol_audit: dict[str, object]


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _support_schedule(
    batch_size: int,
    support_pairs: tuple[tuple[int, int], ...],
    repetitions: int,
) -> np.ndarray:
    base = np.asarray(support_pairs * repetitions, dtype="int16")
    schedule = np.empty((batch_size, base.shape[0], 2), dtype="int16")
    for batch_index in range(batch_size):
        order = np.random.permutation(base.shape[0])
        schedule[batch_index] = base[order]
        reverse = np.random.randint(0, 2, size=base.shape[0]).astype(bool)
        schedule[batch_index, reverse] = schedule[batch_index, reverse, ::-1]
    return schedule


def simulate_covert_feedback_population(
    model: MiconiKayRetroModulRNN,
    config: MiconiKaySourceConfig,
    *,
    support_pairs: tuple[tuple[int, int], ...],
    support_repetitions: int,
) -> CovertFeedbackPopulation:
    """Generate virtual subjects without reading any human test choice.

    The visible Liu relation determines whether the network's covert response
    was correct.  That signed internal prediction error is present on the
    feedback step, whose fast-weight write is blocked.  The following delay
    step is allowed to write.  Hidden state and eligibility reset per support
    observation; episode fast weights persist.
    """

    config.validate()
    if config.min_items != 8 or config.max_items != 8:
        raise ValueError("covert-feedback transfer requires exactly eight items")
    if config.trial_steps != 4 or config.response_step != 1:
        raise ValueError("candidate requires the pinned four-step source timing")
    expected_trials = len(support_pairs) * support_repetitions
    if config.train_trials != expected_trials or config.test_trials != 0:
        raise ValueError("config trial count must equal the repeated support schedule")
    if len(set(support_pairs)) != len(support_pairs):
        raise ValueError("support pairs must be unique before repetition")
    if any(not (0 <= left < right < 8) for left, right in support_pairs):
        raise ValueError("support pairs must be canonical eight-item pairs")

    model.eval()
    device = model.w.device
    cue_data = generate_cue_data(config, 8)
    cue_array = np.asarray(cue_data, dtype="int8")
    schedule = _support_schedule(config.batch_size, support_pairs, support_repetitions)
    state = _EvaluationState(
        hidden=model.initialZeroState(config.batch_size),
        eligibility=model.initialZeroET(config.batch_size),
        plastic_weights=model.initialZeroPlasticWeights(config.batch_size),
    )
    blank_inputs = torch.zeros(
        config.batch_size,
        config.input_size,
        device=device,
        dtype=model.w.dtype,
    )
    covert_responses = np.empty(
        (config.batch_size, expected_trials), dtype="int8"
    )
    signed_prediction_errors = np.empty_like(covert_responses)
    feedback_fast_delta = np.empty(
        (config.batch_size, expected_trials), dtype="float32"
    )
    delay_fast_delta = np.empty_like(feedback_fast_delta)

    with torch.no_grad():
        for _ in range(2):
            _, _, _, hidden, eligibility, plastic = model(
                blank_inputs,
                state.hidden,
                state.eligibility,
                state.plastic_weights,
            )
            state = _EvaluationState(hidden, eligibility, plastic)
        episode_step = 0
        reward = np.zeros(config.batch_size, dtype="float32")
        previous_actions = np.zeros(config.batch_size, dtype="int32")
        for trial_index in range(expected_trials):
            state.hidden = model.initialZeroState(config.batch_size)
            state.eligibility = model.initialZeroET(config.batch_size)
            pairs = schedule[:, trial_index, :]
            cues = [
                [list(map(int, pair)), 8, -1, -1]
                for pair in pairs
            ]
            reward.fill(0.0)
            for step_index in range(config.trial_steps):
                inputs = build_step_inputs(
                    config,
                    8,
                    cue_data,
                    cues,
                    reward,
                    previous_actions,
                    step_index,
                    episode_step,
                    device=device,
                )
                previous_plastic = state.plastic_weights
                logits, _, _, hidden, eligibility, plastic = model(
                    inputs,
                    state.hidden,
                    state.eligibility,
                    previous_plastic,
                )
                if step_index == config.response_step + 1:
                    # The source write-timing diagnostic selected a structural
                    # feedback/delay separation, not a fitted numeric gate.
                    plastic = previous_plastic
                state = _EvaluationState(hidden, eligibility, plastic)
                actions = torch.distributions.Categorical(
                    torch.softmax(logits, dim=1)
                ).sample()
                previous_actions = actions.detach().cpu().numpy().astype("int32")
                next_reward = np.zeros(config.batch_size, dtype="float32")
                if step_index == config.response_step:
                    correct_actions = (pairs[:, 0] < pairs[:, 1]).astype("int32")
                    correct = previous_actions == correct_actions
                    next_reward = np.where(correct, 1.0, -1.0).astype("float32")
                    covert_responses[:, trial_index] = previous_actions.astype("int8")
                    signed_prediction_errors[:, trial_index] = next_reward.astype("int8")
                if step_index == config.response_step + 1:
                    feedback_fast_delta[:, trial_index] = (
                        state.plastic_weights - previous_plastic
                    ).flatten(1).norm(dim=1).cpu().numpy()
                if step_index == config.response_step + 2:
                    delay_fast_delta[:, trial_index] = (
                        state.plastic_weights - previous_plastic
                    ).flatten(1).norm(dim=1).cpu().numpy()
                reward = next_reward
                episode_step += 1

        query_numpy = np.random.get_state()
        query_torch_cpu = torch.get_rng_state()
        query_torch_device = (
            torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        )
        canonical_pairs = np.asarray(
            [(left, right) for left in range(8) for right in range(left + 1, 8)],
            dtype="int16",
        )
        probability_columns: list[np.ndarray] = []
        forced_pair_violations = 0
        total_trials = expected_trials + 1
        for left, right in canonical_pairs:
            np.random.set_state(query_numpy)
            torch.set_rng_state(query_torch_cpu)
            if query_torch_device is not None:
                torch.cuda.set_rng_state(query_torch_device, device)
            query = _run_segment(
                model,
                config,
                cue_data,
                _clone_evaluation_state(state),
                allowed_items=range(8),
                train_trials=0,
                test_trials=1,
                lesion="none",
                forced_test_pair=(int(left), int(right)),
                pre_segment_blank_steps=0,
                episode_total_trials=total_trials,
                episode_step_offset=expected_trials * config.trial_steps,
            )
            forced_pair_violations += sum(
                set(map(int, pair)) != {int(left), int(right)}
                for pair in query.test_pairs[:, 0, :]
            )
            probability_columns.append(
                query.test_correct_choice_probability[:, 0].copy()
            )
        probabilities = np.stack(probability_columns, axis=1).astype(
            "float32", copy=False
        )

    return CovertFeedbackPopulation(
        canonical_pairs=canonical_pairs,
        correct_choice_probability=probabilities,
        protocol_audit={
            "support_schedule_sha256": _array_sha256(schedule),
            "cue_data_sha256": _array_sha256(cue_array),
            "covert_response_sha256": _array_sha256(covert_responses),
            "signed_prediction_error_sha256": _array_sha256(
                signed_prediction_errors
            ),
            "post_support_fast_weight_sha256": _tensor_sha256(
                state.plastic_weights
            ),
            "probability_sha256": tensor_sha256(probabilities),
            "feedback_step_fast_delta_max": float(feedback_fast_delta.max()),
            "delay_step_fast_delta_mean": float(delay_fast_delta.mean()),
            "support_trials": expected_trials,
            "support_repetitions": support_repetitions,
            "human_test_choices_read": False,
            "core_or_adapter_training": False,
            "feedback_step_write_blocked": True,
            "delay_step_write_allowed": True,
            "forced_query_pair_violations": int(forced_pair_violations),
        },
    )
