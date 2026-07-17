"""Behavioral source-task evaluation for published Miconi & Kay checkpoints.

This module ports the single-list and three-episode linked-list protocols from
the authors' ``main.ipynb`` and the pinned readable evaluator.  It deliberately
stops at behavioral source identity: recoded-reinstatement probes and their
time-specific causal lesions are separate, stricter mechanism gates.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from math import sqrt
from collections.abc import Callable
from typing import Literal

import numpy as np
import torch

from fsrl.miconi_kay.source.config import MiconiKaySourceConfig
from fsrl.miconi_kay.source.model import MiconiKayRetroModulRNN
from fsrl.miconi_kay.source.task import build_step_inputs, generate_cue_data


FastStateLesion = Literal[
    "none",
    "no_fast_weight_updates",
    "reset_fast_weights_each_trial",
    "reset_fast_weights_between_segments",
    "block_bridge_reward_step_updates",
    "block_bridge_delay_step_updates",
]
LinkCondition = Literal["true_bridge", "sham"]
HiddenIntervention = Callable[
    [int, int, np.ndarray, torch.Tensor], torch.Tensor
]


@dataclass
class _EvaluationState:
    hidden: torch.Tensor
    eligibility: torch.Tensor
    plastic_weights: torch.Tensor


@dataclass
class _SegmentResult:
    state: _EvaluationState
    test_pairs: np.ndarray
    test_correct: np.ndarray
    test_correct_choice_probability: np.ndarray
    all_pairs_sha256: str
    all_pairs: np.ndarray
    hidden_by_trial_step: np.ndarray | None = None
    response_actions: np.ndarray | None = None
    correct_by_trial: np.ndarray | None = None


@dataclass
class SourceFullTrace:
    """Full single-list trace used only by representation-level probes."""

    cue_data: np.ndarray
    pairs: np.ndarray
    hidden_by_trial_step: np.ndarray
    response_actions: np.ndarray
    correct_by_trial: np.ndarray
    forced_first_exposure_mask: np.ndarray
    target_trial_index: int
    barred_pair: tuple[int, int]


@dataclass
class SourceBehaviorResult:
    """Compact result for one source evaluation condition."""

    condition: str
    lesion: FastStateLesion
    metrics: dict[str, object]
    protocol_audit: dict[str, object]
    mean_absolute_fast_weight: float


@dataclass
class ExhaustiveLinkedListResult:
    """All cross-list queries cloned from one post-bridge state per episode."""

    condition: str
    canonical_pairs: np.ndarray
    sampled_correct: np.ndarray
    correct_choice_probability: np.ndarray
    protocol_audit: dict[str, object]


def _initial_state(
    model: MiconiKayRetroModulRNN, batch_size: int
) -> _EvaluationState:
    return _EvaluationState(
        hidden=model.initialZeroState(batch_size),
        eligibility=model.initialZeroET(batch_size),
        plastic_weights=model.initialZeroPlasticWeights(batch_size),
    )


def _sample_pair(indices: range, adjacent_only: bool) -> list[int]:
    pair = list(np.random.choice(indices, 2, replace=False))
    while adjacent_only and abs(int(pair[0]) - int(pair[1])) > 1:
        pair = list(np.random.choice(indices, 2, replace=False))
    return [int(pair[0]), int(pair[1])]


def _pair_for_trial(
    *,
    item_count: int,
    allowed_items: range,
    trial_index: int,
    train_trials: int,
    link_condition: LinkCondition | None,
) -> list[int]:
    # The notebook samples (and, for training, rejects until adjacent) before
    # overwriting the pair in the bridge/sham segment.  Those apparently
    # wasted NumPy draws are part of fixed-seed parity and affect the test pair.
    sampled_pair = _sample_pair(
        allowed_items,
        adjacent_only=trial_index < train_trials,
    )
    if link_condition is not None and trial_index < train_trials:
        if link_condition == "true_bridge":
            midpoint = item_count // 2
            return (
                [midpoint - 1, midpoint]
                if int(np.random.randint(2))
                else [midpoint, midpoint - 1]
            )
        # Exact original-code sham pair.  The paper describes this control as
        # a non-linking pair; we retain indices [1, 2] rather than relabeling it.
        return [item_count // 2 - 3, item_count // 2 - 2]
    return sampled_pair


def _advance(
    model: MiconiKayRetroModulRNN,
    inputs: torch.Tensor,
    state: _EvaluationState,
    lesion: FastStateLesion,
    *,
    block_update: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, _EvaluationState]:
    previous_plastic_weights = state.plastic_weights
    logits, value, _, hidden, eligibility, plastic_weights = model(
        inputs,
        state.hidden,
        state.eligibility,
        previous_plastic_weights,
    )
    if lesion == "no_fast_weight_updates" or block_update:
        plastic_weights = previous_plastic_weights
    return logits, value, _EvaluationState(hidden, eligibility, plastic_weights)


def _run_segment(
    model: MiconiKayRetroModulRNN,
    base_config: MiconiKaySourceConfig,
    cue_data: list[list[np.ndarray]],
    state: _EvaluationState,
    *,
    allowed_items: range,
    train_trials: int,
    test_trials: int,
    lesion: FastStateLesion,
    link_condition: LinkCondition | None = None,
    record_full_trace: bool = False,
    first_exposure_probe: tuple[int, tuple[int, int]] | None = None,
    hidden_intervention: HiddenIntervention | None = None,
    forced_test_pair: tuple[int, int] | None = None,
    pre_segment_blank_steps: int = 2,
    episode_total_trials: int | None = None,
    episode_step_offset: int = 0,
) -> _SegmentResult:
    """Run one source episode segment, preserving the notebook chronology."""

    config = replace(
        base_config,
        train_trials=train_trials,
        test_trials=test_trials,
    )
    config.validate()
    if pre_segment_blank_steps < 0 or episode_step_offset < 0:
        raise ValueError("blank-step count and episode-step offset must be non-negative")
    total_trials = config.num_trials if episode_total_trials is None else episode_total_trials
    if total_trials < config.num_trials or total_trials <= 0:
        raise ValueError("episode_total_trials must cover every executed trial")
    if episode_step_offset + config.episode_steps > total_trials * config.trial_steps:
        raise ValueError("episode step offset exceeds the declared episode duration")
    input_config = (
        config
        if total_trials == config.num_trials
        else replace(config, train_trials=total_trials, test_trials=0)
    )
    batch_size = config.batch_size
    item_count = config.max_items
    device = model.w.device
    if forced_test_pair is not None:
        if test_trials <= 0:
            raise ValueError("forced_test_pair requires at least one test trial")
        if (
            len(set(forced_test_pair)) != 2
            or min(forced_test_pair) < 0
            or max(forced_test_pair) >= item_count
        ):
            raise ValueError("forced_test_pair contains invalid item indices")

    if lesion == "reset_fast_weights_between_segments":
        state.plastic_weights = model.initialZeroPlasticWeights(batch_size)

    reward = np.zeros(batch_size, dtype="float32")
    previous_actions = np.zeros(batch_size, dtype="int32")
    blank_inputs = torch.zeros(
        batch_size,
        config.input_size,
        device=device,
        dtype=model.w.dtype,
    )
    # These blank steps occur before every episode in the original linked-list
    # protocol and can themselves transform the carried fast state.
    for _ in range(pre_segment_blank_steps):
        _, _, state = _advance(model, blank_inputs, state, lesion)

    all_pairs: list[list[list[int]]] = []
    test_pairs: list[np.ndarray] = []
    test_correct: list[np.ndarray] = []
    test_correct_probability: list[np.ndarray] = []
    hidden_history: list[list[np.ndarray]] = []
    response_history: list[np.ndarray] = []
    correct_history: list[np.ndarray] = []
    episode_step = episode_step_offset

    for trial_index in range(config.num_trials):
        state.hidden = model.initialZeroState(batch_size)
        state.eligibility = model.initialZeroET(batch_size)
        if lesion == "reset_fast_weights_each_trial":
            state.plastic_weights = model.initialZeroPlasticWeights(batch_size)

        pairs: list[list[int]] = []
        for batch_index in range(batch_size):
            pair = _pair_for_trial(
                item_count=item_count,
                allowed_items=allowed_items,
                trial_index=trial_index,
                train_trials=train_trials,
                link_condition=link_condition,
            )
            if (
                first_exposure_probe is not None
                and batch_index > batch_size // 2
            ):
                target_trial, barred_pair = first_exposure_probe
                if trial_index == target_trial:
                    pair = (
                        [barred_pair[0], barred_pair[1]]
                        if int(np.random.randint(2))
                        else [barred_pair[1], barred_pair[0]]
                    )
                elif trial_index < target_trial:
                    # Exact HALFNOBARREDPAIRUNTILT18 schedule: discard the
                    # already sampled pair and draw again until it is adjacent
                    # but not the barred pair.
                    while True:
                        pair = _sample_pair(allowed_items, adjacent_only=True)
                        if set(pair) != set(barred_pair):
                            break
            if forced_test_pair is not None and trial_index >= train_trials:
                first, second = forced_test_pair
                pair = (
                    [first, second]
                    if batch_index < batch_size // 2
                    else [second, first]
                )
            pairs.append(pair)
        all_pairs.append(pairs)
        correct_actions = np.asarray(
            [1 if pair[0] < pair[1] else 0 for pair in pairs],
            dtype="int64",
        )
        cues: list[list[object]] = [
            [pair, item_count]
            + [-1 for _ in range(config.trial_steps - 2)]
            for pair in pairs
        ]

        response_actions: np.ndarray | None = None
        response_correct_probability: np.ndarray | None = None
        trial_hidden: list[np.ndarray] = []
        for step_index in range(config.trial_steps):
            inputs = build_step_inputs(
                input_config,
                item_count,
                cue_data,
                cues,
                reward,
                previous_actions,
                step_index,
                episode_step,
                device=device,
            )
            is_bridge_training = (
                link_condition == "true_bridge" and trial_index < train_trials
            )
            # Paper step numbers are one-based: Python step 2 is reward
            # delivery (paper step 3), and Python step 3 is the following
            # delay (paper step 4).  Only bridge-trial updates are blocked, so
            # acquisition of the two component lists is held fixed.
            block_update = is_bridge_training and (
                (
                    lesion == "block_bridge_reward_step_updates"
                    and step_index == config.response_step + 1
                )
                or (
                    lesion == "block_bridge_delay_step_updates"
                    and step_index == config.response_step + 2
                )
            )
            logits, _, state = _advance(
                model,
                inputs,
                state,
                lesion,
                block_update=block_update,
            )
            if hidden_intervention is not None:
                intervened_hidden = hidden_intervention(
                    trial_index,
                    step_index,
                    np.asarray(pairs, dtype="int16"),
                    state.hidden,
                )
                if intervened_hidden.shape != state.hidden.shape:
                    raise ValueError("hidden intervention changed tensor shape")
                if not torch.isfinite(intervened_hidden).all():
                    raise ValueError("hidden intervention produced non-finite values")
                state.hidden = intervened_hidden
            if record_full_trace:
                trial_hidden.append(
                    state.hidden.detach().cpu().numpy().astype("float32", copy=True)
                )
            probabilities = torch.softmax(logits, dim=1)
            actions = torch.distributions.Categorical(probabilities).sample()
            previous_actions = actions.detach().cpu().numpy()

            reward = np.zeros(batch_size, dtype="float32")
            if step_index == config.response_step:
                response_actions = previous_actions.copy()
                correct_index = torch.from_numpy(correct_actions).to(device)
                response_correct_probability = (
                    probabilities.gather(1, correct_index[:, None])[:, 0]
                    .detach()
                    .cpu()
                    .numpy()
                )
                reward = np.where(
                    response_actions == correct_actions,
                    config.reward_amount,
                    -config.reward_amount,
                ).astype("float32")
            episode_step += 1

        if trial_index >= train_trials:
            if response_actions is None or response_correct_probability is None:
                raise RuntimeError("test trial had no response step")
            test_pairs.append(np.asarray(pairs, dtype="int16"))
            test_correct.append(
                (response_actions == correct_actions).astype("int8")
            )
            test_correct_probability.append(response_correct_probability)
        if record_full_trace:
            if response_actions is None:
                raise RuntimeError("recorded trial had no response")
            hidden_history.append(trial_hidden)
            response_history.append(response_actions.astype("int8", copy=True))
            correct_history.append(
                (response_actions == correct_actions).astype("int8")
            )

    all_pairs_array = np.asarray(all_pairs, dtype="int16").transpose(1, 0, 2)
    if test_pairs:
        test_pair_array = np.asarray(test_pairs, dtype="int16").transpose(1, 0, 2)
        test_correct_array = np.asarray(test_correct, dtype="int8").transpose(1, 0)
        test_probability_array = np.asarray(
            test_correct_probability, dtype="float32"
        ).transpose(1, 0)
    else:
        test_pair_array = np.empty((batch_size, 0, 2), dtype="int16")
        test_correct_array = np.empty((batch_size, 0), dtype="int8")
        test_probability_array = np.empty((batch_size, 0), dtype="float32")

    return _SegmentResult(
        state=state,
        test_pairs=test_pair_array,
        test_correct=test_correct_array,
        test_correct_choice_probability=test_probability_array,
        all_pairs_sha256=hashlib.sha256(all_pairs_array.tobytes()).hexdigest(),
        all_pairs=all_pairs_array,
        hidden_by_trial_step=(
            np.asarray(hidden_history, dtype="float32").transpose(2, 0, 1, 3)
            if record_full_trace
            else None
        ),
        response_actions=(
            np.asarray(response_history, dtype="int8").transpose(1, 0)
            if record_full_trace
            else None
        ),
        correct_by_trial=(
            np.asarray(correct_history, dtype="int8").transpose(1, 0)
            if record_full_trace
            else None
        ),
    )


def collect_single_list_trace(
    model: MiconiKayRetroModulRNN,
    config: MiconiKaySourceConfig,
    *,
    target_trial_index: int = 18,
    barred_pair: tuple[int, int] = (3, 4),
    hidden_intervention: HiddenIntervention | None = None,
) -> SourceFullTrace:
    """Collect the authors' first-exposure schedule for representation probes."""

    config.validate()
    if config.min_items != config.max_items or config.max_items != 8:
        raise ValueError("reinstatement trace requires exactly eight items")
    if not 0 <= target_trial_index < config.train_trials:
        raise ValueError("target trial must lie in the adjacent-pair training period")
    if abs(barred_pair[0] - barred_pair[1]) != 1:
        raise ValueError("barred pair must be adjacent")
    model.eval()
    cue_data = generate_cue_data(config, config.max_items)
    state = _initial_state(model, config.batch_size)
    with torch.no_grad():
        segment = _run_segment(
            model,
            config,
            cue_data,
            state,
            allowed_items=range(config.max_items),
            train_trials=config.train_trials,
            test_trials=config.test_trials,
            lesion="none",
            record_full_trace=True,
            first_exposure_probe=(target_trial_index, barred_pair),
            hidden_intervention=hidden_intervention,
        )
    if (
        segment.hidden_by_trial_step is None
        or segment.response_actions is None
        or segment.correct_by_trial is None
    ):
        raise RuntimeError("full source trace was not recorded")
    forced_mask = np.arange(config.batch_size) > config.batch_size // 2
    target_pairs = segment.all_pairs[forced_mask, target_trial_index]
    if not all(set(map(int, pair)) == set(barred_pair) for pair in target_pairs):
        raise AssertionError("forced first-exposure target pair was not presented")
    prior_pairs = segment.all_pairs[forced_mask, :target_trial_index]
    if any(set(map(int, pair)) == set(barred_pair) for pair in prior_pairs.reshape(-1, 2)):
        raise AssertionError("barred pair leaked into the pre-target schedule")
    return SourceFullTrace(
        cue_data=np.asarray(cue_data, dtype="float32"),
        pairs=segment.all_pairs,
        hidden_by_trial_step=segment.hidden_by_trial_step,
        response_actions=segment.response_actions,
        correct_by_trial=segment.correct_by_trial,
        forced_first_exposure_mask=forced_mask,
        target_trial_index=target_trial_index,
        barred_pair=barred_pair,
    )


def _wilson_interval(successes: int, count: int) -> list[float] | None:
    if count == 0:
        return None
    z = 1.959963984540054
    proportion = successes / count
    denominator = 1 + z * z / count
    center = (proportion + z * z / (2 * count)) / denominator
    half_width = (
        z
        * sqrt(
            proportion * (1 - proportion) / count
            + z * z / (4 * count * count)
        )
        / denominator
    )
    return [center - half_width, center + half_width]


def _binary_metric(values: np.ndarray) -> dict[str, object]:
    flattened = values.astype("int64", copy=False).reshape(-1)
    count = int(flattened.size)
    successes = int(flattened.sum())
    return {
        "accuracy": None if count == 0 else successes / count,
        "successes": successes,
        "count": count,
        "wilson95": _wilson_interval(successes, count),
    }


def _behavior_metrics(
    pairs: np.ndarray,
    correct: np.ndarray,
    correct_choice_probability: np.ndarray,
    *,
    item_count: int,
) -> dict[str, object]:
    pair_rows = pairs.reshape(-1, 2)
    correctness = correct.reshape(-1)
    probability = correct_choice_probability.reshape(-1)
    midpoint = item_count // 2
    left = pair_rows[:, 0] < midpoint
    right = pair_rows[:, 1] < midpoint
    cross_list = left != right
    distance = np.abs(pair_rows[:, 0] - pair_rows[:, 1])

    per_distance = {
        str(symbolic_distance): _binary_metric(
            correctness[distance == symbolic_distance]
        )
        for symbolic_distance in range(1, item_count)
    }
    per_pair: dict[str, object] = {}
    for first in range(item_count):
        for second in range(first + 1, item_count):
            mask = (
                ((pair_rows[:, 0] == first) & (pair_rows[:, 1] == second))
                | ((pair_rows[:, 0] == second) & (pair_rows[:, 1] == first))
            )
            per_pair[f"{first}-{second}"] = _binary_metric(correctness[mask])

    cross_pair_metrics = [
        per_pair[f"{first}-{second}"]
        for first in range(midpoint)
        for second in range(midpoint, item_count)
    ]
    cross_pair_intervals = [
        metric["wilson95"]
        for metric in cross_pair_metrics
        if metric["wilson95"] is not None
    ]
    cross_pair_accuracies = [
        float(metric["accuracy"])
        for metric in cross_pair_metrics
        if metric["accuracy"] is not None
    ]
    cross_pair_coherence = {
        "expected_pair_count": midpoint * (item_count - midpoint),
        "observed_pair_count": len(cross_pair_accuracies),
        "minimum_trials_per_pair": min(
            int(metric["count"]) for metric in cross_pair_metrics
        ),
        "macro_accuracy": (
            None
            if not cross_pair_accuracies
            else float(np.mean(cross_pair_accuracies))
        ),
        "minimum_pair_accuracy": (
            None if not cross_pair_accuracies else min(cross_pair_accuracies)
        ),
        "minimum_wilson95_lower": (
            None
            if not cross_pair_intervals
            else min(float(interval[0]) for interval in cross_pair_intervals)
        ),
        "reliably_at_or_below_chance_pair_count": sum(
            float(interval[1]) <= 0.5 for interval in cross_pair_intervals
        ),
    }

    return {
        "overall": _binary_metric(correctness),
        "adjacent": _binary_metric(correctness[distance == 1]),
        "nonadjacent": _binary_metric(correctness[distance > 1]),
        "within_list": _binary_metric(correctness[~cross_list]),
        "cross_list": _binary_metric(correctness[cross_list]),
        "cross_list_pair_coherence": cross_pair_coherence,
        "mean_correct_choice_probability": (
            None if probability.size == 0 else float(probability.mean())
        ),
        "per_symbolic_distance": per_distance,
        "per_unordered_pair": per_pair,
    }


def evaluate_single_list(
    model: MiconiKayRetroModulRNN,
    config: MiconiKaySourceConfig,
    *,
    lesion: FastStateLesion = "none",
) -> SourceBehaviorResult:
    """Run the 20-adjacent-train/10-unrestricted-test source condition."""

    config.validate()
    if config.min_items != config.max_items:
        raise ValueError("source evaluation requires a fixed item count")
    model.eval()
    cue_data = generate_cue_data(config, config.max_items)
    state = _initial_state(model, config.batch_size)
    with torch.no_grad():
        segment = _run_segment(
            model,
            config,
            cue_data,
            state,
            allowed_items=range(config.max_items),
            train_trials=config.train_trials,
            test_trials=config.test_trials,
            lesion=lesion,
        )

    train_pairs = segment.all_pairs[:, : config.train_trials, :]
    train_nonadjacent = int(
        np.sum(np.abs(train_pairs[:, :, 0] - train_pairs[:, :, 1]) != 1)
    )
    return SourceBehaviorResult(
        condition="single_list",
        lesion=lesion,
        metrics=_behavior_metrics(
            segment.test_pairs,
            segment.test_correct,
            segment.test_correct_choice_probability,
            item_count=config.max_items,
        ),
        protocol_audit={
            "train_trials": config.train_trials,
            "test_trials": config.test_trials,
            "training_nonadjacent_violations": train_nonadjacent,
            "all_pairs_sha256": segment.all_pairs_sha256,
            "test_pairs_sha256": hashlib.sha256(
                segment.test_pairs.tobytes()
            ).hexdigest(),
            "test_correct_sha256": hashlib.sha256(
                segment.test_correct.tobytes()
            ).hexdigest(),
        },
        mean_absolute_fast_weight=float(
            segment.state.plastic_weights.abs().mean().cpu()
        ),
    )


def evaluate_linked_lists(
    model: MiconiKayRetroModulRNN,
    config: MiconiKaySourceConfig,
    *,
    link_condition: LinkCondition,
    lesion: FastStateLesion = "none",
) -> SourceBehaviorResult:
    """Run the exact three-segment linked-list or non-linking sham protocol."""

    config.validate()
    item_count = config.max_items
    if config.min_items != item_count or item_count != 8:
        raise ValueError("published linked-list evaluation requires exactly 8 items")
    if config.trial_steps != 4 or config.response_step != 1:
        raise ValueError(
            "published linked-list evaluation requires four steps with response at step 1"
        )
    model.eval()
    cue_data = generate_cue_data(config, item_count)
    state = _initial_state(model, config.batch_size)
    midpoint = item_count // 2
    segment_specs = (
        (range(0, midpoint), 10, 0, None),
        (range(midpoint, item_count), 10, 0, None),
        (
            range(0, item_count),
            4 if link_condition == "true_bridge" else 1,
            1,
            link_condition,
        ),
    )
    segments: list[_SegmentResult] = []
    with torch.no_grad():
        for allowed_items, train_trials, test_trials, segment_condition in segment_specs:
            segment = _run_segment(
                model,
                config,
                cue_data,
                state,
                allowed_items=allowed_items,
                train_trials=train_trials,
                test_trials=test_trials,
                lesion=lesion,
                link_condition=segment_condition,
            )
            state = segment.state
            segments.append(segment)

    first_pairs = segments[0].all_pairs
    second_pairs = segments[1].all_pairs
    final_pairs = segments[2].all_pairs
    bridge_train_trials = segment_specs[2][1]
    expected_bridge = {3, 4}
    expected_sham = [1, 2]
    final_training = final_pairs[:, :bridge_train_trials, :]
    if link_condition == "true_bridge":
        final_training_violations = sum(
            set(map(int, pair)) != expected_bridge
            for pair in final_training.reshape(-1, 2)
        )
    else:
        final_training_violations = int(
            np.sum(np.any(final_training != expected_sham, axis=2))
        )

    test_segment = segments[2]
    return SourceBehaviorResult(
        condition=f"linked_list_{link_condition}",
        lesion=lesion,
        metrics=_behavior_metrics(
            test_segment.test_pairs,
            test_segment.test_correct,
            test_segment.test_correct_choice_probability,
            item_count=item_count,
        ),
        protocol_audit={
            "segments": [
                {"train_trials": 10, "test_trials": 0, "items": [0, 1, 2, 3]},
                {"train_trials": 10, "test_trials": 0, "items": [4, 5, 6, 7]},
                {
                    "train_trials": bridge_train_trials,
                    "test_trials": 1,
                    "training_pair": (
                        [3, 4] if link_condition == "true_bridge" else [1, 2]
                    ),
                },
            ],
            "first_segment_range_violations": int(
                np.sum((first_pairs < 0) | (first_pairs >= midpoint))
            ),
            "second_segment_range_violations": int(
                np.sum((second_pairs < midpoint) | (second_pairs >= item_count))
            ),
            "first_segment_nonadjacent_violations": int(
                np.sum(np.abs(first_pairs[:, :, 0] - first_pairs[:, :, 1]) != 1)
            ),
            "second_segment_nonadjacent_violations": int(
                np.sum(np.abs(second_pairs[:, :, 0] - second_pairs[:, :, 1]) != 1)
            ),
            "final_training_pair_violations": int(final_training_violations),
            "segment_pair_sha256": [segment.all_pairs_sha256 for segment in segments],
            "test_pairs_sha256": hashlib.sha256(
                test_segment.test_pairs.tobytes()
            ).hexdigest(),
            "test_correct_sha256": hashlib.sha256(
                test_segment.test_correct.tobytes()
            ).hexdigest(),
            "fast_weights_carried_across_segments": lesion
            not in {
                "no_fast_weight_updates",
                "reset_fast_weights_between_segments",
            },
            "bridge_update_block": (
                "paper_step_3_reward_delivery"
                if lesion == "block_bridge_reward_step_updates"
                else (
                    "paper_step_4_delay"
                    if lesion == "block_bridge_delay_step_updates"
                    else None
                )
            ),
        },
        mean_absolute_fast_weight=float(state.plastic_weights.abs().mean().cpu()),
    )


def _clone_evaluation_state(state: _EvaluationState) -> _EvaluationState:
    return _EvaluationState(
        hidden=state.hidden.clone(),
        eligibility=state.eligibility.clone(),
        plastic_weights=state.plastic_weights.clone(),
    )


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def evaluate_linked_lists_exhaustive(
    model: MiconiKayRetroModulRNN,
    config: MiconiKaySourceConfig,
    *,
    link_condition: LinkCondition,
) -> ExhaustiveLinkedListResult:
    """Query every cross-list pair from cloned, identical post-link states.

    The original source protocol samples one unrestricted test pair per
    episode.  That is sufficient for aggregate accuracy but gives different
    episode-fast states to different unordered pairs.  This diagnostic runs
    the two component lists and the bridge/sham training once, then clones the
    resulting fast state and presents each of the 16 cross-list pairs.  The
    time channel exactly matches the held-out test trial of the original final
    segment; no extra blank steps occur between bridge training and query.
    """

    config.validate()
    item_count = config.max_items
    if config.min_items != item_count or item_count != 8:
        raise ValueError("exhaustive linked-list evaluation requires exactly 8 items")
    if config.trial_steps != 4 or config.response_step != 1:
        raise ValueError(
            "exhaustive linked-list evaluation requires four steps with response at step 1"
        )
    model.eval()
    cue_data = generate_cue_data(config, item_count)
    state = _initial_state(model, config.batch_size)
    midpoint = item_count // 2

    with torch.no_grad():
        first = _run_segment(
            model,
            config,
            cue_data,
            state,
            allowed_items=range(0, midpoint),
            train_trials=10,
            test_trials=0,
            lesion="none",
        )
        second = _run_segment(
            model,
            config,
            cue_data,
            first.state,
            allowed_items=range(midpoint, item_count),
            train_trials=10,
            test_trials=0,
            lesion="none",
        )
        bridge_train_trials = 4 if link_condition == "true_bridge" else 1
        final_total_trials = bridge_train_trials + 1
        bridge = _run_segment(
            model,
            config,
            cue_data,
            second.state,
            allowed_items=range(item_count),
            train_trials=bridge_train_trials,
            test_trials=0,
            lesion="none",
            link_condition=link_condition,
            episode_total_trials=final_total_trials,
        )

        numpy_query_state = np.random.get_state()
        torch_cpu_query_state = torch.get_rng_state()
        device = model.w.device
        torch_device_query_state = (
            torch.cuda.get_rng_state(device)
            if device.type == "cuda"
            else None
        )
        canonical_pairs = np.asarray(
            [
                (left, right)
                for left in range(midpoint)
                for right in range(midpoint, item_count)
            ],
            dtype="int16",
        )
        correct_columns: list[np.ndarray] = []
        probability_columns: list[np.ndarray] = []
        forced_pair_violations = 0
        for left, right in canonical_pairs:
            np.random.set_state(numpy_query_state)
            torch.set_rng_state(torch_cpu_query_state)
            if torch_device_query_state is not None:
                torch.cuda.set_rng_state(torch_device_query_state, device)
            query = _run_segment(
                model,
                config,
                cue_data,
                _clone_evaluation_state(bridge.state),
                allowed_items=range(item_count),
                train_trials=0,
                test_trials=1,
                lesion="none",
                forced_test_pair=(int(left), int(right)),
                pre_segment_blank_steps=0,
                episode_total_trials=final_total_trials,
                episode_step_offset=bridge_train_trials * config.trial_steps,
            )
            forced_pair_violations += sum(
                set(map(int, pair)) != {int(left), int(right)}
                for pair in query.test_pairs[:, 0, :]
            )
            correct_columns.append(query.test_correct[:, 0].copy())
            probability_columns.append(
                query.test_correct_choice_probability[:, 0].copy()
            )

    bridge_training = bridge.all_pairs
    expected_training_pair = {3, 4} if link_condition == "true_bridge" else {1, 2}
    bridge_training_violations = sum(
        set(map(int, pair)) != expected_training_pair
        for pair in bridge_training.reshape(-1, 2)
    )
    return ExhaustiveLinkedListResult(
        condition=f"linked_list_{link_condition}_exhaustive",
        canonical_pairs=canonical_pairs,
        sampled_correct=np.stack(correct_columns, axis=1).astype("int8", copy=False),
        correct_choice_probability=np.stack(probability_columns, axis=1).astype(
            "float32", copy=False
        ),
        protocol_audit={
            "component_segment_pair_sha256": [
                first.all_pairs_sha256,
                second.all_pairs_sha256,
            ],
            "bridge_training_pair_sha256": bridge.all_pairs_sha256,
            "bridge_training_trials": bridge_train_trials,
            "bridge_training_pair_violations": int(bridge_training_violations),
            "post_bridge_fast_weight_sha256": _tensor_sha256(
                bridge.state.plastic_weights
            ),
            "post_bridge_mean_absolute_fast_weight": float(
                bridge.state.plastic_weights.abs().mean().cpu()
            ),
            "query_count_per_episode": int(canonical_pairs.shape[0]),
            "query_common_random_numbers": True,
            "query_extra_blank_steps": 0,
            "query_episode_step_offset": bridge_train_trials * config.trial_steps,
            "query_episode_total_steps": final_total_trials * config.trial_steps,
            "balanced_query_orientation": True,
            "forced_query_pair_violations": int(forced_pair_violations),
            "cue_data_sha256": hashlib.sha256(
                np.asarray(cue_data, dtype="float32").tobytes()
            ).hexdigest(),
        },
    )


def accuracy_difference(
    left: dict[str, object], right: dict[str, object]
) -> dict[str, object]:
    """Unpaired normal-approximation interval for two simulation proportions."""

    left_count = int(left["count"])
    right_count = int(right["count"])
    if left_count == 0 or right_count == 0:
        return {"estimate": None, "normal95": None, "counts": [left_count, right_count]}
    left_accuracy = float(left["accuracy"])
    right_accuracy = float(right["accuracy"])
    difference = left_accuracy - right_accuracy
    standard_error = sqrt(
        left_accuracy * (1 - left_accuracy) / left_count
        + right_accuracy * (1 - right_accuracy) / right_count
    )
    half_width = 1.959963984540054 * standard_error
    return {
        "estimate": difference,
        "normal95": [difference - half_width, difference + half_width],
        "counts": [left_count, right_count],
    }
