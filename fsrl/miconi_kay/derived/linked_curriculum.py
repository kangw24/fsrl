"""Source-only linked-list curriculum for an M&K-derived checkpoint.

This module is deliberately outside :mod:`fsrl.miconi_kay.source`: it changes
the outer training distribution and therefore must not inherit source parity.
It retains the source trial chronology and actor-critic semantics:

``pair -> sampled action -> action-contingent reward -> next-step input ->
neuromodulated fast-weight update``.

No Liu graph, human choice, participant identifier, teacher sign, or endpoint
metric is available to this code.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch

from fsrl.miconi_kay.source.config import MiconiKaySourceConfig
from fsrl.miconi_kay.source.model import MiconiKayRetroModulRNN
from fsrl.miconi_kay.source.task import build_step_inputs, generate_cue_data


@dataclass(frozen=True)
class LinkedCurriculumConfig:
    component_trials: int = 10
    bridge_trials: int = 4
    query_loss_multiplier: float = 3.0

    def validate(self) -> None:
        if self.component_trials <= 0 or self.bridge_trials <= 0:
            raise ValueError("component and bridge trial counts must be positive")
        if self.query_loss_multiplier <= 0:
            raise ValueError("query loss multiplier must be positive")


@dataclass
class _DifferentiableState:
    hidden: torch.Tensor
    eligibility: torch.Tensor
    plastic_weights: torch.Tensor


@dataclass
class _StepRecords:
    log_probabilities: list[torch.Tensor]
    values: list[torch.Tensor]
    probabilities: list[torch.Tensor]
    rewards: list[np.ndarray]


@dataclass
class LinkedCurriculumTrace:
    query_pairs: np.ndarray
    query_response_actions: np.ndarray
    query_rewards_after_response: np.ndarray
    query_feedback_reward_inputs: np.ndarray
    query_feedback_action_inputs: np.ndarray


@dataclass
class LinkedCurriculumStats:
    loss: torch.Tensor
    loss_value: float
    query_accuracy: float
    minimum_pair_accuracy: float
    mean_absolute_post_bridge_fast_weight: float
    trace: LinkedCurriculumTrace | None = None


def _initial_state(
    model: MiconiKayRetroModulRNN, batch_size: int
) -> _DifferentiableState:
    return _DifferentiableState(
        hidden=model.initialZeroState(batch_size),
        eligibility=model.initialZeroET(batch_size),
        plastic_weights=model.initialZeroPlasticWeights(batch_size),
    )


def _blank_between_segments(
    model: MiconiKayRetroModulRNN,
    source: MiconiKaySourceConfig,
    state: _DifferentiableState,
) -> _DifferentiableState:
    blank = torch.zeros(
        source.batch_size,
        source.input_size,
        device=model.w.device,
        dtype=model.w.dtype,
    )
    for _ in range(2):
        _, _, _, hidden, eligibility, plastic_weights = model(
            blank, state.hidden, state.eligibility, state.plastic_weights
        )
        state = _DifferentiableState(hidden, eligibility, plastic_weights)
    return state


def _sample_adjacent_pairs(
    allowed_items: range,
    batch_size: int,
) -> list[list[int]]:
    pairs: list[list[int]] = []
    for _ in range(batch_size):
        pair = list(np.random.choice(allowed_items, 2, replace=False))
        while abs(int(pair[0]) - int(pair[1])) > 1:
            pair = list(np.random.choice(allowed_items, 2, replace=False))
        pairs.append([int(pair[0]), int(pair[1])])
    return pairs


def _bridge_pairs(batch_size: int) -> list[list[int]]:
    return [
        [3, 4] if int(np.random.randint(2)) else [4, 3]
        for _ in range(batch_size)
    ]


def _trial_cues(pairs: list[list[int]], source: MiconiKaySourceConfig) -> list[list[object]]:
    return [
        [pair, 8] + [-1 for _ in range(source.trial_steps - 2)]
        for pair in pairs
    ]


def _correct_actions(pairs: list[list[int]]) -> np.ndarray:
    return np.asarray([1 if pair[0] < pair[1] else 0 for pair in pairs], dtype="int64")


def _run_segment(
    model: MiconiKayRetroModulRNN,
    source: MiconiKaySourceConfig,
    cue_data: list[list[np.ndarray]],
    state: _DifferentiableState,
    *,
    trial_count: int,
    total_trials_for_time: int,
    allowed_items: range | None = None,
    bridge_only: bool = False,
) -> tuple[_DifferentiableState, _StepRecords]:
    if bridge_only == (allowed_items is not None):
        raise ValueError("choose exactly one of bridge_only or allowed_items")
    time_config = replace(
        source,
        train_trials=total_trials_for_time,
        test_trials=0,
    )
    reward = np.zeros(source.batch_size, dtype="float32")
    previous_actions = np.zeros(source.batch_size, dtype="int32")
    records = _StepRecords([], [], [], [])
    state = _blank_between_segments(model, source, state)
    episode_step = 0

    for _ in range(trial_count):
        state.hidden = model.initialZeroState(source.batch_size)
        state.eligibility = model.initialZeroET(source.batch_size)
        pairs = (
            _bridge_pairs(source.batch_size)
            if bridge_only
            else _sample_adjacent_pairs(allowed_items, source.batch_size)  # type: ignore[arg-type]
        )
        correct = _correct_actions(pairs)
        cues = _trial_cues(pairs, source)
        for step_index in range(source.trial_steps):
            inputs = build_step_inputs(
                time_config,
                8,
                cue_data,
                cues,
                reward,
                previous_actions,
                step_index,
                episode_step,
                device=model.w.device,
            )
            logits, value, _, hidden, eligibility, plastic_weights = model(
                inputs,
                state.hidden,
                state.eligibility,
                state.plastic_weights,
            )
            state = _DifferentiableState(hidden, eligibility, plastic_weights)
            probabilities = torch.softmax(logits, dim=1)
            distribution = torch.distributions.Categorical(probabilities)
            actions = distribution.sample()
            records.log_probabilities.append(distribution.log_prob(actions))
            records.values.append(value)
            records.probabilities.append(probabilities)
            previous_actions = actions.detach().cpu().numpy()
            reward = np.zeros(source.batch_size, dtype="float32")
            if step_index == source.response_step:
                reward = np.where(
                    previous_actions == correct,
                    source.reward_amount,
                    -source.reward_amount,
                ).astype("float32")
            records.rewards.append(reward.copy())
            episode_step += 1
    return state, records


def _actor_critic_loss(
    source: MiconiKaySourceConfig,
    records: _StepRecords,
) -> torch.Tensor:
    if not records.log_probabilities:
        raise ValueError("actor-critic segment contains no steps")
    device = records.values[0].device
    dtype = records.values[0].dtype
    batch_size = source.batch_size
    loss = sum(
        source.action_concentration_cost * probabilities.pow(2).sum() / batch_size
        for probabilities in records.probabilities
    )
    value_loss: torch.Tensor | float = 0.0
    discounted_return = torch.zeros(batch_size, device=device, dtype=dtype)
    for step in reversed(range(len(records.rewards))):
        discounted_return = (
            source.discount * discounted_return
            + torch.from_numpy(records.rewards[step]).detach().to(device)
        )
        advantage = discounted_return - records.values[step][:, 0]
        value_loss = value_loss + advantage.pow(2).sum() / batch_size
        loss = loss - (
            records.log_probabilities[step] * advantage.detach()
        ).sum() / batch_size
    if not isinstance(loss, torch.Tensor) or not isinstance(value_loss, torch.Tensor):
        raise RuntimeError("actor-critic objective is not differentiable")
    return (
        loss + source.value_loss_coefficient * value_loss
    ) / len(records.rewards)


def _query_pair_for_batch(
    left: int,
    right: int,
    batch_size: int,
) -> list[list[int]]:
    return [
        [left, right] if batch_index < batch_size // 2 else [right, left]
        for batch_index in range(batch_size)
    ]


def _run_query_branch(
    model: MiconiKayRetroModulRNN,
    source: MiconiKaySourceConfig,
    cue_data: list[list[np.ndarray]],
    post_bridge_state: _DifferentiableState,
    *,
    pair: tuple[int, int],
    bridge_trials: int,
    record_trace: bool,
) -> tuple[torch.Tensor, np.ndarray, dict[str, np.ndarray] | None]:
    total_trials = bridge_trials + 1
    time_config = replace(source, train_trials=total_trials, test_trials=0)
    state = _DifferentiableState(
        hidden=model.initialZeroState(source.batch_size),
        eligibility=model.initialZeroET(source.batch_size),
        plastic_weights=post_bridge_state.plastic_weights.clone(),
    )
    reward = np.zeros(source.batch_size, dtype="float32")
    previous_actions = np.zeros(source.batch_size, dtype="int32")
    pairs = _query_pair_for_batch(pair[0], pair[1], source.batch_size)
    correct = _correct_actions(pairs)
    cues = _trial_cues(pairs, source)
    records = _StepRecords([], [], [], [])
    response_actions: np.ndarray | None = None
    response_rewards: np.ndarray | None = None
    feedback_reward_inputs: np.ndarray | None = None
    feedback_action_inputs: np.ndarray | None = None
    action_offset = source.stimulus_bits + source.additional_input_count

    for step_index in range(source.trial_steps):
        inputs = build_step_inputs(
            time_config,
            8,
            cue_data,
            cues,
            reward,
            previous_actions,
            step_index,
            bridge_trials * source.trial_steps + step_index,
            device=model.w.device,
        )
        if record_trace and step_index == source.response_step + 1:
            feedback_reward_inputs = (
                inputs[:, source.stimulus_bits + 2].detach().cpu().numpy().copy()
            )
            feedback_action_inputs = (
                inputs[:, action_offset : action_offset + 2]
                .detach()
                .cpu()
                .numpy()
                .copy()
            )
        logits, value, _, hidden, eligibility, plastic_weights = model(
            inputs, state.hidden, state.eligibility, state.plastic_weights
        )
        state = _DifferentiableState(hidden, eligibility, plastic_weights)
        probabilities = torch.softmax(logits, dim=1)
        distribution = torch.distributions.Categorical(probabilities)
        actions = distribution.sample()
        records.log_probabilities.append(distribution.log_prob(actions))
        records.values.append(value)
        records.probabilities.append(probabilities)
        previous_actions = actions.detach().cpu().numpy()
        reward = np.zeros(source.batch_size, dtype="float32")
        if step_index == source.response_step:
            response_actions = previous_actions.copy()
            reward = np.where(
                response_actions == correct,
                source.reward_amount,
                -source.reward_amount,
            ).astype("float32")
            response_rewards = reward.copy()
        records.rewards.append(reward.copy())

    if (
        response_actions is None
        or response_rewards is None
    ):
        raise RuntimeError("query branch did not execute its response step")
    trace = None
    if record_trace:
        if feedback_reward_inputs is None or feedback_action_inputs is None:
            raise RuntimeError("query feedback inputs were not recorded")
        trace = {
            "actions": response_actions,
            "rewards": response_rewards,
            "feedback_rewards": feedback_reward_inputs,
            "feedback_actions": feedback_action_inputs,
        }
    return _actor_critic_loss(source, records), response_actions == correct, trace


def run_linked_curriculum_episode(
    source: MiconiKaySourceConfig,
    curriculum: LinkedCurriculumConfig,
    model: MiconiKayRetroModulRNN,
    *,
    record_trace: bool = False,
) -> LinkedCurriculumStats:
    """Run one differentiable source-only linked-list meta-training episode."""

    source.validate()
    curriculum.validate()
    if source.min_items != 8 or source.max_items != 8:
        raise ValueError("linked curriculum requires exactly eight items")
    if source.trial_steps != 4 or source.response_step != 1:
        raise ValueError("linked curriculum requires the source four-step trial")

    cue_data = generate_cue_data(source, 8)
    state = _initial_state(model, source.batch_size)
    state, first_records = _run_segment(
        model,
        source,
        cue_data,
        state,
        trial_count=curriculum.component_trials,
        total_trials_for_time=curriculum.component_trials,
        allowed_items=range(0, 4),
    )
    state, second_records = _run_segment(
        model,
        source,
        cue_data,
        state,
        trial_count=curriculum.component_trials,
        total_trials_for_time=curriculum.component_trials,
        allowed_items=range(4, 8),
    )
    state, bridge_records = _run_segment(
        model,
        source,
        cue_data,
        state,
        trial_count=curriculum.bridge_trials,
        total_trials_for_time=curriculum.bridge_trials + 1,
        bridge_only=True,
    )

    base_loss = (
        _actor_critic_loss(source, first_records)
        + _actor_critic_loss(source, second_records)
        + _actor_critic_loss(source, bridge_records)
    ) / 3.0
    canonical_pairs = [
        (left, right) for left in range(4) for right in range(4, 8)
    ]
    query_losses: list[torch.Tensor] = []
    query_correct: list[np.ndarray] = []
    query_actions: list[np.ndarray] = []
    query_rewards: list[np.ndarray] = []
    feedback_rewards: list[np.ndarray] = []
    feedback_actions: list[np.ndarray] = []
    for pair in canonical_pairs:
        query_loss, correct, trace = _run_query_branch(
            model,
            source,
            cue_data,
            state,
            pair=pair,
            bridge_trials=curriculum.bridge_trials,
            record_trace=record_trace,
        )
        query_losses.append(query_loss)
        query_correct.append(correct.astype("float32"))
        if trace is not None:
            query_actions.append(trace["actions"])
            query_rewards.append(trace["rewards"])
            feedback_rewards.append(trace["feedback_rewards"])
            feedback_actions.append(trace["feedback_actions"])

    query_loss = torch.stack(query_losses).mean()
    loss = base_loss + curriculum.query_loss_multiplier * query_loss
    loss = loss + (
        state.plastic_weights.square().mean() * source.plastic_weight_penalty
    )
    correctness = np.stack(query_correct, axis=1)
    pair_accuracy = correctness.mean(axis=0)
    trace_payload = None
    if record_trace:
        trace_payload = LinkedCurriculumTrace(
            query_pairs=np.asarray(canonical_pairs, dtype="int16"),
            query_response_actions=np.stack(query_actions, axis=1).astype("int8"),
            query_rewards_after_response=np.stack(query_rewards, axis=1).astype(
                "float32"
            ),
            query_feedback_reward_inputs=np.stack(feedback_rewards, axis=1).astype(
                "float32"
            ),
            query_feedback_action_inputs=np.stack(feedback_actions, axis=1).astype(
                "float32"
            ),
        )
    return LinkedCurriculumStats(
        loss=loss,
        loss_value=float(loss.detach().cpu()),
        query_accuracy=float(correctness.mean()),
        minimum_pair_accuracy=float(pair_accuracy.min()),
        mean_absolute_post_bridge_fast_weight=float(
            state.plastic_weights.detach().abs().mean().cpu()
        ),
        trace=trace_payload,
    )
