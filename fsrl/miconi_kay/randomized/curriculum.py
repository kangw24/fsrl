"""Randomized, reward-only list-linking curriculum for a new source candidate.

Unlike linked-source-v1, training does not enumerate the exhaustive 8-item
challenge.  Each episode samples an even item count, bridge duration, and a
small number of cross-list queries.  The online chronology remains M&K-style:

``pair -> sampled action -> chosen-action reward -> next-step feedback``.

This module has no Liu graph, human response, participant identifier,
``teacher_sign``, direct correct-action loss, or unchosen-action reward.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch

from fsrl.miconi_kay.source.config import MiconiKaySourceConfig
from fsrl.miconi_kay.source.model import MiconiKayRetroModulRNN
from fsrl.miconi_kay.source.task import build_step_inputs, generate_cue_data


@dataclass(frozen=True)
class RandomizedLinkedCurriculumConfig:
    # The exact 8-item, four-bridge exhaustive challenge is held out.
    item_counts: tuple[int, ...] = (4, 6)
    component_trials: int = 10
    minimum_bridge_trials: int = 1
    maximum_bridge_trials: int = 3
    query_branches: int = 4
    query_loss_multiplier: float = 3.0

    def validate(self) -> None:
        if not self.item_counts:
            raise ValueError("item_counts cannot be empty")
        if any(count < 4 or count % 2 for count in self.item_counts):
            raise ValueError("item_counts must contain even values >= 4")
        if self.component_trials <= 0 or self.query_branches <= 0:
            raise ValueError("component_trials and query_branches must be positive")
        if not 0 < self.minimum_bridge_trials <= self.maximum_bridge_trials:
            raise ValueError("bridge-trial range must be positive and ordered")
        if self.query_loss_multiplier <= 0:
            raise ValueError("query_loss_multiplier must be positive")


@dataclass
class _State:
    hidden: torch.Tensor
    eligibility: torch.Tensor
    plastic_weights: torch.Tensor


@dataclass
class _Records:
    log_probabilities: list[torch.Tensor]
    values: list[torch.Tensor]
    probabilities: list[torch.Tensor]
    rewards: list[np.ndarray]


@dataclass
class RandomizedLinkedCurriculumTrace:
    item_count: int
    bridge_trials: int
    query_pairs: np.ndarray
    query_response_actions: np.ndarray
    query_rewards_after_response: np.ndarray
    query_feedback_reward_inputs: np.ndarray
    query_feedback_action_inputs: np.ndarray


@dataclass
class RandomizedLinkedCurriculumStats:
    loss: torch.Tensor
    loss_value: float
    item_count: int
    bridge_trials: int
    query_accuracy: float
    mean_correct_choice_probability: float
    mean_absolute_post_bridge_fast_weight: float
    trace: RandomizedLinkedCurriculumTrace | None = None


def _initial_state(model: MiconiKayRetroModulRNN, batch_size: int) -> _State:
    return _State(
        hidden=model.initialZeroState(batch_size),
        eligibility=model.initialZeroET(batch_size),
        plastic_weights=model.initialZeroPlasticWeights(batch_size),
    )


def _blank_between_segments(
    model: MiconiKayRetroModulRNN,
    source: MiconiKaySourceConfig,
    state: _State,
) -> _State:
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
        state = _State(hidden, eligibility, plastic_weights)
    return state


def _sample_adjacent_pairs(indices: range, batch_size: int) -> list[list[int]]:
    if len(indices) < 2:
        raise ValueError("a component list must contain at least two items")
    pairs: list[list[int]] = []
    for _ in range(batch_size):
        pair = list(np.random.choice(indices, 2, replace=False))
        while abs(int(pair[0]) - int(pair[1])) != 1:
            pair = list(np.random.choice(indices, 2, replace=False))
        pairs.append([int(pair[0]), int(pair[1])])
    return pairs


def _bridge_pairs(item_count: int, batch_size: int) -> list[list[int]]:
    midpoint = item_count // 2
    return [
        [midpoint - 1, midpoint]
        if int(np.random.randint(2))
        else [midpoint, midpoint - 1]
        for _ in range(batch_size)
    ]


def _sample_cross_pairs(item_count: int, batch_size: int) -> list[list[int]]:
    midpoint = item_count // 2
    pairs: list[list[int]] = []
    for _ in range(batch_size):
        left = int(np.random.randint(0, midpoint))
        right = int(np.random.randint(midpoint, item_count))
        pairs.append(
            [left, right] if int(np.random.randint(2)) else [right, left]
        )
    return pairs


def _trial_cues(
    pairs: list[list[int]], item_count: int, source: MiconiKaySourceConfig
) -> list[list[object]]:
    return [
        [pair, item_count] + [-1 for _ in range(source.trial_steps - 2)]
        for pair in pairs
    ]


def _correct_actions(pairs: list[list[int]]) -> np.ndarray:
    return np.asarray(
        [1 if pair[0] < pair[1] else 0 for pair in pairs], dtype="int64"
    )


def _run_segment(
    model: MiconiKayRetroModulRNN,
    source: MiconiKaySourceConfig,
    cue_data: list[list[np.ndarray]],
    state: _State,
    *,
    item_count: int,
    trial_count: int,
    total_trials_for_time: int,
    allowed_items: range | None = None,
    bridge_only: bool = False,
) -> tuple[_State, _Records]:
    if bridge_only == (allowed_items is not None):
        raise ValueError("choose exactly one of bridge_only or allowed_items")
    time_config = replace(source, train_trials=total_trials_for_time, test_trials=0)
    reward = np.zeros(source.batch_size, dtype="float32")
    previous_actions = np.zeros(source.batch_size, dtype="int32")
    records = _Records([], [], [], [])
    state = _blank_between_segments(model, source, state)
    episode_step = 0

    for _ in range(trial_count):
        state.hidden = model.initialZeroState(source.batch_size)
        state.eligibility = model.initialZeroET(source.batch_size)
        pairs = (
            _bridge_pairs(item_count, source.batch_size)
            if bridge_only
            else _sample_adjacent_pairs(allowed_items, source.batch_size)  # type: ignore[arg-type]
        )
        correct = _correct_actions(pairs)
        cues = _trial_cues(pairs, item_count, source)
        for step_index in range(source.trial_steps):
            inputs = build_step_inputs(
                time_config,
                item_count,
                cue_data,
                cues,
                reward,
                previous_actions,
                step_index,
                episode_step,
                device=model.w.device,
            )
            logits, value, _, hidden, eligibility, plastic_weights = model(
                inputs, state.hidden, state.eligibility, state.plastic_weights
            )
            state = _State(hidden, eligibility, plastic_weights)
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
    source: MiconiKaySourceConfig, records: _Records
) -> torch.Tensor:
    if not records.log_probabilities:
        raise ValueError("actor-critic segment contains no steps")
    batch_size = source.batch_size
    loss = sum(
        source.action_concentration_cost * probabilities.pow(2).sum() / batch_size
        for probabilities in records.probabilities
    )
    value_loss: torch.Tensor | float = 0.0
    discounted_return = torch.zeros(
        batch_size,
        device=records.values[0].device,
        dtype=records.values[0].dtype,
    )
    for step_index in reversed(range(len(records.rewards))):
        discounted_return = (
            source.discount * discounted_return
            + torch.from_numpy(records.rewards[step_index])
            .detach()
            .to(records.values[0].device)
        )
        advantage = discounted_return - records.values[step_index][:, 0]
        value_loss = value_loss + advantage.pow(2).sum() / batch_size
        loss = loss - (
            records.log_probabilities[step_index] * advantage.detach()
        ).sum() / batch_size
    if not isinstance(loss, torch.Tensor) or not isinstance(value_loss, torch.Tensor):
        raise RuntimeError("actor-critic objective is not differentiable")
    return (loss + source.value_loss_coefficient * value_loss) / len(
        records.rewards
    )


def _run_query_branch(
    model: MiconiKayRetroModulRNN,
    source: MiconiKaySourceConfig,
    cue_data: list[list[np.ndarray]],
    post_bridge_state: _State,
    *,
    item_count: int,
    bridge_trials: int,
    record_trace: bool,
) -> tuple[
    torch.Tensor,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray] | None,
]:
    total_trials = bridge_trials + 1
    time_config = replace(source, train_trials=total_trials, test_trials=0)
    state = _State(
        hidden=model.initialZeroState(source.batch_size),
        eligibility=model.initialZeroET(source.batch_size),
        plastic_weights=post_bridge_state.plastic_weights.clone(),
    )
    reward = np.zeros(source.batch_size, dtype="float32")
    previous_actions = np.zeros(source.batch_size, dtype="int32")
    pairs = _sample_cross_pairs(item_count, source.batch_size)
    correct = _correct_actions(pairs)
    cues = _trial_cues(pairs, item_count, source)
    records = _Records([], [], [], [])
    response_actions: np.ndarray | None = None
    response_rewards: np.ndarray | None = None
    response_correct_probability: np.ndarray | None = None
    feedback_reward_inputs: np.ndarray | None = None
    feedback_action_inputs: np.ndarray | None = None
    action_offset = source.stimulus_bits + source.additional_input_count

    for step_index in range(source.trial_steps):
        inputs = build_step_inputs(
            time_config,
            item_count,
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
        state = _State(hidden, eligibility, plastic_weights)
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
            correct_tensor = torch.from_numpy(correct).to(model.w.device)
            response_correct_probability = (
                probabilities.gather(1, correct_tensor[:, None])[:, 0]
                .detach()
                .cpu()
                .numpy()
                .copy()
            )
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
        or response_correct_probability is None
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
    return (
        _actor_critic_loss(source, records),
        response_actions == correct,
        response_correct_probability,
        np.asarray(pairs, dtype="int16"),
        trace,
    )


def run_randomized_linked_episode(
    source: MiconiKaySourceConfig,
    curriculum: RandomizedLinkedCurriculumConfig,
    model: MiconiKayRetroModulRNN,
    *,
    item_count: int | None = None,
    bridge_trials: int | None = None,
    record_trace: bool = False,
) -> RandomizedLinkedCurriculumStats:
    """Run one differentiable randomized linked-list source episode."""

    source.validate()
    curriculum.validate()
    if source.trial_steps != 4 or source.response_step != 1:
        raise ValueError("randomized curriculum requires the source four-step trial")
    selected_item_count = (
        int(np.random.choice(curriculum.item_counts))
        if item_count is None
        else int(item_count)
    )
    if selected_item_count not in curriculum.item_counts:
        raise ValueError("forced item_count is outside the curriculum")
    if not source.min_items <= selected_item_count <= source.max_items:
        raise ValueError("source item range does not cover the sampled curriculum")
    selected_bridge_trials = (
        int(
            np.random.randint(
                curriculum.minimum_bridge_trials,
                curriculum.maximum_bridge_trials + 1,
            )
        )
        if bridge_trials is None
        else int(bridge_trials)
    )
    if not (
        curriculum.minimum_bridge_trials
        <= selected_bridge_trials
        <= curriculum.maximum_bridge_trials
    ):
        raise ValueError("forced bridge_trials is outside the curriculum")

    midpoint = selected_item_count // 2
    cue_data = generate_cue_data(source, selected_item_count)
    state = _initial_state(model, source.batch_size)
    state, first_records = _run_segment(
        model,
        source,
        cue_data,
        state,
        item_count=selected_item_count,
        trial_count=curriculum.component_trials,
        total_trials_for_time=curriculum.component_trials,
        allowed_items=range(0, midpoint),
    )
    state, second_records = _run_segment(
        model,
        source,
        cue_data,
        state,
        item_count=selected_item_count,
        trial_count=curriculum.component_trials,
        total_trials_for_time=curriculum.component_trials,
        allowed_items=range(midpoint, selected_item_count),
    )
    state, bridge_records = _run_segment(
        model,
        source,
        cue_data,
        state,
        item_count=selected_item_count,
        trial_count=selected_bridge_trials,
        total_trials_for_time=selected_bridge_trials + 1,
        bridge_only=True,
    )
    base_loss = (
        _actor_critic_loss(source, first_records)
        + _actor_critic_loss(source, second_records)
        + _actor_critic_loss(source, bridge_records)
    ) / 3.0

    query_losses: list[torch.Tensor] = []
    query_correct: list[np.ndarray] = []
    query_probabilities: list[np.ndarray] = []
    query_pairs: list[np.ndarray] = []
    query_actions: list[np.ndarray] = []
    query_rewards: list[np.ndarray] = []
    feedback_rewards: list[np.ndarray] = []
    feedback_actions: list[np.ndarray] = []
    for _ in range(curriculum.query_branches):
        query_loss, correct, probability, pairs, trace = _run_query_branch(
            model,
            source,
            cue_data,
            state,
            item_count=selected_item_count,
            bridge_trials=selected_bridge_trials,
            record_trace=record_trace,
        )
        query_losses.append(query_loss)
        query_correct.append(correct.astype("float32"))
        query_probabilities.append(probability.astype("float32"))
        query_pairs.append(pairs)
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
    correct_probabilities = np.stack(query_probabilities, axis=1)
    trace_payload = None
    if record_trace:
        trace_payload = RandomizedLinkedCurriculumTrace(
            item_count=selected_item_count,
            bridge_trials=selected_bridge_trials,
            query_pairs=np.stack(query_pairs, axis=1),
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
    return RandomizedLinkedCurriculumStats(
        loss=loss,
        loss_value=float(loss.detach().cpu()),
        item_count=selected_item_count,
        bridge_trials=selected_bridge_trials,
        query_accuracy=float(correctness.mean()),
        mean_correct_choice_probability=float(correct_probabilities.mean()),
        mean_absolute_post_bridge_fast_weight=float(
            state.plastic_weights.detach().abs().mean().cpu()
        ),
        trace=trace_payload,
    )
