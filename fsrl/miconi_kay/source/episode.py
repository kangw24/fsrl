"""Differentiable source-task episode with action-contingent feedback."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from fsrl.miconi_kay.source.config import MiconiKaySourceConfig
from fsrl.miconi_kay.source.model import MiconiKayRetroModulRNN
from fsrl.miconi_kay.source.task import (
    build_step_inputs,
    generate_cue_data,
    prepare_trial,
)


@dataclass
class EpisodeTrace:
    """Optional audit trace; tensors are detached CPU snapshots."""

    cue_pairs: list[list[list[int]]]
    inputs: list[torch.Tensor]
    actions: list[torch.Tensor]
    rewards_after_step: list[np.ndarray]
    neuromodulator: list[torch.Tensor]


@dataclass
class EpisodeStats:
    loss: torch.Tensor
    loss_value: float
    loss_objective: float
    test_reward_mean: float
    num_test_trials: int
    test_accuracy: float | None
    adjacent_test_accuracy: float | None
    nonadjacent_test_accuracy: float | None
    final_plastic_weights: torch.Tensor
    trace: EpisodeTrace | None = None


def run_episode(
    config: MiconiKaySourceConfig,
    model: MiconiKayRetroModulRNN,
    item_count: int,
    *,
    record_trace: bool = False,
) -> EpisodeStats:
    """Run one source episode while preserving the readable trainer semantics."""

    config.validate()
    if item_count not in config.item_count_range:
        raise ValueError(f"item_count {item_count} is outside configured range")
    device = model.w.device
    hidden = model.initialZeroState(config.batch_size)
    eligibility = model.initialZeroET(config.batch_size)
    plastic_weights = model.initialZeroPlasticWeights(config.batch_size)
    cue_data = generate_cue_data(config, item_count)

    reward = np.zeros(config.batch_size, dtype="float32")
    test_reward_sum = np.zeros(config.batch_size)
    rewards: list[np.ndarray] = []
    values: list[torch.Tensor] = []
    log_probabilities: list[torch.Tensor] = []
    completed_trials = np.zeros(config.batch_size)
    previous_actions = np.zeros(config.batch_size, dtype="int32")

    num_test_trials = 0
    num_test_correct = 0.0
    num_adjacent_test = 0.0
    num_adjacent_test_correct = 0.0
    num_nonadjacent_test = 0.0
    num_nonadjacent_test_correct = 0.0

    loss: torch.Tensor | float = 0.0
    value_loss: torch.Tensor | float = 0.0

    trace = (
        EpisodeTrace([], [], [], [], []) if record_trace else None
    )

    # Two blank steps before the episode, as in the readable source.
    blank_inputs = torch.zeros(
        config.batch_size,
        config.input_size,
        device=device,
        dtype=model.w.dtype,
        requires_grad=False,
    )
    for _ in range(2):
        _, _, _, hidden, eligibility, plastic_weights = model(
            blank_inputs, hidden, eligibility, plastic_weights
        )

    episode_step_index = 0
    for trial_index in range(config.num_trials):
        # Neural activity and eligibility are trial-local; plastic weights are
        # episode-local and therefore deliberately not reset here.
        hidden = model.initialZeroState(config.batch_size)
        eligibility = model.initialZeroET(config.batch_size)
        cues, cue_pairs, correct_order, adjacent = prepare_trial(
            config, item_count, completed_trials
        )
        if trace is not None:
            trace.cue_pairs.append(
                [[int(pair[0]), int(pair[1])] for pair in cue_pairs]
            )

        correct_answer = np.zeros(config.batch_size)
        for step_index in range(config.trial_steps):
            inputs = build_step_inputs(
                config,
                item_count,
                cue_data,
                cues,
                reward,
                previous_actions,
                step_index,
                episode_step_index,
                device=device,
            )
            raw_actions, value, daout, hidden, eligibility, plastic_weights = model(
                inputs, hidden, eligibility, plastic_weights
            )

            probabilities = torch.softmax(raw_actions, dim=1)
            distribution = torch.distributions.Categorical(probabilities)
            sampled_actions = distribution.sample()
            # The public readable implementation records all step log-probs,
            # even though only the response-step action controls reward.
            log_probabilities.append(distribution.log_prob(sampled_actions))
            previous_actions = sampled_actions.detach().cpu().numpy()

            reward = np.zeros(config.batch_size, dtype="float32")
            for batch_index in range(config.batch_size):
                if step_index == config.response_step:
                    correct_answer[batch_index] = 1
                    chose_item_one = previous_actions[batch_index] == 1
                    correct_choice = bool(correct_order[batch_index])
                    if (correct_choice and chose_item_one) or (
                        (not correct_choice) and not chose_item_one
                    ):
                        reward[batch_index] += config.reward_amount
                    else:
                        reward[batch_index] -= config.reward_amount
                        correct_answer[batch_index] = 0
                if step_index == config.trial_steps - 1:
                    completed_trials[batch_index] += 1

            rewards.append(reward.copy())
            values.append(value)
            if trial_index >= config.num_trials - config.test_trials:
                test_reward_sum += reward

            loss = (
                loss
                + config.action_concentration_cost
                * probabilities.pow(2).sum()
                / config.batch_size
            )
            if trace is not None:
                trace.inputs.append(inputs.detach().cpu().clone())
                trace.actions.append(sampled_actions.detach().cpu().clone())
                trace.rewards_after_step.append(reward.copy())
                trace.neuromodulator.append(daout.detach().cpu().clone())
            episode_step_index += 1

        if trial_index >= config.num_trials - config.test_trials:
            # Preserved from the source. The final step reward is normally zero.
            test_reward_sum += reward
            num_test_trials += config.batch_size
            num_test_correct += float(np.sum(correct_answer))
            num_adjacent_test += float(np.sum(adjacent))
            num_adjacent_test_correct += float(np.sum(adjacent * correct_answer))
            num_nonadjacent_test += float(np.sum(1 - adjacent))
            num_nonadjacent_test_correct += float(
                np.sum((1 - adjacent) * correct_answer)
            )

    discounted_return = torch.zeros(
        config.batch_size, device=device, dtype=model.w.dtype, requires_grad=False
    )
    for backward_step in reversed(range(config.episode_steps)):
        discounted_return = (
            config.discount * discounted_return
            + torch.from_numpy(rewards[backward_step]).detach().to(device)
        )
        advantage = discounted_return - values[backward_step][:, 0]
        value_loss = value_loss + advantage.pow(2).sum() / config.batch_size
        loss_multiplier = (
            config.test_loss_multiplier
            if backward_step
            > config.episode_steps - config.trial_steps * config.test_trials
            else 1.0
        )
        loss = (
            loss
            - loss_multiplier
            * (log_probabilities[backward_step] * advantage.detach()).sum()
            / config.batch_size
        )

    if not isinstance(loss, torch.Tensor) or not isinstance(value_loss, torch.Tensor):
        raise RuntimeError("episode did not produce a differentiable objective")
    loss_objective = float(loss.detach().cpu())
    loss = loss + config.value_loss_coefficient * value_loss
    loss = loss / config.episode_steps
    loss = loss + torch.mean(plastic_weights**2) * config.plastic_weight_penalty

    test_accuracy = (
        None if num_test_trials == 0 else num_test_correct / num_test_trials
    )
    adjacent_accuracy = (
        None
        if num_adjacent_test == 0
        else num_adjacent_test_correct / num_adjacent_test
    )
    nonadjacent_accuracy = (
        None
        if num_nonadjacent_test == 0
        else num_nonadjacent_test_correct / num_nonadjacent_test
    )

    return EpisodeStats(
        loss=loss,
        loss_value=float(loss.detach().cpu()),
        loss_objective=loss_objective,
        test_reward_mean=float(test_reward_sum.mean()),
        num_test_trials=num_test_trials,
        test_accuracy=test_accuracy,
        adjacent_test_accuracy=adjacent_accuracy,
        nonadjacent_test_accuracy=nonadjacent_accuracy,
        final_plastic_weights=plastic_weights.detach(),
        trace=trace,
    )
