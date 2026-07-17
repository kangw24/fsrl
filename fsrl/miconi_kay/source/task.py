"""Stimulus generation and trial encoding for the M&K source task."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from fsrl.miconi_kay.source.config import MiconiKaySourceConfig


def generate_cue_data(
    config: MiconiKaySourceConfig, item_count: int
) -> list[list[np.ndarray]]:
    """Generate unique random binary cues for every batch element."""

    cue_data: list[list[np.ndarray]] = []
    for _ in range(config.batch_size):
        batch_cues: list[np.ndarray] = []
        for cue_index in range(item_count):
            batch_cues.append(sample_unique_cue(config, batch_cues, cue_index))
        cue_data.append(batch_cues)
    return cue_data


def sample_unique_cue(
    config: MiconiKaySourceConfig,
    existing_cues: Sequence[np.ndarray],
    cue_index: int,
) -> np.ndarray:
    attempts = 0
    while True:
        attempts += 1
        if attempts > 10_000:
            raise ValueError("Could not generate a full list of different cues")
        candidate = np.random.randint(2, size=config.cue_size) * 2 - 1
        if all(
            np.mean(existing_cues[index] == candidate) <= 0.66
            for index in range(cue_index)
        ):
            return candidate


def sample_trial_pair(item_count: int, is_train_trial: bool) -> list[int]:
    cue_pair = list(np.random.choice(range(item_count), 2, replace=False))
    if is_train_trial:
        while abs(cue_pair[0] - cue_pair[1]) > 1:
            cue_pair = list(np.random.choice(range(item_count), 2, replace=False))
    return [int(cue_pair[0]), int(cue_pair[1])]


def build_step_inputs(
    config: MiconiKaySourceConfig,
    item_count: int,
    cue_data: list[list[np.ndarray]],
    cues: list[list[object]],
    reward: np.ndarray,
    previous_actions: np.ndarray,
    step_index: int,
    episode_step_index: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    inputs = np.zeros((config.batch_size, config.input_size), dtype="float32")

    for batch_index in range(config.batch_size):
        cue = cues[batch_index][step_index]
        if isinstance(cue, (list, tuple, np.ndarray)):
            inputs[batch_index, : config.stimulus_bits - 1] = np.concatenate(
                (
                    cue_data[batch_index][int(cue[0])][:],
                    cue_data[batch_index][int(cue[1])][:],
                )
            )
        elif cue == item_count:
            inputs[batch_index, config.stimulus_bits - 1] = 1.0

        inputs[batch_index, config.stimulus_bits + 0] = 1.0
        inputs[batch_index, config.stimulus_bits + 1] = (
            episode_step_index / config.episode_steps
        )
        inputs[batch_index, config.stimulus_bits + 2] = reward[batch_index]

        if step_index == config.response_step + 1:
            action_offset = config.stimulus_bits + config.additional_input_count
            inputs[
                batch_index,
                action_offset + int(previous_actions[batch_index]),
            ] = 1.0

    return torch.from_numpy(inputs).detach().to(device)


def prepare_trial(
    config: MiconiKaySourceConfig,
    item_count: int,
    completed_trials: np.ndarray,
) -> tuple[list[list[object]], list[list[int]], np.ndarray, np.ndarray]:
    cues: list[list[object]] = []
    cue_pairs: list[list[int]] = []
    correct_order = np.zeros(config.batch_size)
    adjacent = np.zeros(config.batch_size)

    for batch_index in range(config.batch_size):
        if completed_trials[batch_index] != int(completed_trials[batch_index]):
            raise ValueError("completed trial count must be integral")
        cue_pair = sample_trial_pair(
            item_count,
            completed_trials[batch_index] < config.train_trials,
        )
        correct_order[batch_index] = 1 if cue_pair[0] < cue_pair[1] else 0
        adjacent[batch_index] = 1 if abs(cue_pair[0] - cue_pair[1]) == 1 else 0
        cue_pairs.append(cue_pair)
        cues.append(
            [cue_pair, item_count]
            + [-1 for _ in range(config.trial_steps - 2)]
        )

    return cues, cue_pairs, correct_order, adjacent
