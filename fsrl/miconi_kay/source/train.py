"""Meta-training loop for the isolated M&K source task."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from fsrl.miconi_kay.source.checkpoint import save_source_checkpoint
from fsrl.miconi_kay.source.config import MiconiKaySourceConfig
from fsrl.miconi_kay.source.episode import run_episode
from fsrl.miconi_kay.source.model import MiconiKayRetroModulRNN


@dataclass(frozen=True)
class SourceTrainingSummary:
    """Detached per-episode scalars safe to retain across long meta-training."""

    episode_index: int
    loss_value: float
    loss_objective: float
    test_reward_mean: float
    test_accuracy: float | None
    adjacent_test_accuracy: float | None
    nonadjacent_test_accuracy: float | None
    mean_absolute_fast_weight: float


def set_source_seed(seed: int) -> None:
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_source_model(
    config: MiconiKaySourceConfig,
    output_dir: str | Path,
    *,
    device: torch.device | str | None = None,
) -> tuple[MiconiKayRetroModulRNN, list[SourceTrainingSummary]]:
    config.validate()
    set_source_seed(config.rng_seed)
    resolved_device = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = MiconiKayRetroModulRNN(config.to_model_dict()).to(resolved_device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )
    history: list[SourceTrainingSummary] = []
    test_rewards: list[float] = []

    for episode_index in range(config.num_episodes):
        item_count = int(np.random.choice(list(config.item_count_range)))
        optimizer.zero_grad(set_to_none=True)
        stats = run_episode(config, model, item_count)
        stats.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        if episode_index >= config.optimizer_warmup_episodes:
            optimizer.step()
        history.append(
            SourceTrainingSummary(
                episode_index=episode_index,
                loss_value=stats.loss_value,
                loss_objective=stats.loss_objective,
                test_reward_mean=stats.test_reward_mean,
                test_accuracy=stats.test_accuracy,
                adjacent_test_accuracy=stats.adjacent_test_accuracy,
                nonadjacent_test_accuracy=stats.nonadjacent_test_accuracy,
                mean_absolute_fast_weight=float(
                    stats.final_plastic_weights.abs().mean().cpu()
                ),
            )
        )
        test_rewards.append(stats.test_reward_mean)

        if (
            config.save_every > 0
            and episode_index > 0
            and episode_index % config.save_every == 0
        ):
            save_source_checkpoint(
                model,
                config,
                output_dir,
                episode_index=episode_index,
                test_rewards=test_rewards,
            )

    return model, history
