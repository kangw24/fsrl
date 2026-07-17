"""CLI for the isolated Miconi & Kay source-task trainer."""

from __future__ import annotations

import argparse
from pathlib import Path

from fsrl.miconi_kay.source.checkpoint import save_source_checkpoint
from fsrl.miconi_kay.source.config import MiconiKaySourceConfig
from fsrl.miconi_kay.source.train import train_source_model


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the isolated Miconi & Kay source-task plastic RNN."
    )
    parser.add_argument("--num-episodes", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--print-every", type=int, default=101)
    parser.add_argument("--warmup-episodes", type=int, default=101)
    parser.add_argument("--min-items", type=int, default=4)
    parser.add_argument("--max-items", type=int, default=8)
    parser.add_argument("--train-trials", type=int, default=20)
    parser.add_argument("--test-trials", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/miconi_kay_source")
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> None:
    parsed = parse_args(args)
    config = MiconiKaySourceConfig(
        rng_seed=parsed.seed,
        hidden_size=parsed.hidden_size,
        batch_size=parsed.batch_size,
        num_episodes=parsed.num_episodes,
        save_every=parsed.save_every,
        print_every=parsed.print_every,
        optimizer_warmup_episodes=parsed.warmup_episodes,
        min_items=parsed.min_items,
        max_items=parsed.max_items,
        train_trials=parsed.train_trials,
        test_trials=parsed.test_trials,
    )
    model, history = train_source_model(
        config, parsed.output_dir, device=parsed.device
    )
    rewards = [stats.test_reward_mean for stats in history]
    save_source_checkpoint(
        model,
        config,
        parsed.output_dir,
        episode_index=max(-1, config.num_episodes - 1),
        test_rewards=rewards,
    )
    if history:
        final = history[-1]
        print(
            "source smoke/train complete: "
            f"loss={final.loss_value:.6f}, test_accuracy={final.test_accuracy}"
        )
    else:
        print("source initialization complete; no episodes requested")


if __name__ == "__main__":
    main()
