"""Configuration for the isolated Miconi & Kay source task."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


VERITAS_UPSTREAM_COMMIT = "afd31c923c1d77bef56c1d551f062179b55c487f"
VERITAS_UPSTREAM_URL = "https://github.com/VeriTas-arch/fsrl"
MICONI_UPSTREAM_COMMIT = "fbdebf06f6d335bcd4473a4fa24926208a98910f"
MICONI_UPSTREAM_URL = "https://github.com/ThomasMiconi/TransitiveInference"


@dataclass(frozen=True)
class MiconiKaySourceConfig:
    """Source-task parameters matching the readable ``simple_neo`` trainer.

    The defaults intentionally preserve code-level behavior, including the
    four-step trial, response at step 1, 20 adjacent training trials, and 10
    unrestricted test trials.  Small values may be supplied for smoke tests,
    but changing them defines a test configuration rather than the published
    source condition.
    """

    rng_seed: int = -1
    reward_amount: float = 1.0
    # Present in the readable source's config/model dictionary as ``wp`` but
    # not read by its episode or model code.  Preserve it as provenance, not
    # as a behavior that the upstream does not implement.
    legacy_unused_wp: float = 0.0
    action_concentration_cost: float = 0.1
    value_loss_coefficient: float = 0.1
    discount: float = 0.9
    hidden_size: int = 200
    batch_size: int = 32
    gradient_clip: float = 2.0
    adam_epsilon: float = 1e-6
    num_episodes: int = 30_000
    save_every: int = 200
    print_every: int = 101
    cue_size: int = 15
    trial_steps: int = 4
    response_step: int = 1
    train_trials: int = 20
    test_trials: int = 10
    test_loss_multiplier: float = 3.0
    weight_decay: float = 0.0
    learning_rate: float = 1e-4
    plastic_weight_penalty: float = 1e-4
    min_items: int = 4
    max_items: int = 8
    optimizer_warmup_episodes: int = 101

    @property
    def item_count_range(self) -> range:
        return range(self.min_items, self.max_items + 1)

    @property
    def num_trials(self) -> int:
        return self.train_trials + self.test_trials

    @property
    def episode_steps(self) -> int:
        return self.num_trials * self.trial_steps

    @property
    def stimulus_bits(self) -> int:
        return 2 * self.cue_size + 1

    @property
    def output_size(self) -> int:
        return 2

    @property
    def additional_input_count(self) -> int:
        return 4

    @property
    def input_size(self) -> int:
        return (
            self.stimulus_bits
            + self.additional_input_count
            + self.output_size
        )

    def validate(self) -> None:
        errors: list[str] = []
        if self.batch_size <= 0 or self.hidden_size <= 0 or self.cue_size <= 0:
            errors.append("batch_size, hidden_size, and cue_size must be positive")
        if self.num_episodes < 0:
            errors.append("num_episodes must be non-negative")
        if self.train_trials < 0 or self.test_trials < 0 or self.num_trials <= 0:
            errors.append("trial counts must be non-negative with a positive total")
        if self.trial_steps <= self.response_step + 1:
            errors.append("trial_steps must include a post-response feedback step")
        if self.response_step < 0:
            errors.append("response_step must be non-negative")
        if self.min_items < 2 or self.max_items < self.min_items:
            errors.append("item range must contain at least two items")
        if self.test_loss_multiplier <= 0:
            errors.append("test_loss_multiplier must be positive")
        if not 0 <= self.discount <= 1:
            errors.append("discount must lie in [0, 1]")
        if self.optimizer_warmup_episodes < 0:
            errors.append("optimizer_warmup_episodes must be non-negative")
        if errors:
            raise ValueError("Invalid MiconiKaySourceConfig: " + "; ".join(errors))

    def to_model_dict(self) -> dict[str, Any]:
        """Return the parameter names used by the readable source model."""

        return {
            "rngseed": self.rng_seed,
            "rew": self.reward_amount,
            "wp": self.legacy_unused_wp,
            "bent": self.action_concentration_cost,
            "blossv": self.value_loss_coefficient,
            "gr": self.discount,
            "hs": self.hidden_size,
            "bs": self.batch_size,
            "gc": self.gradient_clip,
            "eps": self.adam_epsilon,
            "nbiter": self.num_episodes,
            "save_every": self.save_every,
            "pe": self.print_every,
            "nbcuesrange": self.item_count_range,
            "cs": self.cue_size,
            "triallen": self.trial_steps,
            "nbtraintrials": self.train_trials,
            "nbtesttrials": self.test_trials,
            "nbtrials": self.num_trials,
            "eplen": self.episode_steps,
            "testlmult": self.test_loss_multiplier,
            "l2": self.weight_decay,
            "lr": self.learning_rate,
            "lpw": self.plastic_weight_penalty,
            "outputsize": self.output_size,
            "inputsize": self.input_size,
        }

    def to_manifest_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "num_trials": self.num_trials,
                "episode_steps": self.episode_steps,
                "input_size": self.input_size,
                "veritas_upstream_commit": VERITAS_UPSTREAM_COMMIT,
                "veritas_upstream_url": VERITAS_UPSTREAM_URL,
                "miconi_upstream_commit": MICONI_UPSTREAM_COMMIT,
                "miconi_upstream_url": MICONI_UPSTREAM_URL,
            }
        )
        return payload
