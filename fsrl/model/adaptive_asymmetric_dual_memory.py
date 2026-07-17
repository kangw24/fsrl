"""Adaptive asymmetric item learning plus direct-pair episodic memory.

This is a sign-only, episode-local process candidate.  It implements the
population-shared entropy allocation rule preregistered as AADM-v1 and adds a
separately lesionable direct-pair memory.  It has no participant, cue, pair, or
rank-specific slow parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

from fsrl.device import DEVICE


AsymmetryMode = Literal["adaptive", "symmetric", "winner_only"]


def _logit(probability: float) -> float:
    if probability <= 0.0:
        return -20.0
    if probability >= 1.0:
        return 20.0
    return math.log(probability / (1.0 - probability))


@dataclass(frozen=True)
class AdaptiveAsymmetricDualMemoryConfig:
    max_items: int = 8
    cue_dim: int = 15
    base_learning_rate_init: float = 0.30
    difference_weight_init: float = 0.50
    entropy_temperature_init: float = 1.0
    pair_write_init: float = 0.50
    local_scale_init: float = 1.0
    item_scale_init: float = 1.0
    decision_lapse_init: float = 0.02
    encoding_noise_init: float = 0.05

    def __post_init__(self) -> None:
        if self.max_items < 4:
            raise ValueError("max_items must be at least 4")
        if self.cue_dim <= 0:
            raise ValueError("cue_dim must be positive")
        for name, value in (
            ("base_learning_rate_init", self.base_learning_rate_init),
            ("difference_weight_init", self.difference_weight_init),
            ("pair_write_init", self.pair_write_init),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be inside (0, 1)")
        if self.entropy_temperature_init <= 0.0:
            raise ValueError("entropy_temperature_init must be positive")
        if self.local_scale_init <= 0.0 or self.item_scale_init <= 0.0:
            raise ValueError("readout scales must be positive")
        if not 0.0 <= self.decision_lapse_init < 1.0:
            raise ValueError("decision_lapse_init must be in [0, 1)")
        if self.encoding_noise_init <= 0.0:
            raise ValueError("encoding_noise_init must be positive")


@dataclass
class AdaptiveAsymmetricDualMemoryState:
    item_values: torch.Tensor
    local_evidence: torch.Tensor
    local_precision: torch.Tensor
    update_count: torch.Tensor

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.item_values,
            self.local_evidence,
            self.local_precision,
            self.update_count,
        )


@dataclass(frozen=True)
class AdaptiveAsymmetricIntervention:
    local_route_enabled: bool = True
    global_route_enabled: bool = True
    local_update_enabled: bool = True
    item_update_enabled: bool = True
    asymmetry_mode: AsymmetryMode = "adaptive"

    def __post_init__(self) -> None:
        if self.asymmetry_mode not in ("adaptive", "symmetric", "winner_only"):
            raise ValueError("unknown asymmetry mode")


@dataclass
class AdaptiveAsymmetricEncodingTrace:
    correct_relation_probability: torch.Tensor
    entropy: torch.Tensor
    winner_learning_rate: torch.Tensor
    loser_learning_rate: torch.Tensor
    winner_delta: torch.Tensor
    loser_delta: torch.Tensor
    write_strength: torch.Tensor


@dataclass
class AdaptiveAsymmetricConsolidationTrace:
    event: str
    proposed_scores: torch.Tensor
    score_change: torch.Tensor
    uncertainty: torch.Tensor


@dataclass
class AdaptiveAsymmetricQueryTrace:
    logits: torch.Tensor
    probability_first: torch.Tensor
    uncertainty: torch.Tensor
    local_logit: torch.Tensor
    global_logit: torch.Tensor
    retrieval_strength: torch.Tensor
    global_reliability: torch.Tensor


class AdaptiveAsymmetricDualMemory(nn.Module):
    """Q-adapt-style item cache with an independent direct-pair trace."""

    _LEGAL_EVENTS = frozenset(("phase_transition", "visible_boundary"))

    def __init__(self, config: AdaptiveAsymmetricDualMemoryConfig):
        super().__init__()
        self.config = config
        self.base_learning_rate_logit = nn.Parameter(
            torch.tensor(_logit(config.base_learning_rate_init))
        )
        self.difference_weight_logit = nn.Parameter(
            torch.tensor(_logit(config.difference_weight_init))
        )
        self.entropy_temperature_log = nn.Parameter(
            torch.tensor(math.log(config.entropy_temperature_init))
        )
        self.pair_write_logit = nn.Parameter(torch.tensor(_logit(config.pair_write_init)))
        self.local_scale_log = nn.Parameter(torch.tensor(math.log(config.local_scale_init)))
        self.item_scale_log = nn.Parameter(torch.tensor(math.log(config.item_scale_init)))
        self.decision_lapse_logit = nn.Parameter(
            torch.tensor(_logit(config.decision_lapse_init))
        )
        self.encoding_noise_log = nn.Parameter(
            torch.tensor(math.log(config.encoding_noise_init))
        )
        self.to(DEVICE)

    @property
    def base_learning_rate(self) -> torch.Tensor:
        return torch.sigmoid(self.base_learning_rate_logit)

    @property
    def difference_weight(self) -> torch.Tensor:
        return torch.sigmoid(self.difference_weight_logit)

    @property
    def entropy_temperature(self) -> torch.Tensor:
        return torch.exp(self.entropy_temperature_log)

    @property
    def pair_write(self) -> torch.Tensor:
        return torch.sigmoid(self.pair_write_logit)

    @property
    def local_scale(self) -> torch.Tensor:
        return torch.exp(self.local_scale_log)

    @property
    def item_scale(self) -> torch.Tensor:
        return torch.exp(self.item_scale_log)

    @property
    def decision_lapse(self) -> torch.Tensor:
        return torch.sigmoid(self.decision_lapse_logit)

    @property
    def encoding_noise(self) -> torch.Tensor:
        return torch.exp(self.encoding_noise_log)

    def initial_state(self, batch_size: int) -> AdaptiveAsymmetricDualMemoryState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        reference = self.base_learning_rate_logit
        kwargs = {"device": reference.device, "dtype": reference.dtype}
        n_items = self.config.max_items
        return AdaptiveAsymmetricDualMemoryState(
            item_values=torch.zeros(batch_size, n_items, **kwargs),
            local_evidence=torch.zeros(batch_size, n_items, n_items, **kwargs),
            local_precision=torch.zeros(batch_size, n_items, n_items, **kwargs),
            update_count=torch.zeros(
                batch_size, device=reference.device, dtype=torch.long
            ),
        )

    def _validate_pair_inputs(
        self,
        state: AdaptiveAsymmetricDualMemoryState,
        cue_features: torch.Tensor,
        left_cue: torch.Tensor,
        right_cue: torch.Tensor,
    ) -> None:
        batch_size = state.item_values.shape[0]
        expected = (batch_size, self.config.max_items, self.config.cue_dim)
        if cue_features.shape != expected:
            raise ValueError(f"cue_features must have shape {expected}")
        if left_cue.shape != (batch_size,) or right_cue.shape != (batch_size,):
            raise ValueError("cue indices must have shape [batch]")
        if torch.any(left_cue == right_cue):
            raise ValueError("a pair cannot contain the same cue twice")
        for name, cue in (("left", left_cue), ("right", right_cue)):
            if torch.any(cue < 0) or torch.any(cue >= self.config.max_items):
                raise ValueError(f"{name} cue outside configured range")

    @staticmethod
    def _binary_entropy(probability: torch.Tensor) -> torch.Tensor:
        probability = torch.clamp(probability, min=1e-7, max=1.0 - 1e-7)
        return -probability * torch.log2(probability) - (
            1.0 - probability
        ) * torch.log2(1.0 - probability)

    def observe_relation(
        self,
        state: AdaptiveAsymmetricDualMemoryState,
        cue_features: torch.Tensor,
        left_cue: torch.Tensor,
        right_cue: torch.Tensor,
        sign: torch.Tensor,
        *,
        intervention: Any | None = None,
        stochastic: bool = True,
        generator: torch.Generator | None = None,
    ) -> tuple[AdaptiveAsymmetricDualMemoryState, AdaptiveAsymmetricEncodingTrace]:
        """Update after feedback; the current choice must already have occurred."""

        if intervention is None:
            intervention = AdaptiveAsymmetricIntervention()
        self._validate_pair_inputs(state, cue_features, left_cue, right_cue)
        if sign.shape != left_cue.shape or not bool(torch.all(torch.abs(sign) == 1)):
            raise ValueError("sign must contain only -1/+1 with shape [batch]")
        sign = sign.to(dtype=self.base_learning_rate_logit.dtype)
        left_cue = left_cue.long()
        right_cue = right_cue.long()
        winner = torch.where(sign > 0, left_cue, right_cue)
        loser = torch.where(sign > 0, right_cue, left_cue)
        batch = torch.arange(sign.shape[0], device=sign.device)
        winner_value = state.item_values[batch, winner]
        loser_value = state.item_values[batch, loser]
        correct_probability = torch.sigmoid(
            (winner_value - loser_value) / self.entropy_temperature
        )
        entropy = self._binary_entropy(correct_probability)
        mode = getattr(intervention, "asymmetry_mode", "adaptive")
        if mode == "symmetric":
            allocation = torch.zeros_like(entropy)
        elif mode == "winner_only":
            allocation = torch.ones_like(entropy)
        elif mode == "adaptive":
            allocation = entropy
        else:
            raise ValueError("unknown asymmetry mode")

        rate_noise = torch.zeros_like(entropy)
        if stochastic:
            rate_noise = torch.randn(
                entropy.shape,
                device=entropy.device,
                dtype=entropy.dtype,
                generator=generator,
            ) * self.encoding_noise
        resource = torch.sigmoid(self.base_learning_rate_logit + rate_noise)
        winner_rate = resource * (1.0 + allocation) * 0.5
        loser_rate = resource * (1.0 - allocation) * 0.5
        weighted_difference = self.difference_weight * (
            winner_value - loser_value
        )
        winner_delta = winner_rate * torch.relu(
            1.0 - weighted_difference - winner_value
        )
        loser_delta = -loser_rate * torch.relu(
            1.0 - weighted_difference + loser_value
        )

        item_values = state.item_values
        if getattr(intervention, "item_update_enabled", True):
            item_change = torch.zeros_like(item_values)
            item_change.scatter_add_(1, winner[:, None], winner_delta[:, None])
            item_change.scatter_add_(1, loser[:, None], loser_delta[:, None])
            item_values = item_values + item_change

        local_evidence = state.local_evidence
        local_precision = state.local_precision
        write_strength = torch.sigmoid(self.pair_write_logit + rate_noise)
        if getattr(intervention, "local_update_enabled", True):
            left = F.one_hot(left_cue, num_classes=self.config.max_items).to(sign.dtype)
            right = F.one_hot(right_cue, num_classes=self.config.max_items).to(sign.dtype)
            oriented_edge = left[:, :, None] * right[:, None, :] - right[:, :, None] * left[:, None, :]
            undirected_edge = torch.abs(oriented_edge)
            local_evidence = local_evidence + (
                write_strength * sign
            )[:, None, None] * oriented_edge
            local_precision = local_precision + write_strength[:, None, None] * undirected_edge

        next_state = AdaptiveAsymmetricDualMemoryState(
            item_values=item_values,
            local_evidence=local_evidence,
            local_precision=local_precision,
            update_count=state.update_count + 1,
        )
        return next_state, AdaptiveAsymmetricEncodingTrace(
            correct_relation_probability=correct_probability,
            entropy=entropy,
            winner_learning_rate=winner_rate,
            loser_learning_rate=loser_rate,
            winner_delta=winner_delta,
            loser_delta=loser_delta,
            write_strength=write_strength,
        )

    def consolidate(
        self,
        state: AdaptiveAsymmetricDualMemoryState,
        event: Any,
        *,
        stochastic: bool = True,
        generator: torch.Generator | None = None,
    ) -> tuple[AdaptiveAsymmetricDualMemoryState, AdaptiveAsymmetricConsolidationTrace]:
        """A visible boundary is observed but online item state is not rewritten."""

        del stochastic, generator
        value = event.value if isinstance(event, Enum) else None
        if value not in self._LEGAL_EVENTS:
            raise ValueError(
                "consolidation requires an explicit phase transition or visible boundary"
            )
        zero_change = torch.zeros_like(state.item_values)
        uncertainty = 1.0 - torch.abs(torch.tanh(state.item_values))
        return state, AdaptiveAsymmetricConsolidationTrace(
            event=str(value),
            proposed_scores=state.item_values,
            score_change=zero_change,
            uncertainty=uncertainty,
        )

    def query_pair(
        self,
        state: AdaptiveAsymmetricDualMemoryState,
        cue_features: torch.Tensor,
        left_cue: torch.Tensor,
        right_cue: torch.Tensor,
        *,
        intervention: Any | None = None,
    ) -> AdaptiveAsymmetricQueryTrace:
        if intervention is None:
            intervention = AdaptiveAsymmetricIntervention()
        self._validate_pair_inputs(state, cue_features, left_cue, right_cue)
        batch = torch.arange(left_cue.shape[0], device=left_cue.device)
        left_cue = left_cue.long()
        right_cue = right_cue.long()
        evidence = state.local_evidence[batch, left_cue, right_cue]
        precision = state.local_precision[batch, left_cue, right_cue]
        local_mean = evidence / torch.clamp(precision, min=1e-8)
        retrieval = 1.0 - torch.exp(-precision)
        local_logit = self.local_scale * retrieval * local_mean
        item_difference = (
            state.item_values[batch, left_cue]
            - state.item_values[batch, right_cue]
        )
        global_logit = self.item_scale * item_difference
        if not getattr(intervention, "local_route_enabled", True):
            local_logit = torch.zeros_like(local_logit)
        if not getattr(intervention, "global_route_enabled", True):
            global_logit = torch.zeros_like(global_logit)
        probability = torch.sigmoid(local_logit + global_logit)
        lapse = self.decision_lapse
        probability = lapse * 0.5 + (1.0 - lapse) * probability
        probability = torch.clamp(probability, min=1e-7, max=1.0 - 1e-7)
        item_probability = torch.sigmoid(torch.abs(item_difference))
        item_reliability = 1.0 - self._binary_entropy(item_probability)
        return AdaptiveAsymmetricQueryTrace(
            logits=torch.stack((torch.log(probability), torch.log1p(-probability)), dim=1),
            probability_first=probability,
            uncertainty=1.0 - 2.0 * torch.abs(probability - 0.5),
            local_logit=local_logit,
            global_logit=global_logit,
            retrieval_strength=retrieval,
            global_reliability=item_reliability,
        )
