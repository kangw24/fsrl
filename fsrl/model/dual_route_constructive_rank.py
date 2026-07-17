"""Dual-route, sign-only constructive ranking process prototype.

The model has no participant, item, pair, or rank-specific slow parameter.
Every episode starts with empty local edge memory, a zero global geometry, and
zero RetroModulRNN fast state.  A full RetroModulRNN acts as the shared
encoding/write controller; a permutation-equivariant iterative integrator
consolidates local evidence into scalar global scores only at a legal visible
event.  Queries read both routes without changing any state.

This module is an engineered, falsifiable algorithmic candidate.  Its
existence does not imply that the brain implements these exact equations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn.functional as F
from torch import nn

from fsrl.model.retro_modul_rnn import RetroModulRNN


class ConsolidationEvent(str, Enum):
    """Only events that can be visible to a participant are legal."""

    PHASE_TRANSITION = "phase_transition"
    VISIBLE_BOUNDARY = "visible_boundary"


@dataclass(frozen=True)
class DualRouteConstructiveRankConfig:
    max_items: int = 8
    cue_dim: int = 15
    controller_hidden: int = 32
    controller_steps: int = 3
    integrator_steps: int = 8
    encoding_noise_init: float = 0.05
    consolidation_noise_init: float = 0.05
    decision_lapse_init: float = 0.02
    local_scale_init: float = 1.0
    global_scale_init: float = 1.0
    consolidation_rate_init: float = 0.8

    def __post_init__(self) -> None:
        if self.max_items < 3:
            raise ValueError("max_items must be at least 3")
        if self.cue_dim <= 0 or self.controller_hidden <= 0:
            raise ValueError("cue_dim and controller_hidden must be positive")
        if self.controller_steps < 3:
            raise ValueError("controller_steps must allow encode, eligibility, write")
        if self.integrator_steps <= 0:
            raise ValueError("integrator_steps must be positive")
        for name, value in (
            ("encoding_noise_init", self.encoding_noise_init),
            ("consolidation_noise_init", self.consolidation_noise_init),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.decision_lapse_init < 1.0:
            raise ValueError("decision_lapse_init must be in [0, 1)")
        if self.local_scale_init <= 0.0 or self.global_scale_init <= 0.0:
            raise ValueError("readout scales must be positive")
        if not 0.0 < self.consolidation_rate_init < 1.0:
            raise ValueError("consolidation_rate_init must be inside (0, 1)")

    @property
    def controller_input_dim(self) -> int:
        # pair sum, absolute difference, sign-aligned difference, two event
        # bits (relation/blank), and a constant bias channel.
        return 3 * self.cue_dim + 3


@dataclass
class DualRouteConstructiveRankState:
    controller_hidden: torch.Tensor
    controller_eligibility: torch.Tensor
    controller_plastic_weights: torch.Tensor
    local_evidence: torch.Tensor
    local_precision: torch.Tensor
    global_scores: torch.Tensor
    global_uncertainty: torch.Tensor
    consolidation_count: torch.Tensor

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.controller_hidden,
            self.controller_eligibility,
            self.controller_plastic_weights,
            self.local_evidence,
            self.local_precision,
            self.global_scores,
            self.global_uncertainty,
            self.consolidation_count,
        )


@dataclass(frozen=True)
class DualRouteIntervention:
    local_route_enabled: bool = True
    global_route_enabled: bool = True
    controller_enabled: bool = True
    plastic_write_enabled: bool = True
    eligibility_enabled: bool = True


@dataclass
class EncodingTrace:
    write_strength: torch.Tensor
    controller_output: torch.Tensor
    plastic_delta_norm: torch.Tensor


@dataclass
class ConsolidationTrace:
    event: ConsolidationEvent
    proposed_scores: torch.Tensor
    score_change: torch.Tensor
    uncertainty: torch.Tensor


@dataclass
class QueryTrace:
    logits: torch.Tensor
    probability_first: torch.Tensor
    uncertainty: torch.Tensor
    local_logit: torch.Tensor
    global_logit: torch.Tensor
    retrieval_strength: torch.Tensor
    global_reliability: torch.Tensor


def _logit(probability: float) -> float:
    if probability <= 0.0:
        return -20.0
    if probability >= 1.0:
        return 20.0
    return math.log(probability / (1.0 - probability))


class DualRouteConstructiveRank(nn.Module):
    """A fresh local-memory plus consolidated-global-geometry process."""

    def __init__(self, config: DualRouteConstructiveRankConfig):
        super().__init__()
        self.config = config
        controller_config = {
            "inputsize": config.controller_input_dim,
            "outputsize": 2,
            "hs": config.controller_hidden,
            "bs": 1,
            "subject_embedding_dim": 0,
            "pw_init_std": 0.0,
            "use_plasticity": True,
        }
        self.controller = RetroModulRNN(controller_config)
        self.base_write_logit = nn.Parameter(torch.zeros(()))
        self.integrator_step_logits = nn.Parameter(
            torch.zeros(config.integrator_steps)
        )
        self.consolidation_rate_logit = nn.Parameter(
            torch.tensor(_logit(config.consolidation_rate_init))
        )
        self.local_scale_log = nn.Parameter(
            torch.tensor(math.log(config.local_scale_init))
        )
        self.global_scale_log = nn.Parameter(
            torch.tensor(math.log(config.global_scale_init))
        )
        self.encoding_noise_log = nn.Parameter(
            torch.tensor(math.log(config.encoding_noise_init))
        )
        self.consolidation_noise_log = nn.Parameter(
            torch.tensor(math.log(config.consolidation_noise_init))
        )
        self.decision_lapse_logit = nn.Parameter(
            torch.tensor(_logit(config.decision_lapse_init))
        )
        # RetroModulRNN follows the repository-wide device policy internally.
        # Keep the wrapper parameters on that same device so fast state cannot
        # become a silent CPU/CUDA mixture.
        self.to(self.controller.w.device)

    @property
    def encoding_noise(self) -> torch.Tensor:
        return torch.exp(self.encoding_noise_log)

    @property
    def consolidation_noise(self) -> torch.Tensor:
        return torch.exp(self.consolidation_noise_log)

    @property
    def decision_lapse(self) -> torch.Tensor:
        return torch.sigmoid(self.decision_lapse_logit)

    @property
    def local_scale(self) -> torch.Tensor:
        return torch.exp(self.local_scale_log)

    @property
    def global_scale(self) -> torch.Tensor:
        return torch.exp(self.global_scale_log)

    def initial_state(self, batch_size: int) -> DualRouteConstructiveRankState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        reference = self.base_write_logit
        kwargs = {"device": reference.device, "dtype": reference.dtype}
        n = self.config.max_items
        return DualRouteConstructiveRankState(
            controller_hidden=self.controller.initialZeroState(batch_size),
            controller_eligibility=self.controller.initialZeroET(batch_size),
            controller_plastic_weights=self.controller.initialZeroPlasticWeights(
                batch_size
            ),
            local_evidence=torch.zeros(batch_size, n, n, **kwargs),
            local_precision=torch.zeros(batch_size, n, n, **kwargs),
            global_scores=torch.zeros(batch_size, n, **kwargs),
            global_uncertainty=torch.ones(batch_size, n, **kwargs),
            consolidation_count=torch.zeros(
                batch_size, device=reference.device, dtype=torch.long
            ),
        )

    def _validate_pair_inputs(
        self,
        state: DualRouteConstructiveRankState,
        cue_features: torch.Tensor,
        left_cue: torch.Tensor,
        right_cue: torch.Tensor,
    ) -> None:
        batch_size = state.local_evidence.shape[0]
        expected = (
            batch_size,
            self.config.max_items,
            self.config.cue_dim,
        )
        if cue_features.shape != expected:
            raise ValueError(f"cue_features must have shape {expected}")
        if left_cue.shape != (batch_size,) or right_cue.shape != (batch_size,):
            raise ValueError("cue indices must have shape [batch]")
        if torch.any(left_cue == right_cue):
            raise ValueError("a pair cannot contain the same cue twice")
        if torch.any(left_cue < 0) or torch.any(
            left_cue >= self.config.max_items
        ):
            raise ValueError("left cue outside configured range")
        if torch.any(right_cue < 0) or torch.any(
            right_cue >= self.config.max_items
        ):
            raise ValueError("right cue outside configured range")

    @staticmethod
    def _gather_cues(features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        gather_index = indices[:, None, None].expand(-1, 1, features.shape[-1])
        return features.gather(1, gather_index).squeeze(1)

    def _relation_input(
        self,
        cue_features: torch.Tensor,
        left_cue: torch.Tensor,
        right_cue: torch.Tensor,
        sign: torch.Tensor,
    ) -> torch.Tensor:
        left = self._gather_cues(cue_features, left_cue)
        right = self._gather_cues(cue_features, right_cue)
        aligned_difference = sign[:, None] * (left - right)
        inputs = torch.cat(
            (left + right, torch.abs(left - right), aligned_difference), dim=1
        )
        event_and_bias = torch.zeros(
            inputs.shape[0], 3, device=inputs.device, dtype=inputs.dtype
        )
        event_and_bias[:, 0] = 1.0
        event_and_bias[:, 2] = 1.0
        return torch.cat((inputs, event_and_bias), dim=1)

    def _blank_input(self, batch_size: int) -> torch.Tensor:
        inputs = torch.zeros(
            batch_size,
            self.config.controller_input_dim,
            device=self.base_write_logit.device,
            dtype=self.base_write_logit.dtype,
        )
        inputs[:, -2] = 1.0
        inputs[:, -1] = 1.0
        return inputs

    def observe_relation(
        self,
        state: DualRouteConstructiveRankState,
        cue_features: torch.Tensor,
        left_cue: torch.Tensor,
        right_cue: torch.Tensor,
        sign: torch.Tensor,
        *,
        intervention: DualRouteIntervention | None = None,
        stochastic: bool = True,
        generator: torch.Generator | None = None,
    ) -> tuple[DualRouteConstructiveRankState, EncodingTrace]:
        """Encode one visible binary relation; no magnitude is accepted."""

        if intervention is None:
            intervention = DualRouteIntervention()
        self._validate_pair_inputs(state, cue_features, left_cue, right_cue)
        if sign.shape != left_cue.shape or not bool(torch.all(torch.abs(sign) == 1)):
            raise ValueError("sign must contain only -1/+1 with shape [batch]")
        sign = sign.to(dtype=self.base_write_logit.dtype)
        batch_size = sign.shape[0]

        hidden = self.controller.initialZeroState(batch_size)
        eligibility = self.controller.initialZeroET(batch_size)
        plastic_before = state.controller_plastic_weights
        plastic = plastic_before
        controller_output = torch.zeros(
            batch_size, 2, device=sign.device, dtype=sign.dtype
        )
        for step_index in range(self.config.controller_steps):
            inputs = (
                self._relation_input(
                    cue_features, left_cue.long(), right_cue.long(), sign
                )
                if step_index == 0
                else self._blank_input(batch_size)
            )
            (
                controller_output,
                _,
                _,
                hidden,
                eligibility,
                plastic,
            ) = self.controller(
                inputs,
                hidden,
                eligibility,
                plastic,
                teacher_da=None,
                plastic_write_enabled=(
                    intervention.plastic_write_enabled
                    and step_index == self.config.controller_steps - 1
                ),
                eligibility_update_enabled=intervention.eligibility_enabled,
            )

        controller_logit = controller_output[:, 0] - controller_output[:, 1]
        if not intervention.controller_enabled:
            controller_logit = torch.zeros_like(controller_logit)
        noise = torch.zeros_like(controller_logit)
        if stochastic:
            noise = torch.randn(
                controller_logit.shape,
                device=controller_logit.device,
                dtype=controller_logit.dtype,
                generator=generator,
            ) * self.encoding_noise
        write_strength = torch.sigmoid(
            self.base_write_logit + controller_logit + noise
        )

        left = F.one_hot(
            left_cue.long(), num_classes=self.config.max_items
        ).to(sign.dtype)
        right = F.one_hot(
            right_cue.long(), num_classes=self.config.max_items
        ).to(sign.dtype)
        oriented_edge = left[:, :, None] * right[:, None, :] - right[
            :, :, None
        ] * left[:, None, :]
        undirected_edge = torch.abs(oriented_edge)
        evidence = state.local_evidence + (
            write_strength * sign
        )[:, None, None] * oriented_edge
        precision = state.local_precision + write_strength[
            :, None, None
        ] * undirected_edge

        next_state = DualRouteConstructiveRankState(
            controller_hidden=hidden,
            controller_eligibility=eligibility,
            controller_plastic_weights=plastic,
            local_evidence=evidence,
            local_precision=precision,
            global_scores=state.global_scores,
            global_uncertainty=state.global_uncertainty,
            consolidation_count=state.consolidation_count,
        )
        trace = EncodingTrace(
            write_strength=write_strength,
            controller_output=controller_output,
            plastic_delta_norm=(plastic - plastic_before).flatten(1).norm(dim=1),
        )
        return next_state, trace

    def consolidate(
        self,
        state: DualRouteConstructiveRankState,
        event: ConsolidationEvent,
        *,
        stochastic: bool = True,
        generator: torch.Generator | None = None,
    ) -> tuple[DualRouteConstructiveRankState, ConsolidationTrace]:
        """Construct a scalar global geometry at a legal visible event."""

        if not isinstance(event, ConsolidationEvent):
            raise ValueError(
                "consolidation requires an explicit phase transition or visible boundary"
            )
        precision = state.local_precision
        relation = state.local_evidence / torch.clamp(precision, min=1e-8)
        degree = precision.sum(dim=2)
        scores = state.global_scores
        for step_logit in self.integrator_step_logits:
            predicted = torch.tanh(scores[:, :, None] - scores[:, None, :])
            residual = precision * (relation - predicted)
            delta = residual.sum(dim=2) / torch.clamp(degree, min=1.0)
            scores = scores + torch.sigmoid(step_logit) * delta
            scores = scores - scores.mean(dim=1, keepdim=True)
        proposed = scores
        if stochastic:
            noise = torch.randn(
                proposed.shape,
                device=proposed.device,
                dtype=proposed.dtype,
                generator=generator,
            )
            proposed = proposed + self.consolidation_noise * noise / torch.sqrt(
                degree + 1.0
            )
            proposed = proposed - proposed.mean(dim=1, keepdim=True)
        rate = torch.sigmoid(self.consolidation_rate_logit)
        global_scores = (1.0 - rate) * state.global_scores + rate * proposed
        final_prediction = torch.tanh(
            proposed[:, :, None] - proposed[:, None, :]
        )
        conflict = (
            precision * torch.abs(relation - final_prediction)
        ).sum(dim=2) / torch.clamp(degree, min=1.0)
        # Repeated opposing signs can cancel to a zero mean relation.  A
        # residual-only score would call that agreement with tied scores and
        # become spuriously certain.  Track within-edge sign disagreement
        # separately from integrator residual.
        edge_consistency = torch.abs(state.local_evidence) / torch.clamp(
            precision, min=1e-8
        )
        ambiguity = (precision * (1.0 - edge_consistency)).sum(
            dim=2
        ) / torch.clamp(degree, min=1.0)
        coverage_uncertainty = 1.0 / torch.sqrt(degree + 1.0)
        uncertainty = torch.clamp(
            coverage_uncertainty + conflict + ambiguity, max=1.0
        )
        next_state = DualRouteConstructiveRankState(
            controller_hidden=state.controller_hidden,
            controller_eligibility=state.controller_eligibility,
            controller_plastic_weights=state.controller_plastic_weights,
            local_evidence=state.local_evidence,
            local_precision=state.local_precision,
            global_scores=global_scores,
            global_uncertainty=uncertainty,
            consolidation_count=state.consolidation_count + 1,
        )
        return next_state, ConsolidationTrace(
            event=event,
            proposed_scores=proposed,
            score_change=global_scores - state.global_scores,
            uncertainty=uncertainty,
        )

    def query_pair(
        self,
        state: DualRouteConstructiveRankState,
        cue_features: torch.Tensor,
        left_cue: torch.Tensor,
        right_cue: torch.Tensor,
        *,
        intervention: DualRouteIntervention | None = None,
    ) -> QueryTrace:
        """Read local and global routes without modifying any fast state."""

        if intervention is None:
            intervention = DualRouteIntervention()
        self._validate_pair_inputs(state, cue_features, left_cue, right_cue)
        batch = torch.arange(left_cue.shape[0], device=left_cue.device)
        left_cue = left_cue.long()
        right_cue = right_cue.long()
        evidence = state.local_evidence[batch, left_cue, right_cue]
        precision = state.local_precision[batch, left_cue, right_cue]
        local_mean = evidence / torch.clamp(precision, min=1e-8)
        retrieval = 1.0 - torch.exp(-precision)
        local_logit = self.local_scale * retrieval * local_mean
        global_difference = (
            state.global_scores[batch, left_cue]
            - state.global_scores[batch, right_cue]
        )
        global_reliability = 1.0 - 0.5 * (
            state.global_uncertainty[batch, left_cue]
            + state.global_uncertainty[batch, right_cue]
        )
        global_reliability = torch.clamp(global_reliability, min=0.0, max=1.0)
        global_logit = self.global_scale * global_reliability * global_difference
        if not intervention.local_route_enabled:
            local_logit = torch.zeros_like(local_logit)
        if not intervention.global_route_enabled:
            global_logit = torch.zeros_like(global_logit)
        raw_logit = local_logit + global_logit
        probability = torch.sigmoid(raw_logit)
        lapse = self.decision_lapse
        probability = lapse * 0.5 + (1.0 - lapse) * probability
        probability = torch.clamp(probability, min=1e-7, max=1.0 - 1e-7)
        logits = torch.stack((torch.log(probability), torch.log1p(-probability)), dim=1)
        uncertainty = 1.0 - 2.0 * torch.abs(probability - 0.5)
        return QueryTrace(
            logits=logits,
            probability_first=probability,
            uncertainty=uncertainty,
            local_logit=local_logit,
            global_logit=global_logit,
            retrieval_strength=retrieval,
            global_reliability=global_reliability,
        )
