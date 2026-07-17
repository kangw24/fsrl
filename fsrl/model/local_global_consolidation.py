"""RNN-free local-memory plus visible-boundary consolidation candidate.

LocalGlobalConsolidation-v1 (LGC-v1) is the exact computational submodel used
by the preregistered DCR-v1 constant-gate control.  It has no recurrent neural
controller, eligibility trace, or plastic neural weights.  A shared scalar
write strength stores sign-only pair evidence; a permutation-equivariant
iterative solver constructs scalar item scores only at participant-visible
events.  Queries read local and global state without writing either.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from fsrl.device import DEVICE


@dataclass(frozen=True)
class LocalGlobalConsolidationConfig:
    max_items: int = 8
    cue_dim: int = 15
    integrator_steps: int = 8
    encoding_noise_init: float = 0.05
    consolidation_noise_init: float = 0.05
    decision_lapse_init: float = 0.02
    write_strength_init: float = 0.5
    local_scale_init: float = 1.0
    global_scale_init: float = 1.0
    consolidation_rate_init: float = 0.8

    def __post_init__(self) -> None:
        if self.max_items < 3:
            raise ValueError("max_items must be at least 3")
        if self.cue_dim <= 0 or self.integrator_steps <= 0:
            raise ValueError("cue_dim and integrator_steps must be positive")
        if self.encoding_noise_init <= 0.0 or self.consolidation_noise_init <= 0.0:
            raise ValueError("noise scales must be positive")
        if not 0.0 <= self.decision_lapse_init < 1.0:
            raise ValueError("decision_lapse_init must be in [0, 1)")
        if not 0.0 < self.write_strength_init < 1.0:
            raise ValueError("write_strength_init must be inside (0, 1)")
        if self.local_scale_init <= 0.0 or self.global_scale_init <= 0.0:
            raise ValueError("readout scales must be positive")
        if not 0.0 < self.consolidation_rate_init < 1.0:
            raise ValueError("consolidation_rate_init must be inside (0, 1)")


@dataclass
class LocalGlobalConsolidationState:
    local_evidence: torch.Tensor
    local_precision: torch.Tensor
    global_scores: torch.Tensor
    global_uncertainty: torch.Tensor
    consolidation_count: torch.Tensor

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.local_evidence,
            self.local_precision,
            self.global_scores,
            self.global_uncertainty,
            self.consolidation_count,
        )


@dataclass(frozen=True)
class LocalGlobalIntervention:
    local_route_enabled: bool = True
    global_route_enabled: bool = True


@dataclass
class LocalGlobalEncodingTrace:
    write_strength: torch.Tensor


@dataclass
class LocalGlobalConsolidationTrace:
    event: str
    proposed_scores: torch.Tensor
    score_change: torch.Tensor
    uncertainty: torch.Tensor


@dataclass
class LocalGlobalQueryTrace:
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


class LocalGlobalConsolidation(nn.Module):
    """Minimal dual-route process selected after DCR-v1 plasticity failed."""

    _LEGAL_EVENTS = frozenset(("phase_transition", "visible_boundary"))

    def __init__(self, config: LocalGlobalConsolidationConfig):
        super().__init__()
        self.config = config
        self.base_write_logit = nn.Parameter(
            torch.tensor(_logit(config.write_strength_init))
        )
        self.integrator_step_logits = nn.Parameter(
            torch.zeros(config.integrator_steps)
        )
        self.consolidation_rate_logit = nn.Parameter(
            torch.tensor(_logit(config.consolidation_rate_init))
        )
        self.local_scale_log = nn.Parameter(torch.tensor(math.log(config.local_scale_init)))
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
        self.to(DEVICE)

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

    def initial_state(self, batch_size: int) -> LocalGlobalConsolidationState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        reference = self.base_write_logit
        kwargs = {"device": reference.device, "dtype": reference.dtype}
        n_items = self.config.max_items
        return LocalGlobalConsolidationState(
            local_evidence=torch.zeros(batch_size, n_items, n_items, **kwargs),
            local_precision=torch.zeros(batch_size, n_items, n_items, **kwargs),
            global_scores=torch.zeros(batch_size, n_items, **kwargs),
            global_uncertainty=torch.ones(batch_size, n_items, **kwargs),
            consolidation_count=torch.zeros(
                batch_size, device=reference.device, dtype=torch.long
            ),
        )

    def _validate_pair_inputs(
        self,
        state: LocalGlobalConsolidationState,
        cue_features: torch.Tensor,
        left_cue: torch.Tensor,
        right_cue: torch.Tensor,
    ) -> None:
        batch_size = state.local_evidence.shape[0]
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

    def observe_relation(
        self,
        state: LocalGlobalConsolidationState,
        cue_features: torch.Tensor,
        left_cue: torch.Tensor,
        right_cue: torch.Tensor,
        sign: torch.Tensor,
        *,
        intervention: Any | None = None,
        stochastic: bool = True,
        generator: torch.Generator | None = None,
    ) -> tuple[LocalGlobalConsolidationState, LocalGlobalEncodingTrace]:
        """Store one binary relation with one population-shared write gate."""

        del intervention
        self._validate_pair_inputs(state, cue_features, left_cue, right_cue)
        if sign.shape != left_cue.shape or not bool(torch.all(torch.abs(sign) == 1)):
            raise ValueError("sign must contain only -1/+1 with shape [batch]")
        sign = sign.to(dtype=self.base_write_logit.dtype)
        noise = torch.zeros_like(sign)
        if stochastic:
            noise = torch.randn(
                sign.shape,
                device=sign.device,
                dtype=sign.dtype,
                generator=generator,
            ) * self.encoding_noise
        write_strength = torch.sigmoid(self.base_write_logit + noise)
        left = F.one_hot(left_cue.long(), num_classes=self.config.max_items).to(sign.dtype)
        right = F.one_hot(right_cue.long(), num_classes=self.config.max_items).to(sign.dtype)
        oriented_edge = left[:, :, None] * right[:, None, :] - right[:, :, None] * left[:, None, :]
        undirected_edge = torch.abs(oriented_edge)
        return (
            LocalGlobalConsolidationState(
                local_evidence=state.local_evidence
                + (write_strength * sign)[:, None, None] * oriented_edge,
                local_precision=state.local_precision
                + write_strength[:, None, None] * undirected_edge,
                global_scores=state.global_scores,
                global_uncertainty=state.global_uncertainty,
                consolidation_count=state.consolidation_count,
            ),
            LocalGlobalEncodingTrace(write_strength=write_strength),
        )

    @classmethod
    def _event_value(cls, event: Any) -> str:
        value = event.value if isinstance(event, Enum) else None
        if value not in cls._LEGAL_EVENTS:
            raise ValueError(
                "consolidation requires an explicit phase transition or visible boundary"
            )
        return str(value)

    def consolidate(
        self,
        state: LocalGlobalConsolidationState,
        event: Any,
        *,
        stochastic: bool = True,
        generator: torch.Generator | None = None,
    ) -> tuple[LocalGlobalConsolidationState, LocalGlobalConsolidationTrace]:
        event_value = self._event_value(event)
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
        final_prediction = torch.tanh(proposed[:, :, None] - proposed[:, None, :])
        conflict = (
            precision * torch.abs(relation - final_prediction)
        ).sum(dim=2) / torch.clamp(degree, min=1.0)
        edge_consistency = torch.abs(state.local_evidence) / torch.clamp(
            precision, min=1e-8
        )
        ambiguity = (precision * (1.0 - edge_consistency)).sum(dim=2) / torch.clamp(
            degree, min=1.0
        )
        coverage_uncertainty = 1.0 / torch.sqrt(degree + 1.0)
        uncertainty = torch.clamp(
            coverage_uncertainty + conflict + ambiguity, max=1.0
        )
        next_state = LocalGlobalConsolidationState(
            local_evidence=state.local_evidence,
            local_precision=state.local_precision,
            global_scores=global_scores,
            global_uncertainty=uncertainty,
            consolidation_count=state.consolidation_count + 1,
        )
        return next_state, LocalGlobalConsolidationTrace(
            event=event_value,
            proposed_scores=proposed,
            score_change=global_scores - state.global_scores,
            uncertainty=uncertainty,
        )

    def query_pair(
        self,
        state: LocalGlobalConsolidationState,
        cue_features: torch.Tensor,
        left_cue: torch.Tensor,
        right_cue: torch.Tensor,
        *,
        intervention: Any | None = None,
    ) -> LocalGlobalQueryTrace:
        if intervention is None:
            intervention = LocalGlobalIntervention()
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
        if not getattr(intervention, "local_route_enabled", True):
            local_logit = torch.zeros_like(local_logit)
        if not getattr(intervention, "global_route_enabled", True):
            global_logit = torch.zeros_like(global_logit)
        probability = torch.sigmoid(local_logit + global_logit)
        lapse = self.decision_lapse
        probability = lapse * 0.5 + (1.0 - lapse) * probability
        probability = torch.clamp(probability, min=1e-7, max=1.0 - 1e-7)
        return LocalGlobalQueryTrace(
            logits=torch.stack((torch.log(probability), torch.log1p(-probability)), dim=1),
            probability_first=probability,
            uncertainty=1.0 - 2.0 * torch.abs(probability - 0.5),
            local_logit=local_logit,
            global_logit=global_logit,
            retrieval_strength=retrieval,
            global_reliability=global_reliability,
        )


LGC_PARAMETER_NAMES = frozenset(
    (
        "base_write_logit",
        "integrator_step_logits",
        "consolidation_rate_logit",
        "local_scale_log",
        "global_scale_log",
        "encoding_noise_log",
        "consolidation_noise_log",
        "decision_lapse_logit",
    )
)


def load_dcr_constant_gate_state(
    model: LocalGlobalConsolidation, checkpoint: dict[str, Any]
) -> None:
    """Load only behaviorally reachable constant-gate parameters from DCR."""

    if checkpoint.get("regime") != "constant_gate":
        raise ValueError("only a frozen DCR constant_gate checkpoint can initialize LGC")
    source = checkpoint["state_dict"]
    selected = {name: source[name] for name in LGC_PARAMETER_NAMES}
    missing, unexpected = model.load_state_dict(selected, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"LGC state mismatch: missing={missing}, unexpected={unexpected}")
