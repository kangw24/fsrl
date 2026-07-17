"""Direction- and norm-controlled interventions for M&K source traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch.nn import functional as F

from fsrl.miconi_kay.source.evaluation import SourceFullTrace
from fsrl.miconi_kay.source.reinstatement import RecodedVectors


AblationCondition = Literal[
    "target_recoded_step1",
    "recoded_step0_time_control",
    "recoded_step2_time_control",
    "feedforward_step1_control",
    "label_permuted_step1_control",
    "batch_permuted_step1_control",
    "random_step1_control",
]


@dataclass
class InterventionAudit:
    condition: str
    applied: bool = False
    trial_index: int | None = None
    step_index: int | None = None
    selected_count: int = 0
    reference_norm_mean: float | None = None
    delta_norm_mean: float | None = None
    post_hidden_norm_mean: float | None = None
    maximum_relative_delta_norm_error: float | None = None
    maximum_relative_post_norm_error: float | None = None
    reference_was_clamped_count: int = 0


def _pair_axes(
    representations: torch.Tensor,
    pairs: np.ndarray,
) -> torch.Tensor:
    if representations.ndim != 3:
        raise ValueError("representations must have items × batch × hidden shape")
    device = representations.device
    pair_tensor = torch.as_tensor(pairs, device=device, dtype=torch.long)
    if pair_tensor.shape != (representations.shape[1], 2):
        raise ValueError("pair array and representation batch disagree")
    batch = torch.arange(representations.shape[1], device=device)
    return (
        representations[pair_tensor[:, 0], batch]
        - representations[pair_tensor[:, 1], batch]
    )


def recoded_projection_norms(
    trace: SourceFullTrace,
    vectors: RecodedVectors,
    *,
    python_step: int = 1,
) -> torch.Tensor:
    """Norm removed by an exact recoded-axis projection in the baseline trace."""

    device = vectors.item_vectors.device
    hidden = torch.as_tensor(
        trace.hidden_by_trial_step[:, trace.target_trial_index, python_step, :],
        device=device,
        dtype=vectors.item_vectors.dtype,
    )
    pairs = trace.pairs[:, trace.target_trial_index]
    axes = F.normalize(_pair_axes(vectors.item_vectors, pairs), dim=-1)
    norms = torch.abs((hidden * axes).sum(dim=-1))
    mask = torch.as_tensor(
        trace.forced_first_exposure_mask, device=device, dtype=torch.bool
    )
    return torch.where(mask, norms, torch.zeros_like(norms)).detach()


def _orthogonal_component(
    axes: torch.Tensor,
    hidden_unit: torch.Tensor,
) -> torch.Tensor:
    orthogonal = axes - (axes * hidden_unit).sum(dim=-1, keepdim=True) * hidden_unit
    norm = torch.linalg.vector_norm(orthogonal, dim=-1, keepdim=True)
    fallback = torch.roll(hidden_unit, shifts=1, dims=-1)
    fallback = fallback - (
        fallback * hidden_unit
    ).sum(dim=-1, keepdim=True) * hidden_unit
    fallback = F.normalize(fallback, dim=-1)
    return torch.where(norm > 1e-8, orthogonal / norm.clamp_min(1e-12), fallback)


class MatchedHiddenAblation:
    """Remove a hidden component after a chosen model step.

    The target condition removes the exact projection onto the signed recoded
    pair axis.  Controls rotate their specified axis so that its alignment with
    the current hidden state equals the target projection magnitude.  Thus
    target and controls have equal perturbation norm and equal post-lesion
    hidden norm for every selected episode.

    The evaluator calls this after logits and the within-call eligibility
    update have been computed.  At Python step 1 this preserves the current
    response while changing the state that enters steps 2 and 3.
    """

    _STEP_BY_CONDITION = {
        "target_recoded_step1": 1,
        "recoded_step0_time_control": 0,
        "recoded_step2_time_control": 2,
        "feedforward_step1_control": 1,
        "label_permuted_step1_control": 1,
        "batch_permuted_step1_control": 1,
        "random_step1_control": 1,
    }

    def __init__(
        self,
        *,
        condition: AblationCondition,
        recoded_vectors: torch.Tensor,
        feedforward_vectors: torch.Tensor,
        selected_mask: np.ndarray,
        reference_projection_norms: torch.Tensor,
        target_trial_index: int,
        random_seed: int = 701,
    ) -> None:
        if condition not in self._STEP_BY_CONDITION:
            raise ValueError(f"unknown ablation condition: {condition}")
        if recoded_vectors.shape != feedforward_vectors.shape:
            raise ValueError("recoded and feedforward representations disagree")
        if recoded_vectors.ndim != 3:
            raise ValueError("representations must have items × batch × hidden shape")
        batch_size = recoded_vectors.shape[1]
        if selected_mask.shape != (batch_size,):
            raise ValueError("selected mask has the wrong batch dimension")
        if tuple(reference_projection_norms.shape) != (batch_size,):
            raise ValueError("reference projection norms have the wrong shape")
        self.condition = condition
        self.recoded_vectors = recoded_vectors.detach()
        self.feedforward_vectors = feedforward_vectors.detach()
        self.selected_mask = selected_mask.astype(bool, copy=True)
        self.reference_projection_norms = reference_projection_norms.detach()
        self.target_trial_index = target_trial_index
        self.random_seed = random_seed
        self.audit = InterventionAudit(condition=condition)

        generator = torch.Generator(device=recoded_vectors.device)
        generator.manual_seed(random_seed)
        self.random_axes = torch.randn(
            batch_size,
            recoded_vectors.shape[-1],
            generator=generator,
            device=recoded_vectors.device,
            dtype=recoded_vectors.dtype,
        )

    @property
    def target_step(self) -> int:
        return self._STEP_BY_CONDITION[self.condition]

    def _axes(self, pairs: np.ndarray) -> torch.Tensor:
        if self.condition == "feedforward_step1_control":
            return _pair_axes(self.feedforward_vectors, pairs)
        if self.condition == "label_permuted_step1_control":
            return _pair_axes(
                torch.roll(self.recoded_vectors, shifts=1, dims=0), pairs
            )
        if self.condition == "batch_permuted_step1_control":
            # Preserve ordinal labels and the distribution of optimized
            # vectors while assigning every episode another episode's random
            # cue-specific representation.
            return _pair_axes(
                torch.roll(self.recoded_vectors, shifts=1, dims=1), pairs
            )
        if self.condition == "random_step1_control":
            return self.random_axes
        return _pair_axes(self.recoded_vectors, pairs)

    def __call__(
        self,
        trial_index: int,
        step_index: int,
        pairs: np.ndarray,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        if trial_index != self.target_trial_index or step_index != self.target_step:
            return hidden
        if self.audit.applied:
            raise RuntimeError("hidden ablation was applied more than once")

        device = hidden.device
        mask = torch.as_tensor(self.selected_mask, device=device, dtype=torch.bool)
        selected = torch.nonzero(mask, as_tuple=False).flatten()
        axes = F.normalize(self._axes(pairs)[selected], dim=-1)
        current = hidden[selected]
        hidden_norm = torch.linalg.vector_norm(current, dim=-1).clamp_min(1e-12)
        hidden_unit = current / hidden_norm[:, None]
        reference = self.reference_projection_norms.to(device)[selected]

        if self.condition == "target_recoded_step1":
            coefficients = (current * axes).sum(dim=-1)
            oriented_axes = torch.where(
                (coefficients >= 0)[:, None], axes, -axes
            )
            delta_norm = coefficients.abs()
            clamped = torch.zeros_like(delta_norm, dtype=torch.bool)
        else:
            maximum = hidden_norm * (1.0 - 1e-6)
            delta_norm = torch.minimum(reference, maximum)
            clamped = reference > maximum
            cosine = (delta_norm / hidden_norm).clamp(0.0, 1.0 - 1e-6)
            orthogonal = _orthogonal_component(axes, hidden_unit)
            oriented_axes = (
                cosine[:, None] * hidden_unit
                + torch.sqrt((1.0 - cosine.square()).clamp_min(0.0))[:, None]
                * orthogonal
            )

        delta = delta_norm[:, None] * oriented_axes
        intervened_selected = current - delta
        result = hidden.clone()
        result[selected] = intervened_selected

        actual_delta_norm = torch.linalg.vector_norm(delta, dim=-1)
        post_norm = torch.linalg.vector_norm(intervened_selected, dim=-1)
        expected_post_norm = torch.sqrt(
            (hidden_norm.square() - delta_norm.square()).clamp_min(0.0)
        )
        relative_delta_error = (
            (actual_delta_norm - reference).abs()
            / reference.clamp_min(1e-8)
        )
        relative_post_error = (
            (post_norm - expected_post_norm).abs()
            / expected_post_norm.clamp_min(1e-8)
        )
        self.audit = InterventionAudit(
            condition=self.condition,
            applied=True,
            trial_index=trial_index,
            step_index=step_index,
            selected_count=int(selected.numel()),
            reference_norm_mean=float(reference.mean().detach().cpu()),
            delta_norm_mean=float(actual_delta_norm.mean().detach().cpu()),
            post_hidden_norm_mean=float(post_norm.mean().detach().cpu()),
            maximum_relative_delta_norm_error=float(
                relative_delta_error.max().detach().cpu()
            ),
            maximum_relative_post_norm_error=float(
                relative_post_error.max().detach().cpu()
            ),
            reference_was_clamped_count=int(clamped.sum().detach().cpu()),
        )
        return result
