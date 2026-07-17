"""Recoded-representation optimization and held-out activity probes.

The optimizer follows the construction used in the authors' ``main.ipynb``:
it sees only slow recurrent weights, plasticity coefficients, the output
decision vector, and isolated-item feedforward representations.  Episode
activity, pair identities, responses, and correctness are used only after the
recoded vectors have been fitted.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fsrl.miconi_kay.source.config import MiconiKaySourceConfig
from fsrl.miconi_kay.source.evaluation import SourceFullTrace
from fsrl.miconi_kay.source.model import MiconiKayRetroModulRNN


@dataclass(frozen=True)
class RecodedOptimizationConfig:
    steps: int = 1000
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    seed: int = 101
    restarts: int = 3
    record_every: int = 50

    def validate(self) -> None:
        if self.steps <= 0 or self.restarts <= 0 or self.record_every <= 0:
            raise ValueError("optimization steps, restarts, and record interval must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning rate must be positive and weight decay non-negative")
        if self.seed < 0:
            raise ValueError("optimization seed must be non-negative")


@dataclass
class RecodedVectors:
    item_vectors: torch.Tensor
    output_vector: torch.Tensor
    final_loss: float
    final_mean_cosine: float
    selected_restart: int
    restart_summaries: list[dict[str, Any]]
    loss_trajectory: list[dict[str, float | int]]
    sign_flipped: bool = False
    early_response_correlation: float | None = None


@dataclass
class ReinstatementProbeResult:
    report: dict[str, Any]
    sample_metrics: dict[str, np.ndarray]


def tensor_sha256(tensor: torch.Tensor | np.ndarray) -> str:
    array = (
        tensor.detach().cpu().contiguous().numpy()
        if isinstance(tensor, torch.Tensor)
        else np.ascontiguousarray(tensor)
    )
    return hashlib.sha256(array.tobytes()).hexdigest()


def extract_feedforward_item_representations(
    model: MiconiKayRetroModulRNN,
    config: MiconiKaySourceConfig,
    cue_data: np.ndarray,
) -> torch.Tensor:
    """Return isolated-item step-1 representations with zero recurrent state."""

    cues = torch.as_tensor(cue_data, device=model.w.device, dtype=model.w.dtype)
    if cues.shape != (config.batch_size, config.max_items, config.cue_size):
        raise ValueError(
            "cue_data shape mismatch: "
            f"expected {(config.batch_size, config.max_items, config.cue_size)}, "
            f"got {tuple(cues.shape)}"
        )
    item_count = config.max_items
    batch_size = config.batch_size
    representations: list[torch.Tensor] = []
    with torch.no_grad():
        for item_index in range(item_count):
            inputs = torch.zeros(
                batch_size,
                config.input_size,
                device=model.w.device,
                dtype=model.w.dtype,
            )
            inputs[:, : config.cue_size] = cues[:, item_index, :]
            inputs[:, config.stimulus_bits] = 1.0
            hidden = model.initialZeroState(batch_size)
            eligibility = model.initialZeroET(batch_size)
            plastic = model.initialZeroPlasticWeights(batch_size)
            _, _, _, hidden, _, _ = model(
                inputs, hidden, eligibility, plastic
            )
            representations.append(hidden.detach())
    return torch.stack(representations, dim=0)


def recoded_projection(
    feedforward: torch.Tensor,
    item_vectors: torch.Tensor,
    output_vector: torch.Tensor,
    recurrent_weight: torch.Tensor,
    plasticity_coefficient: torch.Tensor,
) -> torch.Tensor:
    """Memory-efficient form of the notebook's batched outer-product equation.

    Computes ``(w + alpha * outer(v2, v1)) @ feedforward`` without materializing
    an ``items × batch × hidden × hidden`` tensor.
    """

    if feedforward.shape != item_vectors.shape:
        raise ValueError("feedforward and item-vector tensors must have equal shape")
    if output_vector.ndim != 1:
        raise ValueError("output vector must be one-dimensional")
    base = torch.matmul(feedforward, recurrent_weight.T)
    plastic_drive = torch.matmul(
        item_vectors * feedforward,
        plasticity_coefficient.T,
    )
    return base + plastic_drive * output_vector.view(1, 1, -1)


def _set_optimization_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _optimize_one_restart(
    model: MiconiKayRetroModulRNN,
    feedforward: torch.Tensor,
    config: RecodedOptimizationConfig,
    restart: int,
) -> RecodedVectors:
    device = model.w.device
    _set_optimization_seed(config.seed + restart, device)
    item_vectors = nn.Parameter(
        0.01 * (torch.rand_like(feedforward) - 0.5)
    )
    output_vector = nn.Parameter(
        0.01 * (
            torch.rand(
                model.hidden_size,
                device=device,
                dtype=model.w.dtype,
            )
            - 0.5
        )
    )
    optimizer = torch.optim.Adam(
        (item_vectors, output_vector),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    recurrent = model.w.detach()
    alpha = model.alpha.detach()
    decision = (
        model.h2o.weight[1].detach() - model.h2o.weight[0].detach()
    )
    trajectory: list[dict[str, float | int]] = []
    with torch.enable_grad():
        for step in range(config.steps):
            optimizer.zero_grad(set_to_none=True)
            projected = recoded_projection(
                feedforward,
                item_vectors,
                output_vector,
                recurrent,
                alpha,
            )
            cosine = F.cosine_similarity(
                projected,
                decision.view(1, 1, -1),
                dim=-1,
            )
            loss = -cosine.sum()
            loss.backward()
            optimizer.step()
            if (
                step == 0
                or step == config.steps - 1
                or (step + 1) % config.record_every == 0
            ):
                trajectory.append(
                    {
                        "step": step + 1,
                        "loss": float(loss.detach().cpu()),
                        "mean_cosine": float(cosine.detach().mean().cpu()),
                    }
                )
        # Report the objective of the returned, post-update tensors rather than
        # the pre-update objective from the final Adam iteration.
        with torch.no_grad():
            projected = recoded_projection(
                feedforward,
                item_vectors,
                output_vector,
                recurrent,
                alpha,
            )
            cosine = F.cosine_similarity(
                projected,
                decision.view(1, 1, -1),
                dim=-1,
            )
            final_loss = float((-cosine.sum()).cpu())
            final_mean_cosine = float(cosine.mean().cpu())
    return RecodedVectors(
        item_vectors=item_vectors.detach(),
        output_vector=output_vector.detach(),
        final_loss=final_loss,
        final_mean_cosine=final_mean_cosine,
        selected_restart=restart,
        restart_summaries=[],
        loss_trajectory=trajectory,
    )


def optimize_recoded_vectors(
    model: MiconiKayRetroModulRNN,
    feedforward: torch.Tensor,
    config: RecodedOptimizationConfig,
) -> RecodedVectors:
    """Fit a fixed number of restarts and select by optimization objective only."""

    config.validate()
    expected = (8, feedforward.shape[1], model.hidden_size)
    if tuple(feedforward.shape) != expected:
        raise ValueError(f"expected feedforward shape {expected}, got {tuple(feedforward.shape)}")
    candidates = [
        _optimize_one_restart(model, feedforward, config, restart)
        for restart in range(config.restarts)
    ]
    selected = min(candidates, key=lambda value: value.final_loss)
    selected.restart_summaries = [
        {
            "restart": candidate.selected_restart,
            "seed": config.seed + candidate.selected_restart,
            "final_loss": candidate.final_loss,
            "final_mean_cosine": candidate.final_mean_cosine,
        }
        for candidate in candidates
    ]
    return selected


def _pearson_rows(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_centered = left - left.mean(dim=-1, keepdim=True)
    right_centered = right - right.mean(dim=-1, keepdim=True)
    numerator = (left_centered * right_centered).sum(dim=-1)
    denominator = torch.sqrt(
        left_centered.square().sum(dim=-1)
        * right_centered.square().sum(dim=-1)
    ).clamp_min(1e-12)
    return numerator / denominator


def _pearson_numpy(left: np.ndarray, right: np.ndarray) -> float:
    if left.size != right.size or left.size < 3:
        return float("nan")
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = np.sqrt(
        np.sum(left_centered**2) * np.sum(right_centered**2)
    )
    return float(np.sum(left_centered * right_centered) / denominator) if denominator else float("nan")


def orient_recoded_vectors(
    vectors: RecodedVectors,
    trace: SourceFullTrace,
    *,
    early_trial_index: int = 5,
) -> RecodedVectors:
    """Resolve the joint sign ambiguity using the authors' early-response rule."""

    if not 0 <= early_trial_index < trace.hidden_by_trial_step.shape[1]:
        raise ValueError("early trial index is outside the trace")
    # Python step 2 corresponds to paper step 3 (reward/action input).
    hidden = torch.as_tensor(
        trace.hidden_by_trial_step[:, early_trial_index, 2, :],
        device=vectors.output_vector.device,
        dtype=vectors.output_vector.dtype,
    )
    similarities = F.cosine_similarity(
        hidden,
        vectors.output_vector.view(1, -1),
        dim=-1,
    ).detach().cpu().numpy()
    responses = 2 * trace.response_actions[:, early_trial_index].astype("float64") - 1
    correlation = _pearson_numpy(similarities.astype("float64"), responses)
    flip = bool(np.isfinite(correlation) and correlation < 0)
    if flip:
        vectors.item_vectors = -vectors.item_vectors
        vectors.output_vector = -vectors.output_vector
        correlation = -correlation
    vectors.sign_flipped = flip
    vectors.early_response_correlation = correlation
    return vectors


def _similarities(
    representations: torch.Tensor,
    hidden: torch.Tensor,
) -> torch.Tensor:
    # representations: items × batch × hidden; hidden: batch × steps × hidden
    rep = F.normalize(representations, dim=-1)
    activity = F.normalize(hidden, dim=-1)
    return torch.einsum("ibh,bsh->sib", rep, activity)


def summarize_samples(
    values: np.ndarray, *, seed: int, bootstrap_samples: int
) -> dict[str, Any]:
    values = np.asarray(values, dtype="float64")
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": None, "std": None, "count": 0, "bootstrap95": None}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(bootstrap_samples, values.size))
    bootstrap_means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "count": int(values.size),
        "bootstrap95": [
            float(np.quantile(bootstrap_means, 0.025)),
            float(np.quantile(bootstrap_means, 0.975)),
        ],
    }


def _gather_pair_contrast(
    similarities: torch.Tensor,
    pairs: np.ndarray,
    batch_indices: np.ndarray,
) -> torch.Tensor:
    # similarities: steps × items × batch
    device = similarities.device
    batch = torch.as_tensor(batch_indices, device=device, dtype=torch.long)
    first = torch.as_tensor(pairs[:, 0], device=device, dtype=torch.long)
    second = torch.as_tensor(pairs[:, 1], device=device, dtype=torch.long)
    return 0.5 * (
        similarities[:, first, batch] - similarities[:, second, batch]
    )


def probe_reinstatement(
    trace: SourceFullTrace,
    feedforward: torch.Tensor,
    vectors: RecodedVectors,
    *,
    bootstrap_seed: int = 1001,
    bootstrap_samples: int = 2000,
) -> ReinstatementProbeResult:
    """Evaluate time, item-label, sensory, and neighboring-item predictions."""

    device = vectors.item_vectors.device
    target = trace.target_trial_index
    selected = np.flatnonzero(trace.forced_first_exposure_mask)
    pairs = trace.pairs[selected, target]
    hidden = torch.as_tensor(
        trace.hidden_by_trial_step[:, target, :, :],
        device=device,
        dtype=vectors.item_vectors.dtype,
    )
    recoded = _similarities(vectors.item_vectors, hidden)
    original = _similarities(feedforward, hidden)
    permuted = _similarities(torch.roll(vectors.item_vectors, shifts=1, dims=0), hidden)

    presented = _gather_pair_contrast(recoded, pairs, selected)
    presented_original = _gather_pair_contrast(original, pairs, selected)
    presented_permuted = _gather_pair_contrast(permuted, pairs, selected)

    left_neighbor = np.where(pairs[:, 0] == 3, 2, 5)
    right_neighbor = np.where(pairs[:, 0] == 3, 5, 2)
    neighbor_pairs = np.stack((left_neighbor, right_neighbor), axis=1)
    neighbor = _gather_pair_contrast(recoded, neighbor_pairs, selected)

    central_unoriented_pairs = np.tile(np.asarray([[3, 4]]), (selected.size, 1))
    unoriented = _gather_pair_contrast(recoded, central_unoriented_pairs, selected)

    # The notebook uses cosine similarity only for the early-trial sign rule.
    # Its formal target-trial output-vector probe uses row-wise Pearson
    # correlation (``np.corrcoef``), so retain that distinction here.
    v2_similarity = _pearson_rows(
        hidden,
        vectors.output_vector.view(1, 1, -1).expand_as(hidden),
    ).detach().cpu().numpy()
    responses = 2 * trace.response_actions[:, target].astype("float64") - 1
    v2_response_correlations = [
        _pearson_numpy(v2_similarity[:, step], responses)
        for step in range(v2_similarity.shape[1])
    ]

    item_ff_pearson = _pearson_rows(
        vectors.item_vectors,
        feedforward,
    ).detach().cpu().numpy()

    presented_np = presented.detach().cpu().numpy()
    original_np = presented_original.detach().cpu().numpy()
    permuted_np = presented_permuted.detach().cpu().numpy()
    neighbor_np = neighbor.detach().cpu().numpy()
    unoriented_np = unoriented.detach().cpu().numpy()
    target_step = 1
    other_steps = [0, 2, 3]
    sample_metrics = {
        "presented_target": presented_np[target_step],
        "neighbor_target": neighbor_np[target_step],
        "target_minus_other_max": (
            presented_np[target_step]
            - np.max(presented_np[other_steps], axis=0)
        ),
        "target_minus_original": presented_np[target_step] - original_np[target_step],
        "target_minus_label_permuted": (
            presented_np[target_step] - permuted_np[target_step]
        ),
    }

    report = {
        "selected_first_exposure_count": int(selected.size),
        "target_trial_index_zero_based": target,
        "target_pair": list(trace.barred_pair),
        "optimization": {
            "final_loss": vectors.final_loss,
            "final_mean_cosine": vectors.final_mean_cosine,
            "selected_restart": vectors.selected_restart,
            "restart_summaries": vectors.restart_summaries,
            "loss_trajectory": vectors.loss_trajectory,
            "sign_flipped": vectors.sign_flipped,
            "early_response_correlation": vectors.early_response_correlation,
        },
        "recoded_distinct_from_feedforward": {
            "mean_absolute_pearson": float(np.mean(np.abs(item_ff_pearson))),
            "mean_signed_pearson": float(np.mean(item_ff_pearson)),
        },
        "presented_recoded_by_python_step": [
            summarize_samples(
                presented_np[step],
                seed=bootstrap_seed + step,
                bootstrap_samples=bootstrap_samples,
            )
            for step in range(4)
        ],
        "neighbor_recoded_by_python_step": [
            summarize_samples(
                neighbor_np[step],
                seed=bootstrap_seed + 10 + step,
                bootstrap_samples=bootstrap_samples,
            )
            for step in range(4)
        ],
        "presented_original_by_python_step": [
            summarize_samples(
                original_np[step],
                seed=bootstrap_seed + 20 + step,
                bootstrap_samples=bootstrap_samples,
            )
            for step in range(4)
        ],
        "presented_label_permuted_by_python_step": [
            summarize_samples(
                permuted_np[step],
                seed=bootstrap_seed + 30 + step,
                bootstrap_samples=bootstrap_samples,
            )
            for step in range(4)
        ],
        "unoriented_central_by_python_step": [
            summarize_samples(
                unoriented_np[step],
                seed=bootstrap_seed + 40 + step,
                bootstrap_samples=bootstrap_samples,
            )
            for step in range(4)
        ],
        "paired_contrasts": {
            name: summarize_samples(
                values,
                seed=bootstrap_seed + 50 + index,
                bootstrap_samples=bootstrap_samples,
            )
            for index, (name, values) in enumerate(sample_metrics.items())
        },
        "recoded_output_response_correlation_by_python_step": v2_response_correlations,
        "hashes": {
            "feedforward": tensor_sha256(feedforward),
            "recoded_item_vectors": tensor_sha256(vectors.item_vectors),
            "recoded_output_vector": tensor_sha256(vectors.output_vector),
            "target_hidden": tensor_sha256(hidden),
            "target_pairs": tensor_sha256(pairs),
        },
        "leakage_firewall": {
            "optimizer_inputs": [
                "slow recurrent weight w",
                "plasticity coefficient alpha",
                "output decision vector",
                "isolated-item feedforward representations",
            ],
            "optimizer_forbidden_inputs": [
                "episode hidden activity",
                "episode pair identities",
                "responses",
                "correctness",
            ],
        },
    }
    return ReinstatementProbeResult(report=report, sample_metrics=sample_metrics)
