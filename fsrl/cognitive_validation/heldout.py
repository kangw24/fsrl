"""Held-out raw-choice scoring for shared latent-state population priors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import permutations
from math import log
from pathlib import Path
from typing import Iterable

import numpy as np

from fsrl.data.liu2026_human import build_liu2026_human_episode


@dataclass(frozen=True)
class ChoicePanel:
    subject_ids: np.ndarray
    canonical_pairs: np.ndarray
    # One means choosing the objectively higher public film index.
    chose_higher: np.ndarray
    provenance: dict[str, object]


@dataclass(frozen=True)
class PredictiveEvaluation:
    candidate: str
    subject_ids: np.ndarray
    subject_log_predictive: np.ndarray
    subject_log_predictive_per_choice: np.ndarray
    subject_brier: np.ndarray
    subject_accuracy: np.ndarray
    subject_posterior_ess: np.ndarray | None
    choices_per_subject: int
    conditioning_blocks: tuple[int, ...]
    scored_blocks: tuple[int, ...]

    def summary(self) -> dict[str, object]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, np.ndarray):
                payload[key] = value.tolist()
        payload.update(
            {
                "total_log_predictive": float(self.subject_log_predictive.sum()),
                "mean_log_predictive_per_choice": float(
                    self.subject_log_predictive.sum()
                    / (self.subject_ids.size * self.choices_per_subject)
                ),
                "mean_bits_per_choice_over_chance": float(
                    (
                        self.subject_log_predictive.sum()
                        + self.subject_ids.size
                        * self.choices_per_subject
                        * log(2.0)
                    )
                    / (self.subject_ids.size * self.choices_per_subject * log(2.0))
                ),
                "mean_brier": float(self.subject_brier.mean()),
                "mean_accuracy": float(self.subject_accuracy.mean()),
                "median_posterior_ess": (
                    None
                    if self.subject_posterior_ess is None
                    else float(np.median(self.subject_posterior_ess))
                ),
                "minimum_posterior_ess": (
                    None
                    if self.subject_posterior_ess is None
                    else float(np.min(self.subject_posterior_ess))
                ),
            }
        )
        return payload


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + float(np.log(np.exp(values - maximum).sum()))


def build_choice_panel(paths: Iterable[str | Path]) -> ChoicePanel:
    """Load validated Liu choices into subject x block x pair binary form."""

    record = build_liu2026_human_episode(paths)
    pairs = np.asarray(record.query_set, dtype="int16")
    pair_to_index = {tuple(map(int, pair)): i for i, pair in enumerate(pairs)}
    subject_ids = np.asarray(
        sorted({response.batch_index for response in record.test_responses}),
        dtype="int32",
    )
    subject_to_index = {int(subject): i for i, subject in enumerate(subject_ids)}
    choices = np.full((subject_ids.size, 10, pairs.shape[0]), -1, dtype="int8")
    for response in record.test_responses:
        subject_index = subject_to_index[int(response.batch_index)]
        pair_index = pair_to_index[tuple(map(int, response.pair))]
        if choices[subject_index, response.block_id, pair_index] != -1:
            raise ValueError("duplicate subject/block/pair choice")
        choices[subject_index, response.block_id, pair_index] = int(response.correct)
    if np.any(choices < 0):
        raise ValueError("choice panel is incomplete")
    return ChoicePanel(
        subject_ids=subject_ids,
        canonical_pairs=pairs,
        chose_higher=choices,
        provenance=record.provenance,
    )


def _block_indices(blocks: tuple[int, ...]) -> np.ndarray:
    if not blocks or any(block not in range(1, 11) for block in blocks):
        raise ValueError("blocks must be a non-empty subset of 1..10")
    if len(set(blocks)) != len(blocks):
        raise ValueError("blocks must not repeat")
    return np.asarray([block - 1 for block in blocks], dtype="int64")


def _count_log_likelihood(
    probabilities: np.ndarray, successes: np.ndarray, trials: int
) -> np.ndarray:
    clipped = np.clip(probabilities.astype("float64"), 1e-6, 1.0 - 1e-6)
    return (
        successes[None, :] * np.log(clipped)
        + (trials - successes)[None, :] * np.log1p(-clipped)
    ).sum(axis=1)


def evaluate_particle_prior(
    panel: ChoicePanel,
    probabilities: np.ndarray,
    *,
    candidate: str,
    conditioning_blocks: tuple[int, ...],
    scored_blocks: tuple[int, ...],
) -> PredictiveEvaluation:
    """Condition a shared particle prior, then score untouched choices."""

    probabilities = np.asarray(probabilities, dtype="float64")
    if probabilities.ndim != 2 or probabilities.shape[1] != panel.canonical_pairs.shape[0]:
        raise ValueError("probabilities must be particle x canonical pair")
    if probabilities.shape[0] <= 0 or not np.isfinite(probabilities).all():
        raise ValueError("candidate probabilities must be finite and non-empty")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("candidate probabilities must lie in [0,1]")
    fit_index = _block_indices(conditioning_blocks)
    test_index = _block_indices(scored_blocks)
    if set(fit_index.tolist()) & set(test_index.tolist()):
        raise ValueError("conditioning and scored blocks must be disjoint")
    fit_trials = int(fit_index.size)
    test_trials = int(test_index.size)
    choices_per_subject = test_trials * panel.canonical_pairs.shape[0]
    log_prior = -log(probabilities.shape[0])

    log_predictive = np.empty(panel.subject_ids.size, dtype="float64")
    brier = np.empty(panel.subject_ids.size, dtype="float64")
    accuracy = np.empty(panel.subject_ids.size, dtype="float64")
    ess = np.empty(panel.subject_ids.size, dtype="float64")
    for subject_index in range(panel.subject_ids.size):
        fit_successes = panel.chose_higher[subject_index, fit_index, :].sum(axis=0)
        test_choices = panel.chose_higher[subject_index, test_index, :]
        test_successes = test_choices.sum(axis=0)
        fit_ll = _count_log_likelihood(probabilities, fit_successes, fit_trials)
        log_normalizer = _logsumexp(fit_ll + log_prior)
        log_posterior = fit_ll + log_prior - log_normalizer
        posterior = np.exp(log_posterior)
        ess[subject_index] = 1.0 / float(np.square(posterior).sum())
        test_ll = _count_log_likelihood(probabilities, test_successes, test_trials)
        log_predictive[subject_index] = _logsumexp(log_posterior + test_ll)
        posterior_mean = posterior @ probabilities
        expanded = np.broadcast_to(posterior_mean, test_choices.shape)
        brier[subject_index] = float(np.square(test_choices - expanded).mean())
        accuracy[subject_index] = float(
            ((expanded >= 0.5) == test_choices.astype(bool)).mean()
        )
    return PredictiveEvaluation(
        candidate=candidate,
        subject_ids=panel.subject_ids.copy(),
        subject_log_predictive=log_predictive,
        subject_log_predictive_per_choice=log_predictive / choices_per_subject,
        subject_brier=brier,
        subject_accuracy=accuracy,
        subject_posterior_ess=ess,
        choices_per_subject=choices_per_subject,
        conditioning_blocks=conditioning_blocks,
        scored_blocks=scored_blocks,
    )


def particle_prior_log_marginal(
    panel: ChoicePanel,
    probabilities: np.ndarray,
    *,
    blocks: tuple[int, ...],
) -> float:
    """Marginal log likelihood under a shared uniform particle prior."""

    probabilities = np.asarray(probabilities, dtype="float64")
    if probabilities.ndim != 2 or probabilities.shape[1] != panel.canonical_pairs.shape[0]:
        raise ValueError("probabilities must be particle x canonical pair")
    if probabilities.shape[0] <= 0 or not np.isfinite(probabilities).all():
        raise ValueError("candidate probabilities must be finite and non-empty")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("candidate probabilities must lie in [0,1]")
    block_index = _block_indices(blocks)
    trials = int(block_index.size)
    log_prior = -log(probabilities.shape[0])
    total = 0.0
    for subject_index in range(panel.subject_ids.size):
        successes = panel.chose_higher[subject_index, block_index, :].sum(axis=0)
        total += _logsumexp(
            _count_log_likelihood(probabilities, successes, trials) + log_prior
        )
    return float(total)


def temperature_scale_probabilities(
    probabilities: np.ndarray, temperature: float
) -> np.ndarray:
    """Apply one shared decision temperature without changing latent states."""

    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    values = np.asarray(probabilities, dtype="float64")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("probabilities must be finite and lie in [0,1]")
    clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped) - np.log1p(-clipped)
    scaled_logits = logits / temperature
    # Stable sigmoid for both signs.
    output = np.empty_like(scaled_logits)
    positive = scaled_logits >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-scaled_logits[positive]))
    exponential = np.exp(scaled_logits[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def evaluate_independent_binary_prior(
    panel: ChoicePanel,
    *,
    lapse: float,
    conditioning_blocks: tuple[int, ...],
    scored_blocks: tuple[int, ...],
) -> PredictiveEvaluation:
    """Exact non-transitive ceiling with one binary preference per pair."""

    if not 0 < lapse < 0.5:
        raise ValueError("lapse must lie in (0, 0.5)")
    fit_index = _block_indices(conditioning_blocks)
    test_index = _block_indices(scored_blocks)
    if set(fit_index.tolist()) & set(test_index.tolist()):
        raise ValueError("conditioning and scored blocks must be disjoint")
    fit_trials = int(fit_index.size)
    test_trials = int(test_index.size)
    choices_per_subject = test_trials * panel.canonical_pairs.shape[0]
    orientations = np.asarray([lapse, 1.0 - lapse], dtype="float64")
    log_predictive = np.empty(panel.subject_ids.size, dtype="float64")
    brier = np.empty(panel.subject_ids.size, dtype="float64")
    accuracy = np.empty(panel.subject_ids.size, dtype="float64")
    for subject_index in range(panel.subject_ids.size):
        fit_successes = panel.chose_higher[subject_index, fit_index, :].sum(axis=0)
        test_choices = panel.chose_higher[subject_index, test_index, :]
        test_successes = test_choices.sum(axis=0)
        subject_log_predictive = 0.0
        posterior_means = np.empty(panel.canonical_pairs.shape[0], dtype="float64")
        for pair_index in range(panel.canonical_pairs.shape[0]):
            fit_ll = (
                fit_successes[pair_index] * np.log(orientations)
                + (fit_trials - fit_successes[pair_index]) * np.log1p(-orientations)
            )
            log_posterior = fit_ll - log(2.0)
            log_posterior -= _logsumexp(log_posterior)
            posterior = np.exp(log_posterior)
            posterior_means[pair_index] = float(posterior @ orientations)
            test_ll = (
                test_successes[pair_index] * np.log(orientations)
                + (test_trials - test_successes[pair_index])
                * np.log1p(-orientations)
            )
            subject_log_predictive += _logsumexp(log_posterior + test_ll)
        log_predictive[subject_index] = subject_log_predictive
        expanded = np.broadcast_to(posterior_means, test_choices.shape)
        brier[subject_index] = float(np.square(test_choices - expanded).mean())
        accuracy[subject_index] = float(
            ((expanded >= 0.5) == test_choices.astype(bool)).mean()
        )
    return PredictiveEvaluation(
        candidate="independent_binary_preferences_lapse05",
        subject_ids=panel.subject_ids.copy(),
        subject_log_predictive=log_predictive,
        subject_log_predictive_per_choice=log_predictive / choices_per_subject,
        subject_brier=brier,
        subject_accuracy=accuracy,
        subject_posterior_ess=None,
        choices_per_subject=choices_per_subject,
        conditioning_blocks=conditioning_blocks,
        scored_blocks=scored_blocks,
    )


def make_uniform_permutation_prior(n_items: int, *, lapse: float) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate a uniform transitive latent-ranking prior."""

    if n_items < 2 or not 0 < lapse < 0.5:
        raise ValueError("invalid item count or lapse")
    pairs = np.asarray(
        [(left, right) for left in range(n_items) for right in range(left + 1, n_items)],
        dtype="int16",
    )
    orderings = np.asarray(list(permutations(range(n_items))), dtype="int16")
    positions = np.empty_like(orderings)
    positions[np.arange(orderings.shape[0])[:, None], orderings] = np.arange(n_items)
    higher_preferred = positions[:, pairs[:, 1]] < positions[:, pairs[:, 0]]
    probabilities = np.where(higher_preferred, 1.0 - lapse, lapse).astype("float32")
    return pairs, probabilities


def paired_bootstrap_interval(
    values: np.ndarray, *, seed: int, bootstrap_samples: int = 2000
) -> dict[str, object]:
    values = np.asarray(values, dtype="float64")
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be a finite non-empty vector")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(bootstrap_samples, values.size))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "bootstrap95": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
        "count": int(values.size),
    }
