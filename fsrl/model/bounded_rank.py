"""Bounded sequential inference over explicit global-rank hypotheses.

This is a cognitive-level competitor, not a neural implementation.  Every
episode starts from the same uniform prior and freshly sampled particles.  No
participant identifier, stored ranking, true rank, or test response enters the
state.  Observations update a limited particle population online; ESS-triggered
resampling makes the process path dependent, and resample-move rejuvenation
constructs new permutations rather than merely selecting a prestored answer.

An optional phase-boundary commitment freezes one posterior hypothesis for
read-only repeated queries.  That operation is intentionally explicit because
it is a strong, falsifiable explanation of coherent individual rankings: it
predicts an abrupt stabilization of pair choices and near-zero circular triads.
It must compete with the uncommitted posterior-mixture variant in human data.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class BoundedRankConfig:
    n_items: int
    n_particles: int = 64
    observation_scale: float = 0.20
    resample_ess_fraction: float = 0.75
    rejuvenation_steps: int = 2
    decision_lapse: float = 0.02
    observation_mode: str = "signed_distance"
    sign_error_probability: float = 0.01

    def __post_init__(self) -> None:
        if self.n_items < 3:
            raise ValueError("n_items must be at least 3")
        if self.n_particles < 2:
            raise ValueError("n_particles must be at least 2")
        if self.observation_scale <= 0.0:
            raise ValueError("observation_scale must be positive")
        if not 0.0 <= self.resample_ess_fraction <= 1.0:
            raise ValueError("resample_ess_fraction must be in [0, 1]")
        if self.rejuvenation_steps < 0:
            raise ValueError("rejuvenation_steps cannot be negative")
        if not 0.0 <= self.decision_lapse < 1.0:
            raise ValueError("decision_lapse must be in [0, 1)")
        if self.observation_mode not in {"signed_distance", "sign_only"}:
            raise ValueError(
                "observation_mode must be 'signed_distance' or 'sign_only'"
            )
        if not 0.0 < self.sign_error_probability < 0.5:
            raise ValueError("sign_error_probability must be inside (0, 0.5)")


@dataclass(frozen=True)
class RankObservation:
    left_item: torch.Tensor
    right_item: torch.Tensor
    signed_distance: torch.Tensor


@dataclass
class BoundedRankState:
    """Episode-local state; none of its fields are optimized parameters."""

    particles: torch.Tensor
    log_weights: torch.Tensor
    history: tuple[RankObservation, ...]
    committed_permutation: torch.Tensor | None = None
    resample_count: torch.Tensor | None = None


class BoundedRankParticleFilter:
    """Sequential Monte Carlo over a finite population of rank permutations."""

    def __init__(self, config: BoundedRankConfig):
        self.config = config

    def initial_state(
        self,
        batch_size: int,
        *,
        generator: torch.Generator,
        device: torch.device | str = "cpu",
    ) -> BoundedRankState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        device = torch.device(device)
        # Sampling random keys gives independent uniform permutations without
        # exposing a privileged center ranking.
        keys = torch.rand(
            batch_size,
            self.config.n_particles,
            self.config.n_items,
            generator=generator,
            device=device,
        )
        particles = torch.argsort(keys, dim=-1)
        log_weights = torch.full(
            (batch_size, self.config.n_particles),
            -math.log(self.config.n_particles),
            device=device,
        )
        return BoundedRankState(
            particles=particles,
            log_weights=log_weights,
            history=(),
            committed_permutation=None,
            resample_count=torch.zeros(batch_size, dtype=torch.long, device=device),
        )

    @staticmethod
    def _validate_items(
        particles: torch.Tensor, left_item: torch.Tensor, right_item: torch.Tensor
    ) -> None:
        batch_size, _, n_items = particles.shape
        if left_item.shape != (batch_size,) or right_item.shape != (batch_size,):
            raise ValueError("item indices must have shape [batch]")
        if bool((left_item == right_item).any()):
            raise ValueError("a relation must contain two different items")
        if bool((left_item < 0).any() or (left_item >= n_items).any()):
            raise ValueError("left item index is out of range")
        if bool((right_item < 0).any() or (right_item >= n_items).any()):
            raise ValueError("right item index is out of range")

    def _observation_log_likelihood(
        self, particles: torch.Tensor, observation: RankObservation
    ) -> torch.Tensor:
        positions = torch.argsort(particles, dim=-1)
        batch_size, n_particles, _ = particles.shape
        left = observation.left_item[:, None, None].expand(batch_size, n_particles, 1)
        right = observation.right_item[:, None, None].expand(batch_size, n_particles, 1)
        left_position = positions.gather(2, left).squeeze(-1)
        right_position = positions.gather(2, right).squeeze(-1)
        predicted = (right_position - left_position).float() / float(
            self.config.n_items - 1
        )
        if self.config.observation_mode == "sign_only":
            observed_sign = torch.sign(
                observation.signed_distance[:, None].to(predicted.dtype)
            )
            agrees = predicted * observed_sign > 0
            correct_log_probability = math.log(
                1.0 - self.config.sign_error_probability
            )
            error_log_probability = math.log(
                self.config.sign_error_probability
            )
            return torch.where(
                agrees,
                torch.full_like(predicted, correct_log_probability),
                torch.full_like(predicted, error_log_probability),
            )
        residual = (
            predicted - observation.signed_distance[:, None].to(predicted.dtype)
        )
        return -0.5 * (residual / self.config.observation_scale).square()

    def _history_log_likelihood(
        self, particles: torch.Tensor, history: tuple[RankObservation, ...]
    ) -> torch.Tensor:
        total = torch.zeros(
            particles.shape[:2], device=particles.device, dtype=torch.float32
        )
        for observation in history:
            total = total + self._observation_log_likelihood(particles, observation)
        return total

    def _rejuvenate(
        self,
        particles: torch.Tensor,
        history: tuple[RankObservation, ...],
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Symmetric adjacent-swap Metropolis moves targeting all observations."""
        if self.config.rejuvenation_steps == 0 or not history:
            return particles
        batch_size, n_particles, n_items = particles.shape
        current = particles
        current_log_likelihood = self._history_log_likelihood(current, history)
        for _ in range(self.config.rejuvenation_steps):
            swap_position = torch.randint(
                0,
                n_items - 1,
                (batch_size, n_particles),
                generator=generator,
                device=particles.device,
            )
            proposed = current.clone()
            left_value = proposed.gather(2, swap_position[:, :, None]).clone()
            right_value = proposed.gather(2, (swap_position + 1)[:, :, None]).clone()
            proposed.scatter_(2, swap_position[:, :, None], right_value)
            proposed.scatter_(2, (swap_position + 1)[:, :, None], left_value)
            proposed_log_likelihood = self._history_log_likelihood(proposed, history)
            log_acceptance = torch.minimum(
                torch.zeros_like(current_log_likelihood),
                proposed_log_likelihood - current_log_likelihood,
            )
            log_uniform = torch.log(
                torch.rand(
                    batch_size,
                    n_particles,
                    generator=generator,
                    device=particles.device,
                ).clamp_min(1e-12)
            )
            accept = log_uniform < log_acceptance
            current = torch.where(accept[:, :, None], proposed, current)
            current_log_likelihood = torch.where(
                accept, proposed_log_likelihood, current_log_likelihood
            )
        return current

    def observe(
        self,
        state: BoundedRankState,
        left_item: torch.Tensor,
        right_item: torch.Tensor,
        signed_distance: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> BoundedRankState:
        """Assimilate one batch of physically observed signed relationships."""
        self._validate_items(state.particles, left_item, right_item)
        batch_size = state.particles.shape[0]
        if signed_distance.shape != (batch_size,):
            raise ValueError("signed_distance must have shape [batch]")
        if bool((signed_distance == 0).any()):
            raise ValueError("observed relationships must have nonzero distance")
        observation = RankObservation(
            left_item=left_item.clone(),
            right_item=right_item.clone(),
            signed_distance=signed_distance.clone(),
        )
        history = state.history + (observation,)
        log_weights = state.log_weights + self._observation_log_likelihood(
            state.particles, observation
        )
        log_weights = log_weights - torch.logsumexp(log_weights, dim=1, keepdim=True)
        weights = torch.exp(log_weights)
        ess = 1.0 / weights.square().sum(dim=1)
        resample = ess < (
            self.config.resample_ess_fraction * self.config.n_particles
        )
        particles = state.particles.clone()
        resample_count = (
            torch.zeros(batch_size, dtype=torch.long, device=particles.device)
            if state.resample_count is None
            else state.resample_count.clone()
        )
        for batch_index in torch.nonzero(resample, as_tuple=False).flatten().tolist():
            indices = torch.multinomial(
                weights[batch_index],
                self.config.n_particles,
                replacement=True,
                generator=generator,
            )
            particles[batch_index] = particles[batch_index, indices]
            log_weights[batch_index].fill_(-math.log(self.config.n_particles))
            resample_count[batch_index] += 1
        # Rejuvenation only follows resampling; untouched batches preserve
        # exact sequential-importance order invariance.
        if bool(resample.any()) and self.config.rejuvenation_steps:
            rejuvenated = self._rejuvenate(
                particles[resample],
                tuple(
                    RankObservation(
                        left_item=obs.left_item[resample],
                        right_item=obs.right_item[resample],
                        signed_distance=obs.signed_distance[resample],
                    )
                    for obs in history
                ),
                generator=generator,
            )
            particles[resample] = rejuvenated
        return BoundedRankState(
            particles=particles,
            log_weights=log_weights,
            history=history,
            committed_permutation=None,
            resample_count=resample_count,
        )

    def commit(
        self,
        state: BoundedRankState,
        *,
        generator: torch.Generator,
        strategy: str = "sample",
    ) -> BoundedRankState:
        """Freeze one global hypothesis at a support-to-query phase boundary."""
        weights = torch.softmax(state.log_weights, dim=1)
        if strategy == "sample":
            indices = torch.multinomial(
                weights, 1, replacement=True, generator=generator
            ).squeeze(1)
        elif strategy == "map":
            indices = weights.argmax(dim=1)
        else:
            raise ValueError("commit strategy must be 'sample' or 'map'")
        batch = torch.arange(state.particles.shape[0], device=state.particles.device)
        committed = state.particles[batch, indices].clone()
        return BoundedRankState(
            particles=state.particles,
            log_weights=state.log_weights,
            history=state.history,
            committed_permutation=committed,
            resample_count=state.resample_count,
        )

    def query_probability(
        self,
        state: BoundedRankState,
        left_item: torch.Tensor,
        right_item: torch.Tensor,
    ) -> torch.Tensor:
        """Return P(left is stronger) without mutating episode state."""
        self._validate_items(state.particles, left_item, right_item)
        batch_size, n_particles, _ = state.particles.shape
        if state.committed_permutation is not None:
            positions = torch.argsort(state.committed_permutation, dim=-1)
            batch = torch.arange(batch_size, device=positions.device)
            probability = (
                positions[batch, left_item] < positions[batch, right_item]
            ).float()
        else:
            positions = torch.argsort(state.particles, dim=-1)
            left = left_item[:, None, None].expand(batch_size, n_particles, 1)
            right = right_item[:, None, None].expand(batch_size, n_particles, 1)
            left_first = (
                positions.gather(2, left).squeeze(-1)
                < positions.gather(2, right).squeeze(-1)
            ).float()
            probability = (
                torch.softmax(state.log_weights, dim=1) * left_first
            ).sum(dim=1)
        lapse = self.config.decision_lapse
        return lapse * 0.5 + (1.0 - lapse) * probability
