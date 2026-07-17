"""Online, inspectable implementation of Constructive Global Rank 3.2."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ModelRun:
    """Final total order and pairwise probability readout."""

    order_strong_to_weak: list[int]
    choice_prob: dict[tuple[int, int], float] = field(default_factory=dict)


@dataclass
class CGR32State:
    """Inspectable state retained within one learning episode."""

    anchor: np.ndarray
    encoding_seed: int
    precision_inverse: np.ndarray
    axis: np.ndarray
    exposure_counts: dict[tuple[int, int], int] = field(default_factory=dict)
    encoded_counts: dict[tuple[int, int], int] = field(default_factory=dict)
    relation_direction: dict[tuple[int, int], tuple[int, int]] = field(
        default_factory=dict
    )
    relation_magnitude: dict[tuple[int, int], float] = field(default_factory=dict)
    order_strong_to_weak: list[int] = field(default_factory=list)
    support_steps: int = 0
    order_trajectory: list[tuple[int, ...]] = field(default_factory=list)
    encoded_relation_trajectory: list[int] = field(default_factory=list)
    block_trajectory: list[int] = field(default_factory=list)


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0))))


class CGR32:
    """Construct a global order through online encoding and RLS updates.

    The model reads only model-visible support observations. Query targets,
    rewards, true ranks and generator metadata are outside its interface.
    """

    def __init__(
        self,
        n_items: int,
        *,
        sigma_a: float = 0.30,
        encoding_probability: float = 0.45,
        beta: float = 12.0,
        ridge: float = 1e-6,
    ) -> None:
        if n_items < 2:
            raise ValueError("n_items must be at least two")
        if sigma_a < 0.0:
            raise ValueError("sigma_a must be non-negative")
        if not 0.0 <= encoding_probability <= 1.0:
            raise ValueError("encoding_probability must lie in [0, 1]")
        if beta < 0.0:
            raise ValueError("beta must be non-negative")
        if ridge <= 0.0:
            raise ValueError("ridge must be positive")
        self.n_items = int(n_items)
        self.sigma_a = float(sigma_a)
        self.encoding_probability = float(encoding_probability)
        self.beta = float(beta)
        self.ridge = float(ridge)

    def initialize_state(self, rng: np.random.Generator) -> CGR32State:
        anchor = np.asarray(
            rng.normal(0.0, self.sigma_a, self.n_items), dtype=float
        )
        encoding_seed = int(rng.integers(0, 2**32, dtype=np.uint64))
        initial_order = self._preferred_linear_extension(anchor, [])
        return CGR32State(
            anchor=anchor,
            encoding_seed=encoding_seed,
            precision_inverse=np.eye(self.n_items, dtype=float) / self.ridge,
            axis=np.zeros(self.n_items, dtype=float),
            order_strong_to_weak=initial_order,
            order_trajectory=[tuple(initial_order)],
            encoded_relation_trajectory=[0],
            block_trajectory=[-1],
        )

    def _parse_support_trial(self, trial) -> tuple[int, int, float]:
        obs = trial.observation
        if obs.sign >= 0:
            high_cue, low_cue = int(obs.left_cue), int(obs.right_cue)
        else:
            high_cue, low_cue = int(obs.right_cue), int(obs.left_cue)
        if high_cue == low_cue:
            raise ValueError("support relation must contain two distinct cues")
        if not (0 <= high_cue < self.n_items and 0 <= low_cue < self.n_items):
            raise ValueError("support cue is outside the model item range")
        magnitude = getattr(obs, "magnitude", None)
        if magnitude is None:
            raise ValueError("CGR 3.2 requires model-visible magnitude")
        magnitude = float(magnitude)
        if not np.isfinite(magnitude) or magnitude <= 0.0:
            raise ValueError("magnitude must be finite and positive")
        return high_cue, low_cue, magnitude

    @staticmethod
    def _encoding_uniform(
        encoding_seed: int, key: tuple[int, int], exposure_number: int
    ) -> float:
        pair_seed = np.random.SeedSequence(
            [encoding_seed, key[0], key[1], exposure_number]
        )
        return float(np.random.default_rng(pair_seed).random())

    def _recalled_relations(self, state: CGR32State):
        return [
            (
                *state.relation_direction[key],
                state.relation_magnitude[key],
                state.encoded_counts[key],
            )
            for key in sorted(state.encoded_counts)
            if state.encoded_counts[key] > 0
        ]

    def support_step(self, state: CGR32State, trial) -> CGR32State:
        """Consume one learning presentation and update the prefix state."""

        high, low, magnitude = self._parse_support_trial(trial)
        key = tuple(sorted((high, low)))
        existing = state.relation_direction.get(key)
        if existing is not None and existing != (high, low):
            raise ValueError("contradictory support signs for one cue pair")
        state.relation_direction[key] = (high, low)
        state.relation_magnitude[key] = magnitude

        exposure = state.exposure_counts.get(key, 0) + 1
        state.exposure_counts[key] = exposure
        encoded = (
            self._encoding_uniform(state.encoding_seed, key, exposure)
            < self.encoding_probability
        )

        if encoded:
            state.encoded_counts[key] = state.encoded_counts.get(key, 0) + 1
            observation = np.zeros(self.n_items, dtype=float)
            observation[high] = 1.0
            observation[low] = -1.0

            px = state.precision_inverse @ observation
            gain = px / (1.0 + float(observation @ px))
            prediction_error = magnitude - float(observation @ state.axis)
            state.axis = state.axis + gain * prediction_error
            state.precision_inverse = state.precision_inverse - np.outer(
                gain, observation @ state.precision_inverse
            )
            state.precision_inverse = 0.5 * (
                state.precision_inverse + state.precision_inverse.T
            )

        state.order_strong_to_weak = self._preferred_linear_extension(
            state.axis + state.anchor,
            self._recalled_relations(state),
        )
        state.support_steps += 1
        state.order_trajectory.append(tuple(state.order_strong_to_weak))
        state.encoded_relation_trajectory.append(len(state.encoded_counts))
        state.block_trajectory.append(int(trial.block_index))
        return state

    def query_probability(
        self, state: CGR32State, left_cue: int, right_cue: int
    ) -> float:
        """Return P(left is higher) without changing state."""

        left_cue, right_cue = int(left_cue), int(right_cue)
        if left_cue == right_cue:
            raise ValueError("query must contain two distinct cues")
        if not (
            0 <= left_cue < self.n_items and 0 <= right_cue < self.n_items
        ):
            raise ValueError("query cue is outside the model item range")
        rank = {
            cue: index for index, cue in enumerate(state.order_strong_to_weak)
        }
        scale = float(self.n_items - 1)
        left_value = (self.n_items - 1 - rank[left_cue]) / scale
        right_value = (self.n_items - 1 - rank[right_cue]) / scale
        return _sigmoid(self.beta * (left_value - right_value))

    def query_step(self, state: CGR32State, trial) -> float:
        obs = trial.observation
        return self.query_probability(state, obs.left_cue, obs.right_cue)

    def run_subject_with_state(self, subject_task, rng: np.random.Generator):
        state = self.initialize_state(rng)
        for trial in subject_task.support_trials:
            self.support_step(state, trial)
        return self._model_run_from_state(state), state

    def run_subject(self, subject_task, rng: np.random.Generator) -> ModelRun:
        run, _ = self.run_subject_with_state(subject_task, rng)
        return run

    def _model_run_from_state(self, state: CGR32State) -> ModelRun:
        probabilities = {
            (first, second): self.query_probability(state, first, second)
            for first in range(self.n_items)
            for second in range(first + 1, self.n_items)
        }
        return ModelRun(list(state.order_strong_to_weak), probabilities)

    def _preferred_linear_extension(self, scores, recalled) -> list[int]:
        """Topological sort, using scores to choose among unconstrained cues."""

        outgoing = [set() for _ in range(self.n_items)]
        indegree = [0] * self.n_items
        for high, low, _magnitude, _count in recalled:
            if low not in outgoing[high]:
                outgoing[high].add(low)
                indegree[low] += 1

        available = {cue for cue, degree in enumerate(indegree) if degree == 0}
        order: list[int] = []
        while available:
            cue = max(available, key=lambda item: (float(scores[item]), -item))
            available.remove(cue)
            order.append(cue)
            for lower in sorted(outgoing[cue]):
                indegree[lower] -= 1
                if indegree[lower] == 0:
                    available.add(lower)
        if len(order) != self.n_items:
            raise ValueError("recalled relations contain a directed cycle")
        return order

    def batch_axis_for_audit(self, state: CGR32State) -> np.ndarray:
        """Solve the encoded endpoint directly to audit the online RLS state."""

        recalled = self._recalled_relations(state)
        if not recalled:
            return np.zeros(self.n_items, dtype=float)
        design = np.zeros((len(recalled), self.n_items), dtype=float)
        target = np.zeros(len(recalled), dtype=float)
        for row, (high, low, magnitude, count) in enumerate(recalled):
            weight = float(np.sqrt(count))
            design[row, high] = weight
            design[row, low] = -weight
            target[row] = weight * magnitude
        return np.linalg.solve(
            design.T @ design + self.ridge * np.eye(self.n_items),
            design.T @ target,
        )
