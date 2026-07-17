"""Symbolic eight-item task used to evaluate CGR 3.2."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


# Weak-to-strong positions on the abstract A..H axis.
SUPPORT_POSITION_PAIRS = (
    (0, 5),
    (1, 2),
    (1, 4),
    (2, 6),
    (3, 5),
    (3, 6),
    (4, 7),
    (0, 7),
)


@dataclass(frozen=True)
class SupportObservation:
    """Information visible to the model during one learning presentation."""

    left_cue: int
    right_cue: int
    sign: int
    magnitude: float


@dataclass(frozen=True)
class QueryObservation:
    """Information visible to the model during a no-feedback query."""

    left_cue: int
    right_cue: int


@dataclass(frozen=True)
class SupportTrial:
    observation: SupportObservation
    block_index: int
    position_pair: tuple[int, int]


@dataclass(frozen=True)
class QueryTrial:
    observation: QueryObservation
    block_index: int
    position_pair: tuple[int, int]
    correct_choice: int


@dataclass(frozen=True)
class LiuSubjectTask:
    subject_index: int
    true_rank: tuple[int, ...]
    support_trials: tuple[SupportTrial, ...]
    query_trials: tuple[QueryTrial, ...]


def _cue_at_position(true_rank: tuple[int, ...], weak_position: int) -> int:
    """Map a weak-to-strong position to a cue in strong-to-weak true_rank."""

    return int(true_rank[len(true_rank) - 1 - weak_position])


def build_subject_tasks(
    *,
    n_subjects: int,
    rng: np.random.Generator | None = None,
    support_blocks: int = 4,
    query_blocks: int = 10,
    randomize_true_rank: bool = True,
) -> tuple[LiuSubjectTask, ...]:
    """Build independently randomized eight-item task instances.

    Generator metadata is stored for evaluation, but the model interface reads
    only each trial's ``observation`` field.
    """

    if n_subjects <= 0:
        raise ValueError("n_subjects must be positive")
    if support_blocks <= 0 or query_blocks <= 0:
        raise ValueError("block counts must be positive")
    if rng is None:
        rng = np.random.default_rng()

    n_items = 8
    query_pairs = tuple(combinations(range(n_items), 2))
    tasks: list[LiuSubjectTask] = []

    for subject_index in range(n_subjects):
        if randomize_true_rank:
            true_rank = tuple(int(cue) for cue in rng.permutation(n_items))
        else:
            true_rank = tuple(range(n_items))

        support_trials: list[SupportTrial] = []
        for block_index in range(support_blocks):
            for pair_index in rng.permutation(len(SUPPORT_POSITION_PAIRS)):
                low_position, high_position = SUPPORT_POSITION_PAIRS[int(pair_index)]
                low_cue = _cue_at_position(true_rank, low_position)
                high_cue = _cue_at_position(true_rank, high_position)
                if float(rng.random()) < 0.5:
                    left_cue, right_cue, sign = low_cue, high_cue, -1
                else:
                    left_cue, right_cue, sign = high_cue, low_cue, 1
                support_trials.append(
                    SupportTrial(
                        observation=SupportObservation(
                            left_cue=left_cue,
                            right_cue=right_cue,
                            sign=sign,
                            magnitude=(high_position - low_position)
                            / float(n_items - 1),
                        ),
                        block_index=block_index,
                        position_pair=(low_position, high_position),
                    )
                )

        query_trials: list[QueryTrial] = []
        for block_index in range(query_blocks):
            for pair_index in rng.permutation(len(query_pairs)):
                low_position, high_position = query_pairs[int(pair_index)]
                low_cue = _cue_at_position(true_rank, low_position)
                high_cue = _cue_at_position(true_rank, high_position)
                if float(rng.random()) < 0.5:
                    left_cue, right_cue, correct_choice = low_cue, high_cue, 1
                else:
                    left_cue, right_cue, correct_choice = high_cue, low_cue, 0
                query_trials.append(
                    QueryTrial(
                        observation=QueryObservation(left_cue, right_cue),
                        block_index=block_index,
                        position_pair=(low_position, high_position),
                        correct_choice=correct_choice,
                    )
                )

        tasks.append(
            LiuSubjectTask(
                subject_index=subject_index,
                true_rank=true_rank,
                support_trials=tuple(support_trials),
                query_trials=tuple(query_trials),
            )
        )

    return tuple(tasks)
