"""Shared, auditable evaluation utilities for CGR endpoint candidates.

This module owns the Liu endpoint metric definitions and query-sampling
protocols used by the active CGR-v3.1/v3.2 audits.  Models never import this
module: it belongs to the evaluator side of the information firewall.

Two simulators are intentionally retained:

``simulate_legacy_pair_aggregate``
    Reproduces the frozen v3.1 report protocol (one model/choice RNG and one
    binomial draw per pair).  It exists for historical report reproducibility.

``simulate_query_chronology``
    Executes all query trials in experimental order with isolated model and
    choice RNGs.  This is the current protocol for v3.2 and future audits.
"""

from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import kendalltau, pearsonr


N_ITEMS = 8
ALL_PAIRS = tuple(combinations(range(N_ITEMS), 2))
SUPPORT_PAIRS = frozenset(
    {
        (0, 5),
        (1, 2),
        (1, 4),
        (2, 6),
        (3, 5),
        (3, 6),
        (4, 7),
        (0, 7),
    }
)
TASK_SEED = 971001
MODEL_SEED = TASK_SEED + 999
CHOICE_SEED_OFFSET = 1_000_003
LAPSE = 0.08
HUMAN_FILENAMES = (
    "preregistered_experiment_data.csv",
    "replication_experiment_data.csv",
)


def load_liu_human_choices(
    project_root: Path,
    filenames: Sequence[str] = HUMAN_FILENAMES,
):
    """Load per-subject, per-pair correctness fractions from frozen raw CSVs."""

    subjects = {}
    for filename in filenames:
        path = project_root / "outputs" / "liu2026_human_audit" / "raw" / filename
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                subject = int(row["id"])
                first = int(row["film_index_1"]) - 1
                second = int(row["film_index_2"]) - 1
                chosen = int(row["film_choose_index"]) - 1
                pair = tuple(sorted((first, second)))
                correct = int(chosen == max(first, second))
                subjects.setdefault(subject, {}).setdefault(pair, []).append(correct)
    return {
        subject: {pair: float(np.mean(values)) for pair, values in pairs.items()}
        for subject, pairs in subjects.items()
    }


def fit_beta_label(values) -> str:
    """Apply the frozen paper-style 0.01/0.99 boundary clipping rule."""

    clipped = np.clip(np.asarray(values, dtype=float), 0.01, 0.99)
    try:
        alpha, beta, _, _ = beta_dist.fit(clipped, floc=0, fscale=1.0)
    except Exception:
        return "fit_failed"
    if alpha < 1.0 and beta < 1.0:
        return "bimodal"
    if alpha > 1.0 and beta > 1.0:
        return "unimodal"
    if alpha > 1.0 and beta < 1.0:
        return "high_accuracy"
    return "other"


def compute_endpoint_metrics(
    subject_accuracy,
    *,
    n_items: int = N_ITEMS,
    support_pairs=SUPPORT_PAIRS,
):
    """Compute the frozen endpoint summary from subject-by-pair accuracies."""

    all_pairs = tuple(combinations(range(n_items), 2))
    subject_ids = list(subject_accuracy)
    if not subject_ids:
        raise ValueError("subject_accuracy must contain at least one subject")
    pair_accuracy = {
        pair: float(np.mean([subject_accuracy[s][pair] for s in subject_ids]))
        for pair in all_pairs
    }
    bimodal = {
        pair
        for pair in all_pairs
        if fit_beta_label([subject_accuracy[s][pair] for s in subject_ids])
        == "bimodal"
    }

    max_triads = (n_items**3 - 4 * n_items) // 24
    rankings = []
    self_consistency = []
    circular_triads = 0
    self_consistent_wrong = 0
    for subject in subject_ids:
        preference = np.full((n_items, n_items), 0.5, dtype=float)
        for low, high in all_pairs:
            accuracy = subject_accuracy[subject][(low, high)]
            preference[high, low] = accuracy
            preference[low, high] = 1.0 - accuracy
        subject_triads = 0
        for first, second, third in combinations(range(n_items), 3):
            clockwise = (
                preference[first, second] > 0.5
                and preference[second, third] > 0.5
                and preference[third, first] > 0.5
            )
            counterclockwise = (
                preference[second, first] > 0.5
                and preference[third, second] > 0.5
                and preference[first, third] > 0.5
            )
            subject_triads += int(clockwise or counterclockwise)
        circular_triads += subject_triads
        self_consistency.append(1.0 - subject_triads / max_triads)
        score = np.array(
            [
                sum(preference[item, other] - 0.5 for other in range(n_items))
                for item in range(n_items)
            ]
        )
        ranking = tuple(np.argsort(-score, kind="mergesort"))
        rankings.append(ranking)
        if subject_triads == 0 and ranking != tuple(range(n_items - 1, -1, -1)):
            self_consistent_wrong += 1

    inter_subject = []
    for first in range(len(rankings)):
        for second in range(first + 1, len(rankings)):
            tau, _ = kendalltau(rankings[first], rankings[second])
            if not np.isnan(tau):
                inter_subject.append(float(tau))

    distance = {}
    for pair, accuracy in pair_accuracy.items():
        distance.setdefault(pair[1] - pair[0], []).append(accuracy)
    learned = [pair_accuracy[pair] for pair in support_pairs]
    unlearned = [pair_accuracy[pair] for pair in all_pairs if pair not in support_pairs]
    return {
        "n": len(subject_ids),
        "pair_accuracy": {str(pair): value for pair, value in pair_accuracy.items()},
        "bimodal_pairs": [list(pair) for pair in sorted(bimodal)],
        "n_bimodal": len(bimodal),
        "self_consistency_mean": float(np.mean(self_consistency)),
        "self_consistent_wrong_fraction": self_consistent_wrong / len(subject_ids),
        "inter_subject_tau_mean": (
            float(np.mean(inter_subject)) if inter_subject else float("nan")
        ),
        "overall_accuracy": float(np.mean(list(pair_accuracy.values()))),
        "learned_accuracy": float(np.mean(learned)),
        "unlearned_accuracy": float(np.mean(unlearned)),
        "circular_triads": circular_triads,
        "distance_accuracy": {
            str(gap): float(np.mean(values)) for gap, values in sorted(distance.items())
        },
    }


def pair_accuracy_correlation(human, model, *, all_pairs=ALL_PAIRS) -> float:
    human_values = np.array([human["pair_accuracy"][str(pair)] for pair in all_pairs])
    model_values = np.array([model["pair_accuracy"][str(pair)] for pair in all_pairs])
    correlation, _ = pearsonr(human_values, model_values)
    return float(correlation)


def probability_left_from_run(run, left_cue: int, right_cue: int) -> float:
    first, second = sorted((int(left_cue), int(right_cue)))
    probability_first = float(run.choice_prob[(first, second)])
    return probability_first if left_cue == first else 1.0 - probability_first


def sample_task_query_accuracy(run, task, choice_rng, *, lapse: float = LAPSE):
    """Execute a task's query chronology without changing model state."""

    per_pair_trials = {pair: [] for pair in ALL_PAIRS}
    for query_trial in task.query_trials:
        obs = query_trial.observation
        probability_left = probability_left_from_run(
            run, obs.left_cue, obs.right_cue
        )
        probability_left = lapse * 0.5 + (1.0 - lapse) * probability_left
        chosen_index = 0 if float(choice_rng.random()) < probability_left else 1
        per_pair_trials[query_trial.position_pair].append(
            int(chosen_index == query_trial.correct_choice)
        )
    missing = [pair for pair, values in per_pair_trials.items() if not values]
    if missing:
        raise ValueError(f"query schedule is missing pairs: {missing}")
    return {pair: float(np.mean(values)) for pair, values in per_pair_trials.items()}


def simulate_query_chronology(
    model,
    tasks,
    *,
    model_seed: int = MODEL_SEED,
    choice_seed: int | None = None,
    lapse: float = LAPSE,
):
    """Simulate endpoint choices with isolated model and choice RNG streams."""

    if choice_seed is None:
        choice_seed = model_seed + CHOICE_SEED_OFFSET
    model_rng = np.random.default_rng(model_seed)
    choice_rng = np.random.default_rng(choice_seed)
    return {
        task.subject_index: sample_task_query_accuracy(
            model.run_subject(task, model_rng), task, choice_rng, lapse=lapse
        )
        for task in tasks
    }


def simulate_legacy_pair_aggregate(
    model,
    tasks,
    seed: int,
    *,
    lapse: float = LAPSE,
):
    """Reproduce the frozen v3.1 aggregate/binomial protocol exactly."""

    rng = np.random.default_rng(seed)
    result = {}
    for task in tasks:
        run = model.run_subject(task, rng)
        subject = {}
        for low_position, high_position in ALL_PAIRS:
            low_cue = task.true_rank[N_ITEMS - 1 - low_position]
            high_cue = task.true_rank[N_ITEMS - 1 - high_position]
            probability_high = probability_left_from_run(run, high_cue, low_cue)
            probability_correct = lapse * 0.5 + (1.0 - lapse) * probability_high
            subject[(low_position, high_position)] = float(
                rng.binomial(10, probability_correct)
            ) / 10.0
        result[task.subject_index] = subject
    return result


def report_provenance(
    *,
    status: str,
    evaluator: str,
    task_seed: int,
    model_seed: int,
    choice_seed: int | None,
    lapse: float,
    parameters: Mapping[str, object],
):
    """Return the common machine-readable provenance block for new reports."""

    return {
        "status": status,
        "evaluator": evaluator,
        "task_seed": int(task_seed),
        "model_seed": int(model_seed),
        "choice_seed": None if choice_seed is None else int(choice_seed),
        "lapse": float(lapse),
        "parameters": dict(parameters),
    }
