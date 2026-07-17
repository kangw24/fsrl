"""Model recovery for the proposed prefix/order/conflict human experiment.

The script compares six frozen path signatures:

* a commutative chain accumulator (decay=1);
* an unfitted leaky edge trace (decay=0.95);
* the frozen Phase 5j immediate recurrent checkpoint;
* the frozen Phase 5j delayed recurrent checkpoint.
* an unfitted bounded global-hypothesis posterior mixture;
* the same bounded process with an explicit MAP commitment at query.

Each simulated participant is assigned to one condition and one stopping block,
as required to prevent prefix probes from changing later learning.  Candidate
models receive the same three global nuisance parameters (intercept, positive
slope, and lapse), so recovery cannot be won merely by matching average
accuracy or confidence scale.  Subject-level logit offsets stress-test the
design against unmodelled heterogeneity.

This is design validation, not human evidence and not a neural-mechanism test.
The a-priori recovery gate is diagonal recovery >= .80 for every candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fsrl.cli.liu2026 import _source_tree_sha256


CONDITIONS = (
    "forward",
    "reverse",
    "interleaved",
    "conflict_early",
    "conflict_late",
)
RECOVERY_GATE = 0.80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--immediate-report", type=Path, required=True)
    parser.add_argument("--delayed-report", type=Path, required=True)
    parser.add_argument("--bounded-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--design",
        choices=("compact", "full_factorial"),
        default="compact",
        help=(
            "compact uses interleaved prefixes plus final order/conflict cells; "
            "full_factorial crosses every condition with every stopping block"
        ),
    )
    parser.add_argument("--simulations-per-model", type=int, default=50)
    parser.add_argument("--subjects-per-cell", type=int, default=20)
    parser.add_argument("--query-repeats", type=int, default=4)
    parser.add_argument("--subject-logit-sd", type=float, default=0.5)
    parser.add_argument("--target-mean-accuracy", type=float, default=0.75)
    parser.add_argument("--generating-lapse", type=float, default=0.04)
    parser.add_argument("--analytic-interleaved-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=26001)
    return parser.parse_args()


def selected_cells(repeats: int, design: str) -> list[tuple[str, int]]:
    if design == "full_factorial":
        return [
            (condition, block)
            for condition in CONDITIONS
            for block in range(1, repeats + 1)
        ]
    if design == "compact":
        # Prefix trajectory is measured only under the ordinary interleaved
        # schedule. Separate final-only groups carry the order and conflict
        # contrasts, avoiding a scientifically unnecessary full factorial.
        return [
            *(("interleaved", block) for block in range(1, repeats + 1)),
            ("forward", repeats),
            ("reverse", repeats),
            ("conflict_early", repeats),
            ("conflict_late", repeats),
        ]
    raise ValueError(f"unknown design: {design}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_neural_signature(path: Path) -> tuple[np.ndarray, list[list[int]], dict]:
    report = json.loads(path.read_text(encoding="utf-8"))
    section = report["intact"]
    pairs = section["query_pairs_rank_positions"]
    profiles = section["pair_probability_profiles"]
    values = []
    for condition in CONDITIONS:
        blocks = profiles[condition]
        if len(blocks) != report["design"]["repeats"]:
            raise ValueError(f"unexpected block count in {path}")
        values.extend(block["mean_correct_probability_by_pair"] for block in blocks)
    array = np.asarray(values, dtype=float)
    if array.shape[1] != len(pairs):
        raise ValueError(f"pair profile width mismatch in {path}")
    return np.clip(array, 1e-5, 1.0 - 1e-5), pairs, report


def load_bounded_signature(
    path: Path, readout: str
) -> tuple[np.ndarray, list[list[int]], dict]:
    report = json.loads(path.read_text(encoding="utf-8"))
    section = report["readouts"][readout]
    pairs = section["query_pairs_rank_positions"]
    profiles = section["pair_probability_profiles"]
    values = []
    for condition in CONDITIONS:
        values.extend(
            block["mean_correct_probability_by_pair"]
            for block in profiles[condition]
        )
    array = np.asarray(values, dtype=float)
    if array.shape[1] != len(pairs):
        raise ValueError(f"pair profile width mismatch in {path}:{readout}")
    return np.clip(array, 1e-5, 1.0 - 1e-5), pairs, report


def analytic_signature(
    *,
    n_items: int,
    repeats: int,
    decay: float,
    interleaved_samples: int,
    seed: int,
    position_edges: list[tuple[int, int]],
    target_edge_index: int,
) -> tuple[np.ndarray, list[list[int]]]:
    """Return cell x pair probabilities for an independent leaky edge trace."""
    n_edges = len(position_edges)
    pairs = [[i, j] for i in range(n_items) for j in range(i + 1, n_items)]
    forward = [
        (repeat, edge)
        for repeat in range(repeats)
        for edge in range(n_edges)
    ]
    reverse = [
        (repeat, edge)
        for repeat in range(repeats)
        for edge in reversed(range(n_edges))
    ]

    incidence = np.zeros((n_edges, n_items), dtype=float)
    for edge_index, (strong, weak) in enumerate(position_edges):
        incidence[edge_index, strong] = 1.0
        incidence[edge_index, weak] = -1.0
    augmented = np.concatenate([incidence, np.ones((1, n_items))], axis=0)

    def encode(schedule, conflict_event=None):
        traces = np.zeros(n_edges, dtype=float)
        blocks = []
        for trial_index, event in enumerate(schedule):
            traces *= decay
            strong, weak = position_edges[event[1]]
            sign = -1.0 if event == conflict_event else 1.0
            traces[event[1]] += sign * float(weak - strong) / float(n_items - 1)
            if (trial_index + 1) % n_edges == 0:
                scores = np.linalg.lstsq(
                    augmented,
                    np.concatenate([traces, np.asarray([0.0])]),
                    rcond=None,
                )[0]
                logits = np.asarray([scores[i] - scores[j] for i, j in pairs])
                blocks.append(expit(logits))
        return np.asarray(blocks)

    rng = np.random.default_rng(seed)
    interleaved = np.zeros((repeats, len(pairs)), dtype=float)
    for _ in range(interleaved_samples):
        schedule = []
        for repeat in range(repeats):
            schedule.extend((repeat, int(edge)) for edge in rng.permutation(n_edges))
        interleaved += encode(schedule)
    interleaved /= interleaved_samples

    profiles = {
        "forward": encode(forward),
        "reverse": encode(reverse),
        "interleaved": interleaved,
        "conflict_early": encode(forward, (0, target_edge_index)),
        "conflict_late": encode(forward, (repeats - 1, target_edge_index)),
    }
    return np.concatenate([profiles[name] for name in CONDITIONS], axis=0), pairs


def transformed_probability(signature: np.ndarray, params: np.ndarray) -> np.ndarray:
    intercept, log_slope, lapse_logit = params
    slope = np.exp(np.clip(log_slope, -5.0, 5.0))
    lapse = 0.20 * expit(lapse_logit)
    latent = intercept + slope * logit(np.clip(signature, 1e-5, 1.0 - 1e-5))
    return lapse * 0.5 + (1.0 - lapse) * expit(latent)


def calibrating_intercept(
    signature: np.ndarray, target: float, lapse: float
) -> float:
    base = logit(np.clip(signature, 1e-5, 1.0 - 1e-5))

    def objective(intercept):
        probability = lapse * 0.5 + (1.0 - lapse) * expit(intercept + base)
        return float(probability.mean() - target)

    return float(brentq(objective, -20.0, 20.0))


def simulate_counts(
    signature: np.ndarray,
    *,
    subjects_per_cell: int,
    query_repeats: int,
    subject_logit_sd: float,
    target_mean_accuracy: float,
    lapse: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a between-subject stopping design and aggregate its counts."""
    intercept = calibrating_intercept(signature, target_mean_accuracy, lapse)
    base = logit(np.clip(signature, 1e-5, 1.0 - 1e-5))
    successes = np.zeros_like(signature, dtype=int)
    totals = np.zeros_like(signature, dtype=int)
    for cell in range(signature.shape[0]):
        offsets = rng.normal(0.0, subject_logit_sd, size=subjects_per_cell)
        probability = lapse * 0.5 + (1.0 - lapse) * expit(
            intercept + base[cell][None, :] + offsets[:, None]
        )
        observations = rng.binomial(query_repeats, probability)
        successes[cell] = observations.sum(axis=0)
        totals[cell] = subjects_per_cell * query_repeats
    return successes, totals


def fit_candidate(
    signature: np.ndarray, successes: np.ndarray, totals: np.ndarray
) -> dict:
    def objective(params):
        probability = np.clip(transformed_probability(signature, params), 1e-8, 1 - 1e-8)
        return float(
            -np.sum(
                successes * np.log(probability)
                + (totals - successes) * np.log1p(-probability)
            )
        )

    bounds = ((-20.0, 20.0), (-5.0, 5.0), (-12.0, 12.0))
    best = minimize(
        objective,
        np.asarray([0.0, 0.0, -2.0]),
        method="L-BFGS-B",
        bounds=bounds,
    )
    # The likelihood is only three-dimensional and the bounded primary start
    # has been stable in formal runs. A second start is used only as a genuine
    # optimizer fallback, avoiding a threefold cost on every successful fit.
    if not best.success:
        fallback = minimize(
            objective,
            np.asarray([-1.0, 0.5, -4.0]),
            method="L-BFGS-B",
            bounds=bounds,
        )
        if fallback.fun < best.fun:
            best = fallback
    if not best.success:
        raise RuntimeError(f"candidate optimization failed: {best.message}")
    intercept, log_slope, lapse_logit = best.x
    return {
        "negative_log_likelihood": float(best.fun),
        "intercept": float(intercept),
        "slope": float(np.exp(np.clip(log_slope, -5.0, 5.0))),
        "lapse": float(0.20 * expit(lapse_logit)),
    }


def main() -> None:
    args = parse_args()
    if not 0.5 < args.target_mean_accuracy < 1.0:
        raise ValueError("target mean accuracy must be between .5 and 1")
    if not 0.0 <= args.generating_lapse < 0.2:
        raise ValueError("generating lapse must be in [0, .2)")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    immediate, pairs, immediate_report = load_neural_signature(args.immediate_report)
    delayed, delayed_pairs, delayed_report = load_neural_signature(args.delayed_report)
    if pairs != delayed_pairs:
        raise ValueError("neural reports use different query-pair coordinates")
    n_items = int(immediate_report["design"]["n_items"])
    repeats = int(immediate_report["design"]["repeats"])
    position_edges = [
        tuple(edge)
        for edge in immediate_report["design"].get(
            "support_position_edges",
            [[position, position + 1] for position in range(n_items - 1)],
        )
    ]
    target_edge_index = int(
        immediate_report["design"].get(
            "conflict_edge_index", (n_items - 1) // 2
        )
    )
    if delayed_report["design"]["n_items"] != n_items or delayed_report["design"]["repeats"] != repeats:
        raise ValueError("neural reports use different task designs")
    delayed_edges = delayed_report["design"].get(
        "support_position_edges",
        [[position, position + 1] for position in range(n_items - 1)],
    )
    if delayed_edges != [list(edge) for edge in position_edges]:
        raise ValueError("neural reports use different support graphs")
    bounded_mixture, bounded_pairs, bounded_report = load_bounded_signature(
        args.bounded_report, "posterior_mixture"
    )
    bounded_commitment, bounded_commitment_pairs, bounded_report_again = load_bounded_signature(
        args.bounded_report, "map_commitment"
    )
    if bounded_report_again["source_tree_sha256"] != bounded_report["source_tree_sha256"]:
        raise ValueError("bounded readouts came from inconsistent reports")
    if pairs != bounded_pairs or pairs != bounded_commitment_pairs:
        raise ValueError("bounded and neural reports use different pair coordinates")
    if bounded_report["design"]["n_items"] != n_items or bounded_report["design"]["repeats"] != repeats:
        raise ValueError("bounded and neural reports use different task designs")
    bounded_edges = bounded_report["design"].get(
        "support_position_edges",
        [[position, position + 1] for position in range(n_items - 1)],
    )
    if bounded_edges != [list(edge) for edge in position_edges]:
        raise ValueError("bounded and neural reports use different support graphs")

    commutative, analytic_pairs = analytic_signature(
        n_items=n_items,
        repeats=repeats,
        decay=1.0,
        interleaved_samples=args.analytic_interleaved_samples,
        seed=args.seed + 1,
        position_edges=position_edges,
        target_edge_index=target_edge_index,
    )
    leaky, leaky_pairs = analytic_signature(
        n_items=n_items,
        repeats=repeats,
        decay=0.95,
        interleaved_samples=args.analytic_interleaved_samples,
        seed=args.seed + 2,
        position_edges=position_edges,
        target_edge_index=target_edge_index,
    )
    if pairs != analytic_pairs or pairs != leaky_pairs:
        raise ValueError("analytic and neural candidates use different pair coordinates")

    candidates = {
        "commutative_chain": commutative,
        "leaky_chain_decay_0.95": leaky,
        "phase5j_immediate": immediate,
        "phase5j_delayed": delayed,
        "bounded_posterior_mixture": bounded_mixture,
        "bounded_map_commitment": bounded_commitment,
    }
    expected_shape = (len(CONDITIONS) * repeats, len(pairs))
    if any(signature.shape != expected_shape for signature in candidates.values()):
        raise ValueError("candidate signatures have inconsistent dimensions")
    cells = selected_cells(repeats, args.design)
    condition_index = {condition: index for index, condition in enumerate(CONDITIONS)}
    cell_indices = [
        condition_index[condition] * repeats + (block - 1)
        for condition, block in cells
    ]
    candidates = {
        label: signature[cell_indices]
        for label, signature in candidates.items()
    }

    rng = np.random.default_rng(args.seed)
    labels = list(candidates)
    confusion = np.zeros((len(labels), len(labels)), dtype=int)
    margins = {label: [] for label in labels}
    fit_failures = 0
    for generator_index, generator_label in enumerate(labels):
        for _ in range(args.simulations_per_model):
            successes, totals = simulate_counts(
                candidates[generator_label],
                subjects_per_cell=args.subjects_per_cell,
                query_repeats=args.query_repeats,
                subject_logit_sd=args.subject_logit_sd,
                target_mean_accuracy=args.target_mean_accuracy,
                lapse=args.generating_lapse,
                rng=rng,
            )
            fits = {}
            try:
                for candidate_label in labels:
                    fits[candidate_label] = fit_candidate(
                        candidates[candidate_label], successes, totals
                    )
            except RuntimeError:
                fit_failures += 1
                continue
            ordered = sorted(
                fits, key=lambda label: fits[label]["negative_log_likelihood"]
            )
            winner = ordered[0]
            confusion[generator_index, labels.index(winner)] += 1
            margins[generator_label].append(
                fits[ordered[1]]["negative_log_likelihood"]
                - fits[ordered[0]]["negative_log_likelihood"]
            )

    valid_per_row = confusion.sum(axis=1)
    proportions = np.divide(
        confusion,
        valid_per_row[:, None],
        out=np.zeros_like(confusion, dtype=float),
        where=valid_per_row[:, None] > 0,
    )
    diagonal = np.diag(proportions)
    gate_pass = bool(np.all(diagonal >= RECOVERY_GATE) and fit_failures == 0)
    report = {
        "status": "simulated design validation; not human evidence",
        "source_tree_sha256": _source_tree_sha256(PROJECT_ROOT),
        "immediate_report_sha256": sha256(args.immediate_report),
        "delayed_report_sha256": sha256(args.delayed_report),
        "bounded_report_sha256": sha256(args.bounded_report),
        "design": {
            "name": args.design,
            "between_subject_cells": [
                {"condition": condition, "stopping_block": block}
                for condition, block in cells
            ],
            "number_of_cells": len(cells),
            "query_pairs": pairs,
            "support_position_edges": [list(edge) for edge in position_edges],
            "conflict_edge_index": target_edge_index,
            "between_subject_prefix_design": True,
            "simulations_per_model": args.simulations_per_model,
            "subjects_per_condition_by_stopping_block": args.subjects_per_cell,
            "total_subjects": len(cells) * args.subjects_per_cell,
            "query_repeats_per_pair": args.query_repeats,
            "subject_logit_sd": args.subject_logit_sd,
            "generating_target_mean_accuracy": args.target_mean_accuracy,
            "generating_lapse": args.generating_lapse,
            "global_nuisance_parameters_fit_equally_to_all_candidates": [
                "intercept",
                "positive slope",
                "lapse in [0, .2]",
            ],
        },
        "candidate_labels": labels,
        "recovery_count_rows_generate_columns_fit": confusion.tolist(),
        "recovery_proportion_rows_generate_columns_fit": proportions.tolist(),
        "diagonal_recovery": {
            label: float(diagonal[index]) for index, label in enumerate(labels)
        },
        "median_winner_nll_margin": {
            label: float(np.median(values)) if values else float("nan")
            for label, values in margins.items()
        },
        "fit_failures": fit_failures,
        "a_priori_gate": {
            "minimum_diagonal_recovery_each_candidate": RECOVERY_GATE,
            "requires_zero_fit_failures": True,
            "passed": gate_pass,
        },
        "claim_limit": (
            "Passing shows only that the proposed behavioral cells can recover these "
            "six candidate mean signatures under the stated simulation. It does not "
            "establish that any candidate is cognitively correct, identify replay, or "
            "cover unimplemented global-hypothesis models."
        ),
    }
    report_path = args.output_dir / "model_recovery_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
