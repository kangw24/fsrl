"""Retrospective developmental audit of CGR-v3.1 against Liu choices.

This script deliberately retains the frozen legacy aggregate/binomial sampling
protocol so its historical report remains reproducible.  Shared metric and data
definitions live in :mod:`fsrl.analysis.cgr_evaluation`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fsrl.analysis.cgr_evaluation import (  # noqa: E402
    LAPSE,
    MODEL_SEED,
    N_ITEMS,
    TASK_SEED,
    compute_endpoint_metrics,
    load_liu_human_choices,
    pair_accuracy_correlation,
    report_provenance,
    simulate_legacy_pair_aggregate,
)
from fsrl.model.constructive_global_rank import (  # noqa: E402
    ConstructiveGlobalRankCompression,
    ConstructiveGlobalRankMemoryConstrained,
)
from fsrl.task.liu2026 import build_liu2026_symbolic_subject_tasks  # noqa: E402


def main():
    human = compute_endpoint_metrics(load_liu_human_choices(ROOT))
    tasks = build_liu2026_symbolic_subject_tasks(
        n_subjects=human["n"],
        rng=np.random.default_rng(TASK_SEED),
        nbcues=N_ITEMS,
        randomize_true_rank=True,
        include_magnitude=True,
    )
    candidates = {
        "v3": ConstructiveGlobalRankCompression(
            N_ITEMS, sigma_a=0.30, beta=12.0
        ),
        "v3_1": ConstructiveGlobalRankMemoryConstrained(
            N_ITEMS,
            sigma_a=0.30,
            encoding_probability=0.45,
            beta=12.0,
        ),
        "v3_1_anchor_lesion": ConstructiveGlobalRankMemoryConstrained(
            N_ITEMS,
            sigma_a=0.0,
            encoding_probability=0.45,
            beta=12.0,
        ),
        "v3_1_memory_lesion": ConstructiveGlobalRankMemoryConstrained(
            N_ITEMS,
            sigma_a=0.30,
            encoding_probability=0.0,
            beta=12.0,
        ),
        "v3_1_perfect_memory": ConstructiveGlobalRankMemoryConstrained(
            N_ITEMS,
            sigma_a=0.30,
            encoding_probability=1.0,
            beta=12.0,
        ),
        "v3_1_commitment_readout_lesion": ConstructiveGlobalRankMemoryConstrained(
            N_ITEMS,
            sigma_a=0.30,
            encoding_probability=0.45,
            beta=0.0,
        ),
    }

    model_metrics = {}
    for name, model in candidates.items():
        cohort = simulate_legacy_pair_aggregate(model, tasks, MODEL_SEED)
        model_metrics[name] = compute_endpoint_metrics(cohort)
        model_metrics[name]["pair_accuracy_pearson_r_vs_human"] = (
            pair_accuracy_correlation(human, model_metrics[name])
        )

    parameters = {
        "sigma_a": 0.30,
        "encoding_probability_per_presentation": 0.45,
        "implied_four_presentation_recall_probability": 1.0 - 0.55**4,
        "beta": 12.0,
    }
    report = {
        **report_provenance(
            status="retrospective_developmental_not_confirmatory",
            evaluator="legacy_pair_aggregate",
            task_seed=TASK_SEED,
            model_seed=MODEL_SEED,
            choice_seed=None,
            lapse=LAPSE,
            parameters=parameters,
        ),
        # Kept for compatibility with the pre-maintenance report schema.
        "model_seed_base": MODEL_SEED,
        "frozen_candidate": parameters,
        "human": human,
        "models": model_metrics,
    }
    output = ROOT / "outputs" / "cgr_v3_1_vs_liu_humans" / "development_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== CGR-v3.1 retrospective developmental audit ===")
    for name in ("v3", "v3_1"):
        values = model_metrics[name]
        print(
            f"{name}: overall={values['overall_accuracy']:.3f} "
            f"learned={values['learned_accuracy']:.3f} "
            f"unlearned={values['unlearned_accuracy']:.3f} "
            f"inter_tau={values['inter_subject_tau_mean']:.3f} "
            f"selfc={values['self_consistency_mean']:.3f} "
            f"bimodal={values['n_bimodal']} "
            f"pair_r={values['pair_accuracy_pearson_r_vs_human']:.3f}"
        )
    print(f"report: {output}")


if __name__ == "__main__":
    main()
