"""Retrospective endpoint and process audit for online CGR-v3.2.

The public Liu behavior contains only post-learning choices, so prefix metrics
below are prospective model predictions, not human-validated process fits.
The script compares v3.2 endpoints with batch v3.1 and records the online order
trajectory after each of the four support blocks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fsrl.analysis.cgr_evaluation import (
    ALL_PAIRS,
    CHOICE_SEED_OFFSET,
    LAPSE,
    MODEL_SEED,
    N_ITEMS,
    TASK_SEED,
    compute_endpoint_metrics,
    load_liu_human_choices,
    pair_accuracy_correlation,
    report_provenance,
    simulate_query_chronology,
)
from fsrl.model.constructive_global_rank import (
    ConstructiveGlobalRankMemoryConstrained,
    ConstructiveGlobalRankOnlineMemory,
)
from fsrl.task.liu2026 import build_liu2026_symbolic_subject_tasks

def position_vector(order):
    positions = np.empty(len(order), dtype=int)
    for rank, cue in enumerate(order):
        positions[cue] = rank
    return positions


def deterministic_pair_accuracy(order, task):
    rank = {cue: index for index, cue in enumerate(order)}
    correct = []
    for low_position, high_position in ALL_PAIRS:
        low_cue = task.true_rank[N_ITEMS - 1 - low_position]
        high_cue = task.true_rank[N_ITEMS - 1 - high_position]
        correct.append(float(rank[high_cue] < rank[low_cue]))
    return float(np.mean(correct))


def simulate_online(model, tasks, seed):
    model_rng = np.random.default_rng(seed)
    choice_rng = np.random.default_rng(seed + CHOICE_SEED_OFFSET)
    subject_accuracy = {}
    block_accuracy = {str(block): [] for block in range(4)}
    block_final_tau = {str(block): [] for block in range(4)}
    block_encoded_relations = {str(block): [] for block in range(4)}
    revision_counts = []
    final_stability_steps = []
    endpoint_axis_max_abs_differences = []
    endpoint_order_mismatches = 0

    for task in tasks:
        _run, state = model.run_subject_with_state(task, model_rng)
        summaries = {
            key: (high, low, magnitude)
            for key, high, low, magnitude, _repetitions in model._group_support(task)
        }
        recalled = []
        for key, count in sorted(state.encoded_counts.items()):
            high, low, magnitude = summaries[key]
            recalled.append((high, low, magnitude, count))
        batch_axis = model._fit_recalled_axis(recalled)
        batch_order = model._preferred_linear_extension(
            batch_axis + state.anchor, recalled
        )
        endpoint_axis_max_abs_differences.append(
            float(np.max(np.abs(batch_axis - state.axis)))
        )
        endpoint_order_mismatches += int(
            batch_order != state.order_strong_to_weak
        )
        final_positions = position_vector(state.order_strong_to_weak)
        trajectory = state.order_trajectory
        revision_counts.append(
            sum(trajectory[index] != trajectory[index - 1] for index in range(1, len(trajectory)))
        )
        last_change = 0
        for index in range(1, len(trajectory)):
            if trajectory[index] != trajectory[index - 1]:
                last_change = index
        final_stability_steps.append(last_change)

        for block in range(4):
            step = (block + 1) * 8
            prefix_order = trajectory[step]
            block_accuracy[str(block)].append(
                deterministic_pair_accuracy(prefix_order, task)
            )
            tau, _ = kendalltau(position_vector(prefix_order), final_positions)
            block_final_tau[str(block)].append(float(tau))
            block_encoded_relations[str(block)].append(
                state.encoded_relation_trajectory[step]
            )

        per_pair_trials = {pair: [] for pair in ALL_PAIRS}
        for query_trial in task.query_trials:
            probability_left = model.query_step(state, query_trial)
            probability_left = LAPSE * 0.5 + (1.0 - LAPSE) * probability_left
            chosen_index = 0 if float(choice_rng.random()) < probability_left else 1
            per_pair_trials[query_trial.position_pair].append(
                int(chosen_index == query_trial.correct_choice)
            )
        subject_accuracy[task.subject_index] = {
            pair: float(np.mean(values)) for pair, values in per_pair_trials.items()
        }

    process = {
        "mean_order_revisions_across_32_support_trials": float(
            np.mean(revision_counts)
        ),
        "mean_last_order_change_support_step": float(
            np.mean(final_stability_steps)
        ),
        "fraction_stable_by_block": {
            str(block): float(
                np.mean(np.asarray(final_stability_steps) <= (block + 1) * 8)
            )
            for block in range(4)
        },
        "prefix_deterministic_accuracy": {
            block: float(np.mean(values)) for block, values in block_accuracy.items()
        },
        "prefix_tau_to_final_order": {
            block: float(np.mean(values)) for block, values in block_final_tau.items()
        },
        "mean_encoded_unique_relations": {
            block: float(np.mean(values))
            for block, values in block_encoded_relations.items()
        },
        "endpoint_online_batch_equivalence": {
            "max_axis_abs_difference": float(
                np.max(endpoint_axis_max_abs_differences)
            ),
            "order_mismatches": int(endpoint_order_mismatches),
            "subjects": len(tasks),
            "interpretation": (
                "online RLS and batch ridge are the same endpoint conditional "
                "on identical encoded observations and anchor"
            ),
        },
    }
    return subject_accuracy, process


def main():
    human = compute_endpoint_metrics(load_liu_human_choices(ROOT))
    tasks = build_liu2026_symbolic_subject_tasks(
        n_subjects=human["n"],
        rng=np.random.default_rng(TASK_SEED),
        nbcues=N_ITEMS,
        randomize_true_rank=True,
        include_magnitude=True,
    )

    v31 = ConstructiveGlobalRankMemoryConstrained(
        N_ITEMS, sigma_a=0.30, encoding_probability=0.45, beta=12.0
    )
    v32 = ConstructiveGlobalRankOnlineMemory(
        N_ITEMS, sigma_a=0.30, encoding_probability=0.45, beta=12.0
    )
    v31_metrics = compute_endpoint_metrics(
        simulate_query_chronology(v31, tasks, model_seed=MODEL_SEED)
    )
    v32_subjects, process = simulate_online(v32, tasks, MODEL_SEED)
    v32_metrics = compute_endpoint_metrics(v32_subjects)
    v31_metrics["pair_accuracy_pearson_r_vs_human"] = pair_accuracy_correlation(
        human, v31_metrics
    )
    v32_metrics["pair_accuracy_pearson_r_vs_human"] = pair_accuracy_correlation(
        human, v32_metrics
    )

    lesions = {}
    lesion_models = {
        "anchor_lesion": ConstructiveGlobalRankOnlineMemory(
            N_ITEMS, sigma_a=0.0, encoding_probability=0.45, beta=12.0
        ),
        "memory_lesion": ConstructiveGlobalRankOnlineMemory(
            N_ITEMS, sigma_a=0.30, encoding_probability=0.0, beta=12.0
        ),
        "perfect_memory": ConstructiveGlobalRankOnlineMemory(
            N_ITEMS, sigma_a=0.30, encoding_probability=1.0, beta=12.0
        ),
        "commitment_readout_lesion": ConstructiveGlobalRankOnlineMemory(
            N_ITEMS, sigma_a=0.30, encoding_probability=0.45, beta=0.0
        ),
    }
    for name, model in lesion_models.items():
        cohort, _lesion_process = simulate_online(model, tasks, MODEL_SEED)
        lesions[name] = compute_endpoint_metrics(cohort)
        lesions[name]["pair_accuracy_pearson_r_vs_human"] = (
            pair_accuracy_correlation(human, lesions[name])
        )

    parameters = {
        "sigma_a": 0.30,
        "encoding_probability_per_presentation": 0.45,
        "beta": 12.0,
        "update": "recursive_least_squares_after_each_encoded_support_trial",
        "query_update": False,
        "meta_update_inside_episode": False,
    }
    report = {
        **report_provenance(
            status=(
                "retrospective_endpoint_developmental_"
                "process_predictions_unvalidated"
            ),
            evaluator="query_chronology_isolated_rng",
            task_seed=TASK_SEED,
            model_seed=MODEL_SEED,
            choice_seed=MODEL_SEED + CHOICE_SEED_OFFSET,
            lapse=LAPSE,
            parameters=parameters,
        ),
        # Kept as a readable alias for the model card/report consumer.
        "candidate": parameters,
        "human": human,
        "batch_v3_1_independent_monte_carlo_reference": v31_metrics,
        "online_v3_2": v32_metrics,
        "endpoint_comparison_interpretation": (
            "v3.1-v3.2 summary differences are finite-cohort Monte Carlo "
            "differences, not evidence for an online endpoint advantage; the "
            "matched latent endpoint equivalence is reported under process predictions"
        ),
        "online_process_predictions": process,
        "lesions": lesions,
    }
    output = ROOT / "outputs" / "cgr_v3_2_vs_liu_humans" / "development_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== CGR-v3.2 online retrospective audit ===")
    print("v3.1/v3.2 summary deltas are not online-mechanism effects")
    for name, values in (("v3.1", v31_metrics), ("v3.2", v32_metrics)):
        print(
            f"{name}: overall={values['overall_accuracy']:.3f} "
            f"learned={values['learned_accuracy']:.3f} "
            f"unlearned={values['unlearned_accuracy']:.3f} "
            f"tau={values['inter_subject_tau_mean']:.3f} "
            f"selfc={values['self_consistency_mean']:.3f} "
            f"bimodal={values['n_bimodal']} "
            f"pair_r={values['pair_accuracy_pearson_r_vs_human']:.3f}"
        )
    print("process:", json.dumps(process, indent=2))
    print(f"report: {output}")


if __name__ == "__main__":
    main()
