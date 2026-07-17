import numpy as np
from scipy import stats

from fsrl.analysis.liu_effects import subject_error_consistency
from fsrl.analysis.matrix import pair_key
from fsrl.episode.types import TestResponse


def kendall_tau(true_rank: list[int], estimated_rank: list[int]) -> float:
    """与真实排序的 Kendall τ。"""
    n = len(true_rank)
    true_pos = {cue: idx for idx, cue in enumerate(true_rank)}
    est_pos = {cue: idx for idx, cue in enumerate(estimated_rank)}
    cues = list(range(n))
    tau, _ = stats.kendalltau(
        [true_pos[c] for c in cues],
        [est_pos[c] for c in cues],
    )
    return float(tau) if not np.isnan(tau) else 0.0


def spearman_rho(true_rank: list[int], estimated_rank: list[int]) -> float:
    """与真实排序的 Spearman ρ。"""
    n = len(true_rank)
    true_pos = {cue: idx for idx, cue in enumerate(true_rank)}
    est_pos = {cue: idx for idx, cue in enumerate(estimated_rank)}
    cues = list(range(n))
    rho, _ = stats.spearmanr(
        [true_pos[c] for c in cues],
        [est_pos[c] for c in cues],
    )
    return float(rho) if not np.isnan(rho) else 0.0


def analyze_per_subject_rankings(
    responses: list[TestResponse],
    nbcues: int,
    true_rank: list[int],
    subject_true_ranks: dict[int, list[int]] | None = None,
    tau_wrong_threshold: float = 0.999,
    exclude_all_pairs_majority_correct: bool = False,
    min_subject_acc: float | None = 0.5,
) -> dict:
    """
    逐被试 HodgeRank（对齐 Liu 2026 Fig 3C/E）。

    Args:
        subject_true_ranks: 若提供，按 batch_index 查找每个被试自己所在 episode 的
            真实排序；否则所有被试都使用传入的 ``true_rank``。
    """
    from fsrl.analysis.hodge import (
        detect_circular_triads,
        hodge_rank_scores,
        subjective_rank_from_scores,
    )
    from fsrl.analysis.matrix import build_response_matrix

    subjects = sorted({r.batch_index for r in responses})
    per_subject: list[dict] = []
    ranks: list[list[int]] = []
    n_excluded_below_accuracy = 0

    for batch in subjects:
        sub_resp = [r for r in responses if r.batch_index == batch]
        if not sub_resp:
            continue
        acc = float(np.mean([r.correct for r in sub_resp]))
        if min_subject_acc is not None and acc < min_subject_acc:
            n_excluded_below_accuracy += 1
            continue
        R_s = build_response_matrix(sub_resp, nbcues)
        strict_triads = detect_circular_triads(R_s, nbcues)
        # Liu's released Fig. 3b classifications are reproduced when a 5:5
        # pair is allowed in either direction: a participant is inconsistent
        # if any tie resolution admits a circular triad.
        triads = detect_circular_triads(R_s, nbcues, ties_as_both=True)
        scores, curl_s = hodge_rank_scores(R_s, nbcues)
        rank_s = subjective_rank_from_scores(scores)
        subj_true_rank = (
            subject_true_ranks.get(int(batch), true_rank)
            if subject_true_ranks
            else true_rank
        )
        tau = kendall_tau(subj_true_rank, rank_s)
        rho = spearman_rho(subj_true_rank, rank_s)
        pair_acc: dict[tuple[int, int], list[int]] = {}
        for resp in sub_resp:
            pair_acc.setdefault(resp.pair, []).append(int(resp.correct))
        pair_accuracy = {p: float(np.mean(v)) for p, v in pair_acc.items()}
        ec = subject_error_consistency(pair_accuracy)
        tied_pairs = [pair for pair, accuracy in pair_accuracy.items() if accuracy == 0.5]
        expected_pairs = nbcues * (nbcues - 1) // 2
        all_pairs_majority_correct = (
            len(pair_accuracy) == expected_pairs
            and all(accuracy > 0.5 for accuracy in pair_accuracy.values())
        )
        max_circular_triads = (
            (nbcues ** 3 - 4 * nbcues) // 24
            if nbcues % 2 == 0
            else (nbcues ** 3 - nbcues) // 24
        )
        self_consistency_coefficient = 1.0 - len(triads) / max_circular_triads
        per_subject.append(
            {
                "batch_index": int(batch),
                "subjective_rank": rank_s,
                "true_rank": list(subj_true_rank),
                "kendall_tau_vs_true": tau,
                "spearman_rho_vs_true": rho,
                "curl_ratio": curl_s,
                "test_accuracy": acc,
                "n_circular_triads": len(triads),
                "n_strict_circular_triads": len(strict_triads),
                "n_tied_pairs": len(tied_pairs),
                "tied_pairs": [list(pair) for pair in tied_pairs],
                "error_consistency": ec,
                "pairwise_response_consistency": ec,
                "all_pairs_majority_correct": all_pairs_majority_correct,
                "self_consistency_coefficient": self_consistency_coefficient,
                "globally_self_consistent": len(triads) == 0,
                "self_consistent_wrong": (
                    tau < tau_wrong_threshold and len(triads) == 0
                ),
            }
        )
        ranks.append(rank_s)

    n_excluded_all_pairs_majority_correct = 0
    if exclude_all_pairs_majority_correct:
        n_excluded_all_pairs_majority_correct = sum(
            1 for item in per_subject if item["all_pairs_majority_correct"]
        )
        per_subject = [
            item
            for item in per_subject
            if not item["all_pairs_majority_correct"]
        ]
        ranks = [item["subjective_rank"] for item in per_subject]

    if not per_subject:
        return {
            "per_subject_rankings": [],
            "mean_per_subject_tau_vs_true": float("nan"),
            "mean_inter_subject_tau": float("nan"),
            "n_subjects_with_wrong_rank": 0,
            "n_subjects_self_consistent_wrong": 0,
            "n_subjects_rank_analysis": 0,
            "n_subjects_excluded_all_pairs_majority_correct": (
                n_excluded_all_pairs_majority_correct
            ),
            "n_subjects_excluded_below_accuracy": n_excluded_below_accuracy,
        }

    pairwise: list[float] = []
    for i in range(len(ranks)):
        for j in range(i + 1, len(ranks)):
            pairwise.append(kendall_tau(ranks[i], ranks[j]))

    n_wrong = sum(
        1 for item in per_subject if item["kendall_tau_vs_true"] < tau_wrong_threshold
    )
    n_sc_wrong = sum(1 for item in per_subject if item["self_consistent_wrong"])

    return {
        "per_subject_rankings": per_subject,
        "mean_per_subject_tau_vs_true": float(
            np.mean([item["kendall_tau_vs_true"] for item in per_subject])
        ),
        "mean_inter_subject_tau": float(np.mean(pairwise)) if pairwise else float("nan"),
        "n_subjects_with_wrong_rank": n_wrong,
        "n_subjects_self_consistent_wrong": n_sc_wrong,
        "n_subjects_rank_analysis": len(per_subject),
        "n_subjects_excluded_all_pairs_majority_correct": (
            n_excluded_all_pairs_majority_correct
        ),
        "n_subjects_excluded_below_accuracy": n_excluded_below_accuracy,
    }
