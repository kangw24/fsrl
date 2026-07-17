"""mode-support 一致性诊断。

用于回答 Q1：虚拟被试最终形成的排序（以及 LatentRank 选用的 mode）
是否真正受到 support evidence 的约束，而不是随机全序的产物。
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from fsrl.analysis.per_subject import analyze_per_subject_rankings
from fsrl.episode.types import EpisodeRecord


def support_consistency_score(
    subjective_rank: list[int],
    supervision_set: list,
    reference_rank: list[int] | None = None,
) -> float:
    """计算一个主观排序满足多少 support pair。

    Args:
        subjective_rank: cue 索引从强到弱的排列。
        supervision_set: canonical cue pair 列表。tuple 顺序只标识 pair，不能编码
            哪一项更强。
        reference_rank: evidence 所规定的 strongest-to-weakest 排序。提供时，
            support 方向由该排序决定；仅为兼容旧调用，省略时才把 tuple 第一项
            解释为更强。

    Returns:
        被满足的 support pair 比例（0–1）。
    """
    if not supervision_set:
        return float("nan")
    pos = {cue: idx for idx, cue in enumerate(subjective_rank)}
    reference_pos = (
        {cue: idx for idx, cue in enumerate(reference_rank)}
        if reference_rank is not None
        else None
    )
    correct = 0
    for i, j in supervision_set:
        i = int(i)
        j = int(j)
        expected_i_before_j = (
            reference_pos[i] < reference_pos[j]
            if reference_pos is not None
            else True
        )
        actual_i_before_j = pos[i] < pos[j]
        if actual_i_before_j == expected_i_before_j:
            correct += 1
    return correct / len(supervision_set)


def _random_consistency_distribution(
    nbcues: int,
    supervision_set: list,
    n_samples: int = 2000,
    rng=None,
    reference_rank: list[int] | None = None,
):
    """随机全局排序下 support consistency 的零分布。"""
    if rng is None:
        rng = np.random.default_rng()
    scores = []
    cues = list(range(nbcues))
    for _ in range(n_samples):
        perm = rng.permutation(cues).tolist()
        scores.append(
            support_consistency_score(perm, supervision_set, reference_rank)
        )
    return np.array(scores)


def analyze_mode_support_consistency(
    record: EpisodeRecord,
    n_random_samples: int = 2000,
    rng=None,
) -> dict:
    """分析每个被试的主观排序与 support evidence 的一致性。

    Returns:
        dict with keys:
          - mean_actual_consistency
          - std_actual_consistency
          - mean_random_consistency
          - random_ci_low / random_ci_high (95%)
          - delta (actual - random)
          - t_stat / p_value (单样本 t 检验，检验 actual 是否高于 random)
          - n_subjects
          - per_subject: list of {batch_index, mode, actual_consistency}
    """
    if rng is None:
        rng = np.random.default_rng(42)

    nbcues = record.nbcues
    true_rank = list(record.true_rank)
    supervision_set = record.supervision_set
    subject_support_pairs = getattr(record, "subject_support_pairs", {}) or {}

    per_subject_result = analyze_per_subject_rankings(
        record.test_responses,
        nbcues,
        true_rank,
        subject_true_ranks=getattr(record, "subject_true_ranks", None),
    )

    subject_modes = getattr(record, "subject_modes", {}) or {}

    actual_scores = []
    details = []
    for item in per_subject_result["per_subject_rankings"]:
        batch_index = item["batch_index"]
        rank = item["subjective_rank"]
        observed_support = subject_support_pairs.get(batch_index, supervision_set)
        reference_rank = (
            record.subject_true_ranks.get(batch_index, true_rank)
            if record.subject_true_ranks
            else true_rank
        )
        score = support_consistency_score(
            rank, observed_support, reference_rank
        )
        actual_scores.append(score)
        details.append({
            "batch_index": batch_index,
            "mode": int(subject_modes.get(batch_index, -1)),
            "actual_consistency": float(score),
            "subjective_rank": rank,
            "n_observed_support_pairs": len(observed_support),
        })

    actual_scores = np.array(actual_scores)
    random_scores = _random_consistency_distribution(
        nbcues,
        supervision_set,
        n_samples=n_random_samples,
        rng=rng,
        reference_rank=true_rank,
    )

    mean_actual = float(np.mean(actual_scores)) if len(actual_scores) else float("nan")
    std_actual = float(np.std(actual_scores)) if len(actual_scores) else float("nan")
    mean_random = float(np.mean(random_scores))
    ci_low = float(np.percentile(random_scores, 2.5))
    ci_high = float(np.percentile(random_scores, 97.5))
    delta = mean_actual - mean_random

    # 单样本 t 检验：实际被试的 consistency 是否显著高于随机零分布均值
    if len(actual_scores) > 1 and not np.isclose(np.std(actual_scores), 0.0):
        t_stat, p_value = stats.ttest_1samp(actual_scores, mean_random)
        # scipy 默认双边；我们关心的是 actual > random，转为单边
        p_value = p_value / 2.0 if t_stat > 0 else 1.0 - p_value / 2.0
    elif len(actual_scores) > 1:
        if mean_actual > mean_random:
            t_stat, p_value = float("inf"), 0.0
        elif mean_actual < mean_random:
            t_stat, p_value = float("-inf"), 1.0
        else:
            t_stat, p_value = 0.0, 1.0
    else:
        t_stat = float("nan")
        p_value = float("nan")

    return {
        "mean_actual_consistency": mean_actual,
        "std_actual_consistency": std_actual,
        "mean_random_consistency": mean_random,
        "random_ci_low": ci_low,
        "random_ci_high": ci_high,
        "delta": delta,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "n_subjects": len(actual_scores),
        "n_supervision_pairs": len(supervision_set),
        "per_subject": details,
    }
