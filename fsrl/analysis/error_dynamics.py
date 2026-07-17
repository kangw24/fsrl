"""Trial / block 级错误动态分析。

用于回答 Q2：被试内部的错误模式是静态全序的产物，
还是随 block / trial 演化的过程性动态。
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from fsrl.episode.types import EpisodeRecord


def _block_accuracy(responses, n_blocks: int):
    """返回每个 block 的准确率序列。"""
    block_acc = []
    for b in range(n_blocks):
        br = [r for r in responses if r.block_id == b]
        if br:
            block_acc.append(float(np.mean([r.correct for r in br])))
        else:
            block_acc.append(float("nan"))
    return np.array(block_acc)


def _pair_block_accuracy(responses, pair: tuple, n_blocks: int):
    """返回某个 pair 在每个 block 上的准确率序列。"""
    seq = []
    for b in range(n_blocks):
        br = [r for r in responses if r.block_id == b and r.pair == pair]
        if br:
            seq.append(float(np.mean([r.correct for r in br])))
        else:
            seq.append(float("nan"))
    return np.array(seq)


def _compare_static_vs_drift(block_acc: np.ndarray):
    """用同一批目标点和 AICc 比较静态均值模型与 AR(1) 模型。

    Returns:
        dict with ss_static, ss_drift, drift_better, slope, intercept, r_value
    """
    valid = ~np.isnan(block_acc)
    if valid.sum() < 4:
        return None
    y = block_acc[valid]
    if len(y) < 3:
        return None
    targets = y[1:]
    # 两个模型必须在相同的 n-1 个目标点上比较。静态模型有 1 个参数，
    # AR(1) 有截距和斜率 2 个参数；AICc 对短 block 序列惩罚额外自由度。
    static_mean = float(np.mean(targets))
    ss_static = float(np.sum((targets - static_mean) ** 2))
    if np.isclose(np.std(y[:-1]), 0.0):
        slope = 0.0
        intercept = float(np.mean(targets))
        r_value = 0.0
    else:
        slope, intercept, r_value, _, _ = stats.linregress(y[:-1], y[1:])
    pred = intercept + slope * y[:-1]
    ss_drift = float(np.sum((y[1:] - pred) ** 2))
    n_obs = len(targets)

    def aicc(rss: float, n: int, k: int) -> float:
        rss = max(rss, np.finfo(float).tiny)
        aic = n * np.log(rss / n) + 2 * k
        correction_denom = n - k - 1
        if correction_denom <= 0:
            return float("inf")
        return float(aic + (2 * k * (k + 1)) / correction_denom)

    aicc_static = aicc(ss_static, n_obs, 1)
    aicc_drift = aicc(ss_drift, n_obs, 2)
    return {
        "ss_static": ss_static,
        "ss_drift": ss_drift,
        "aicc_static": aicc_static,
        "aicc_drift": aicc_drift,
        "delta_aicc_drift_minus_static": aicc_drift - aicc_static,
        "drift_better": aicc_drift < aicc_static,
        "ar1_slope": float(slope),
        "ar1_intercept": float(intercept),
        "ar1_r": float(r_value),
    }


def analyze_error_dynamics(record: EpisodeRecord) -> dict:
    """对每个被试的 block-level 准确率比较静态 vs drift 模型。

    Returns:
        dict with:
          - n_subjects
          - n_drift_better, n_static_better, proportion_drift_better
          - mean_ar1_slope, mean_ar1_r
          - per_subject list
    """
    responses = record.test_responses
    if not responses:
        return {"n_subjects": 0}

    n_blocks = max(r.block_id for r in responses) + 1
    subjects = sorted({r.batch_index for r in responses})

    per_subject = []
    drift_better = 0
    slopes = []
    rs = []
    for subj in subjects:
        sub_resp = [r for r in responses if r.batch_index == subj]
        block_acc = _block_accuracy(sub_resp, n_blocks)
        comp = _compare_static_vs_drift(block_acc)
        if comp is None:
            continue
        if comp["drift_better"]:
            drift_better += 1
        slopes.append(comp["ar1_slope"])
        rs.append(comp["ar1_r"])
        per_subject.append(
            {
                "batch_index": int(subj),
                "block_accuracy": block_acc.tolist(),
                **comp,
            }
        )

    n = len(per_subject)
    return {
        "n_subjects": n,
        "n_drift_better": drift_better,
        "n_static_better": n - drift_better,
        "proportion_drift_better": drift_better / n if n else float("nan"),
        "mean_ar1_slope": float(np.mean(slopes)) if slopes else float("nan"),
        "mean_ar1_r": float(np.mean(rs)) if rs else float("nan"),
        "per_subject": per_subject,
    }


def analyze_pair_error_dynamics(record: EpisodeRecord, top_k: int = 10) -> dict:
    """对每个 pair 计算跨 block 的准确率稳定性（drift 比例）。"""
    responses = record.test_responses
    if not responses:
        return {"n_pairs": 0}

    n_blocks = max(r.block_id for r in responses) + 1
    pairs = sorted({r.pair for r in responses})

    pair_results = []
    for pair in pairs:
        seq = _pair_block_accuracy(responses, pair, n_blocks)
        comp = _compare_static_vs_drift(seq)
        if comp is None:
            continue
        pair_results.append(
            {
                "pair": pair,
                "block_accuracy": seq.tolist(),
                **comp,
            }
        )

    # 按漂移最强排序
    pair_results.sort(key=lambda x: x["ss_static"] - x["ss_drift"], reverse=True)
    return {
        "n_pairs": len(pair_results),
        "n_drift_better_pairs": sum(1 for x in pair_results if x["drift_better"]),
        "top_drift_pairs": pair_results[:top_k],
    }
