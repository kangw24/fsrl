"""
传递推理任务的响应结构分析（对齐 Mermaid ANALYSIS_STAGE）。

流程：响应矩阵 R → 监督/未监督分组 → 错误模式 → circular triad → HodgeRank → 主观排序
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

from simple_neo import EpisodeRecord, TestResponse, log

ALPHABET = [chr(i) for i in range(ord("A"), ord("Z") + 1)]

_CJK_FONT_CANDIDATES = (
    "Microsoft YaHei",
    "SimHei",
    "DengXian",
    "PingFang SC",
    "Noto Sans CJK SC",
    "WenQuanYi Micro Hei",
    "Arial Unicode MS",
)


def configure_matplotlib_chinese() -> str | None:
    """选用系统里第一个可用的中文字体，避免 DejaVu Sans 缺字警告。"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CJK_FONT_CANDIDATES:
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name
    for font in font_manager.fontManager.ttflist:
        if any(
            key in font.name
            for key in ("YaHei", "SimHei", "PingFang", "Noto Sans CJK", "WenQuanYi", "Heiti")
        ):
            matplotlib.rcParams["font.sans-serif"] = [font.name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return font.name
    matplotlib.rcParams["axes.unicode_minus"] = False
    return None


configure_matplotlib_chinese()


@dataclass
class AnalysisReport:
    """单次 episode 分析结果。"""

    nbcues: int
    response_matrix: np.ndarray
    acc_supervised: float
    acc_unsupervised: float
    n_supervised_pairs: int
    n_unsupervised_pairs: int
    error_by_pair: dict[str, float]
    stable_errors: list[str]
    circular_triads: list[list[int]]
    n_circular_triads: int
    curl_ratio: float
    global_consistency: bool
    subjective_scores: np.ndarray
    subjective_rank: list[int]
    true_rank: list[int]
    kendall_tau_vs_true: float
    spearman_rho_vs_true: float
    test_accuracy: float
    # Liu et al. 2026 扩展（Hypothesis I + Fig 1F–G）
    n_subjects: int = 0
    serial_position_acc: dict = field(default_factory=dict)
    distance_acc: dict = field(default_factory=dict)
    pair_beta_fits: list[dict] = field(default_factory=list)
    subject_beta_fits: list[dict] = field(default_factory=list)
    n_pair_bimodal: int = 0
    n_pair_high_accuracy: int = 0
    n_pair_unimodal: int = 0
    n_subject_bimodal: int = 0
    mean_error_consistency: float = float("nan")
    n_subjects_consistent_error_80: int = 0
    # 逐被试 HodgeRank（Liu Fig 3C/E：个体化主观排序）
    per_subject_rankings: list = field(default_factory=list)
    mean_per_subject_tau_vs_true: float = float("nan")
    mean_inter_subject_tau: float = float("nan")
    n_subjects_with_wrong_rank: int = 0
    n_subjects_self_consistent_wrong: int = 0

    def to_json_dict(self):
        """转为可 JSON 序列化的字典。"""
        data = asdict(self)
        data["response_matrix"] = self.response_matrix.tolist()
        data["subjective_scores"] = self.subjective_scores.tolist()
        return _json_sanitize(data)


def rank_positions(true_rank: list[int]) -> dict[int, int]:
    """cue -> 秩位置（0=最强）。"""
    return {cue: idx for idx, cue in enumerate(true_rank)}


def ranking_distance(i: int, j: int, true_rank: list[int]) -> int:
    """真序秩差（symbolic distance）。"""
    pos = rank_positions(true_rank)
    return abs(pos[i] - pos[j])


def build_subject_pair_accuracy(
    responses: list[TestResponse],
    min_subject_acc: float | None = 0.5,
) -> tuple[dict[int, dict[tuple[int, int], float]], list[int]]:
    """
    每个 batch 元素视为一个「被试」，汇总各 pair 准确率。
    min_subject_acc: 排除平均准确率低于此值的被试（论文 <0.5）；None 表示不过滤。
    """
    buckets: dict[int, dict[tuple[int, int], list[int]]] = {}
    for resp in responses:
        buckets.setdefault(resp.batch_index, {}).setdefault(resp.pair, []).append(
            int(resp.correct)
        )

    subject_acc: dict[int, dict[tuple[int, int], float]] = {}
    for batch, pairs in buckets.items():
        subject_acc[batch] = {p: float(np.mean(vals)) for p, vals in pairs.items()}

    if min_subject_acc is not None and subject_acc:
        kept = []
        for batch, pairs in list(subject_acc.items()):
            if pairs and float(np.mean(list(pairs.values()))) >= min_subject_acc:
                kept.append(batch)
            else:
                del subject_acc[batch]
        return subject_acc, kept
    return subject_acc, list(subject_acc.keys())


def fit_beta_profile(values: list[float] | np.ndarray) -> dict | None:
    """
    对 [0,1] 上的比例数据做 Beta 分布 MLE 拟合（Liu Fig 2C–D）。
    返回 α, β 及 profile 分类：bimodal / high_accuracy / unimodal / other
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 3:
        return None
    eps = 1e-3
    arr = np.clip(arr, eps, 1.0 - eps)
    try:
        alpha, beta_param, _, _ = stats.beta.fit(arr, floc=0.0, fscale=1.0)
    except Exception:
        return None

    if alpha < 1.0 and beta_param < 1.0:
        profile = "bimodal"
    elif alpha > 1.0 and beta_param < 1.0:
        profile = "high_accuracy"
    elif alpha > 1.0 and beta_param > 1.0:
        profile = "unimodal"
    else:
        profile = "other"

    return {
        "alpha": float(alpha),
        "beta": float(beta_param),
        "profile": profile,
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
    }


def analyze_pair_level_beta(
    subject_pair_acc: dict[int, dict[tuple[int, int], float]],
    subjects: list[int],
    all_pairs: list[tuple[int, int]],
) -> list[dict]:
    """跨被试 pair 级 Beta 拟合（Hypothesis I, Fig 2D）。"""
    fits = []
    for pair in all_pairs:
        accs = [
            subject_pair_acc[s][pair]
            for s in subjects
            if pair in subject_pair_acc.get(s, {})
        ]
        fit = fit_beta_profile(accs)
        if fit is None:
            continue
        i, j = pair
        fits.append(
            {
                "pair": pair_key(i, j),
                "rank_distance": abs(i - j),
                "subject_accuracies": accs,
                **fit,
            }
        )
    return fits


def analyze_subject_level_beta(
    subject_pair_acc: dict[int, dict[tuple[int, int], float]],
    subjects: list[int],
) -> list[dict]:
    """被试级 accuracy profile 的 Beta 拟合（prereg Hypothesis I）。"""
    fits = []
    for batch in subjects:
        pairs = subject_pair_acc.get(batch, {})
        if len(pairs) < 3:
            continue
        accs = list(pairs.values())
        fit = fit_beta_profile(accs)
        if fit is None:
            continue
        fits.append({"batch_index": int(batch), "n_pairs": len(accs), **fit})
    return fits


def subject_error_consistency(subject_pairs: dict[tuple[int, int], float]) -> float:
    """单被试错误一致性：各 pair 上 max(acc, 1-acc) 的均值（Fig 2E）。"""
    if not subject_pairs:
        return float("nan")
    vals = [max(acc, 1.0 - acc) for acc in subject_pairs.values()]
    return float(np.mean(vals))


def analyze_error_consistency(
    subject_pair_acc: dict[int, dict[tuple[int, int], float]],
    subjects: list[int],
    error_threshold: float = 0.8,
) -> tuple[float, int]:
    """群平均错误一致性 + 至少一个 pair 错误率>=threshold 的被试数。"""
    consistencies = []
    n_consistent_error = 0
    for batch in subjects:
        pairs = subject_pair_acc.get(batch, {})
        consistencies.append(subject_error_consistency(pairs))
        if any((1.0 - acc) >= error_threshold for acc in pairs.values()):
            n_consistent_error += 1
    mean_c = float(np.mean(consistencies)) if consistencies else float("nan")
    return mean_c, n_consistent_error


def _accuracy_by_serial_position(
    responses: list[TestResponse],
    true_rank: list[int],
    pair_filter: set[tuple[int, int]] | None = None,
) -> dict[int, float]:
    """Fig 1F：按 cue 秩位置的平均准确率（该位置参与的所有 trial）。"""
    pos_map = rank_positions(true_rank)
    nbcues = len(true_rank)
    buckets: dict[int, list[int]] = {p: [] for p in range(nbcues)}
    for resp in responses:
        if pair_filter is not None and resp.pair not in pair_filter:
            continue
        i, j = resp.pair
        buckets[pos_map[i]].append(int(resp.correct))
        buckets[pos_map[j]].append(int(resp.correct))
    return {
        p: float(np.mean(vals)) if vals else float("nan")
        for p, vals in buckets.items()
    }


def _accuracy_by_rank_distance(
    responses: list[TestResponse],
    true_rank: list[int],
    pair_filter: set[tuple[int, int]] | None = None,
) -> dict[int, float]:
    """Fig 1G：按秩差（symbolic distance）的平均准确率。"""
    buckets: dict[int, list[int]] = {}
    for resp in responses:
        if pair_filter is not None and resp.pair not in pair_filter:
            continue
        d = ranking_distance(resp.pair[0], resp.pair[1], true_rank)
        buckets.setdefault(d, []).append(int(resp.correct))
    return {
        d: float(np.mean(vals)) if vals else float("nan") for d, vals in sorted(buckets.items())
    }


def analyze_liu_behavioral_effects(
    responses: list[TestResponse],
    true_rank: list[int],
    supervision_set,
    query_set,
    min_subject_acc: float | None = 0.5,
) -> dict:
    """Liu 2026：跨被试 Beta、serial position、distance effect。"""
    s_pairs = supervised_pair_set(supervision_set)
    q_pairs = query_pair_set(query_set)
    unsup_pairs = q_pairs - s_pairs
    all_pairs = sorted(q_pairs)

    subject_pair_acc, subjects = build_subject_pair_accuracy(
        responses, min_subject_acc=min_subject_acc
    )

    pair_beta = analyze_pair_level_beta(subject_pair_acc, subjects, all_pairs)
    subject_beta = analyze_subject_level_beta(subject_pair_acc, subjects)
    mean_ec, n_ec80 = analyze_error_consistency(subject_pair_acc, subjects)

    n_bimodal = sum(1 for f in pair_beta if f["profile"] == "bimodal")
    n_high = sum(1 for f in pair_beta if f["profile"] == "high_accuracy")
    n_unimodal = sum(1 for f in pair_beta if f["profile"] == "unimodal")
    n_subj_bimodal = sum(1 for f in subject_beta if f["profile"] == "bimodal")

    serial_all = _accuracy_by_serial_position(responses, true_rank)
    serial_s = _accuracy_by_serial_position(responses, true_rank, s_pairs)
    serial_u = _accuracy_by_serial_position(responses, true_rank, unsup_pairs)
    dist_all = _accuracy_by_rank_distance(responses, true_rank)
    dist_s = _accuracy_by_rank_distance(responses, true_rank, s_pairs)
    dist_u = _accuracy_by_rank_distance(responses, true_rank, unsup_pairs)

    serial_position_acc = {
        "all": serial_all,
        "supervised": serial_s,
        "unsupervised": serial_u,
    }
    distance_acc = {
        "all": dist_all,
        "supervised": dist_s,
        "unsupervised": dist_u,
    }

    return {
        "n_subjects": len(subjects),
        "subjects": subjects,
        "serial_position_acc": serial_position_acc,
        "distance_acc": distance_acc,
        "pair_beta_fits": pair_beta,
        "subject_beta_fits": subject_beta,
        "n_pair_bimodal": n_bimodal,
        "n_pair_high_accuracy": n_high,
        "n_pair_unimodal": n_unimodal,
        "n_subject_bimodal": n_subj_bimodal,
        "mean_error_consistency": mean_ec,
        "n_subjects_consistent_error_80": n_ec80,
        "subject_pair_acc": subject_pair_acc,
    }


def _json_sanitize(obj):
    """把 numpy 标量/数组递归转为原生 Python 类型，供 json.dump 使用。"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    return obj


def canonical_pair(i: int, j: int) -> tuple[int, int]:
    """统一为 (min, max) 无序对。"""
    return (i, j) if i < j else (j, i)


def pair_key(i: int, j: int) -> str:
    """pair 的字符串键，如 '0-2'。"""
    a, b = canonical_pair(i, j)
    return f"{a}-{b}"


def prefers_matrix(R: np.ndarray, i: int, j: int, threshold: float = 0.5) -> bool:
    """R[i,j] 表示 i 被偏好于 j 的比例（i<j 时取上三角）。"""
    if i == j:
        return False
    a, b = canonical_pair(i, j)
    return R[a, b] > threshold if i == a else R[a, b] < (1.0 - threshold)


def build_response_matrix(responses: list[TestResponse], nbcues: int) -> np.ndarray:
    """
    构造 pairwise 响应矩阵 R。
    R[i,j]（i<j）= agent 认为 i ≻ j 的比例。
    """
    counts = np.zeros((nbcues, nbcues), dtype=np.float64)
    totals = np.zeros((nbcues, nbcues), dtype=np.float64)

    for resp in responses:
        i, j = resp.pair
        assert i < j
        totals[i, j] += 1
        if resp.prefers_i_over_j:
            counts[i, j] += 1

    R = np.full((nbcues, nbcues), 0.5, dtype=np.float64)
    mask = totals > 0
    R[mask] = counts[mask] / totals[mask]
    for i in range(nbcues):
        R[i, i] = 0.5
    return R


def supervised_pair_set(supervision_set) -> set[tuple[int, int]]:
    """监督集 S 转为 canonical pair 集合。"""
    pairs = set()
    for item in supervision_set:
        if len(item) == 3:
            i, j, _ = item
        else:
            i, j = item
        pairs.add(canonical_pair(i, j))
    return pairs


def query_pair_set(query_set) -> set[tuple[int, int]]:
    """查询集 Q。"""
    return {canonical_pair(i, j) for i, j in query_set}


def split_supervised_unsupervised(
    responses: list[TestResponse],
    supervision_set,
    query_set,
) -> tuple[float, float, int, int]:
    """计算已监督 pair (S) 与未监督 pair (Q\\S) 的准确率。"""
    s_pairs = supervised_pair_set(supervision_set)
    q_pairs = query_pair_set(query_set)
    unsup_pairs = q_pairs - s_pairs

    correct_s, total_s = 0, 0
    correct_u, total_u = 0, 0
    for resp in responses:
        if resp.pair in s_pairs:
            total_s += 1
            correct_s += int(resp.correct)
        elif resp.pair in unsup_pairs:
            total_u += 1
            correct_u += int(resp.correct)

    acc_s = correct_s / total_s if total_s else float("nan")
    acc_u = correct_u / total_u if total_u else float("nan")
    return acc_s, acc_u, len(s_pairs), len(unsup_pairs)


def analyze_error_patterns(
    responses: list[TestResponse], nbcues: int, stable_threshold: float = 0.6
) -> tuple[dict[str, float], list[str]]:
    """
    按 pair 统计错误率；稳定错误 = 错误率 >= stable_threshold 且至少出现 2 次。
    """
    errors: dict[tuple[int, int], list[int]] = {}
    for resp in responses:
        key = resp.pair
        errors.setdefault(key, []).append(0 if resp.correct else 1)

    error_by_pair: dict[str, float] = {}
    stable_errors: list[str] = []
    for pair, vals in errors.items():
        rate = float(np.mean(vals))
        key = pair_key(*pair)
        error_by_pair[key] = rate
        if rate >= stable_threshold and len(vals) >= 2:
            stable_errors.append(key)

    return error_by_pair, stable_errors


def detect_circular_triads(
    R: np.ndarray, nbcues: int, threshold: float = 0.5
) -> list[list[int]]:
    """
    检测 circular triad：a≻b, b≻c, c≻a 形成环流（与全序不一致）。
    """
    triads: list[list[int]] = []
    for a in range(nbcues):
        for b in range(nbcues):
            if a == b:
                continue
            for c in range(nbcues):
                if c in (a, b):
                    continue
                if (
                    prefers_matrix(R, a, b, threshold)
                    and prefers_matrix(R, b, c, threshold)
                    and prefers_matrix(R, c, a, threshold)
                ):
                    triple = sorted([a, b, c])
                    if triple not in triads:
                        triads.append(triple)
    return triads


def pairwise_observations_from_R(R: np.ndarray, nbcues: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从 R 构造边观测 y_e = 2*R[i,j]-1（i<j）及关联矩阵行。
    返回 edges (m,2), y (m,), 用于最小二乘 ranking。
    """
    edges = []
    y = []
    for i in range(nbcues):
        for j in range(i + 1, nbcues):
            edges.append((i, j))
            y.append(2.0 * R[i, j] - 1.0)
    return np.array(edges, dtype=int), np.array(y, dtype=np.float64), np.arange(len(edges))


def hodge_rank_scores(R: np.ndarray, nbcues: int) -> tuple[np.ndarray, float]:
    """
    HodgeRank 风格重建：将 pairwise 偏好分解为梯度（全序一致）+ 环流分量。
    返回主观势函数 scores 与 curl_ratio（环流能量占比）。
    """
    edges, y, _ = pairwise_observations_from_R(R, nbcues)
    m = len(edges)
    n = nbcues

    # 关联矩阵 B: m x n，边 e=(i,j) 满足 f_i - f_j = y（y>0 表示 i≻j）
    rows, cols, data = [], [], []
    for e, (i, j) in enumerate(edges):
        rows.extend([e, e])
        cols.extend([i, j])
        data.extend([1.0, -1.0])
    B = csr_matrix((data, (rows, cols)), shape=(m, n))

    # 最小二乘 min ||B f - y||^2；固定 gauge：f[0]=0
    # 使用 lsqr 解超定系统
    f, *_ = lsqr(B, y)[:1]
    f = np.asarray(f, dtype=np.float64)

    gradient_flow = B @ f
    curl = y - gradient_flow
    curl_ratio = float(np.sum(curl**2) / (np.sum(y**2) + 1e-10))

    return f, curl_ratio


def subjective_rank_from_scores(scores: np.ndarray) -> list[int]:
    """势函数越大 → 隐含秩越高（越强）；返回从强到弱的 cue 编号列表。"""
    return list(np.argsort(-scores))


def analyze_per_subject_rankings(
    responses: list[TestResponse],
    nbcues: int,
    true_rank: list[int],
    tau_wrong_threshold: float = 0.999,
) -> dict:
    """
    逐被试 HodgeRank（对齐 Liu 2026 Fig 3C/E）。
    检测个体主观排序是否偏离真序，以及被试间排序相似度。
    """
    subjects = sorted({r.batch_index for r in responses})
    per_subject: list[dict] = []
    ranks: list[list[int]] = []

    for batch in subjects:
        sub_resp = [r for r in responses if r.batch_index == batch]
        if not sub_resp:
            continue
        R_s = build_response_matrix(sub_resp, nbcues)
        triads = detect_circular_triads(R_s, nbcues)
        scores, curl_s = hodge_rank_scores(R_s, nbcues)
        rank_s = subjective_rank_from_scores(scores)
        tau = kendall_tau(true_rank, rank_s)
        rho = spearman_rho(true_rank, rank_s)
        acc = float(np.mean([r.correct for r in sub_resp]))
        pair_acc: dict[tuple[int, int], list[int]] = {}
        for resp in sub_resp:
            pair_acc.setdefault(resp.pair, []).append(int(resp.correct))
        ec = subject_error_consistency(
            {p: float(np.mean(v)) for p, v in pair_acc.items()}
        )
        per_subject.append(
            {
                "batch_index": int(batch),
                "subjective_rank": rank_s,
                "kendall_tau_vs_true": tau,
                "spearman_rho_vs_true": rho,
                "curl_ratio": curl_s,
                "test_accuracy": acc,
                "n_circular_triads": len(triads),
                "error_consistency": ec,
                "self_consistent_wrong": (
                    tau < tau_wrong_threshold and len(triads) == 0
                ),
            }
        )
        ranks.append(rank_s)

    if not per_subject:
        return {
            "per_subject_rankings": [],
            "mean_per_subject_tau_vs_true": float("nan"),
            "mean_inter_subject_tau": float("nan"),
            "n_subjects_with_wrong_rank": 0,
            "n_subjects_self_consistent_wrong": 0,
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
    }


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


def analyze_episode(record: EpisodeRecord, curl_threshold: float = 0.15) -> AnalysisReport:
    """对单次 episode 记录运行完整分析链。"""
    nbcues = record.nbcues
    responses = record.test_responses

    R = build_response_matrix(responses, nbcues)
    acc_s, acc_u, n_s, n_u = split_supervised_unsupervised(
        responses, record.supervision_set, record.query_set
    )
    error_by_pair, stable_errors = analyze_error_patterns(responses, nbcues)
    triads = detect_circular_triads(R, nbcues)
    scores, curl_ratio = hodge_rank_scores(R, nbcues)
    subj_rank = subjective_rank_from_scores(scores)
    true_rank = list(record.true_rank)

    tau = kendall_tau(true_rank, subj_rank)
    rho = spearman_rho(true_rank, subj_rank)
    test_acc = float(np.mean([r.correct for r in responses])) if responses else float("nan")

    liu = analyze_liu_behavioral_effects(
        responses, true_rank, record.supervision_set, record.query_set
    )
    per_subj = analyze_per_subject_rankings(responses, nbcues, true_rank)

    return AnalysisReport(
        nbcues=nbcues,
        response_matrix=R,
        acc_supervised=acc_s,
        acc_unsupervised=acc_u,
        n_supervised_pairs=n_s,
        n_unsupervised_pairs=n_u,
        error_by_pair=error_by_pair,
        stable_errors=stable_errors,
        circular_triads=triads,
        n_circular_triads=len(triads),
        curl_ratio=curl_ratio,
        global_consistency=curl_ratio < curl_threshold,
        subjective_scores=scores,
        subjective_rank=subj_rank,
        true_rank=true_rank,
        kendall_tau_vs_true=tau,
        spearman_rho_vs_true=rho,
        test_accuracy=test_acc,
        n_subjects=liu["n_subjects"],
        serial_position_acc=liu["serial_position_acc"],
        distance_acc=liu["distance_acc"],
        pair_beta_fits=liu["pair_beta_fits"],
        subject_beta_fits=liu["subject_beta_fits"],
        n_pair_bimodal=liu["n_pair_bimodal"],
        n_pair_high_accuracy=liu["n_pair_high_accuracy"],
        n_pair_unimodal=liu["n_pair_unimodal"],
        n_subject_bimodal=liu["n_subject_bimodal"],
        mean_error_consistency=liu["mean_error_consistency"],
        n_subjects_consistent_error_80=liu["n_subjects_consistent_error_80"],
        per_subject_rankings=per_subj["per_subject_rankings"],
        mean_per_subject_tau_vs_true=per_subj["mean_per_subject_tau_vs_true"],
        mean_inter_subject_tau=per_subj["mean_inter_subject_tau"],
        n_subjects_with_wrong_rank=per_subj["n_subjects_with_wrong_rank"],
        n_subjects_self_consistent_wrong=per_subj["n_subjects_self_consistent_wrong"],
    )


def save_analysis_report(report: AnalysisReport, output_dir: Path) -> Path:
    """保存 JSON 报告。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "analysis_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.to_json_dict(), f, indent=2, ensure_ascii=False)
    log(f"[analysis] 已保存报告: {path}")
    return path


def plot_analysis_report(report: AnalysisReport, output_dir: Path) -> None:
    """生成分析图表。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    nbcues = report.nbcues
    labels = ALPHABET[:nbcues]

    # 1. 响应矩阵热图
    fig, ax = plt.subplots(figsize=(6, 5))
    R_disp = display_response_matrix(report.response_matrix)
    im = ax.imshow(R_disp, vmin=0, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(nbcues))
    ax.set_yticks(range(nbcues))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    ax.set_title("Pairwise 响应矩阵 R\n上三角 P(i优于j)，下三角 P(j优于i)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(output_dir / "heatmap_R.png", dpi=200)
    fig.savefig(output_dir / "heatmap_R.pdf")
    plt.close(fig)

    # 2. 监督 vs 未监督准确率
    fig, ax = plt.subplots(figsize=(4, 4))
    names = ["S (已监督)", "Q\\S (未监督)"]
    accs = [report.acc_supervised, report.acc_unsupervised]
    ax.bar(names, accs, color=["#4c72b0", "#dd8452"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("准确率")
    ax.set_title("已监督 vs 未监督 pair")
    fig.tight_layout()
    fig.savefig(output_dir / "acc_S_vs_QminusS.png", dpi=200)
    plt.close(fig)

    # 3. 错误率 vs 秩差（与 Fig 1G 一致）
    fig, ax = plt.subplots(figsize=(5, 3))
    distances, error_rates = [], []
    pos_map = rank_positions(report.true_rank)
    for key, err in report.error_by_pair.items():
        i, j = map(int, key.split("-"))
        distances.append(abs(pos_map[i] - pos_map[j]))
        error_rates.append(err)
    if distances:
        ax.scatter(distances, error_rates, alpha=0.6)
        ax.set_xlabel("秩差 |rank(i)-rank(j)|")
        ax.set_ylabel("错误率")
        ax.set_title("错误率 vs symbolic distance")
    fig.tight_layout()
    fig.savefig(output_dir / "error_by_distance.png", dpi=200)
    plt.close(fig)

    # 4. 真实排序 vs 主观排序（按 cue 标签，纵轴为秩位置）
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(nbcues)
    true_pos = [report.true_rank.index(i) for i in range(nbcues)]
    subj_pos = [report.subjective_rank.index(i) for i in range(nbcues)]
    ax.plot(x, true_pos, "o-", label="真实排序", color="green", zorder=1)
    ax.plot(
        x,
        subj_pos,
        "s--",
        label="HodgeRank 主观排序",
        color="orange",
        markersize=7,
        zorder=2,
    )
    if true_pos == subj_pos:
        ax.text(
            0.02,
            0.02,
            "两曲线完全重合",
            transform=ax.transAxes,
            fontsize=9,
            color="gray",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("秩位置（越小越强）")
    ax.set_title(
        f"排序对比 (Kendall τ={report.kendall_tau_vs_true:.3f}, "
        f"curl={report.curl_ratio:.3f})"
    )
    ax.legend()
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_dir / "true_vs_subjective_rank.png", dpi=200)
    plt.close(fig)

    _plot_liu_extended(report, output_dir)
    _plot_per_subject_rankings(report, output_dir)

    log(f"[analysis] 图表已保存至 {output_dir}")


_BETA_COLORS = {
    "bimodal": "#55a868",
    "high_accuracy": "#c44e52",
    "unimodal": "#aaaaaa",
    "other": "#4c72b0",
}


def _sorted_int_keys(curve: dict) -> list[int]:
    """曲线 dict 的键统一为 int 排序（兼容 JSON 字符串键）。"""
    return sorted(int(k) for k in curve.keys())


def _curve_xy(curve: dict) -> tuple[list[int], list[float]]:
    xs = _sorted_int_keys(curve)
    ys = []
    for x in xs:
        if x in curve:
            ys.append(float(curve[x]))
        else:
            ys.append(float(curve[str(x)]))
    return xs, ys


def display_response_matrix(R: np.ndarray) -> np.ndarray:
    """
    热图用矩阵：上三角 P(i≻j)，下三角补 1-P(j≻i)，对角为 nan（不绘制）。
    避免下三角误显示为默认 0.5。
    """
    n = R.shape[0]
    disp = np.full((n, n), np.nan, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            if np.isfinite(R[i, j]):
                disp[i, j] = R[i, j]
                disp[j, i] = 1.0 - R[i, j]
    return disp


def _beta_profile_legend():
    from matplotlib.lines import Line2D

    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=_BETA_COLORS[k],
            label=k,
            markersize=8,
        )
        for k in ("bimodal", "high_accuracy", "unimodal", "other")
    ]


def _pick_exemplar_pair(pair_beta_fits: list[dict]) -> dict:
    """选跨被试离散度最大的 pair 作直方图示例（双峰优先）。"""
    bimodal = [f for f in pair_beta_fits if f["profile"] == "bimodal"]
    pool = bimodal or pair_beta_fits

    def _spread(fit: dict) -> float:
        accs = fit.get("subject_accuracies", [])
        if len(accs) < 2:
            return 0.0
        return float(np.std(accs))

    return max(pool, key=_spread)


def _plot_per_subject_rankings(report: AnalysisReport, output_dir: Path) -> None:
    """逐被试主观排序 vs 真序（Liu Fig 3C/E 风格）。"""
    items = report.per_subject_rankings
    if not items:
        return

    nbcues = report.nbcues
    labels = ALPHABET[:nbcues]
    x = np.arange(nbcues)
    true_pos = [report.true_rank.index(i) for i in range(nbcues)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        x,
        true_pos,
        "o-",
        label="真实排序",
        color="green",
        lw=2.5,
        markersize=8,
        zorder=3,
    )
    for item in items:
        rank = item["subjective_rank"]
        subj_pos = [rank.index(i) for i in range(nbcues)]
        tau = item["kendall_tau_vs_true"]
        color = "#c44e52" if tau < 0.999 else "#4c72b0"
        ax.plot(
            x,
            subj_pos,
            "-",
            color=color,
            alpha=0.45,
            lw=1.0,
            zorder=1,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("秩位置（越小越强）")
    ax.set_title(
        "逐被试 HodgeRank 主观排序\n"
        f"错序={report.n_subjects_with_wrong_rank}, "
        f"自洽错序={report.n_subjects_self_consistent_wrong}, "
        f"被试间 tau_mean={report.mean_inter_subject_tau:.2f}"
    )
    ax.invert_yaxis()
    from matplotlib.lines import Line2D

    legend_items = [
        Line2D([0], [0], color="green", lw=2.5, marker="o", label="真实排序"),
        Line2D([0], [0], color="#4c72b0", lw=1.5, label="被试=真序"),
        Line2D([0], [0], color="#c44e52", lw=1.5, label="被试≠真序"),
    ]
    ax.legend(handles=legend_items, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "per_subject_rankings.png", dpi=200)
    plt.close(fig)


def _plot_liu_extended(report: AnalysisReport, output_dir: Path) -> None:
    """Liu 2026 扩展图：serial position、distance effect、Beta 拟合。"""
    nbcues = report.nbcues
    labels = ALPHABET[:nbcues]

    # 5. Serial position effect (Fig 1F)
    fig, ax = plt.subplots(figsize=(6, 4))
    pos_labels = [f"P{p}" for p in range(nbcues)]
    for key, style, color in (
        ("all", "-o", "#4c72b0"),
        ("supervised", "--s", "#c44e52"),
        ("unsupervised", "--^", "#55a868"),
    ):
        curve = report.serial_position_acc.get(key, {})
        if not curve:
            continue
        xs, ys = _curve_xy(curve)
        label = {"all": "全部", "supervised": "已监督 S", "unsupervised": "未监督 Q\\S"}[key]
        ax.plot(xs, ys, style, color=color, label=label, markersize=5)
    ax.set_xticks(range(nbcues))
    ax.set_xticklabels(pos_labels)
    ax.set_xlabel("秩位置（P0=最强）")
    ax.set_ylabel("准确率")
    ax.set_ylim(0, 1)
    ax.set_title("Serial position effect (Fig 1F)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "serial_position_effect.png", dpi=200)
    plt.close(fig)

    # 6. Symbolic distance effect (Fig 1G)
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, style, color in (
        ("all", "-o", "#4c72b0"),
        ("supervised", "--s", "#c44e52"),
        ("unsupervised", "--^", "#55a868"),
    ):
        curve = report.distance_acc.get(key, {})
        if not curve:
            continue
        xs, ys = _curve_xy(curve)
        label = {"all": "全部", "supervised": "已监督 S", "unsupervised": "未监督 Q\\S"}[key]
        ax.plot(xs, ys, style, color=color, label=label, markersize=5)
    ax.set_xlabel("秩差 |rank(i)-rank(j)|")
    ax.set_ylabel("准确率")
    ax.set_ylim(0, 1)
    ax.set_title("Symbolic distance effect (Fig 1G)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "distance_effect.png", dpi=200)
    plt.close(fig)

    # 7. Pair-level Beta α–β scatter (Fig 2D)
    if report.pair_beta_fits:
        fig, ax = plt.subplots(figsize=(5, 5))
        for fit in report.pair_beta_fits:
            ax.scatter(
                fit["alpha"],
                fit["beta"],
                c=_BETA_COLORS.get(fit["profile"], "#4c72b0"),
                s=40,
                alpha=0.85,
            )
        ax.axvline(1.0, color="k", lw=0.5, ls="--")
        ax.axhline(1.0, color="k", lw=0.5, ls="--")
        ax.set_xlabel("Beta α")
        ax.set_ylabel("Beta β")
        ax.set_title(
            f"Pair-level Beta 拟合 (n={len(report.pair_beta_fits)})\n"
            f"双峰={report.n_pair_bimodal}, 高准确={report.n_pair_high_accuracy}, "
            f"单峰={report.n_pair_unimodal}"
        )
        ax.legend(handles=_beta_profile_legend(), fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "pair_beta_fits.png", dpi=200)
        plt.close(fig)

    # 8. 示例 pair 的跨被试准确率直方图（离散度最大 / 双峰优先）
    if report.pair_beta_fits:
        exemplar = _pick_exemplar_pair(report.pair_beta_fits)
        accs = exemplar.get("subject_accuracies", [])
        if len(accs) >= 3:
            fig, ax = plt.subplots(figsize=(4, 3))
            n_bins = min(12, max(4, len(accs) // 2))
            ax.hist(accs, bins=n_bins, range=(0, 1), color="#55a868", alpha=0.75)
            alpha, beta_p = exemplar["alpha"], exemplar["beta"]
            xs = np.linspace(0.001, 0.999, 200)
            pdf = stats.beta.pdf(xs, alpha, beta_p)
            bin_width = 1.0 / n_bins
            ax.plot(
                xs,
                pdf * len(accs) * bin_width,
                color="#c44e52",
                lw=1.5,
                label=f"Beta({alpha:.1f},{beta_p:.1f})",
            )
            ax.axvline(exemplar["mean"], color="k", ls="--", label=f"均值={exemplar['mean']:.2f}")
            ax.set_xlabel("被试准确率")
            ax.set_ylabel("计数")
            ax.set_title(
                f"跨被试分布: pair {exemplar['pair']} [{exemplar['profile']}]"
            )
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(output_dir / "exemplar_pair_subject_hist.png", dpi=200)
            plt.close(fig)

    # 9. Subject-level Beta scatter
    if report.subject_beta_fits:
        fig, ax = plt.subplots(figsize=(5, 5))
        for fit in report.subject_beta_fits:
            ax.scatter(
                fit["alpha"],
                fit["beta"],
                c=_BETA_COLORS.get(fit["profile"], "#4c72b0"),
                s=40,
                alpha=0.85,
            )
        ax.axvline(1.0, color="k", lw=0.5, ls="--")
        ax.axhline(1.0, color="k", lw=0.5, ls="--")
        ax.set_xlabel("Beta α")
        ax.set_ylabel("Beta β")
        ax.set_title(
            f"被试级 Beta 拟合 (n={report.n_subjects}, 双峰={report.n_subject_bimodal})"
        )
        ax.legend(handles=_beta_profile_legend(), fontsize=8, loc="upper right")
        fig.tight_layout()
        fig.savefig(output_dir / "subject_beta_fits.png", dpi=200)
        plt.close(fig)


def run_full_analysis(record: EpisodeRecord, output_dir: Path) -> AnalysisReport:
    """分析总入口：计算 + 保存报告 + 作图。"""
    log("[analysis] 开始响应结构分析...")
    report = analyze_episode(record)
    save_analysis_report(report, output_dir)
    plot_analysis_report(report, output_dir)
    log(
        f"[analysis] 完成: test_acc={report.test_accuracy:.3f}, "
        f"acc_S={report.acc_supervised:.3f}, acc_Q\\S={report.acc_unsupervised:.3f}, "
        f"triads={report.n_circular_triads}, curl={report.curl_ratio:.3f}, "
        f"Kendall τ={report.kendall_tau_vs_true:.3f}, "
        f"n_subj={report.n_subjects}, pair_bimodal={report.n_pair_bimodal}, "
        f"err_consistency={report.mean_error_consistency:.3f}, "
        f"wrong_rank={report.n_subjects_with_wrong_rank}, "
        f"inter_subj_τ={report.mean_inter_subject_tau:.3f}"
    )
    return report
