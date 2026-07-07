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
import numpy as np
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

from simple_neo import EpisodeRecord, TestResponse, log

ALPHABET = [chr(i) for i in range(ord("A"), ord("Z") + 1)]


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

    def to_json_dict(self):
        """转为可 JSON 序列化的字典。"""
        data = asdict(self)
        data["response_matrix"] = self.response_matrix.tolist()
        data["subjective_scores"] = self.subjective_scores.tolist()
        return _json_sanitize(data)


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
        return {k: _json_sanitize(v) for k, v in obj.items()}
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
    im = ax.imshow(report.response_matrix, vmin=0, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(nbcues))
    ax.set_yticks(range(nbcues))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    ax.set_title("Pairwise 响应矩阵 R\nR[i,j]=P(i≻j)")
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

    # 3. 错误率 vs pair 距离
    fig, ax = plt.subplots(figsize=(5, 3))
    distances, error_rates = [], []
    for key, err in report.error_by_pair.items():
        i, j = map(int, key.split("-"))
        distances.append(abs(i - j))
        error_rates.append(err)
    if distances:
        ax.scatter(distances, error_rates, alpha=0.6)
        ax.set_xlabel("Pair 距离 |i-j|")
        ax.set_ylabel("错误率")
        ax.set_title("局部错误模式")
    fig.tight_layout()
    fig.savefig(output_dir / "error_by_distance.png", dpi=200)
    plt.close(fig)

    # 4. 真实排序 vs 主观排序
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(nbcues)
    true_pos = [report.true_rank.index(i) for i in range(nbcues)]
    subj_pos = [report.subjective_rank.index(i) for i in range(nbcues)]
    ax.plot(x, true_pos, "o-", label="真实排序", color="green")
    ax.plot(x, subj_pos, "s--", label="HodgeRank 主观排序", color="orange")
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

    log(f"[analysis] 图表已保存至 {output_dir}")


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
        f"Kendall τ={report.kendall_tau_vs_true:.3f}"
    )
    return report
