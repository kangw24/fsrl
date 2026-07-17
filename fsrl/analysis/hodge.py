import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

from fsrl.analysis.matrix import prefers_matrix


def detect_circular_triads(
    R: np.ndarray,
    nbcues: int,
    threshold: float = 0.5,
    ties_as_both: bool = False,
) -> list[list[int]]:
    """
    检测 circular triad：a≻b, b≻c, c≻a 形成环流（与全序不一致）。
    """
    def prefers(i: int, j: int) -> bool:
        if not ties_as_both:
            return prefers_matrix(R, i, j, threshold)
        a, b = (i, j) if i < j else (j, i)
        probability_a = float(R[a, b])
        return probability_a >= threshold if i == a else probability_a <= threshold

    triads: set[tuple[int, int, int]] = set()
    for a in range(nbcues):
        for b in range(a + 1, nbcues):
            for c in range(b + 1, nbcues):
                # R 只在上三角存 P(i > j)，下三角是 0.5 占位符。
                # 所有方向都必须经 prefers_matrix 读取，不能把下三角当数据。
                if (
                    prefers(a, b)
                    and prefers(b, c)
                    and prefers(c, a)
                ) or (
                    prefers(b, a)
                    and prefers(a, c)
                    and prefers(c, b)
                ):
                    triads.add((a, b, c))
    return [list(t) for t in triads]


def pairwise_observations_from_R(R: np.ndarray, nbcues: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从 R 构造边观测 y_e = 2*R[i,j]-1（i<j）及关联矩阵行。
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
    """
    edges, y, _ = pairwise_observations_from_R(R, nbcues)
    m = len(edges)
    n = nbcues

    rows, cols, data = [], [], []
    for e, (i, j) in enumerate(edges):
        rows.extend([e, e])
        cols.extend([i, j])
        data.extend([1.0, -1.0])
    B = csr_matrix((data, (rows, cols)), shape=(m, n))

    f, *_ = lsqr(B, y)[:1]
    f = np.asarray(f, dtype=np.float64)

    gradient_flow = B @ f
    curl = y - gradient_flow
    curl_ratio = float(np.sum(curl**2) / (np.sum(y**2) + 1e-10))

    return f, curl_ratio


def subjective_rank_from_scores(scores: np.ndarray) -> list[int]:
    """势函数越大 → 隐含秩越高（越强）；返回从强到弱的 cue 编号列表。"""
    return list(np.argsort(-scores))
