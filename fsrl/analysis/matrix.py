import numpy as np

from fsrl.episode.types import TestResponse


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


def pair_accuracy_from_response_matrix(
    response_matrix: np.ndarray | list[list[float]], true_rank: list[int]
) -> dict[tuple[int, int], float]:
    """Recover pair accuracy from the upper-triangular response matrix.

    ``build_response_matrix`` stores only ``P(lower cue ID is preferred)`` in
    the upper triangle.  The lower triangle is an unused 0.5 placeholder, not
    the complementary probability, so callers must not renormalize the two
    triangles.
    """
    matrix = np.asarray(response_matrix, dtype=np.float64)
    position = {int(cue): idx for idx, cue in enumerate(true_rank)}
    result: dict[tuple[int, int], float] = {}
    for cue_i in range(len(true_rank)):
        for cue_j in range(cue_i + 1, len(true_rank)):
            p_i_over_j = float(matrix[cue_i, cue_j])
            result[(cue_i, cue_j)] = (
                p_i_over_j
                if position[cue_i] < position[cue_j]
                else 1.0 - p_i_over_j
            )
    return result


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


def display_response_matrix(R: np.ndarray) -> np.ndarray:
    """
    热图用矩阵：上三角 P(i≻j)，下三角补 1-P(j≻i)，对角为 nan（不绘制）。
    """
    n = R.shape[0]
    disp = np.full((n, n), np.nan, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            if np.isfinite(R[i, j]):
                disp[i, j] = R[i, j]
                disp[j, i] = 1.0 - R[i, j]
    return disp
