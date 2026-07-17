"""Liu 2026 符号化任务：固定学习关系集 + 全 query set。

核心约束:
  - 所有虚拟被试（batch elements）看到相同的 8 个 support pairs
  - 左右呈现随机化 per-subject, per-trial
  - Query 覆盖全部 28 个 pair，顺序随机
  - Query 阶段无反馈

认知边界:
  - 物体图片只承担任意 item identity，可抽象为 cue 符号
  - 当前主候选的 support observation 只包含二元高低关系，不含具体差值
  - Query 不得接收 relation、正确答案或反馈
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch

from fsrl.device import DEVICE
from fsrl.task.constants import ADDINPUT, NUMRESPONSESTEP
from fsrl.task.cues import sample_unique_cue


# === Liu 2026 的 8 个固定非相邻 support position pairs ===
# 论文与公开 CSV 使用 A..H = 0..7 的 low-to-high 顺序：位置 0 最弱，
# 位置 nbcues-1 最强。仓库内部 true_rank 则是 cue 的 strong-to-weak
# 排列，因此映射时必须反转位置轴。
DEFAULT_LIU2026_SUPPORT_PAIRS = [
    (0, 5),  # A < F, diff=5
    (1, 2),  # B < C, diff=1
    (1, 4),  # B < E, diff=3
    (2, 6),  # C < G, diff=4
    (3, 5),  # D < F, diff=2
    (3, 6),  # D < G, diff=3
    (4, 7),  # E < H, diff=3
    (0, 7),  # A < H, diff=7
]


@dataclass(frozen=True)
class SymbolicSupportObservation:
    """Sign-only model-visible information in one support presentation."""

    left_cue: int
    right_cue: int
    sign: int


@dataclass(frozen=True)
class SymbolicMagnitudeSupportObservation(SymbolicSupportObservation):
    """Magnitude-aware support input for candidates that use displayed bars.

    ``magnitude`` is the displayed relative bar-height difference normalized to
    the task range.  It belongs in a model-facing observation rather than in
    ``position_pair``: humans can see the bar difference, whereas the abstract
    ground-truth positions are generator metadata unavailable to the model.
    """

    magnitude: float


@dataclass(frozen=True)
class SymbolicQueryObservation:
    """主模型在无反馈 query 中可以接收的全部信息。"""

    left_cue: int
    right_cue: int


@dataclass(frozen=True)
class SymbolicSupportTrial:
    """Support observation plus generator metadata unavailable to the model."""

    observation: SymbolicSupportObservation
    block_index: int
    position_pair: tuple[int, int]


@dataclass(frozen=True)
class SymbolicQueryTrial:
    """Query observation plus the outer-loop/agent target."""

    observation: SymbolicQueryObservation
    block_index: int
    position_pair: tuple[int, int]
    correct_choice: int


@dataclass(frozen=True)
class Liu2026SymbolicSubjectTask:
    """One virtual subject's independently randomized symbolic task instance."""

    subject_index: int
    true_rank: tuple[int, ...]
    support_trials: tuple[SymbolicSupportTrial, ...]
    query_trials: tuple[SymbolicQueryTrial, ...]


def _rng_random(rng) -> float:
    if hasattr(rng, "random"):
        return float(rng.random())
    return float(rng.rand())


def _rng_integer(rng, low: int, high: int) -> int:
    """Sample an integer from [low, high) for RandomState or Generator."""

    if hasattr(rng, "integers"):
        return int(rng.integers(low, high))
    return int(rng.randint(low, high))


def _cue_at_weak_to_strong_position(
    true_rank, weak_to_strong_position: int
) -> int:
    """Map Liu's A..H axis to the repository's strong-to-weak permutation."""

    return int(true_rank[len(true_rank) - 1 - weak_to_strong_position])


def build_liu2026_symbolic_subject_tasks(
    *,
    n_subjects: int,
    rng=None,
    nbcues: int = 8,
    support_pairs=DEFAULT_LIU2026_SUPPORT_PAIRS,
    support_blocks: int = 4,
    query_blocks: int = 10,
    randomize_true_rank: bool = True,
    include_magnitude: bool = False,
) -> tuple[Liu2026SymbolicSubjectTask, ...]:
    """Build independent symbolic Liu task instances for a virtual cohort.

    Every subject shares the eight cue identities and the abstract A..H support
    graph, but receives an independent cue-to-position permutation, block order,
    and left/right presentation.  The model-facing support observation contains
    cue identities and sign by default, preserving the preregistered sign-only
    boundary.  Magnitude-aware candidates must explicitly request the displayed
    normalized bar difference with ``include_magnitude=True``.
    """

    if rng is None:
        rng = np.random
    if n_subjects <= 0:
        raise ValueError("n_subjects must be positive")
    if nbcues != 8:
        raise ValueError("the preregistered Liu task requires exactly eight cues")
    if support_blocks <= 0 or query_blocks <= 0:
        raise ValueError("support_blocks and query_blocks must be positive")

    canonical_support_pairs = tuple(
        (int(pair[0]), int(pair[1])) for pair in support_pairs
    )
    if len(canonical_support_pairs) != 8 or len(set(canonical_support_pairs)) != 8:
        raise ValueError("the Liu task requires eight unique support position pairs")
    if any(not (0 <= low < high < nbcues) for low, high in canonical_support_pairs):
        raise ValueError("support pairs must use the weak-to-strong 0..7 axis")

    all_query_pairs = tuple(combinations(range(nbcues), 2))
    subject_tasks = []
    for subject_index in range(n_subjects):
        if randomize_true_rank:
            true_rank = tuple(int(cue) for cue in rng.permutation(nbcues))
        else:
            true_rank = tuple(range(nbcues))

        support_trials = []
        for block_index in range(support_blocks):
            for pair_index in rng.permutation(len(canonical_support_pairs)):
                low_position, high_position = canonical_support_pairs[int(pair_index)]
                low_cue = _cue_at_weak_to_strong_position(true_rank, low_position)
                high_cue = _cue_at_weak_to_strong_position(true_rank, high_position)
                if _rng_random(rng) < 0.5:
                    left_cue, right_cue, sign = low_cue, high_cue, -1
                else:
                    left_cue, right_cue, sign = high_cue, low_cue, 1
                if include_magnitude:
                    observation = SymbolicMagnitudeSupportObservation(
                        left_cue=left_cue,
                        right_cue=right_cue,
                        sign=sign,
                        magnitude=(high_position - low_position)
                        / float(nbcues - 1),
                    )
                else:
                    observation = SymbolicSupportObservation(
                        left_cue=left_cue,
                        right_cue=right_cue,
                        sign=sign,
                    )
                support_trials.append(
                    SymbolicSupportTrial(
                        observation=observation,
                        block_index=block_index,
                        position_pair=(low_position, high_position),
                    )
                )

        query_trials = []
        for block_index in range(query_blocks):
            for pair_index in rng.permutation(len(all_query_pairs)):
                low_position, high_position = all_query_pairs[int(pair_index)]
                low_cue = _cue_at_weak_to_strong_position(true_rank, low_position)
                high_cue = _cue_at_weak_to_strong_position(true_rank, high_position)
                if _rng_random(rng) < 0.5:
                    left_cue, right_cue, correct_choice = low_cue, high_cue, 1
                else:
                    left_cue, right_cue, correct_choice = high_cue, low_cue, 0
                query_trials.append(
                    SymbolicQueryTrial(
                        observation=SymbolicQueryObservation(
                            left_cue=left_cue,
                            right_cue=right_cue,
                        ),
                        block_index=block_index,
                        position_pair=(low_position, high_position),
                        correct_choice=correct_choice,
                    )
                )

        subject_tasks.append(
            Liu2026SymbolicSubjectTask(
                subject_index=subject_index,
                true_rank=true_rank,
                support_trials=tuple(support_trials),
                query_trials=tuple(query_trials),
            )
        )
    return tuple(subject_tasks)


def build_meta_training_symbolic_subject_tasks(
    *,
    n_subjects: int,
    rng=None,
    min_nbcues: int = 5,
    max_nbcues: int = 8,
    min_support_blocks: int = 1,
    max_support_blocks: int = 4,
    max_support_edges: int = 10,
    min_query_fraction: float = 0.5,
) -> tuple[Liu2026SymbolicSubjectTask, ...]:
    """Sample a broad sign-only ranking distribution for outer-loop training.

    This distribution intentionally does not replay Liu's fixed eight-edge
    graph.  Across episodes it varies the active item count, support graph,
    repetitions and query subset.  Within an episode, every virtual subject has
    an independent symbol-to-position mapping, graph and presentation order.
    Only the number of trials is shared so subjects can be batched.
    """

    if rng is None:
        rng = np.random
    if n_subjects <= 0:
        raise ValueError("n_subjects must be positive")
    if not (3 <= min_nbcues <= max_nbcues <= 8):
        raise ValueError("meta-training cue range must satisfy 3 <= min <= max <= 8")
    if not (1 <= min_support_blocks <= max_support_blocks):
        raise ValueError("invalid support block range")
    if not (0.0 < min_query_fraction <= 1.0):
        raise ValueError("min_query_fraction must be inside (0, 1]")

    nbcues = _rng_integer(rng, min_nbcues, max_nbcues + 1)
    support_blocks = _rng_integer(
        rng, min_support_blocks, max_support_blocks + 1
    )
    all_position_pairs = tuple(combinations(range(nbcues), 2))
    min_edges = nbcues - 1
    edge_cap = min(len(all_position_pairs), max(max_support_edges, min_edges))
    edge_count = _rng_integer(rng, min_edges, edge_cap + 1)
    min_queries = max(1, int(np.ceil(min_query_fraction * len(all_position_pairs))))
    query_count = _rng_integer(rng, min_queries, len(all_position_pairs) + 1)

    subject_tasks = []
    for subject_index in range(n_subjects):
        true_rank = tuple(int(cue) for cue in rng.permutation(nbcues))

        # A random spanning tree guarantees that every ordinal position is
        # observed without imposing Liu's particular support topology.  Extra
        # edges are then sampled independently for this subject.
        node_order = [int(node) for node in rng.permutation(nbcues)]
        support_edges = set()
        for node_offset in range(1, nbcues):
            child = node_order[node_offset]
            parent = node_order[_rng_integer(rng, 0, node_offset)]
            support_edges.add(tuple(sorted((child, parent))))
        remaining_edges = [
            pair for pair in all_position_pairs if pair not in support_edges
        ]
        for edge_index in rng.permutation(len(remaining_edges))[
            : edge_count - len(support_edges)
        ]:
            support_edges.add(remaining_edges[int(edge_index)])
        support_edges = tuple(sorted(support_edges))

        support_trials = []
        for block_index in range(support_blocks):
            for pair_index in rng.permutation(len(support_edges)):
                low_position, high_position = support_edges[int(pair_index)]
                low_cue = _cue_at_weak_to_strong_position(true_rank, low_position)
                high_cue = _cue_at_weak_to_strong_position(true_rank, high_position)
                if _rng_random(rng) < 0.5:
                    left_cue, right_cue, sign = low_cue, high_cue, -1
                else:
                    left_cue, right_cue, sign = high_cue, low_cue, 1
                support_trials.append(
                    SymbolicSupportTrial(
                        observation=SymbolicSupportObservation(
                            left_cue=left_cue,
                            right_cue=right_cue,
                            sign=sign,
                        ),
                        block_index=block_index,
                        position_pair=(low_position, high_position),
                    )
                )

        query_trials = []
        selected_query_pairs = [
            all_position_pairs[int(pair_index)]
            for pair_index in rng.permutation(len(all_position_pairs))[:query_count]
        ]
        for low_position, high_position in selected_query_pairs:
            low_cue = _cue_at_weak_to_strong_position(true_rank, low_position)
            high_cue = _cue_at_weak_to_strong_position(true_rank, high_position)
            if _rng_random(rng) < 0.5:
                left_cue, right_cue, correct_choice = low_cue, high_cue, 1
            else:
                left_cue, right_cue, correct_choice = high_cue, low_cue, 0
            query_trials.append(
                SymbolicQueryTrial(
                    observation=SymbolicQueryObservation(
                        left_cue=left_cue,
                        right_cue=right_cue,
                    ),
                    block_index=0,
                    position_pair=(low_position, high_position),
                    correct_choice=correct_choice,
                )
            )

        subject_tasks.append(
            Liu2026SymbolicSubjectTask(
                subject_index=subject_index,
                true_rank=true_rank,
                support_trials=tuple(support_trials),
                query_trials=tuple(query_trials),
            )
        )
    return tuple(subject_tasks)


def published_position_pair_to_rank_positions(pair, nbcues: int) -> tuple[int, int]:
    """Map Liu's weak-to-strong position pair to strong-to-weak indices."""
    low_pos_i, low_pos_j = (int(pair[0]), int(pair[1]))
    if not (0 <= low_pos_i < nbcues and 0 <= low_pos_j < nbcues):
        raise ValueError(f"support position outside 0..{nbcues - 1}: {pair}")
    if low_pos_i == low_pos_j:
        raise ValueError(f"support pair contains the same position twice: {pair}")
    return nbcues - 1 - low_pos_i, nbcues - 1 - low_pos_j


def map_published_support_pair_to_cues(pair, true_rank) -> tuple[int, int, float]:
    """Return cue_i, cue_j and sign using the repository's rank convention."""
    nbcues = len(true_rank)
    rank_pos_i, rank_pos_j = published_position_pair_to_rank_positions(pair, nbcues)
    cue_i = int(true_rank[rank_pos_i])
    cue_j = int(true_rank[rank_pos_j])
    teacher_sign = 1.0 if rank_pos_i < rank_pos_j else -1.0
    return cue_i, cue_j, teacher_sign


def generate_liu2026_cue_data(config, rng=None):
    """生成每个 batch element 的 8 个 cue 向量。

    注意: cue 向量在 episode 内固定。``shared_cue_identities=True`` 时各 batch
    element 共享基础 cue 身份；只有可选的 encoding noise 会产生被试间差异。
    若 config.encoding_noise_std > 0，会为每个 (subject, cue) 加入稳定的高斯噪声，
    模拟编码阶段的个体异质性。
    """
    if rng is None:
        rng = np.random
    cue_data = []
    shared_cues = None
    for batch_index in range(config.bs):
        if getattr(config, "shared_cue_identities", True) and shared_cues is not None:
            cue_data.append(shared_cues.copy())
            continue
        cues = []
        for cue_index in range(config.nbcues):
            candidate = rng.randint(2, size=config.cs) * 2 - 1
            # 确保 cue 之间不太相似
            while any(np.mean(candidate == c) > 0.66 for c in cues):
                candidate = rng.randint(2, size=config.cs) * 2 - 1
            cues.append(candidate)
        cue_array = np.array(cues, dtype="float32")
        if getattr(config, "shared_cue_identities", True):
            shared_cues = cue_array
        cue_data.append(cue_array.copy())

    encoding_noise_std = getattr(config, "encoding_noise_std", 0.0)
    if encoding_noise_std > 0.0:
        noise = rng.normal(
            0.0, encoding_noise_std, size=(config.bs, config.nbcues, config.cs)
        ).astype("float32")
        for batch_index in range(config.bs):
            cue_data[batch_index] = cue_data[batch_index] + noise[batch_index]

    return cue_data


def build_liu2026_support_set(config, true_rank=None, support_order=None, rng=None):
    """构建 support phase trial 序列。

    config.support_pairs 使用论文的 weak-to-strong 位置坐标。
    通过 strong-to-weak 的 true_rank 映射为当前 episode 的实际 cue 对。

    Args:
        true_rank: cue 索引从强到弱的排列。必须提供。
        support_order: 可选的自定义 position-pair 序列，用于诊断实验。
            若提供，则直接使用该序列（不再按 block 随机打乱），长度可任意。

    Returns:
        trials: list of (cue_i, cue_j, teacher_sign) for each trial
        teacher_sign = +1 if cue_i is stronger than cue_j according to
        ``true_rank``, -1 otherwise.  Cue 的整数 ID 本身没有强弱含义。
    """
    if rng is None:
        rng = np.random
    if true_rank is None:
        raise ValueError("true_rank is required to map Liu weak-to-strong support positions")
    trials = []
    if support_order is not None:
        for pair in support_order:
            cue_i, cue_j, teacher_sign = map_published_support_pair_to_cues(
                pair, true_rank
            )
            trials.append((cue_i, cue_j, teacher_sign))
        return trials

    base_pairs = list(config.support_pairs)
    for _ in range(config.support_blocks):
        block_trials = []
        for pair in base_pairs:
            cue_i, cue_j, teacher_sign = map_published_support_pair_to_cues(
                pair, true_rank
            )
            block_trials.append((cue_i, cue_j, teacher_sign))
        rng.shuffle(block_trials)
        trials.extend(block_trials)
    return trials


def build_liu2026_query_set(nbcues=8, true_rank=None):
    """构建全部 28 个 pair 的 query set。

    Query 覆盖所有 pair。这里的 pos_i/pos_j 是仓库内部 strong-to-weak
    rank indices，不是 support_pairs 使用的论文 weak-to-strong 坐标。

    Args:
        nbcues: cue 数量。
        true_rank: cue 索引从强到弱的排列。为 None 时退化为旧行为。

    Returns:
        list of (cue_i, cue_j)，其中 cue_i 比 cue_j 强。
    """
    query_pairs = []
    for pos_i in range(nbcues):
        for pos_j in range(pos_i + 1, nbcues):
            if true_rank is not None:
                cue_i, cue_j = int(true_rank[pos_i]), int(true_rank[pos_j])
            else:
                cue_i, cue_j = pos_i, pos_j
            query_pairs.append((cue_i, cue_j))
    return query_pairs


def prepare_liu2026_support_trial(
    config, cue_data, trial_info, batch_index, rng=None
):
    """准备一个 support phase trial。

    关键: 左右呈现随机化 per-subject。
    Returns:
        cue_vec: [2*cs]
        teacher_da: float (+1 if left is stronger, -1 otherwise)
        pair: (i, j) 原始 pair（i < j）
    """
    cue_i, cue_j, teacher_sign = trial_info

    if rng is None:
        rng = np.random
    # 50% 概率交换左右
    if rng.rand() < 0.5:
        left_cue, right_cue = cue_i, cue_j
        left_stronger = teacher_sign > 0
    else:
        left_cue, right_cue = cue_j, cue_i
        left_stronger = teacher_sign < 0

    cue_vec = np.concatenate(
        [cue_data[batch_index][left_cue], cue_data[batch_index][right_cue]]
    )
    teacher_da = 1.0 if left_stronger else -1.0
    pair = (cue_i, cue_j) if cue_i < cue_j else (cue_j, cue_i)
    return cue_vec, teacher_da, pair


def prepare_liu2026_query_trial(config, cue_data, pair, batch_index, swap_left_right: bool = False):
    """准备一个 query phase trial。

    pair 已经是 (强 cue, 弱 cue)（由 build_liu2026_query_set 保证）。
    Query 阶段可通过 ``swap_left_right=True`` 将强 cue 放在右侧，实现左右随机化。

    Returns:
        cue_vec: [2*cs]
        correct_choice: int (0 = 选择左边, 1 = 选择右边)
        model_pair: (left_cue, right_cue) 传给模型的 cue 对
        canonical_pair: (i, j) 规范 pair（i < j），用于分析
        left_is_first: bool，左边是否为 canonical_pair 的第一个元素
    """
    cue_i, cue_j = pair
    if swap_left_right:
        left_cue, right_cue = cue_j, cue_i
        correct_choice = 1  # 右边强
    else:
        left_cue, right_cue = cue_i, cue_j
        correct_choice = 0  # 左边强

    cue_vec = np.concatenate(
        [cue_data[batch_index][left_cue], cue_data[batch_index][right_cue]]
    )
    model_pair = (left_cue, right_cue)
    canonical_pair = (min(left_cue, right_cue), max(left_cue, right_cue))
    left_is_first = left_cue == canonical_pair[0]
    return cue_vec, correct_choice, model_pair, canonical_pair, left_is_first


def build_liu2026_step_inputs(
    config,
    cue_vec,
    numstep,
    numstep_ep,
    reward=0.0,
    previous_action=None,
    is_go_step=False,
    delta=0.0,
):
    """构造单步输入向量，与现有 RetroModulRNN 的 inputsize 兼容。

    输入结构（与 fsrl/task/trial.py build_step_inputs 一致）:
      [0 : 2*cs]      : cue 向量拼接
      [2*cs]          : Go signal（query 决策步）
      [2*cs+1]        : bias
      [2*cs+2]        : 时间进度
      [2*cs+3]        : reward
      [2*cs+4]        : 当次柱条关系的符号化有符号差值
      [2*cs+5 : ]     : previous action one-hot
    """
    inp = np.zeros(config.inputsize, dtype="float32")

    # Cue 向量
    inp[: config.cs * 2] = cue_vec

    # Go signal
    if is_go_step:
        inp[config.cs * 2] = 1.0

    # Bias
    inp[config.cs * 2 + 1] = 1.0

    # 时间进度
    inp[config.cs * 2 + 2] = numstep_ep / max(config.eplen, 1)

    # Reward
    inp[config.cs * 2 + 3] = reward

    # Distance-aware trial observation；sign-only 条件应另行置换为符号值。
    inp[config.cs * 2 + 4] = delta

    # Previous action
    if previous_action is not None and numstep == NUMRESPONSESTEP + 1:
        inp[config.cs * 2 + ADDINPUT + previous_action] = 1.0

    return torch.from_numpy(inp).to(DEVICE)
