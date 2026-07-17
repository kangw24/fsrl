from dataclasses import dataclass, field

import torch


@dataclass
class EpisodeStats:
    loss: torch.Tensor
    loss_value: float
    loss_objective: float
    test_reward_mean: float
    nbtesttrials: int
    test_perf: float | None
    test_perf_adjacent: float | None
    test_perf_nonadjacent: float | None
    final_pw: torch.Tensor
    final_subject_rank: torch.Tensor | None = None
    test_perf_learned: float | None = None
    test_perf_nonlearned: float | None = None


@dataclass
class TestResponse:
    """单次测试查询的响应记录（用于分析阶段）。"""

    block_id: int
    pair: tuple[int, int]
    action: int
    prefers_i_over_j: bool
    correct: bool
    batch_index: int
    confidence: float | None = None  # 模型对本次选择的确定性：2 * |p - 0.5|
    rt_proxy: float | None = None    # RT 代理：-log(confidence)


@dataclass
class EpisodeRecord:
    """单次 eval episode 的完整记录，供 analysis 使用。"""

    nbcues: int
    supervision_set: list
    query_set: list
    test_responses: list[TestResponse] = field(default_factory=list)
    true_rank: list[int] = field(default_factory=list)
    # 当多个 episode 合并时，用 offset 后的 batch_index 保存每个被试自己的真实排序
    subject_true_ranks: dict[int, list[int]] = field(default_factory=dict)
    # 当模型为 LatentRank/RetroLatentRank 时，保存每个被试最终选用的 mode index
    subject_modes: dict[int, int] = field(default_factory=dict)
    # 过程模型改造：记录本次 episode 实际使用的 support trial 顺序与 dropped pairs
    support_order: list[tuple[int, int, int]] = field(default_factory=list)
    support_dropped_pairs: list[tuple[int, int]] = field(default_factory=list)
    subject_support_pairs: dict[int, list[tuple[int, int]]] = field(
        default_factory=dict
    )
    provenance: dict = field(default_factory=dict)
    test_perf: float | None = None
    test_perf_adjacent: float | None = None
    test_perf_nonadjacent: float | None = None
