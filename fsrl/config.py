from dataclasses import dataclass, field
from enum import Enum
import os
from pathlib import Path

import torch


class ModelType(Enum):
    """Liu 2026 模型条件。"""

    Q_LEARNING = "q_learning"
    VANILLA_RNN = "vanilla_rnn"
    PLASTIC_RNN = "plastic_rnn"
    LATENT_RANK = "latent_rank"
    RETRO_LATENT_RANK = "retro_latent_rank"
    LEAKY_ACCUMULATOR = "leaky_accumulator"
    ONLINE_ORDINAL = "online_ordinal"


@dataclass
class Liu2026Config:
    """Liu et al. (2026) 的符号化 few-shot ranking 配置。

    核心约束：
      - nbcues = 8
      - support_pairs = 8 个固定非相邻 pair
      - support_drop = 0（所有被试看到相同的抽象位置关系集）
      - query 阶段无反馈
      - 不跨 episode 保留 task-specific state

    图片只作为任意 item identity，可抽象为 cue 符号。当前主候选只读取
    学习 trial 的二元高低关系，不读取具体差值；模型不得读取完整
    true_rank、query 答案或人类选择。
    """

    # === 设备 ===
    device: str = field(
        default_factory=lambda: os.environ.get(
            "FSRL_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
        )
    )

    # === 模型选择 ===
    model_type: ModelType = ModelType.PLASTIC_RNN

    # === 刺激 ===
    nbcues: int = 8
    cs: int = 15
    shared_cue_identities: bool = True  # 人类实验中参与者看到同一组 item 身份

    # === Support phase ===
    support_pairs: tuple = field(
        default_factory=lambda: (
            # Published low-to-high ordinal coordinates: A < ... < H.
            (0, 5),  # A < F
            (1, 2),  # B < C
            (1, 4),  # B < E
            (2, 6),  # C < G
            (3, 5),  # D < F
            (3, 6),  # D < G
            (4, 7),  # E < H
            (0, 7),  # A < H
        )
    )
    support_blocks: int = 4
    support_triallen: int = 2
    supervision_drop: int = 0

    # === Query phase ===
    query_blocks: int = 10
    query_triallen: int = 4
    query_feedback: bool = False

    # === 模型架构 ===
    hs: int = 200
    bs: int = 32

    # === Sign-only meta-learning runners ===
    lpw: float = 1e-4
    use_plasticity: bool = True
    use_meta_sign_runner: bool = False
    meta_sign_candidate_name: str = "retro_modul_rnn_meta_sign_v1"
    meta_sign_reset_support_trial_state: bool = False
    meta_sign_reset_query_state: bool = False
    meta_train_min_cues: int = 5
    meta_train_max_cues: int = 8
    meta_train_min_support_blocks: int = 1
    meta_train_max_support_blocks: int = 4
    meta_train_max_support_edges: int = 10
    meta_train_min_query_fraction: float = 0.5

    # === Leaky accumulator baseline ===
    leaky_retention_init: float = 0.9
    leaky_logit_scale_init: float = 1.0

    # === Online ordinal prediction-error candidate ===
    online_learning_rate_init: float = 0.5
    online_retention_init: float = 0.98
    online_logit_scale_init: float = 1.0
    online_encoding_reliability: float = 1.0
    online_query_lapse: float = 0.0

    # === LatentRank 特有 ===
    latent_rank_dim: int = 8
    rank_temperature: float = 0.5  # 无显式 override 时的初始 BTL choice temperature
    use_latent_rank: bool = False
    allow_ground_truth_rank_input: bool = False  # 仅遗留固定排序调试；认知评估必须为 False

    # === LatentRank 排序码本（用于生成多样化全局排序）===
    num_rank_modes: int = 0  # 0 = 不使用排序码本，保持原行为
    rank_mode_temperature: float = 1.0  # mode selector softmax 温度
    rank_diversity_weight: float = 0.0  # 码本多样性正则权重
    rank_adjustment_scale: float = 1.0  # support-derived adjustment 缩放
    use_stochastic_rank: bool = False  # 是否对每个 episode 采样一个硬 mode
    eval_stochastic_mode_sample: bool = False  # eval 时按 mode selector 分布分类采样 mode
    rank_gumbel_scale: float = 0.0  # rank scores 上的 Gumbel 扰动幅度
    rank_score_noise_std: float = 0.0  # eval 时对 rank scores 加 per-subject 高斯噪声（标准差）
    rank_score_noise_position_weight: float = 0.0  # rank score noise 位置加权幅度（0=不加权；>0 中间 rank 噪声更大）
    rank_distribution: str = "gaussian_score"  # "gaussian_score" | "mallows" | "plackett_luce"
    mallows_phi: float = 0.0  # Mallows 分布集中度（越大越集中在中心排列）
    plackett_luce_scale: float = 1.0  # Plackett-Luce 温度参数
    mallows_anchor_mode0: bool = True  # mode 0（真实排序）是否跳过 Mallows/PL 扰动
    mallows_reference: str = "mode"  # "mode" = 围绕被试选定的 mode；"true_rank" = 围绕真实排序
    mallows_mix_weight: float = 0.0  # 以概率把 Mallows 中心替换为真实排序（0=纯 mode，1=纯 true_rank）
    per_subject_position_precision: bool = False  # 位置加权噪声的幅度是否 per-subject 异质
    freeze_subject_embedding: bool = False  # 冻结 subject embedding 为随机
    support_consistency_weight: float = 0.0  # support-consistency loss 权重
    use_hard_ranking_eval: bool = False  # eval 时使用离散 hard ranking
    hard_ranking_noise_prob: float = 0.0  # hard ranking 后随机翻转 pairwise 决策的概率
    hard_ranking_noise_on_learned_pairs: bool = False  # 是否也对已学习的 support pairs 加噪声
    hard_ranking_noise_type: str = "independent"  # independent / distance_dependent / systematic
    hard_ranking_noise_distance_scale: float = 1.0  # distance-dependent 噪声的衰减尺度
    hard_ranking_learned_noise_prob: float | None = None  # 已学习 pair 的独立翻转概率；None 则与未学习 pair 相同
    diversity_use_kendall: bool = False  # diversity loss 使用 Kendall tau 而非余弦相似度

    # === LatentRank 概率化排序评估（替代 hard ranking）===
    use_probabilistic_ranking_eval: bool = False  # eval 时使用连续 rank scores + BTL 逐 trial 采样
    pairwise_choice_temperature: float | None = None  # BTL pairwise 选择温度；None 使用模型学到的 temperature
    block_level_rank_noise_std: float = 0.0  # 每个 test block 初对 rank scores 加高斯漂移
    probabilistic_pair_flip_prob: float = 0.0  # probabilistic eval 中逐 trial 独立翻转 pairwise 决策的概率

    # === Phase 1：pair-dependent noise + 局部 block drift ===
    pairwise_distance_noise_alpha: float = 0.0  # pair-dependent temperature 强度（>0 时秩差小则温度高）
    pairwise_distance_noise_beta: float = 1.0  # pair-dependent temperature 衰减尺度
    local_drift_window_size: int = 0  # 局部 block drift 窗口大小；0=全局漂移

    # === Phase 4：support-phase 贝叶斯后验排序 ===
    use_bayesian_posterior_eval: bool = False  # eval 时用 support evidence 推断排序后验
    bayesian_posterior_phi: float = 1.0  # Mallows 先验集中度
    bayesian_posterior_tau: float = 0.3  # support pair 似然温度
    bayesian_samples_per_block: int = 1  # 每个 block 从后验采样几个排列（>1 时取平均）
    bayesian_max_hypotheses: int = 50000  # n=8 时覆盖全部 8! 排列
    support_observation_noise_prob: float = 0.0  # 每个被试对 support pair 的观测符号随机翻转概率

    # === 任务：是否每个 episode 随机打乱真实 ranking ===
    # True mirrors the participant-level rank counterbalancing in Liu et al.;
    # False keeps a fixed identity ranking for debugging/backward compatibility.
    randomize_true_rank: bool = False

    # === LatentRank 线性扩展码本机制 ===
    linear_extension_init: bool = False  # 用 support 偏序的线性扩展初始化码本
    anchor_true_rank: bool = False  # mode 0 固定为真实排序
    anchor_target_prob: float = 0.07  # 期望选择 mode 0 的被试比例
    mode0_bias: float = 0.0  # mode 0 logit 偏置（提高正确排序被试比例）
    mode_usage_weight: float = 0.0  # mode usage KL loss 权重
    mode_entropy_weight: float = 0.0  # mode 使用熵正则权重
    support_consistency_margin: float = 0.1  # support consistency margin
    project_to_support: bool = False  # 每个优化步后是否投影码本到 support 约束
    asymmetric_diversity: bool = False  # diversity loss 是否排除 mode 0
    freeze_codebook: bool = False  # 冻结整个 rank_codebook
    freeze_mode_selector: bool = False  # 冻结 mode selector

    # === 个体差异来源 ===
    subject_embedding_dim: int = 0
    pw_init_std: float = 0.0
    choice_temperature_heterogeneity: float = 0.0
    encoding_noise_std: float = 0.0

    # === 过程模型改造：mode selector 与 support evidence 约束 ===
    mode_selector_input: str = "context_only"  # "context_only" | "context_plus_subject_prior" | "context_gated_subject"
    subject_embedding_for_mode_selector: bool = False
    selected_mode_support_consistency_weight: float = 0.0
    mode_subject_independence_weight: float = 0.0
    use_sequential_rank_update: bool = False
    rank_update_cell_hidden_dim: int = 64
    learn_rank_update_noise: bool = False
    rank_update_noise_init: float = 0.01
    rank_update_local_mask: bool = False  # RankUpdateCell 只更新被观测 cue

    # === 过程模型改造：pairwise preference accumulator + HodgeRank ===
    use_pairwise_preference_accumulator: bool = False
    pairwise_preference_hidden_dim: int = 64
    hodgerank_eps: float = 0.01
    pairwise_preference_nonnegative: bool = False  # MLP 输出是否强制非负（保留 evidence 方向）
    project_rank_to_support: bool = False          # HodgeRank 输出后是否投影到 support 约束
    pairwise_preference_fixed_weight: float = 0.0  # >0 时忽略 MLP，固定每个 trial 的 preference 幅度
    analytic_pairwise_control: bool = False  # 显式无参数 Hodge 对照；不得作为训练模型汇报

    # === 过程模型诊断：组件 lesion / 干预开关 ===
    lesion_mode_selector_uniform: bool = False      # 强制 mode selector 输出均匀分布
    lesion_single_mode: bool = False                # 强制只使用 mode 0（真实排序）
    lesion_retro_plasticity: bool = False           # 关闭 RetroModulRNN 的 pw 更新
    lesion_pairwise_accumulator_off: bool = False   # 关闭 pairwise preference accumulator
    lesion_sequential_update_off: bool = False      # 保留模块，仅切断 sequential update 路径
    zero_subject_embedding: bool = False            # 把 subject embedding 置零
    lesion_reset_pw_after_support: bool = False     # support phase 后重置 plastic weights

    # === RetroLatentRank 混合模型专属 ===
    retro_context_dim: int | None = None  # RNN hidden -> context 维度；None 默认 hs//2
    retro_detach_per_trial: bool = True   # 每 trial 后 detach et/pw
    retro_support_aux_weight: float = 0.0  # support-phase teacher-sign 辅助损失权重
    retro_rnn_lr_scale: float = 1.0       # RNN 参数学习率缩放
    retro_freeze_rnn: bool = False        # 冻结 RNN，只训练 projection + LatentRank

    # === Meta-train ===
    nbiter: int = 30000
    lr: float = 1e-4
    l2: float = 0.0
    eps: float = 1e-6
    gc: float = 2.0
    warmup: int = 100

    # === 辅助损失 ===
    baux_learn: float = 0.0
    baux_test: float = 0.0  # 默认 0，与人类测试阶段无反馈一致

    # === 日志与保存 ===
    pe: int = 101
    save_every: int = 200
    rngseed: int = 42

    # === 运行时 ===
    output_dir: str = "./outputs/liu2026"
    num_eval_episodes: int = 50
    num_eval_seeds: int = 5
    allow_partial_checkpoint: bool = False

    def validate(self, mode: str | None = None) -> None:
        """Reject configurations whose declared semantics do not match execution."""
        errors: list[str] = []
        normalized_support_pairs = []
        for pair in self.support_pairs:
            if len(pair) != 2:
                errors.append(f"support pair must contain two positions: {pair!r}")
                continue
            i, j = int(pair[0]), int(pair[1])
            if i == j or not (0 <= i < self.nbcues and 0 <= j < self.nbcues):
                errors.append(
                    f"support pair must contain distinct weak-to-strong positions "
                    f"inside 0..{self.nbcues - 1}: {pair!r}"
                )
            normalized_support_pairs.append((min(i, j), max(i, j)))
        if len(set(normalized_support_pairs)) != len(normalized_support_pairs):
            errors.append("support_pairs contains duplicate unordered pairs")
        if self.support_blocks <= 0:
            errors.append("support_blocks must be positive")
        if self.query_blocks <= 0:
            errors.append("query_blocks must be positive")
        if self.query_feedback:
            errors.append("query_feedback=True is unsupported by the Liu 2026 runner")
        if (
            self.use_meta_sign_runner
            and self.meta_sign_reset_support_trial_state
            and self.support_triallen < 3
        ):
            errors.append(
                "trial-reset MetaSign requires support_triallen >= 3 so the "
                "pair can form hidden state and eligibility before the write"
            )
        if self.baux_learn != 0.0:
            errors.append(
                "baux_learn is not implemented by the Liu 2026 runner; "
                "use retro_support_aux_weight for the explicit support objective"
            )
        if self.use_latent_rank:
            errors.append(
                "use_latent_rank is a removed legacy switch; select model_type instead"
            )
        if self.latent_rank_dim != self.nbcues:
            errors.append(
                "latent_rank_dim must equal nbcues because rank scores are item-indexed"
            )
        if self.rank_temperature <= 0:
            errors.append("rank_temperature must be positive")
        if self.randomize_true_rank and self.allow_ground_truth_rank_input:
            errors.append(
                "allow_ground_truth_rank_input leaks the randomized full answer into the model"
            )
        if self.randomize_true_rank and self.anchor_true_rank:
            errors.append(
                "anchor_true_rank is incompatible with randomized ranks: mapping the "
                "anchor to cues would reveal the full answer"
            )
        if self.randomize_true_rank and self.linear_extension_init:
            errors.append(
                "linear_extension_init is positional and cannot be mapped to randomized "
                "cue identities without the full answer"
            )
        if self.mallows_reference == "true_rank" or self.mallows_mix_weight > 0:
            errors.append(
                "true-rank-centered Mallows evaluation is an answer oracle"
            )
        if self.lesion_reset_pw_after_support and self.model_type == ModelType.RETRO_LATENT_RANK:
            errors.append(
                "lesion_reset_pw_after_support is undefined for RetroLatentRank; "
                "use lesion_retro_plasticity"
            )
        if self.analytic_pairwise_control:
            required = {
                "use_pairwise_preference_accumulator": self.use_pairwise_preference_accumulator,
                "pairwise_preference_fixed_weight > 0": self.pairwise_preference_fixed_weight > 0,
                "rank_adjustment_scale == 0": self.rank_adjustment_scale == 0,
                "num_rank_modes <= 1": self.num_rank_modes <= 1,
                "subject_embedding_dim == 0": self.subject_embedding_dim == 0,
                "rank_score_noise_std == 0": self.rank_score_noise_std == 0,
                "rank_score_noise_position_weight == 0": self.rank_score_noise_position_weight == 0,
                "retro_support_aux_weight == 0": self.retro_support_aux_weight == 0,
                "baux_test == 0": self.baux_test == 0,
                "pairwise_choice_temperature is set": self.pairwise_choice_temperature is not None,
            }
            failed = [name for name, ok in required.items() if not ok]
            if failed:
                errors.append(
                    "analytic_pairwise_control violates: " + ", ".join(failed)
                )
            if any(
                (
                    self.use_bayesian_posterior_eval,
                    self.use_hard_ranking_eval,
                    self.use_probabilistic_ranking_eval,
                    self.rank_gumbel_scale > 0,
                    self.block_level_rank_noise_std > 0,
                    self.probabilistic_pair_flip_prob > 0,
                )
            ):
                errors.append(
                    "analytic_pairwise_control cannot include stochastic/legacy eval heads"
                )
            if mode in {"train", "full"}:
                errors.append(
                    "analytic_pairwise_control has no trainable behavioral path; use analysis/eval"
                )
        if errors:
            raise ValueError("Invalid Liu2026Config: " + "; ".join(errors))

    @property
    def num_support_trials(self):
        return self.support_blocks * len(self.support_pairs)

    @property
    def num_query_trials(self):
        return self.query_blocks * (self.nbcues * (self.nbcues - 1) // 2)

    @property
    def total_trials_per_episode(self):
        return self.num_support_trials + self.num_query_trials

    @property
    def eplen(self):
        return (
            self.num_support_trials * self.support_triallen
            + self.num_query_trials * self.query_triallen
        )

    @property
    def outputsize(self):
        return 2

    @property
    def nbstimbits(self):
        return 2 * self.cs + 1

    @property
    def inputsize(self):
        from fsrl.task.constants import ADDINPUT

        return self.nbstimbits + ADDINPUT + self.outputsize

    def to_model_dict(self):
        """Return the dict shape expected by RetroModulRNN."""
        return {
            "rngseed": self.rngseed,
            "hs": self.hs,
            "bs": self.bs,
            "outputsize": self.outputsize,
            "inputsize": self.inputsize,
            "pw_init_std": self.pw_init_std,
            "use_plasticity": self.use_plasticity,
            "subject_embedding_dim": self.subject_embedding_dim,
            "lesion_retro_plasticity": self.lesion_retro_plasticity,
            "zero_subject_embedding": self.zero_subject_embedding,
        }


@dataclass
class TrainConfig:
    rngseed: int = -1
    rew: float = 1.0
    wp: float = 0.0
    bent: float = 0.1
    blossv: float = 0.1
    gr: float = 0.9
    hs: int = 200
    bs: int = 32
    gc: float = 2.0
    eps: float = 1e-6
    nbiter: int = 5000
    save_every: int = 100
    pe: int = 101
    cs: int = 15
    learn_triallen: int = 2
    test_triallen: int = 4
    nb_learn_blocks: int = 4
    nb_test_blocks: int = 10
    supervision_size: int = 8
    testlmult: float = 3.0
    l2: float = 0.0
    lr: float = 1e-4
    lpw: float = 1e-4
    nbcues: int = 8
    analyze_on_save: bool = False
    learn_shuffle_presentation: bool = True
    baux_learn: float = 0.5
    baux_test: float = 1.0
    warmup: int = 100
    pw_init_std: float = 0.0
    persistent_pw: bool = False
    pw_episode_jitter: float = 0.0
    use_constructive_rank: bool = False
    construct_rank_lr: float = 0.3
    construct_rank_init_std: float = 0.1
    construct_rank_mix: float = 1.0
    construct_rank_temperature: float = 1.0
    construct_rank_fixed_pair_rng: bool = False
    rank_episode_jitter: float = 0.0
    construct_rank_update_noise: float = 0.0
    persistent_rank: bool = True
    supervision_drop: int = 0

    @property
    def nbcuesrange(self):
        return range(self.nbcues, self.nbcues + 1)

    @property
    def effective_supervision_size(self) -> int:
        return max(0, self.supervision_size - self.supervision_drop)

    @property
    def n_learn_trials(self):
        """每个 episode 的学习 trial 总数 = block 数 × |S_eff|。"""
        return self.nb_learn_blocks * self.effective_supervision_size

    @property
    def n_test_trials(self):
        """每个 episode 的测试 trial 总数 = block 数 × C(n,2)。"""
        return self.nb_test_blocks * (self.nbcues * (self.nbcues - 1) // 2)

    @property
    def nbtrials(self):
        """每个 episode 的总 trial 数。"""
        return self.n_learn_trials + self.n_test_trials

    @property
    def eplen(self):
        """每个 episode 的总时间步数（学习步 + 测试步）。"""
        return (
            self.n_learn_trials * self.learn_triallen
            + self.n_test_trials * self.test_triallen
        )

    @property
    def triallen(self):
        """兼容旧接口：默认返回测试 trial 步数。"""
        return self.test_triallen

    @property
    def nbtraintrials(self):
        """兼容旧接口：返回学习 trial 总数。"""
        return self.n_learn_trials

    @property
    def nbtesttrials(self):
        """兼容旧接口：返回测试 trial 总数。"""
        return self.n_test_trials

    @property
    def nbstimbits(self):
        return 2 * self.cs + 1

    @property
    def outputsize(self):
        return 2

    @property
    def inputsize(self):
        from fsrl.task.constants import ADDINPUT

        return self.nbstimbits + ADDINPUT + self.outputsize

    def to_model_dict(self):
        """Return the dict shape expected by the original model code."""
        return {
            "rngseed": self.rngseed,
            "rew": self.rew,
            "wp": self.wp,
            "bent": self.bent,
            "blossv": self.blossv,
            "gr": self.gr,
            "hs": self.hs,
            "bs": self.bs,
            "gc": self.gc,
            "eps": self.eps,
            "nbiter": self.nbiter,
            "save_every": self.save_every,
            "pe": self.pe,
            "nbcuesrange": self.nbcuesrange,
            "cs": self.cs,
            "triallen": self.test_triallen,
            "learn_triallen": self.learn_triallen,
            "test_triallen": self.test_triallen,
            "nb_learn_blocks": self.nb_learn_blocks,
            "nb_test_blocks": self.nb_test_blocks,
            "supervision_size": self.supervision_size,
            "nbtraintrials": self.nbtraintrials,
            "nbtesttrials": self.nbtesttrials,
            "nbtrials": self.nbtrials,
            "eplen": self.eplen,
            "testlmult": self.testlmult,
            "l2": self.l2,
            "lr": self.lr,
            "lpw": self.lpw,
            "outputsize": self.outputsize,
            "inputsize": self.inputsize,
        }


def apply_construct_preset(args):
    """Route C：建构性排序训练预设。"""
    if not args.construct_preset:
        return args
    args.use_constructive_rank = True
    args.persistent_pw = True
    args.baux_learn = 0.5
    args.baux_test = 0.0
    if args.pw_init_std == 0.0:
        args.pw_init_std = 0.05
    if args.construct_rank_init_std == 0.1:
        args.construct_rank_init_std = 0.15
    if args.rank_episode_jitter == 0.0:
        args.rank_episode_jitter = 0.02
    if args.construct_rank_update_noise == 0.0:
        args.construct_rank_update_noise = 0.03
    return args


def apply_combo_v2_preset(args):
    """A 持久 rank + B baux_test + C test_blocks=20 + D 温度+固定 pair RNG。"""
    if not getattr(args, "combo_v2_preset", False):
        return args
    args.use_constructive_rank = True
    args.persistent_pw = True
    args.no_persistent_rank = False
    args.baux_learn = 0.5
    args.baux_test = 0.3
    if args.test_blocks == 10:
        args.test_blocks = 20
    if args.supervision_drop == 0:
        args.supervision_drop = 1
    if args.pw_init_std == 0.0:
        args.pw_init_std = 0.1
    if args.construct_rank_init_std == 0.1:
        args.construct_rank_init_std = 0.5
    if args.construct_rank_lr == 0.3:
        args.construct_rank_lr = 0.1
    if args.construct_rank_mix == 1.0:
        args.construct_rank_mix = 0.85
    if args.rank_episode_jitter == 0.0:
        args.rank_episode_jitter = 0.02
    if args.construct_rank_update_noise == 0.0:
        args.construct_rank_update_noise = 0.03
    if args.construct_rank_temperature == 1.0:
        args.construct_rank_temperature = 0.5
    if not args.no_construct_rank_fixed_pair_rng:
        args.construct_rank_fixed_pair_rng = True
    return args


def apply_combo_v3_preset(args):
    """v2 基础上加强个体分化：更大 rank 初值、drop=2、纯 rank 决策、pw/rank 微扰。"""
    if not getattr(args, "combo_v3_preset", False):
        return args
    if not getattr(args, "combo_v2_preset", False):
        args.combo_v2_preset = True
        apply_combo_v2_preset(args)
    args.supervision_drop = 2
    args.construct_rank_init_std = 0.8
    args.construct_rank_lr = 0.08
    args.construct_rank_mix = 1.0
    args.rank_episode_jitter = 0.03
    if args.pw_episode_jitter == 0.0:
        args.pw_episode_jitter = 0.02
    return args


def apply_liu_minimal_preset(args):
    """应用 Liu 2026 最小对齐预设（仅填充用户未显式指定的项）。"""
    if not args.liu_minimal:
        return args
    args.baux_learn = 0.0
    args.baux_test = 0.0
    args.persistent_pw = True
    if args.pw_init_std == 0.0:
        args.pw_init_std = 0.08
    return args
