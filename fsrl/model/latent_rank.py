"""Latent-rank meta-learner：从 support set 推断全局排序潜变量。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from fsrl.device import DEVICE
from fsrl.utils.linear_extensions import generate_linear_extensions


class RankUpdateCell(nn.Module):
    """Trial-level sequential rank update cell。

    输入：
      - r_t: 当前 rank belief [bs, nbcues]
      - obs_vec: support observation 的 one-hot 编码 [bs, nbcues]
                 其中 cue_i 位置为 +teacher_sign，cue_j 位置为 -teacher_sign
      - context: 当前 trial 的上下文 [bs, context_dim]（可选）
    输出：
      - delta_r: rank belief 更新量 [bs, nbcues]
    """

    def __init__(self, nbcues, context_dim, hidden_dim, local_mask=False):
        super().__init__()
        self.nbcues = nbcues
        self.local_mask = local_mask
        input_dim = nbcues + nbcues + context_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nbcues),
        )
        # 可学习的更新门：控制新观测对旧 belief 的覆盖程度
        self.update_gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nbcues),
            nn.Sigmoid(),
        )

    def encode_observation(self, cue_i, cue_j, teacher_sign, bs, device):
        """把 (cue_i, cue_j, teacher_sign) 编码为 [bs, nbcues] 向量。"""
        if isinstance(cue_i, int):
            cue_i = torch.full((bs,), cue_i, dtype=torch.long, device=device)
        if isinstance(cue_j, int):
            cue_j = torch.full((bs,), cue_j, dtype=torch.long, device=device)
        if isinstance(teacher_sign, (int, float)):
            teacher_sign = torch.full(
                (bs,), float(teacher_sign), dtype=torch.float32, device=device
            )
        obs = torch.zeros(bs, self.nbcues, device=device)
        # teacher_sign=+1 表示 cue_i 强于 cue_j：提高 i，降低 j
        obs.scatter_add_(1, cue_i.unsqueeze(1), teacher_sign.unsqueeze(1))
        obs.scatter_add_(1, cue_j.unsqueeze(1), (-teacher_sign).unsqueeze(1))
        return obs

    def forward(self, r_t, cue_i, cue_j, teacher_sign, context=None):
        bs = r_t.shape[0]
        device = r_t.device
        obs_vec = self.encode_observation(cue_i, cue_j, teacher_sign, bs, device)
        inputs = [r_t, obs_vec]
        if context is not None:
            inputs.append(context)
        x = torch.cat(inputs, dim=-1)
        delta = self.mlp(x)
        gate = self.update_gate(x)
        delta = gate * delta
        if self.local_mask:
            # 过程模型归纳偏置：每个 support trial 只应局部更新被观测到的两个 cue
            mask = torch.zeros_like(delta)
            if isinstance(cue_i, int):
                cue_i = torch.full((bs,), cue_i, dtype=torch.long, device=device)
            if isinstance(cue_j, int):
                cue_j = torch.full((bs,), cue_j, dtype=torch.long, device=device)
            mask.scatter_(1, cue_i.unsqueeze(1), 1.0)
            mask.scatter_(1, cue_j.unsqueeze(1), 1.0)
            delta = delta * mask
        return delta


class PairwisePreferenceAccumulator(nn.Module):
    """Pairwise preference accumulator + HodgeRank head。

    输入：
      - trial_encodings: [num_trials, bs, context_dim]
      - support_observations: list of (cue_i, cue_j, teacher_sign)
    输出：
      - rank_scores: [bs, nbcues]

    机制：
      1. 初始化全零 skew-symmetric preference matrix P [bs, nbcues, nbcues]。
      2. 每个 support trial 用 RNN/context 门控一个标量偏好强度 p，
         更新 P[:, i, j] += p，P[:, j, i] -= p。
      3. 用 HodgeRank（图 Laplacian 最小二乘）从 P 恢复连续 rank scores：
         solve (L + eps I) s = b，其中 b_i = sum_j P_{ij}。
    """

    def __init__(self, nbcues, context_dim, hidden_dim, eps=0.01, nonnegative=False, fixed_weight=0.0):
        super().__init__()
        self.nbcues = nbcues
        self.eps = eps
        self.nonnegative = nonnegative
        self.fixed_weight = fixed_weight
        # 输入：trial context + observation one-hot
        input_dim = context_dim + nbcues
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_observation(self, cue_i, cue_j, teacher_sign, bs, device):
        """把 (cue_i, cue_j, teacher_sign) 编码为 [bs, nbcues] 向量。"""
        if isinstance(cue_i, int):
            cue_i = torch.full((bs,), cue_i, dtype=torch.long, device=device)
        if isinstance(cue_j, int):
            cue_j = torch.full((bs,), cue_j, dtype=torch.long, device=device)
        if isinstance(teacher_sign, (int, float)):
            teacher_sign = torch.full(
                (bs,), float(teacher_sign), dtype=torch.float32, device=device
            )
        obs = torch.zeros(bs, self.nbcues, device=device)
        obs.scatter_add_(1, cue_i.unsqueeze(1), teacher_sign.unsqueeze(1))
        obs.scatter_add_(1, cue_j.unsqueeze(1), (-teacher_sign).unsqueeze(1))
        return obs

    def forward(self, trial_encodings, support_observations):
        """
        Args:
            trial_encodings: [num_trials, bs, context_dim]
            support_observations: list of (cue_i, cue_j, teacher_sign)
        Returns:
            rank_scores: [bs, nbcues]
        """
        num_trials, bs, context_dim = trial_encodings.shape
        device = trial_encodings.device
        n = self.nbcues

        P = torch.zeros(bs, n, n, device=device)
        edge_weight = torch.zeros(bs, n, n, device=device)

        for t, (cue_i, cue_j, teacher_sign) in enumerate(support_observations):
            context = trial_encodings[t]
            obs_vec = self.encode_observation(cue_i, cue_j, teacher_sign, bs, device)
            x = torch.cat([context, obs_vec], dim=-1)
            if isinstance(teacher_sign, (int, float)):
                sign = torch.full(
                    (bs,), float(teacher_sign), dtype=torch.float32, device=device
                )
            else:
                sign = teacher_sign.to(device=device, dtype=torch.float32)
            observed = sign.ne(0.0)

            if self.fixed_weight > 0.0:
                # 固定幅度：每个 support trial 贡献等权 evidence
                p = sign * self.fixed_weight
            else:
                raw = self.mlp(x).squeeze(-1)  # [bs]
                if self.nonnegative:
                    # 强制偏好强度非负，evidence 方向由 teacher_sign 显式决定
                    p = sign * F.softplus(raw)
                else:
                    p = raw * observed.to(raw.dtype)
            if isinstance(cue_i, int):
                cue_i_t = torch.full((bs,), cue_i, dtype=torch.long, device=device)
            else:
                cue_i_t = cue_i
            if isinstance(cue_j, int):
                cue_j_t = torch.full((bs,), cue_j, dtype=torch.long, device=device)
            else:
                cue_j_t = cue_j
            P[torch.arange(bs, device=device), cue_i_t, cue_j_t] += p
            P[torch.arange(bs, device=device), cue_j_t, cue_i_t] -= p
            observed_weight = observed.to(P.dtype)
            edge_weight[torch.arange(bs, device=device), cue_i_t, cue_j_t] += observed_weight
            edge_weight[torch.arange(bs, device=device), cue_j_t, cue_i_t] += observed_weight

        # Weighted HodgeRank: repeated observations increase both evidence and
        # graph weight, so identical replays improve precision without
        # spuriously multiplying the inferred score scale.
        degree = edge_weight.sum(dim=-1)  # [bs, n]
        L = -edge_weight + torch.diag_embed(degree)  # [bs, n, n]
        nonzero_edges = edge_weight.gt(0.0).sum(dim=(-2, -1)).clamp(min=1)
        mean_edge_weight = edge_weight.sum(dim=(-2, -1)) / nonzero_edges
        mean_edge_weight = mean_edge_weight.clamp(min=1.0)
        L_reg = L + (
            self.eps
            * mean_edge_weight[:, None, None]
            * torch.eye(n, device=device).unsqueeze(0)
        )
        b = P.sum(dim=-1)  # [bs, n]
        # s: [bs, n, 1]
        try:
            s = torch.linalg.solve(L_reg, b.unsqueeze(-1))
        except RuntimeError:
            # 万一病态，退化为最小二乘
            s = torch.linalg.lstsq(L_reg, b.unsqueeze(-1)).solution
        return s.squeeze(-1)


class LatentRankMetaLearner(nn.Module):
    """从 support set 推断全局排序潜变量，并基于 rank scores 做 pairwise 决策。

    关键设计:
      1. SupportSetEncoder: 将单个 support trial 编码为 context vector
      2. SupportSetAggregator: 将所有 support trials 聚合成 context
      3. RankInference: 从 context (+ subject embedding) 推断 nbcues 个排序分数
      4. 决策网络: 基于 rank scores 做 pairwise choice

    新增（Liu 2026 对齐）:
      - 排序码本 (rank codebook)：多个可学习的全局排序模式
      - mode selector：由 subject embedding 选择/采样一个模式
      - diversity loss：鼓励码本中的模式彼此不同
      - stochastic rank：每个 episode 硬采样一个模式（Gumbel-STE）
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hs = config.hs
        self.bs = config.bs
        self.nbcues = config.nbcues

        # === 编码器: 从单个 support trial 提取特征 ===
        self.trial_encoder = nn.Sequential(
            nn.Linear(config.cs * 2, config.hs),
            nn.ReLU(),
            nn.Linear(config.hs, config.hs // 2),
        )

        # === Support Set 聚合器 ===
        self.context_dim = config.hs // 2
        self.support_attention = nn.MultiheadAttention(
            embed_dim=self.context_dim,
            num_heads=4,
            batch_first=True,
        )

        # === Subject embedding（个体差异来源）===
        self.subject_embedding_dim = getattr(config, "subject_embedding_dim", 0)
        self.freeze_subject_embedding = getattr(config, "freeze_subject_embedding", False)
        if self.subject_embedding_dim > 0:
            self.subject_embedding = nn.Embedding(
                config.bs, self.subject_embedding_dim
            )
            if self.freeze_subject_embedding:
                self.subject_embedding.weight.requires_grad = False

        # === 排序码本（可选）===
        self.num_rank_modes = getattr(config, "num_rank_modes", 0)
        self.rank_mode_temperature = getattr(config, "rank_mode_temperature", 1.0)
        self.rank_diversity_weight = getattr(config, "rank_diversity_weight", 0.0)
        self.rank_adjustment_scale = getattr(config, "rank_adjustment_scale", 1.0)
        self.use_stochastic_rank = getattr(config, "use_stochastic_rank", False)
        self.eval_stochastic_mode_sample = getattr(
            config, "eval_stochastic_mode_sample", False
        )
        self.rank_gumbel_scale = getattr(config, "rank_gumbel_scale", 0.0)
        self.rank_score_noise_std = getattr(config, "rank_score_noise_std", 0.0)
        self.rank_score_noise_position_weight = getattr(
            config, "rank_score_noise_position_weight", 0.0
        )
        self.per_subject_position_precision = getattr(
            config, "per_subject_position_precision", False
        )
        self.rank_distribution = getattr(config, "rank_distribution", "gaussian_score")
        self.mallows_phi = getattr(config, "mallows_phi", 0.0)
        self.plackett_luce_scale = getattr(config, "plackett_luce_scale", 1.0)
        self.mallows_anchor_mode0 = getattr(config, "mallows_anchor_mode0", True)
        self.mallows_reference = getattr(config, "mallows_reference", "mode")
        self.mallows_mix_weight = getattr(config, "mallows_mix_weight", 0.0)
        self.support_consistency_weight = getattr(
            config, "support_consistency_weight", 0.0
        )
        self.use_hard_ranking_eval = getattr(config, "use_hard_ranking_eval", False)
        self.hard_ranking_noise_prob = getattr(config, "hard_ranking_noise_prob", 0.0)
        self.hard_ranking_noise_on_learned_pairs = getattr(
            config, "hard_ranking_noise_on_learned_pairs", False
        )
        self.hard_ranking_noise_type = getattr(
            config, "hard_ranking_noise_type", "independent"
        )
        self.hard_ranking_noise_distance_scale = getattr(
            config, "hard_ranking_noise_distance_scale", 1.0
        )
        self.hard_ranking_learned_noise_prob = getattr(
            config, "hard_ranking_learned_noise_prob", None
        )
        self.diversity_use_kendall = getattr(config, "diversity_use_kendall", False)

        # 概率化排序评估（替代 hard ranking）
        self.use_probabilistic_ranking_eval = getattr(
            config, "use_probabilistic_ranking_eval", False
        )
        self.pairwise_choice_temperature = getattr(
            config, "pairwise_choice_temperature", None
        )
        self.block_level_rank_noise_std = getattr(
            config, "block_level_rank_noise_std", 0.0
        )
        self.probabilistic_pair_flip_prob = getattr(
            config, "probabilistic_pair_flip_prob", 0.0
        )
        self.pairwise_distance_noise_alpha = getattr(
            config, "pairwise_distance_noise_alpha", 0.0
        )
        self.pairwise_distance_noise_beta = getattr(
            config, "pairwise_distance_noise_beta", 1.0
        )
        self.local_drift_window_size = getattr(config, "local_drift_window_size", 0)

        # === Phase 4：support-phase 贝叶斯后验排序 ===
        self.use_bayesian_posterior_eval = getattr(
            config, "use_bayesian_posterior_eval", False
        )
        self.bayesian_posterior_phi = getattr(config, "bayesian_posterior_phi", 1.0)
        self.bayesian_posterior_tau = getattr(config, "bayesian_posterior_tau", 0.3)
        self.bayesian_samples_per_block = getattr(
            config, "bayesian_samples_per_block", 1
        )

        # 线性扩展码本机制
        self.linear_extension_init = getattr(config, "linear_extension_init", False)
        self.anchor_true_rank = getattr(config, "anchor_true_rank", False)
        self.anchor_target_prob = getattr(config, "anchor_target_prob", 0.07)
        self.mode0_bias = getattr(config, "mode0_bias", 0.0)
        self.mode_usage_weight = getattr(config, "mode_usage_weight", 0.0)
        self.mode_entropy_weight = getattr(config, "mode_entropy_weight", 0.0)
        self.support_consistency_margin = getattr(
            config, "support_consistency_margin", 0.1
        )
        self.project_to_support = getattr(config, "project_to_support", False)
        self.asymmetric_diversity = getattr(config, "asymmetric_diversity", False)
        self.freeze_codebook = getattr(config, "freeze_codebook", False)
        self.freeze_mode_selector = getattr(config, "freeze_mode_selector", False)

        # === 过程模型改造：mode selector 与 support evidence 约束 ===
        self.mode_selector_input = getattr(config, "mode_selector_input", "context_only")
        self.subject_embedding_for_mode_selector = getattr(
            config, "subject_embedding_for_mode_selector", False
        )
        self.selected_mode_support_consistency_weight = getattr(
            config, "selected_mode_support_consistency_weight", 0.0
        )
        self.mode_subject_independence_weight = getattr(
            config, "mode_subject_independence_weight", 0.0
        )
        self.use_sequential_rank_update = getattr(
            config, "use_sequential_rank_update", False
        )
        self.rank_update_cell_hidden_dim = getattr(
            config, "rank_update_cell_hidden_dim", 64
        )
        self.learn_rank_update_noise = getattr(config, "learn_rank_update_noise", False)
        self.rank_update_noise_init = getattr(config, "rank_update_noise_init", 0.01)

        # 缓存 support pairs 用于 consistency loss 和 noise 掩码。
        # config uses Liu's weak-to-strong positions; legacy rank codebooks use
        # the repository's strong-to-weak rank indices.
        # 但 forward 中收到的 cue_pair 是 cue 对。这里保留旧语义：
        # support_pairs 用于 consistency/projection；
        # support_cue_pairs_set 在训练固定 ranking 时等于位置对，
        # 在随机 ranking 时应由调用方传入或在训练时动态设置。
        from fsrl.task.liu2026 import published_position_pair_to_rank_positions

        self.support_pairs = [
            tuple(sorted(published_position_pair_to_rank_positions(pair, self.nbcues)))
            for pair in config.support_pairs
        ]
        self.support_pairs_set = {
            (min(i, j), max(i, j)) for i, j in self.support_pairs
        }

        # === 遗留模式开关 ===
        # 当没有任何遗留组件被启用时，默认进入精简过程模型路径，
        # 不再实例化 codebook / mode selector / Mallows-PL 等模块。
        self.legacy_mode = getattr(config, "legacy_mode", None)
        if self.legacy_mode is None:
            self.legacy_mode = (
                self.num_rank_modes > 1
                or self.rank_distribution != "gaussian_score"
                or self.use_hard_ranking_eval
                or self.use_probabilistic_ranking_eval
                or self.use_bayesian_posterior_eval
                or self.project_to_support
                or self.linear_extension_init
                or self.anchor_true_rank
            )

        if self.num_rank_modes > 1 and self.legacy_mode:
            if self.linear_extension_init:
                # 用 support 偏序的线性扩展初始化码本。
                # 码本学习 strong-to-weak 位置排序；每个 episode 再通过
                # true_rank 映射为 cue 排序。
                from fsrl.utils.linear_extensions import generate_linear_extensions

                seed = getattr(config, "rngseed", 42)
                extensions = generate_linear_extensions(
                    self.nbcues,
                    self.support_pairs,
                    max_extensions=max(self.num_rank_modes, 128),
                    seed=seed,
                )
                init_scores = torch.zeros(
                    self.num_rank_modes, self.nbcues, device=DEVICE
                )
                for mode_idx, ext in enumerate(extensions[: self.num_rank_modes]):
                    for pos, position in enumerate(ext):
                        # ext 是从强到弱的位置排列；position 在该位置
                        init_scores[mode_idx, position] = self.nbcues - 1 - pos
            else:
                init_scores = torch.randn(
                    self.num_rank_modes, self.nbcues, device=DEVICE
                )

            # 每个 mode 是一组 nbcues 个排序分数
            self.rank_codebook = nn.Parameter(init_scores)

            # 用于锚定 mode 0 的梯度掩码；始终注册以保持旧 checkpoint 兼容性
            self.register_buffer(
                "_mode0_grad_mask",
                torch.ones(self.num_rank_modes, self.nbcues, device=DEVICE),
            )
            self.rank_codebook.register_hook(
                lambda grad: grad * self._mode0_grad_mask
            )

            if self.anchor_true_rank:
                # mode 0 固定为真实排序 [0,1,...,nbcues-1] 并冻结
                true_scores = torch.arange(
                    self.nbcues - 1, -1, -1, dtype=torch.float32, device=DEVICE
                )
                self.rank_codebook.data[0] = true_scores
                self._mode0_grad_mask[0] = 0.0

            # mode selector 输入：默认只用 context（过程模型目标）
            if self.mode_selector_input == "context_only" or (
                self.subject_embedding_dim > 0
                and not self.subject_embedding_for_mode_selector
            ):
                selector_input_dim = self.context_dim
            elif self.mode_selector_input == "context_plus_subject_prior":
                selector_input_dim = self.context_dim + self.subject_embedding_dim
            elif self.mode_selector_input == "context_gated_subject":
                selector_input_dim = self.context_dim
                # 门控网络：根据 context 决定 subject prior 的混入比例
                self._mode_subject_gate = nn.Sequential(
                    nn.Linear(self.context_dim + self.subject_embedding_dim, config.hs),
                    nn.ReLU(),
                    nn.Linear(config.hs, 1),
                    nn.Sigmoid(),
                )
                self._mode_subject_proj = nn.Linear(
                    self.subject_embedding_dim, self.context_dim
                )
            else:
                selector_input_dim = self.context_dim

            self.mode_selector = nn.Sequential(
                nn.Linear(selector_input_dim, config.hs),
                nn.ReLU(),
                nn.Linear(config.hs, self.num_rank_modes),
            )

            if self.freeze_codebook:
                self.rank_codebook.requires_grad = False
            if self.freeze_mode_selector:
                for p in self.mode_selector.parameters():
                    p.requires_grad = False

        # 缓存最近一次 mode logits，供 usage/entropy loss 使用
        self._last_mode_logits = None

        # Route B：当没有码本时，缓存最近一次 rank_scores 用于 support consistency loss
        self._last_rank_scores = None

        # 过程模型改造：trial-level sequential rank update
        self.rank_update_cell = None
        if self.use_sequential_rank_update:
            self.rank_update_cell = RankUpdateCell(
                self.nbcues,
                self.context_dim,
                self.rank_update_cell_hidden_dim,
                local_mask=getattr(self.config, "rank_update_local_mask", False),
            )
            if self.learn_rank_update_noise:
                import math

                init_log_std = (
                    math.log(self.rank_update_noise_init)
                    if self.rank_update_noise_init > 0
                    else -4.0
                )
                self.rank_update_noise_log_std = nn.Parameter(
                    torch.full((self.nbcues,), init_log_std, device=DEVICE)
                )

        # 过程模型改造：pairwise preference accumulator + HodgeRank
        self.use_pairwise_preference_accumulator = getattr(
            config, "use_pairwise_preference_accumulator", False
        )
        self.pairwise_preference_accumulator = None
        if self.use_pairwise_preference_accumulator:
            self.pairwise_preference_accumulator = PairwisePreferenceAccumulator(
                self.nbcues,
                self.context_dim,
                getattr(config, "pairwise_preference_hidden_dim", 64),
                eps=getattr(config, "hodgerank_eps", 0.01),
                nonnegative=getattr(
                    config, "pairwise_preference_nonnegative", False
                ),
                fixed_weight=getattr(
                    config, "pairwise_preference_fixed_weight", 0.0
                ),
            )

        # 过程模型改造：记录 mode-subject 联合分布，用于 independence loss
        if self.num_rank_modes > 1 and self.mode_subject_independence_weight > 0.0:
            self.register_buffer(
                "_mode_subject_joint_counts",
                torch.zeros(self.bs, self.num_rank_modes, device=DEVICE),
            )
        else:
            self._mode_subject_joint_counts = None

        # === 排列分布（Mallows / Plackett-Luce）===
        if self.rank_distribution != "gaussian_score" and self.legacy_mode:
            from fsrl.model.permutation_dist import PermutationDistribution

            self.permutation_dist = PermutationDistribution(
                self.nbcues,
                self.support_pairs,
                distribution=self.rank_distribution,
                max_extensions=getattr(config, "mallows_max_extensions", 10000),
                seed=getattr(config, "rngseed", 42),
            )

        # === per-subject 位置精度（P0 改进）===
        if (
            self.per_subject_position_precision
            and self.rank_score_noise_position_weight > 0.0
            and self.subject_embedding_dim > 0
        ):
            self.position_precision_mapper = nn.Sequential(
                nn.Linear(self.subject_embedding_dim, self.hs),
                nn.ReLU(),
                nn.Linear(self.hs, 1),
                nn.Softplus(),
            )

        # === Latent Rank 推断器 ===
        rank_input_dim = self.context_dim
        if self.subject_embedding_dim > 0:
            rank_input_dim = self.context_dim + self.subject_embedding_dim

        self.rank_inferencer = nn.Sequential(
            nn.Linear(rank_input_dim, config.hs),
            nn.ReLU(),
            nn.Linear(config.hs, config.nbcues),
        )

        # === 决策网络 ===
        self.decision_network = nn.Sequential(
            nn.Linear(config.cs * 2 + self.context_dim, config.hs),
            nn.ReLU(),
            nn.Linear(config.hs, 2),
        )

        # === 温度参数 ===
        import math

        initial_temperature = max(float(config.rank_temperature), 1e-6)
        raw_temperature = math.log(math.expm1(initial_temperature))
        self.register_parameter(
            "temperature",
            nn.Parameter(torch.tensor(raw_temperature, device=DEVICE)),
        )

    def encode_support_trial(self, cue_vec):
        """编码单个 support trial。

        Args:
            cue_vec: [bs, 2*cs]
        Returns:
            encoding: [bs, context_dim]
        """
        return self.trial_encoder(cue_vec)

    def aggregate_support_set(self, trial_encodings):
        """聚合所有 support trials 成 context vector。

        Args:
            trial_encodings: [num_support_trials, bs, context_dim]
        Returns:
            context: [bs, context_dim]
        """
        attn_out, _ = self.support_attention(
            trial_encodings, trial_encodings, trial_encodings
        )
        context = attn_out.mean(dim=0)
        return context

    def _sample_mode_weights(self, selector_input, training=False):
        """根据 selector_input 生成 mode 权重。

        - 默认（非 stochastic）: deterministic softmax
        - training + stochastic: Gumbel-max + STE 硬采样
        - eval + stochastic: deterministic argmax（保证同一被试每次使用同一 mode）
        """
        mode_logits = self.mode_selector(selector_input)  # [bs, num_modes]
        if self.mode0_bias != 0.0 and self.num_rank_modes > 1:
            mode_logits = mode_logits.clone()
            mode_logits[:, 0] = mode_logits[:, 0] + self.mode0_bias
        self._last_mode_logits = mode_logits

        if training:
            if not self.use_stochastic_rank:
                # 训练时确定性 soft selection
                return F.softmax(mode_logits / self.rank_mode_temperature, dim=-1)
            # Gumbel-max + STE：forward hard，backward through soft
            eps = 1e-10
            uniform = torch.rand_like(mode_logits).clamp(min=eps, max=1.0 - eps)
            gumbel = -torch.log(-torch.log(uniform) + eps)
            perturbed = mode_logits + gumbel
            soft = F.softmax(perturbed / self.rank_mode_temperature, dim=-1)
            hard_idx = perturbed.argmax(dim=-1)
            hard = F.one_hot(hard_idx, self.num_rank_modes).float()
            return hard - soft.detach() + soft

        # eval 路径
        if self.eval_stochastic_mode_sample or self.use_stochastic_rank:
            # eval 时按分布采样/argmax 一个硬 mode
            if self.eval_stochastic_mode_sample:
                probs = F.softmax(
                    mode_logits / self.rank_mode_temperature, dim=-1
                )
                hard_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                hard_idx = mode_logits.argmax(dim=-1)
            self._last_selected_mode = hard_idx
            return F.one_hot(hard_idx, self.num_rank_modes).float()

        # 默认 eval：确定性 soft selection
        return F.softmax(mode_logits / self.rank_mode_temperature, dim=-1)

    def _sequential_rank_update(
        self,
        r_0,
        trial_encodings,
        support_observations,
        training=False,
    ):
        """从初始 rank belief 出发，顺序应用每个 support observation。

        Args:
            r_0: [bs, nbcues] 初始 rank belief
            trial_encodings: [num_trials, bs, context_dim]
            support_observations: list of (cue_i, cue_j, teacher_sign)
            training: 是否训练模式
        Returns:
            r_T: [bs, nbcues]
        """
        r_t = r_0
        for t, (cue_i, cue_j, teacher_sign) in enumerate(support_observations):
            trial_context = trial_encodings[t]
            delta = self.rank_update_cell(r_t, cue_i, cue_j, teacher_sign, trial_context)
            if training and self.learn_rank_update_noise:
                noise_std = torch.exp(self.rank_update_noise_log_std).clamp(min=1e-4)
                noise = torch.randn_like(delta) * noise_std
                r_t = r_t + delta + noise
            else:
                r_t = r_t + delta
        return r_t

    def infer_latent_rank(
        self,
        context,
        subject_idx=None,
        training=False,
        true_rank=None,
        support_observations=None,
        trial_encodings=None,
    ):
        """从 context 推断每个 cue 的排序分数。

        Args:
            context: [bs, context_dim]
            subject_idx: [bs] optional subject embedding index
            training: 是否处于训练模式（影响 stochastic rank 路径）
            true_rank: [bs, nbcues] 或 [nbcues] optional，当前 episode 的真实 ranking。
                       如果提供，把位置分数按 true_rank 映射为 cue 分数。
            support_observations: list of (cue_i, cue_j, teacher_sign)，eval 时用于贝叶斯后验或 sequential update。
            trial_encodings: [num_trials, bs, context_dim]，用于 sequential rank update。
        Returns:
            rank_scores: [bs, nbcues]
        """
        # 拼接 subject embedding
        if subject_idx is not None and self.subject_embedding_dim > 0:
            subj_emb = self.subject_embedding(subject_idx)
            if getattr(self.config, "zero_subject_embedding", False):
                subj_emb = torch.zeros_like(subj_emb)
            context_with_subj = torch.cat([context, subj_emb], dim=-1)
        else:
            subj_emb = None
            context_with_subj = context

        # support-derived adjustment（残差）
        adjustment = self.rank_inferencer(context_with_subj)
        rank_scores = adjustment * self.rank_adjustment_scale

        # 排序码本 base rank（位置分数）
        base_rank_scores = None
        if self.num_rank_modes > 1:
            # 过程模型改造：mode selector 默认只依赖 context，
            # subject embedding 仅作为先验/门控（不决定 mode 身份）
            if self.mode_selector_input == "context_only" or (
                self.subject_embedding_dim > 0
                and not self.subject_embedding_for_mode_selector
            ):
                selector_input = context
            elif self.mode_selector_input == "context_plus_subject_prior" and subj_emb is not None:
                selector_input = torch.cat([context, subj_emb], dim=-1)
            elif self.mode_selector_input == "context_gated_subject" and subj_emb is not None:
                gate_input = torch.cat([context, subj_emb], dim=-1)
                gate = self._mode_subject_gate(gate_input)  # [bs, 1]
                subj_proj = self._mode_subject_proj(subj_emb)  # [bs, context_dim]
                selector_input = gate * context + (1.0 - gate) * subj_proj
            else:
                selector_input = context

            mode_weights = self._sample_mode_weights(selector_input, training=training)

            # 过程模型诊断：lesion mode selector / 强制单 mode
            if getattr(self.config, "lesion_mode_selector_uniform", False):
                mode_weights = torch.ones_like(mode_weights) / self.num_rank_modes
            if getattr(self.config, "lesion_single_mode", False):
                mode_weights = torch.zeros_like(mode_weights)
                mode_weights[:, 0] = 1.0

            # 过程模型改造：记录 mode-subject 联合分布
            if subject_idx is not None and self._mode_subject_joint_counts is not None:
                self._update_mode_subject_joint_counts(mode_weights, subject_idx)

            position_scores = mode_weights @ self.rank_codebook  # [bs, nbcues]

            # 如果提供 true_rank，把位置分数转换为 cue 分数
            if true_rank is not None:
                if true_rank.dim() == 1:
                    # 同一 true_rank 用于整个 batch
                    cue_scores = torch.zeros_like(position_scores)
                    for pos in range(self.nbcues):
                        cue = int(true_rank[pos].item())
                        cue_scores[:, cue] = position_scores[:, pos]
                else:
                    # per-sample true_rank [bs, nbcues]
                    cue_scores = torch.zeros_like(position_scores)
                    for b in range(true_rank.shape[0]):
                        for pos in range(self.nbcues):
                            cue = int(true_rank[b, pos].item())
                            cue_scores[b, cue] = position_scores[b, pos]
                base_rank_scores = cue_scores
                rank_scores = cue_scores + rank_scores
            else:
                base_rank_scores = position_scores
                rank_scores = position_scores + rank_scores

        # 过程模型改造：trial-level sequential rank update
        if (
            self.use_sequential_rank_update
            and self.rank_update_cell is not None
            and trial_encodings is not None
            and support_observations is not None
            and not getattr(self.config, "lesion_sequential_update_off", False)
        ):
            r_0 = base_rank_scores if base_rank_scores is not None else torch.zeros_like(rank_scores)
            r_t_final = self._sequential_rank_update(
                r_0, trial_encodings, support_observations, training=training
            )
            # sequential update 替换 base rank；保留 rank_inferencer 残差
            rank_scores = r_t_final + adjustment * self.rank_adjustment_scale

        # 过程模型改造：pairwise preference accumulator + HodgeRank
        if (
            self.use_pairwise_preference_accumulator
            and self.pairwise_preference_accumulator is not None
            and trial_encodings is not None
            and support_observations is not None
        ):
            if getattr(self.config, "lesion_pairwise_accumulator_off", False):
                # lesion：用零先验替代 HodgeRank，测试 accumulator 必要性
                hodge_scores = torch.zeros_like(rank_scores)
            else:
                hodge_scores = self.pairwise_preference_accumulator(
                    trial_encodings, support_observations
                )
                # 过程模型约束：HodgeRank 输出必须尊重 support evidence
                if getattr(self.config, "project_rank_to_support", False):
                    hodge_scores = self._project_scores_to_support(
                        hodge_scores, support_observations
                    )
            rank_scores = hodge_scores + adjustment * self.rank_adjustment_scale

        # Phase 4：eval 时从 support evidence 推断贝叶斯后验排序
        # 这会替代 codebook 后的 Mallows/高斯/block-drift 路径，直接生成 per-block 排序样本
        if (
            not training
            and self.use_bayesian_posterior_eval
            and support_observations is not None
        ):
            from fsrl.model.bayesian_rank_posterior import BayesianRankPosterior

            n_blocks = getattr(self.config, "query_blocks", 10)
            bs = rank_scores.shape[0]
            # 先验中心：使用当前 mode 的排列；无码本时默认用 true_rank
            center_perms = torch.argsort(-rank_scores, dim=1)

            # 当前选定的 mode（用于跳过 mode 0 的噪声）
            selected = None
            if self.num_rank_modes > 1:
                selected = mode_weights.argmax(dim=-1)

            # 扰动先验中心排序分数，使被试间先验更分散，从而降低被试间 Kendall tau
            if self.rank_score_noise_std > 0.0:
                noise = torch.randn_like(rank_scores) * self.rank_score_noise_std
                # mode 0（真实排序）保持无噪声，保留少数完全正确的被试
                if selected is not None:
                    noise[selected == 0] = 0.0
                rank_scores = rank_scores + noise
                center_perms = torch.argsort(-rank_scores, dim=1)

            posterior = BayesianRankPosterior(
                self.nbcues,
                self.support_pairs,
                max_extensions=getattr(
                    self.config, "bayesian_max_hypotheses", 50000
                ),
                seed=getattr(self.config, "rngseed", 42),
            )
            support_obs_noise_prob = getattr(
                self.config, "support_observation_noise_prob", 0.0
            )

            bayesian_scores = torch.zeros(
                n_blocks, bs, self.nbcues, device=DEVICE
            )
            for b in range(bs):
                pairs = []
                outcomes = []
                for cue_i, cue_j, sign in support_observations:
                    i_b = int(cue_i if isinstance(cue_i, int) else cue_i[b].item())
                    j_b = int(cue_j if isinstance(cue_j, int) else cue_j[b].item())
                    sign_b = float(
                        sign if isinstance(sign, (int, float)) else sign[b].item()
                    )
                    if sign_b != 0.0:
                        pairs.append((i_b, j_b))
                        outcomes.append(sign_b)
                subj_outcomes = torch.tensor(
                    outcomes, dtype=torch.float32, device=DEVICE
                )
                # 为每个被试的 support 观测加独立噪声：模拟编码/记忆对局部证据的扭曲
                if support_obs_noise_prob > 0.0 and pairs:
                    # mode 0（真实排序）的被试保持无噪声，保留少数完全正确的被试
                    if selected is None or selected[b] != 0:
                        flip = torch.rand(len(pairs), device=DEVICE) < support_obs_noise_prob
                        subj_outcomes = torch.where(
                            flip, -subj_outcomes, subj_outcomes
                        )
                posterior.compute_posterior(
                    center_perms[b],
                    self.bayesian_posterior_phi,
                    pairs,
                    subj_outcomes.tolist(),
                    self.bayesian_posterior_tau,
                )
                n_samples = n_blocks * max(1, self.bayesian_samples_per_block)
                perms = posterior.sample(n_samples)
                scores = posterior.to_scores(perms)
                if self.bayesian_samples_per_block > 1:
                    scores = scores.view(
                        n_blocks, self.bayesian_samples_per_block, self.nbcues
                    ).mean(dim=1)
                bayesian_scores[:, b, :] = scores

            self._eval_bayesian_scores = bayesian_scores
            return bayesian_scores[0].contiguous()

        # 可选：rank scores 上的 Gumbel 扰动
        if self.rank_gumbel_scale > 0.0 and (training or self.use_stochastic_rank):
            eps = 1e-10
            uniform = torch.rand_like(rank_scores).clamp(min=eps, max=1.0 - eps)
            gumbel = -torch.log(-torch.log(uniform) + eps)
            rank_scores = rank_scores + self.rank_gumbel_scale * gumbel

        # eval 时用排列分布采样一个扰动后的全局排序，替代或辅助高斯噪声
        if not training and self.rank_distribution != "gaussian_score":
            selected = getattr(self, "_last_selected_mode", None)
            skip = torch.zeros(rank_scores.shape[0], dtype=torch.bool, device=DEVICE)
            if self.mallows_anchor_mode0 and selected is not None:
                skip = selected == 0
            # 确定 Mallows / PL 的中心排列
            mallows_reference = getattr(self.config, "mallows_reference", "mode")
            # 先计算 mode 中心和真实排序中心
            mode_center = torch.argsort(-rank_scores, dim=1)
            if true_rank is not None:
                if true_rank.dim() == 1:
                    true_center = true_rank.long().unsqueeze(0).expand(
                        rank_scores.shape[0], -1
                    )
                else:
                    true_center = true_rank.long()
            else:
                true_center = torch.arange(
                    self.nbcues, device=DEVICE
                ).unsqueeze(0).expand(rank_scores.shape[0], -1)

            if mallows_reference == "true_rank":
                center_perms = true_center
            elif self.mallows_mix_weight > 0.0:
                # 混合参考：部分被试以真实排序为中心，部分以 mode 为中心
                mix = (
                    torch.rand(rank_scores.shape[0], 1, device=DEVICE)
                    < self.mallows_mix_weight
                )
                center_perms = torch.where(mix, true_center, mode_center)
            else:
                center_perms = mode_center
            parameter = (
                self.mallows_phi
                if self.rank_distribution == "mallows"
                else self.plackett_luce_scale
            )
            sampled_perms = self.permutation_dist.sample(center_perms, parameter)
            sampled_perms[skip] = center_perms[skip]
            rank_scores = self.permutation_dist.to_scores(sampled_perms)

        # eval 时加入 per-subject 高斯噪声，使同一 mode 的被试产生略有不同的 HodgeRank 重建
        # mode 0 固定为真实排序，对应少数完全正确的被试，因此跳过噪声以保持其排序一致
        if not training and self.rank_score_noise_std > 0.0:
            noise = torch.randn_like(rank_scores) * self.rank_score_noise_std
            selected = getattr(self, "_last_selected_mode", None)
            if selected is not None:
                noise[selected == 0] = 0.0
            # 位置加权：中间 rank 的 cue 噪声更大，两端（最强/最弱）噪声更小，
            # 从而使远距 pair 方向更稳定、近距 pair 更易翻转，降低 pair-level 双峰比例
            if self.rank_score_noise_position_weight > 0.0:
                positions = torch.argsort(torch.argsort(-rank_scores, dim=1), dim=1).float()
                n = self.nbcues
                center = (n - 1) / 2.0
                shared_weight = 1.0 - ((positions - center) / center).abs()
                if (
                    self.per_subject_position_precision
                    and self.subject_embedding_dim > 0
                    and subject_idx is not None
                    and subj_emb is not None
                ):
                    # Reuse the embedding prepared at the start of this method.
                    # This is essential for the zero_subject_embedding lesion:
                    # reloading the table here would silently restore the cut path.
                    amplitude = self.position_precision_mapper(subj_emb).squeeze(-1)
                    weight = (
                        1.0
                        + self.rank_score_noise_position_weight
                        * shared_weight
                        * amplitude.unsqueeze(-1)
                    )
                else:
                    weight = 1.0 + self.rank_score_noise_position_weight * shared_weight
                noise = noise * weight
            rank_scores = rank_scores + noise

        # 概率化评估：预生成每个 test block 的 rank-score 漂移
        self._eval_block_perturbations = None
        if (
            not training
            and self.use_probabilistic_ranking_eval
            and self.block_level_rank_noise_std > 0.0
        ):
            n_blocks = getattr(self.config, "query_blocks", 10)
            bs = rank_scores.shape[0]
            noise = torch.randn(n_blocks, bs, self.nbcues, device=DEVICE) * self.block_level_rank_noise_std

            if 0 < self.local_drift_window_size < self.nbcues:
                # 局部漂移：每个 block 随机选择一个连续 rank 窗口，只扰动窗口内的 cue
                positions = torch.argsort(torch.argsort(-rank_scores, dim=1), dim=1)
                mask = torch.zeros(n_blocks, bs, self.nbcues, dtype=torch.bool, device=DEVICE)
                max_start = self.nbcues - self.local_drift_window_size
                for blk in range(n_blocks):
                    start = torch.randint(0, max_start + 1, (1,), device=DEVICE).item()
                    in_window = (positions >= start) & (positions < start + self.local_drift_window_size)
                    mask[blk] = in_window
                noise = noise * mask.float()

            # mode 0（真实排序）的被试跳过漂移，保持少数完全正确被试
            selected = getattr(self, "_last_selected_mode", None)
            if selected is not None:
                noise[:, selected == 0, :] = 0.0
            self._eval_block_perturbations = noise

        # 若使用 systematic / distance-dependent 噪声，预生成 per-subject-pair 翻转 mask
        self._eval_pair_flip_mask = None
        self._eval_learned_flip_mask = None
        if not training and self.use_hard_ranking_eval:
            if (
                self.hard_ranking_noise_type != "independent"
                and self.hard_ranking_noise_prob > 0.0
            ):
                self._eval_pair_flip_mask = self._build_pair_flip_mask(
                    rank_scores, self.hard_ranking_noise_prob
                )
            learned_prob = self.hard_ranking_learned_noise_prob
            if learned_prob is None and self.hard_ranking_noise_on_learned_pairs:
                learned_prob = self.hard_ranking_noise_prob
            if learned_prob is not None and learned_prob > 0.0:
                self._eval_learned_flip_mask = self._build_pair_flip_mask(
                    rank_scores, learned_prob
                )

        # Route B：无码本时缓存当前 rank_scores，供 support consistency loss 使用
        if self.num_rank_modes <= 1:
            self._last_rank_scores = rank_scores

        return rank_scores

    def _hard_rank_scores(self, rank_scores):
        """把连续 rank scores 转换为 hard permutation 的 scores（eval 用）。

        Args:
            rank_scores: [bs, nbcues]
        Returns:
            hard_scores: [bs, nbcues] — 最强 cue 得分为 nbcues-1，最弱为 0
        """
        # argsort(-scores) 给出从强到弱的排列；再 argsort 得到每个 cue 的 rank
        ranks = torch.argsort(torch.argsort(-rank_scores, dim=1), dim=1)
        hard_scores = (self.nbcues - 1 - ranks).float()
        return hard_scores

    def _compute_rank_positions(self, rank_scores):
        """返回每个 cue 在排序中的位置（0=最强）。"""
        return torch.argsort(torch.argsort(-rank_scores, dim=1), dim=1)

    def _build_pair_flip_mask(self, rank_scores, noise_prob):
        """为每个被试、每个 cue pair 生成是否翻转的确定性 mask。

        仅在 eval 且非 independent 噪声模式下使用。翻转概率可随真实秩差衰减：
          p(d) = noise_prob * exp(-(d-1) / distance_scale)
        其中 d = |pos_i - pos_j|，相邻 pair d=1 时概率最大。
        """
        bs = rank_scores.shape[0]
        positions = self._compute_rank_positions(rank_scores)  # [bs, nbcues]
        mask = torch.zeros(bs, self.nbcues, self.nbcues, dtype=torch.bool, device=DEVICE)
        if noise_prob <= 0.0:
            return mask

        scale = max(self.hard_ranking_noise_distance_scale, 1e-3)
        idx_i, idx_j = torch.triu_indices(self.nbcues, self.nbcues, offset=1, device=DEVICE)
        pos_i = positions[:, idx_i]  # [bs, n_pairs]
        pos_j = positions[:, idx_j]
        d = (pos_i - pos_j).abs().float()  # [bs, n_pairs]
        prob = noise_prob * torch.exp(-(d - 1.0) / scale)
        prob = prob.clamp(max=0.5)

        # 为每个 (b, pair) 做一次性 Bernoulli 采样，保证同 pair 跨 block 一致
        uniform = torch.rand(bs, idx_i.shape[0], device=DEVICE)
        flip = uniform < prob
        mask[:, idx_i, idx_j] = flip
        mask[:, idx_j, idx_i] = flip
        return mask

    def compute_pairwise_preference(
        self, rank_scores, cue_i, cue_j, temperature=None
    ):
        """基于排序分数计算 pairwise 选择概率。

        P(choose cue_i over cue_j) = sigmoid((s_i - s_j) / temperature)

        Args:
            temperature: 若提供，覆盖模型学习到的 temperature；否则使用 self.temperature。
        """
        if isinstance(cue_i, int):
            s_i = rank_scores[:, cue_i]
            s_j = rank_scores[:, cue_j]
        else:
            s_i = rank_scores.gather(1, cue_i.unsqueeze(1)).squeeze(1)
            s_j = rank_scores.gather(1, cue_j.unsqueeze(1)).squeeze(1)

        if temperature is None:
            temp = F.softplus(self.temperature)
        else:
            temp = torch.tensor(temperature, device=DEVICE)

        # Phase 1：pair-dependent temperature，让秩差小的 pair 决策更随机
        if self.pairwise_distance_noise_alpha > 0.0:
            positions = torch.argsort(torch.argsort(-rank_scores, dim=1), dim=1)
            if isinstance(cue_i, int):
                pos_i = positions[:, cue_i]
                pos_j = positions[:, cue_j]
            else:
                pos_i = positions.gather(1, cue_i.unsqueeze(1)).squeeze(1)
                pos_j = positions.gather(1, cue_j.unsqueeze(1)).squeeze(1)
            d = (pos_i - pos_j).abs().float()
            scale = 1.0 + self.pairwise_distance_noise_alpha * torch.exp(
                -self.pairwise_distance_noise_beta * d
            )
            temp = temp * scale

        diff = (s_i - s_j) / temp
        prob = torch.sigmoid(diff)
        return prob

    def compute_rank_diversity_loss(self):
        """鼓励码本中的排序模式多样化。

        支持两种度量：
          - 默认：rank-difference 余弦相似度
          - diversity_use_kendall=True：pairwise preference 一致性（Kendall tau 代理）

        若 ``asymmetric_diversity=True``，则排除 mode 0，只在错误 mode 之间计算多样性。
        """
        if self.num_rank_modes <= 1 or self.rank_diversity_weight <= 0.0:
            return torch.tensor(0.0, device=DEVICE)

        cb = self.rank_codebook  # [num_modes, nbcues]
        if self.asymmetric_diversity:
            # 排除真实排序 mode 0
            cb = cb[1:]
        num_active = cb.shape[0]
        if num_active <= 1:
            return torch.tensor(0.0, device=DEVICE)

        n = self.nbcues
        idx_i, idx_j = torch.triu_indices(n, n, offset=1, device=DEVICE)

        if self.diversity_use_kendall:
            # 可微的 Kendall tau 代理：pairwise preference 一致性
            s_i = cb[:, idx_i]  # [m, p]
            s_j = cb[:, idx_j]  # [m, p]
            diff = (s_i - s_j) / 0.1
            P = torch.sigmoid(diff)  # [m, p]
            P1 = P.unsqueeze(0)  # [1, m, p]
            P2 = P.unsqueeze(1)  # [m, 1, p]
            agreement = (P1 * P2 + (1 - P1) * (1 - P2)).mean(dim=-1)  # [m, m]
            mask = torch.triu(
                torch.ones(num_active, num_active, device=DEVICE),
                diagonal=1,
            ).bool()
            mean_agreement = agreement[mask].mean()
            return self.rank_diversity_weight * mean_agreement
        else:
            # rank-difference 余弦相似度
            diffs = cb[:, idx_i] - cb[:, idx_j]
            diffs_norm = F.normalize(diffs, dim=1, eps=1e-8)
            similarity = diffs_norm @ diffs_norm.t()  # [m, m]
            mask = torch.triu(
                torch.ones(num_active, num_active, device=DEVICE),
                diagonal=1,
            ).bool()
            mean_similarity = similarity[mask].mean()
            return self.rank_diversity_weight * mean_similarity

    def compute_support_consistency_loss(self):
        """惩罚与 support pairs 冲突的排序模式。

        有码本时：约束每个 mode 的排序分数；
        无码本时（Route B）：约束最近一次推断出的 rank_scores。
        """
        if self.support_consistency_weight <= 0.0:
            return torch.tensor(0.0, device=DEVICE)

        loss = torch.tensor(0.0, device=DEVICE)
        margin = self.support_consistency_margin

        if self.num_rank_modes > 1:
            cb = self.rank_codebook  # [m, n]
            for i, j in self.support_pairs:
                # i 应该强于 j，即 cb[:, i] > cb[:, j] + margin
                diff = cb[:, j] - cb[:, i] + margin
                loss = loss + F.relu(diff).mean()
        else:
            # Route B：约束当前 episode 的 rank_scores
            rank_scores = getattr(self, "_last_rank_scores", None)
            if rank_scores is None:
                return torch.tensor(0.0, device=DEVICE)
            for i, j in self.support_pairs:
                # i 应该强于 j，即 scores[:, i] > scores[:, j] + margin
                diff = rank_scores[:, j] - rank_scores[:, i] + margin
                loss = loss + F.relu(diff).mean()

        return self.support_consistency_weight * loss

    def compute_selected_mode_support_consistency_loss(self):
        """惩罚当前 episode 实际选中的 mode 分布违反 support pairs 的程度。

        与 compute_support_consistency_loss 不同：后者只约束码本参数，
        前者约束的是 _last_mode_logits 在当前 batch 上产生的 mode 分布。
        """
        if (
            self.num_rank_modes <= 1
            or self.selected_mode_support_consistency_weight <= 0.0
            or self._last_mode_logits is None
        ):
            return torch.tensor(0.0, device=DEVICE)

        mode_weights = F.softmax(
            self._last_mode_logits / self.rank_mode_temperature, dim=-1
        )  # [bs, num_modes]
        cb = self.rank_codebook  # [num_modes, nbcues]
        margin = self.support_consistency_margin

        loss = torch.tensor(0.0, device=DEVICE)
        for i, j in self.support_pairs:
            # i 应该强于 j，即对所有 mode m：cb[m, i] > cb[m, j] + margin
            # 加权违反：sum_m weights[b,m] * relu(cb[m,j] - cb[m,i] + margin)
            violation_per_mode = F.relu(cb[:, j] - cb[:, i] + margin)  # [num_modes]
            weighted_violation = (mode_weights * violation_per_mode.unsqueeze(0)).sum(
                dim=-1
            )  # [bs]
            loss = loss + weighted_violation.mean()

        return self.selected_mode_support_consistency_weight * loss

    def _update_mode_subject_joint_counts(self, mode_weights, subject_idx):
        """更新 running estimate P(mode | subject) 用于 independence loss。

        Args:
            mode_weights: [bs, num_modes]
            subject_idx: [bs]
        """
        if self._mode_subject_joint_counts is None:
            return
        # 使用 soft weights 累加，配合 slow decay（这里用简单累加，loss 内部做归一化）
        # 必须 detach，否则会跨 episode 保留计算图，导致显存泄漏。
        with torch.no_grad():
            for b in range(mode_weights.shape[0]):
                s = int(subject_idx[b].item())
                if s < self.bs:
                    self._mode_subject_joint_counts[s] = (
                        self._mode_subject_joint_counts[s] + mode_weights[b].detach()
                    )

    def compute_mode_subject_independence_loss(self):
        """惩罚 mode 选择可被 subject ID 预测。

        维护 P(mode | subject) 的运行估计，并最小化其与 P(mode) 的 KL。
        """
        if (
            self.num_rank_modes <= 1
            or self.mode_subject_independence_weight <= 0.0
            or self._mode_subject_joint_counts is None
        ):
            return torch.tensor(0.0, device=DEVICE)

        joint = self._mode_subject_joint_counts  # [bs, num_modes]
        # 只使用已经见过至少一个样本的被试
        seen = joint.sum(dim=-1) > 0
        if not seen.any():
            return torch.tensor(0.0, device=DEVICE)

        p_mode_given_subject = joint[seen] / (joint[seen].sum(dim=-1, keepdim=True) + 1e-8)
        p_mode = joint.sum(dim=0) / (joint.sum() + 1e-8)  # [num_modes]

        # 平均 KL(P(mode|subject) || P(mode))
        kl = (
            p_mode_given_subject
            * (
                torch.log(p_mode_given_subject + 1e-8)
                - torch.log(p_mode.unsqueeze(0) + 1e-8)
            )
        ).sum(dim=-1).mean()

        return self.mode_subject_independence_weight * kl

    def project_codebook_support(self):
        """将码本投影到 support 约束上（mode 0 若被锚定则跳过）。"""
        if not self.project_to_support or self.num_rank_modes <= 1:
            return

        cb = self.rank_codebook.data
        margin = self.support_consistency_margin
        eps = 1e-4
        for k in range(self.num_rank_modes):
            if self.anchor_true_rank and k == 0:
                continue
            for i, j in self.support_pairs:
                diff = cb[k, i] - cb[k, j]
                if diff <= margin:
                    delta = (margin + eps - diff) / 2.0
                    cb[k, i] = cb[k, i] + delta
                    cb[k, j] = cb[k, j] - delta

    def _project_scores_to_support(self, scores, support_observations, margin=None):
        """将连续 rank scores 投影到 support 约束：i 应强于 j（含 margin）。

        Args:
            scores: [bs, nbcues]
            support_observations: list of (cue_i, cue_j, teacher_sign)
            margin: 最小 margin，默认使用 self.support_consistency_margin
        Returns:
            scores: [bs, nbcues]（原地修改，可微）
        """
        if margin is None:
            margin = self.support_consistency_margin
        eps = 1e-4
        scores = scores.clone()
        bs = scores.shape[0]
        batch = torch.arange(bs, device=scores.device)
        for cue_i, cue_j, teacher_sign in support_observations:
            i = (
                torch.full((bs,), int(cue_i), dtype=torch.long, device=scores.device)
                if isinstance(cue_i, int)
                else cue_i.to(device=scores.device, dtype=torch.long)
            )
            j = (
                torch.full((bs,), int(cue_j), dtype=torch.long, device=scores.device)
                if isinstance(cue_j, int)
                else cue_j.to(device=scores.device, dtype=torch.long)
            )
            sign = (
                torch.full((bs,), float(teacher_sign), device=scores.device)
                if isinstance(teacher_sign, (int, float))
                else teacher_sign.to(device=scores.device, dtype=scores.dtype)
            )
            observed = sign.ne(0.0)
            strong = torch.where(sign > 0, i, j)
            weak = torch.where(sign > 0, j, i)
            diff = scores[batch, strong] - scores[batch, weak]
            violation = F.relu(margin + eps - diff) * observed.to(scores.dtype)
            delta = violation / 2.0
            updates = torch.zeros_like(scores)
            updates.scatter_add_(1, strong.unsqueeze(1), delta.unsqueeze(1))
            updates.scatter_add_(1, weak.unsqueeze(1), -delta.unsqueeze(1))
            scores = scores + updates
        return scores

    def compute_mode_usage_loss(self):
        """让 mode 使用的边际分布逼近目标分布（mode 0 占 anchor_target_prob）。"""
        if (
            self.mode_usage_weight <= 0.0
            or self._last_mode_logits is None
            or self.num_rank_modes <= 1
        ):
            return torch.tensor(0.0, device=DEVICE)

        logits = self._last_mode_logits  # [bs, num_modes]
        mode_weights = F.softmax(logits / self.rank_mode_temperature, dim=-1)
        marginal = mode_weights.mean(dim=0)  # [num_modes]
        num_modes = marginal.shape[0]

        target = torch.full(
            (num_modes,),
            (1.0 - self.anchor_target_prob) / max(num_modes - 1, 1),
            device=DEVICE,
        )
        target[0] = self.anchor_target_prob

        kl = (marginal * (torch.log(marginal + 1e-8) - torch.log(target + 1e-8))).sum()
        return self.mode_usage_weight * kl

    def compute_mode_entropy_loss(self):
        """防止 mode 分布 collapse；以负熵作为正则项，训练时最大化熵。"""
        if (
            self.mode_entropy_weight <= 0.0
            or self._last_mode_logits is None
            or self.num_rank_modes <= 1
        ):
            return torch.tensor(0.0, device=DEVICE)

        logits = self._last_mode_logits  # [bs, num_modes]
        p = F.softmax(logits / self.rank_mode_temperature, dim=-1)
        per_subject_entropy = -(p * torch.log(p + 1e-8)).sum(dim=-1).mean()
        marginal = p.mean(dim=0)
        batch_entropy = -(marginal * torch.log(marginal + 1e-8)).sum()

        # 最小化负熵 = 最大化 mode 分布熵
        return -self.mode_entropy_weight * (per_subject_entropy + batch_entropy)

    def get_aux_loss(self):
        """返回辅助损失（diversity + support consistency + mode usage + mode entropy
        + selected-mode consistency + mode-subject independence）。"""
        div_loss = self.compute_rank_diversity_loss()
        sup_loss = self.compute_support_consistency_loss()
        usage_loss = self.compute_mode_usage_loss()
        entropy_loss = self.compute_mode_entropy_loss()
        selected_sup_loss = self.compute_selected_mode_support_consistency_loss()
        independence_loss = self.compute_mode_subject_independence_loss()
        return (
            div_loss
            + sup_loss
            + usage_loss
            + entropy_loss
            + selected_sup_loss
            + independence_loss
        )

    def forward(
        self,
        inputs,
        hidden,
        context=None,
        rank_scores=None,
        cue_pair=None,
        phase="support",
        support_cue_pairs=None,
        block_id=None,
    ):
        """前向传播。

        Args:
            inputs: [bs, input_size]
            hidden: [bs, hs] RNN 隐藏状态（保留接口，当前未使用）
            context: [bs, context_dim]
            rank_scores: [bs, nbcues]
            cue_pair: (cue_i, cue_j)
            phase: "support" or "query"
            support_cue_pairs: set/list of canonical cue pairs，用于 noise 掩码。
            block_id: int，当前 query block 索引，用于 block-level rank noise。
        """
        if phase == "support":
            h_new = torch.relu(
                self.trial_encoder(inputs[:, : self.config.cs * 2])
            )
            activout = torch.zeros(inputs.shape[0], 2, device=DEVICE)
            return activout, h_new, context, rank_scores

        # query phase
        if rank_scores is not None and cue_pair is not None:
            cue_i, cue_j = cue_pair
            scores_for_pref = rank_scores

            # Phase 4：贝叶斯后验排序，每个 block 从后验中采样一个排列
            if (
                not self.training
                and self.use_bayesian_posterior_eval
                and getattr(self, "_eval_bayesian_scores", None) is not None
                and block_id is not None
            ):
                scores_for_pref = self._eval_bayesian_scores[block_id]

            # 概率化评估：加入 block-level rank drift，模拟跨 block 记忆/注意波动
            elif (
                not self.training
                and self.use_probabilistic_ranking_eval
                and self._eval_block_perturbations is not None
                and block_id is not None
            ):
                scores_for_pref = scores_for_pref + self._eval_block_perturbations[
                    block_id
                ]

            if not self.training and self.use_hard_ranking_eval:
                scores_for_pref = self._hard_rank_scores(scores_for_pref)
            prob_i = self.compute_pairwise_preference(
                scores_for_pref, cue_i, cue_j, temperature=self.pairwise_choice_temperature
            )

            # probabilistic eval：逐 trial 独立随机翻转，引入传递性违反
            if (
                not self.training
                and self.use_probabilistic_ranking_eval
                and self.probabilistic_pair_flip_prob > 0.0
            ):
                flip = (
                    torch.rand_like(prob_i)
                    < self.probabilistic_pair_flip_prob
                )
                prob_i = torch.where(flip, 1.0 - prob_i, prob_i)

            # hard ranking 后加少量噪声，模拟人类在边界 pair 上的不确定性
            if (
                not self.training
                and self.use_hard_ranking_eval
                and (
                    self.hard_ranking_noise_prob > 0.0
                    or (
                        self.hard_ranking_learned_noise_prob is not None
                        and self.hard_ranking_learned_noise_prob > 0.0
                    )
                )
            ):
                is_learned = False
                if cue_pair is not None:
                    ci, cj = cue_pair
                    canonical = (int(min(ci, cj)), int(max(ci, cj)))
                    learned_set = (
                        set(support_cue_pairs)
                        if support_cue_pairs is not None
                        else self.support_pairs_set
                    )
                    is_learned = canonical in learned_set

                # 默认：未学习 pair 加噪声；已学习 pair 是否加噪声由配置决定
                if is_learned:
                    apply_noise = self.hard_ranking_noise_on_learned_pairs
                    effective_prob = (
                        self.hard_ranking_learned_noise_prob
                        if self.hard_ranking_learned_noise_prob is not None
                        else self.hard_ranking_noise_prob
                    )
                    flip_mask = self._eval_learned_flip_mask
                else:
                    apply_noise = True
                    effective_prob = self.hard_ranking_noise_prob
                    flip_mask = self._eval_pair_flip_mask

                if apply_noise and effective_prob > 0.0:
                    # systematic mask 存在时使用它；否则退化为独立翻转
                    if (
                        self.hard_ranking_noise_type == "independent"
                        or flip_mask is None
                    ):
                        flip = torch.rand_like(prob_i) < effective_prob
                    else:
                        ci, cj = cue_pair
                        flip = flip_mask[:, int(ci), int(cj)]
                    # mode 0（真实排序）的被试不加噪声，对应人类中少数完全正确的被试
                    if getattr(self, "_last_selected_mode", None) is not None:
                        skip = self._last_selected_mode == 0
                        flip = flip & ~skip
                    prob_i = torch.where(flip, 1.0 - prob_i, prob_i)
            logits = torch.stack(
                [torch.log(prob_i + 1e-8), torch.log(1 - prob_i + 1e-8)], dim=1
            )
            return logits, hidden, context, rank_scores

        # fallback: decision network
        if context is not None:
            combined = torch.cat([inputs, context], dim=-1)
        else:
            combined = inputs
        activout = self.decision_network(combined)
        return activout, hidden, context, rank_scores

    def initialZeroState(self, batch_size):
        return torch.zeros(batch_size, self.hs, device=DEVICE)

    def initialZeroET(self, batch_size):
        return torch.zeros(batch_size, self.hs, self.hs, device=DEVICE)

    def initialZeroPlasticWeights(self, batch_size):
        return torch.zeros(batch_size, self.hs, self.hs, device=DEVICE)
