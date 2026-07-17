"""RetroModulRNN + LatentRank 混合模型。

RetroModulRNN 负责 episode-level support phase 序列编码，
LatentRankMetaLearner 负责全局排序推断与 query phase 决策。
"""

import copy

import torch
import torch.nn as nn

from fsrl.device import DEVICE
from fsrl.model.latent_rank import LatentRankMetaLearner
from fsrl.model.retro_modul_rnn import RetroModulRNN


class RetroLatentRank(nn.Module):
    """混合模型：可塑 RNN 编码 support sequence，LatentRank 推断全局排序。"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bs = config.bs
        self.nbcues = config.nbcues
        self.hs = config.hs

        # Support-phase 序列编码器
        self.rnn = RetroModulRNN(config.to_model_dict())
        if getattr(config, "lesion_reset_pw_after_support", False):
            raise ValueError(
                "lesion_reset_pw_after_support is undefined for RetroLatentRank: "
                "query decisions consume a frozen context/rank snapshot, not pw. "
                "Use lesion_retro_plasticity to cut fast plasticity during support."
            )

        # RNN hidden -> LatentRank context
        self.context_dim = getattr(config, "retro_context_dim", None)
        if self.context_dim is None:
            self.context_dim = config.hs // 2
        self.context_proj = nn.Sequential(
            nn.Linear(config.hs, config.hs),
            nn.ReLU(),
            nn.Linear(config.hs, self.context_dim),
        )

        # 统一维护 subject embedding；LatentRank 内部仍保留接口但由外部注入
        self.subject_embedding_dim = getattr(config, "subject_embedding_dim", 0)
        if self.subject_embedding_dim > 0:
            self.subject_embedding = nn.Embedding(
                config.bs, self.subject_embedding_dim
            ).to(DEVICE)
            if hasattr(self.rnn, "subject_embedding"):
                self.rnn.subject_embedding = self.subject_embedding
            if getattr(config, "freeze_subject_embedding", False):
                self.subject_embedding.weight.requires_grad = False
            # 覆盖 LatentRank 内部的 embedding，使其共享同一张表
            latent_cfg = copy.copy(config)
            self.latent_rank = LatentRankMetaLearner(latent_cfg)
            if self.latent_rank.subject_embedding_dim > 0:
                self.latent_rank.subject_embedding = self.subject_embedding
        else:
            self.latent_rank = LatentRankMetaLearner(config)

        # Support-phase 辅助头：预测当前 trial 的 teacher sign
        self.support_aux_weight = getattr(config, "retro_support_aux_weight", 0.0)
        self.support_aux_head = (
            nn.Linear(config.hs, 1).to(DEVICE)
            if self.support_aux_weight > 0.0
            else None
        )
        self._support_aux_loss = torch.tensor(0.0, device=DEVICE)

        # 透传 eval 标志，供 runner 查询
        self.use_probabilistic_ranking_eval = (
            self.latent_rank.use_probabilistic_ranking_eval
        )
        self.use_bayesian_posterior_eval = (
            self.latent_rank.use_bayesian_posterior_eval
        )
        self.use_hard_ranking_eval = self.latent_rank.use_hard_ranking_eval

    # ------------------------------------------------------------------
    # Support encoding
    # ------------------------------------------------------------------
    def encode_support_sequence(self, inputs_seq, teacher_da_seq, training=False, return_hidden_seq=False):
        """处理整个 support phase，返回 context [bs, context_dim]。

        Args:
            inputs_seq: list of [bs, inputsize]，长度 = num_support_trials * support_triallen
            teacher_da_seq: list of [bs, 1] 或 None，与 inputs_seq 对齐
            training: 是否训练模式
            return_hidden_seq: 是否返回每个 trial 结束后的 hidden 序列（用于 sequential rank update）
        Returns:
            context: [bs, context_dim]
            hidden_seq: optional, [num_support_trials, bs, hs]
        """
        bs = inputs_seq[0].shape[0]
        hidden = self.rnn.initialZeroState(bs)
        et = self.rnn.initialZeroET(bs)
        pw = self.rnn.initialZeroPlasticWeights(bs)

        detach_per_trial = getattr(self.config, "retro_detach_per_trial", True)
        aux_losses = []
        hidden_seq = [] if return_hidden_seq else None

        for t, inputs in enumerate(inputs_seq):
            td = teacher_da_seq[t]
            activout, valueout, daout, hidden, et, pw = self.rnn(
                inputs, hidden, et, pw, teacher_da=td
            )

            # support auxiliary loss：从 hidden 预测 teacher sign
            if self.support_aux_head is not None and td is not None:
                pred = self.support_aux_head(hidden).squeeze(-1)
                target = (td.squeeze(-1) + 1.0) / 2.0  # +1/-1 -> 1/0
                aux_losses.append(
                    nn.functional.binary_cross_entropy_with_logits(pred, target)
                )

            # 每 trial 结束后截断 et/pw，避免可塑性矩阵图爆炸
            if (t + 1) % self.config.support_triallen == 0:
                if return_hidden_seq:
                    # hidden 作为 sequential rank update 的输入特征；
                    # detach 与否由 retro_detach_per_trial 控制
                    hidden_seq.append(hidden.detach() if detach_per_trial else hidden)
                if detach_per_trial:
                    et = et.detach()
                    pw = pw.detach()

        # 若 RNN 被冻结，截断 hidden 与 RNN 的梯度，避免 support phase 图过大。
        if training and getattr(self.config, "retro_freeze_rnn", False):
            hidden = hidden.detach()

        context = torch.relu(self.context_proj(hidden))

        if aux_losses:
            self._support_aux_loss = torch.stack(aux_losses).mean()
        else:
            self._support_aux_loss = torch.tensor(0.0, device=DEVICE)

        if return_hidden_seq:
            return context, torch.stack(hidden_seq)
        return context

    # ------------------------------------------------------------------
    # LatentRank 透传
    # ------------------------------------------------------------------
    def infer_latent_rank(
        self,
        context,
        subject_idx=None,
        training=False,
        true_rank=None,
        support_observations=None,
        trial_encodings=None,
    ):
        """从 context 推断 rank scores。

        过程模型改造：context 与 subject_idx 分开传递，由 LatentRankMetaLearner
        内部根据 mode_selector_input 决定 subject embedding 的用途，避免
        subject embedding 直接决定 mode 身份。

        如果提供了 RNN hidden_seq（[num_trials, bs, hs]），先投影到 context_dim
        再传给 LatentRankMetaLearner 做 sequential rank update。
        """
        use_process_head = (
            self.latent_rank.use_sequential_rank_update
            or self.latent_rank.use_pairwise_preference_accumulator
        )
        if trial_encodings is not None and use_process_head:
            # hidden_seq 从 RNN hidden 维度投影到 LatentRank context 维度。
            # 是否截断与 RNN 的梯度由 retro_detach_per_trial 控制：
            # true 时避免 OOM；false 时让 support phase 能训练 RNN。
            detach_per_trial = getattr(self.config, "retro_detach_per_trial", True)
            if detach_per_trial:
                trial_encodings = trial_encodings.detach()
            if trial_encodings.dim() == 3:
                # [num_trials, bs, hs] -> [num_trials, bs, context_dim]
                trial_encodings = torch.relu(
                    self.context_proj[0](trial_encodings)
                )
                trial_encodings = self.context_proj[2](trial_encodings)
        return self.latent_rank.infer_latent_rank(
            context,
            subject_idx=subject_idx,
            training=training,
            true_rank=true_rank,
            support_observations=support_observations,
            trial_encodings=trial_encodings,
        )

    # ------------------------------------------------------------------
    # Query forward
    # ------------------------------------------------------------------
    def forward(
        self,
        inputs,
        hidden,
        context=None,
        rank_scores=None,
        cue_pair=None,
        phase="query",
        support_cue_pairs=None,
        block_id=None,
    ):
        """Query phase 完全委托给 LatentRank。"""
        return self.latent_rank(
            inputs,
            hidden=None,
            context=context,
            rank_scores=rank_scores,
            cue_pair=cue_pair,
            phase=phase,
            support_cue_pairs=support_cue_pairs,
            block_id=block_id,
        )

    # ------------------------------------------------------------------
    # 兼容接口
    # ------------------------------------------------------------------
    def get_aux_loss(self):
        base = self.latent_rank.get_aux_loss()
        return base + self.support_aux_weight * self._support_aux_loss

    def project_codebook_support(self):
        self.latent_rank.project_codebook_support()

    def initialZeroState(self, batch_size):
        return self.rnn.initialZeroState(batch_size)

    def initialZeroET(self, batch_size):
        return self.rnn.initialZeroET(batch_size)

    def initialZeroPlasticWeights(self, batch_size):
        return self.rnn.initialZeroPlasticWeights(batch_size)
