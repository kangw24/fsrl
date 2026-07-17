"""Support-phase 贝叶斯后验排序推断。

从 8 个有噪声的 support pair 观测出发，在全排列空间上推断全局排序的后验分布，
用于模拟人类被试从 few-shot 输入在线构造个体化全局排序的过程。
"""

import itertools
import math

import numpy as np
import torch
import torch.nn.functional as F

from fsrl.device import DEVICE
class BayesianRankPosterior:
    """在全排列假设空间上计算排序后验并采样。

    先验：Mallows(center_perm, phi)，中心排列由调用方提供（如被试选定的 mode 或真实排序）。
    似然：对每个 support pair (i, j) 观测 o_ij ∈ {+1, -1}，
          P(o_ij | π) = σ(o_ij * (score_π(i) - score_π(j)) / tau)。
    后验：P(π | D) ∝ P(π) * ∏_k P(o_k | π)。
    """

    def __init__(self, nbcues, support_pairs, max_extensions=50000, seed=42):
        self.nbcues = nbcues
        self.support_pairs = list(support_pairs)
        # 似然已经允许 noisy/contradictory evidence，因此假设空间不能预先
        # 强制 support edges；否则冲突观测永远无法翻转那些边。
        total = math.factorial(nbcues)
        if total <= max_extensions:
            self.extensions = [
                list(permutation)
                for permutation in itertools.permutations(range(nbcues))
            ]
        else:
            rng = np.random.default_rng(seed)
            sampled: set[tuple[int, ...]] = set()
            while len(sampled) < max_extensions:
                sampled.add(tuple(int(x) for x in rng.permutation(nbcues)))
            self.extensions = [list(permutation) for permutation in sorted(sampled)]
        self.n_ext = len(self.extensions)
        # 预计算每个扩展的 scores: 从强到弱得分为 nbcues-1, ..., 0
        self._scores = torch.zeros(self.n_ext, nbcues, device=DEVICE)
        for idx, ext in enumerate(self.extensions):
            for pos, cue in enumerate(ext):
                self._scores[idx, cue] = nbcues - 1 - pos
        self._log_weights = None

    def _mallows_log_prior(self, center_perm, phi):
        """计算每个扩展相对于 center_perm 的 Mallows log 先验权重。

        Args:
            center_perm: [nbcues] 中心排列（从强到弱）。
            phi: 集中度（>0），越大越集中。
        Returns:
            log_prior: [n_ext]
        """
        if phi <= 0.0:
            return torch.zeros(self.n_ext, device=DEVICE)
        # Kendall-tau 距离：中心排列与每个扩展的 discordant pair 数
        center_pos = torch.zeros(self.nbcues, dtype=torch.long, device=DEVICE)
        for pos, cue in enumerate(center_perm):
            center_pos[cue] = pos
        ext_pos = torch.zeros(self.n_ext, self.nbcues, dtype=torch.long, device=DEVICE)
        for idx, ext in enumerate(self.extensions):
            for pos, cue in enumerate(ext):
                ext_pos[idx, cue] = pos
        # 对每个扩展，计算与中心排列的 Kendall tau 距离
        d = torch.zeros(self.n_ext, device=DEVICE)
        for i in range(self.nbcues):
            for j in range(i + 1, self.nbcues):
                # 中心排列中 i 在 j 前（位置更小）
                center_i_first = center_pos[i] < center_pos[j]
                ext_i_first = ext_pos[:, i] < ext_pos[:, j]
                d += (center_i_first != ext_i_first).float()
        return -phi * d

    def _support_log_likelihood(self, pairs, outcomes, tau):
        """计算每个扩展下 support 观测的 log 似然。

        Args:
            pairs: list of (cue_i, cue_j)
            outcomes: list of int, +1 表示 i 强于 j, -1 表示 j 强于 i。
            tau: 证据温度（>0），越大对 support 越不信任。
        Returns:
            log_lik: [n_ext]
        """
        if tau <= 0.0 or len(pairs) == 0:
            return torch.zeros(self.n_ext, device=DEVICE)
        s_i = self._scores[:, [p[0] for p in pairs]]  # [n_ext, n_pairs]
        s_j = self._scores[:, [p[1] for p in pairs]]
        outcomes_t = torch.tensor(outcomes, dtype=torch.float32, device=DEVICE)
        diff = (s_i - s_j) * outcomes_t.unsqueeze(0)  # [n_ext, n_pairs]
        log_lik = F.logsigmoid(diff / tau).sum(dim=1)
        return log_lik

    def compute_posterior(self, center_perm, phi, pairs, outcomes, tau):
        """计算后验分布。

        Args:
            center_perm: [nbcues] 中心排列（从强到弱）。
            phi: Mallows 集中度。
            pairs: support pair 列表。
            outcomes: 对应观测结果列表。
            tau: 似然温度。
        """
        log_prior = self._mallows_log_prior(center_perm, phi)
        log_lik = self._support_log_likelihood(pairs, outcomes, tau)
        log_weights = log_prior + log_lik
        # 数值稳定：减去最大值
        log_weights = log_weights - log_weights.max()
        self._log_weights = log_weights

    def sample(self, n_samples=1):
        """从后验中采样排列。

        Returns:
            perms: [n_samples, nbcues]，每个排列从强到弱。
        """
        if self._log_weights is None:
            raise RuntimeError("请先调用 compute_posterior")
        weights = torch.exp(self._log_weights)
        probs = weights / weights.sum()
        idx = torch.multinomial(probs, n_samples, replacement=True)
        return torch.tensor(
            [self.extensions[i] for i in idx.tolist()],
            dtype=torch.long,
            device=DEVICE,
        )

    def to_scores(self, perms):
        """把排列转换为 rank scores。

        Args:
            perms: [n_samples, nbcues]，从强到弱。
        Returns:
            scores: [n_samples, nbcues]
        """
        n_samples = perms.shape[0]
        scores = torch.zeros(n_samples, self.nbcues, device=DEVICE)
        for pos in range(self.nbcues):
            cue = perms[:, pos]
            scores[torch.arange(n_samples, device=DEVICE), cue] = self.nbcues - 1 - pos
        return scores

    def posterior_mean_scores(self, n_samples=200):
        """从后验采样并返回平均 scores（更平滑，但不产生 trial-level 变异）。"""
        perms = self.sample(n_samples)
        return self.to_scores(perms).mean(dim=0)
