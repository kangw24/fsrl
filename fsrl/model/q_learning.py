"""Independent-value Q-learning baseline。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QLearningBaseline(nn.Module):
    """Classical model: 每个 item 有独立的 Q value，通过 delta rule 更新。"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.nbcues = config.nbcues
        self.bs = config.bs

        # 每个 item 的 Q value: [bs, nbcues]
        self.register_buffer(
            "q_values", torch.zeros(config.bs, config.nbcues)
        )
        self.register_parameter(
            "alpha", nn.Parameter(torch.tensor(0.1))  # learning rate
        )
        self.register_parameter(
            "gamma", nn.Parameter(torch.tensor(1.0))  # inverse temperature
        )

    def reset(self, batch_size=None):
        """每 episode 开始时重置 Q values。"""
        if batch_size is None:
            batch_size = self.bs
        self.q_values = torch.zeros(
            batch_size, self.nbcues, device=self.q_values.device
        )

    def update_from_pair(self, cue_i, cue_j, observed_diff):
        """Q-learning update rule（对齐 Liu 2026 Materials and methods Eq. 3-4）。

        Args:
            cue_i: int
            cue_j: int
            observed_diff: [bs] tensor，D(i,j) = (rank_j - rank_i) / (n-1)，
                          当 i 比 j 强时为正。
        """
        alpha = torch.sigmoid(self.alpha)
        q_i = self.q_values[:, cue_i]
        q_j = self.q_values[:, cue_j]
        # 纸面更新：Qi += α [D(i,j) - (Qi - Qj)/2]
        delta = observed_diff - (q_i - q_j) / 2

        updated = self.q_values.clone()
        updated[:, cue_i] = q_i + alpha * delta
        updated[:, cue_j] = q_j - alpha * delta
        self.q_values = updated

    def choice_logits(self, cue_i, cue_j, temperature=None):
        """Differentiable two-choice logits used for fitting alpha and gamma."""
        gamma = F.softplus(self.gamma)
        if temperature is not None:
            gamma = gamma / temperature
        q_diff = self.q_values[:, cue_i] - self.q_values[:, cue_j]
        signed = gamma * q_diff
        return torch.stack([0.5 * signed, -0.5 * signed], dim=-1)

    def choose(self, cue_i, cue_j, temperature=None):
        """Softmax choice。

        Args:
            temperature: 可选 [bs] tensor，对 gamma 做缩放（>1 更随机）。

        Returns:
            choices: [bs] long tensor，0 表示选择 cue_i（Q 值高者），1 表示选择 cue_j
        """
        logits = self.choice_logits(cue_i, cue_j, temperature=temperature)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)

    def forward(self, *args, **kwargs):
        """Dummy forward for API compatibility。"""
        pass
