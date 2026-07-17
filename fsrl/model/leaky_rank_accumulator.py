"""Transparent sign-only leaky evidence accumulator baseline."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeakyRankAccumulator(nn.Module):
    """Accumulate directed pair evidence and read out Borda-like item scores.

    The model has only two shared slow parameters: evidence retention and query
    logit scale.  It has no participant parameters, recurrent hidden state or
    learned item embeddings.  Its episode state is reset to zero.
    """

    def __init__(self, config):
        super().__init__()
        self.nbcues = int(config.nbcues)
        retention_init = float(config.leaky_retention_init)
        scale_init = float(config.leaky_logit_scale_init)
        if not (0.0 < retention_init < 1.0):
            raise ValueError("leaky_retention_init must be inside (0, 1)")
        if scale_init <= 0.0:
            raise ValueError("leaky_logit_scale_init must be positive")
        self.retention_logit = nn.Parameter(
            torch.tensor(math.log(retention_init / (1.0 - retention_init)))
        )
        self.logit_scale_log = nn.Parameter(torch.tensor(math.log(scale_init)))

    @property
    def retention(self) -> torch.Tensor:
        return torch.sigmoid(self.retention_logit)

    @property
    def logit_scale(self) -> torch.Tensor:
        return torch.exp(self.logit_scale_log)

    def initial_state(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            self.nbcues,
            self.nbcues,
            device=self.retention_logit.device,
        )

    def update_support(
        self,
        state: torch.Tensor,
        left_cue: torch.Tensor,
        right_cue: torch.Tensor,
        sign: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one leaky, antisymmetric pair-evidence update."""

        left = F.one_hot(left_cue.long(), num_classes=self.nbcues).to(state.dtype)
        right = F.one_hot(right_cue.long(), num_classes=self.nbcues).to(state.dtype)
        directed_edge = left[:, :, None] * right[:, None, :]
        antisymmetric_edge = directed_edge - directed_edge.transpose(1, 2)
        return self.retention * state + sign[:, None, None] * antisymmetric_edge

    def query_logits(
        self,
        state: torch.Tensor,
        left_cue: torch.Tensor,
        right_cue: torch.Tensor,
    ) -> torch.Tensor:
        """Read the frozen state without changing it."""

        item_scores = state.sum(dim=2)
        left_score = item_scores.gather(1, left_cue.long()[:, None]).squeeze(1)
        right_score = item_scores.gather(1, right_cue.long()[:, None]).squeeze(1)
        return self.logit_scale * torch.stack((left_score, right_score), dim=1)
