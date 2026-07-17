"""Minimal online ordinal prediction-error process."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class OnlineOrdinalPredictionError(nn.Module):
    """Build an episode-local scalar ordering from sign-only observations.

    The only learned quantities are three population-level slow parameters.
    Item scores are fresh zero state in every episode; there are no item,
    participant, pair or task-specific learned parameters.
    """

    def __init__(self, config):
        super().__init__()
        self.nbcues = int(config.nbcues)
        learning_rate = float(config.online_learning_rate_init)
        retention = float(config.online_retention_init)
        logit_scale = float(config.online_logit_scale_init)
        if not 0.0 < learning_rate < 1.0:
            raise ValueError("online_learning_rate_init must be inside (0, 1)")
        if not 0.0 < retention < 1.0:
            raise ValueError("online_retention_init must be inside (0, 1)")
        if logit_scale <= 0.0:
            raise ValueError("online_logit_scale_init must be positive")
        self.learning_rate_logit = nn.Parameter(
            torch.tensor(math.log(learning_rate / (1.0 - learning_rate)))
        )
        self.retention_logit = nn.Parameter(
            torch.tensor(math.log(retention / (1.0 - retention)))
        )
        self.logit_scale_log = nn.Parameter(torch.tensor(math.log(logit_scale)))

    @property
    def learning_rate(self) -> torch.Tensor:
        return torch.sigmoid(self.learning_rate_logit)

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
            device=self.learning_rate_logit.device,
        )

    @staticmethod
    def prediction_error(
        state: torch.Tensor,
        left_cue: torch.Tensor,
        right_cue: torch.Tensor,
        sign: torch.Tensor,
    ) -> torch.Tensor:
        left_score = state.gather(1, left_cue.long()[:, None]).squeeze(1)
        right_score = state.gather(1, right_cue.long()[:, None]).squeeze(1)
        return sign.to(state.dtype) - torch.tanh(left_score - right_score)

    def update_support(
        self,
        state: torch.Tensor,
        left_cue: torch.Tensor,
        right_cue: torch.Tensor,
        sign: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one decayed, shared prediction-error update."""

        error = self.prediction_error(state, left_cue, right_cue, sign)
        left = F.one_hot(left_cue.long(), num_classes=self.nbcues).to(state.dtype)
        right = F.one_hot(right_cue.long(), num_classes=self.nbcues).to(state.dtype)
        direction = left - right
        next_state = (
            self.retention * state
            + self.learning_rate * error[:, None] * direction
        )
        return next_state, error

    def query_logits(
        self,
        state: torch.Tensor,
        left_cue: torch.Tensor,
        right_cue: torch.Tensor,
    ) -> torch.Tensor:
        """Read the current ordering without modifying it."""

        left_score = state.gather(1, left_cue.long()[:, None]).squeeze(1)
        right_score = state.gather(1, right_cue.long()[:, None]).squeeze(1)
        return self.logit_scale * torch.stack((left_score, right_score), dim=1)
