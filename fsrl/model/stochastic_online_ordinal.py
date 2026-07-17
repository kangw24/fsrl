"""Trial-local stochastic encoding extension of the online ordinal updater."""

import torch

from fsrl.model.online_ordinal import OnlineOrdinalPredictionError


class StochasticEncodingOnlineOrdinal(OnlineOrdinalPredictionError):
    """Add one shared trial-local omission process, never participant state.

    ``encoding_reliability`` is a population-level candidate parameter.  Each
    support trial samples a fresh Bernoulli gate independently for every
    episode.  A closed gate omits the relation update but still applies normal
    retention, so elapsed trials are not silently removed from time.
    """

    def __init__(self, config):
        super().__init__(config)
        reliability = float(config.online_encoding_reliability)
        query_lapse = float(config.online_query_lapse)
        if not 0.0 <= reliability <= 1.0:
            raise ValueError("online_encoding_reliability must be in [0, 1]")
        if not 0.0 <= query_lapse < 1.0:
            raise ValueError("online_query_lapse must be in [0, 1)")
        self.encoding_reliability = reliability
        self.query_lapse = query_lapse

    def sample_encoding_mask(
        self,
        batch_size: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> torch.Tensor:
        return (
            torch.rand(batch_size, generator=generator, device=device)
            < self.encoding_reliability
        ).float()

    def update_support(
        self,
        state: torch.Tensor,
        left_cue: torch.Tensor,
        right_cue: torch.Tensor,
        sign: torch.Tensor,
        *,
        encoding_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if encoding_mask is None:
            if generator is None:
                raise ValueError("generator is required when encoding_mask is not given")
            encoding_mask = self.sample_encoding_mask(
                state.shape[0], generator=generator, device=state.device
            )
        encoding_mask = encoding_mask.to(device=state.device, dtype=state.dtype)
        if encoding_mask.shape != (state.shape[0],):
            raise ValueError("encoding_mask must have shape [batch]")

        error = self.prediction_error(state, left_cue, right_cue, sign)
        left = torch.nn.functional.one_hot(
            left_cue.long(), num_classes=self.nbcues
        ).to(state.dtype)
        right = torch.nn.functional.one_hot(
            right_cue.long(), num_classes=self.nbcues
        ).to(state.dtype)
        direction = left - right
        next_state = (
            self.retention * state
            + self.learning_rate
            * error[:, None]
            * encoding_mask[:, None]
            * direction
        )
        return next_state, error, encoding_mask

    def query_probabilities(
        self,
        state: torch.Tensor,
        left_cue: torch.Tensor,
        right_cue: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = torch.softmax(
            self.query_logits(state, left_cue, right_cue), dim=1
        )
        return self.query_lapse * 0.5 + (1.0 - self.query_lapse) * probabilities
