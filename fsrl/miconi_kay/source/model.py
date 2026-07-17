"""Neuromodulated plastic RNN used by the Miconi & Kay source task."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch import nn


class MiconiKayRetroModulRNN(nn.Module):
    """RNN with neuromodulated recurrent plasticity.

    Attribute names intentionally match the public checkpoints. ``et`` is the
    within-trial Hebbian eligibility trace and ``pw`` is the within-episode
    plastic recurrent matrix.  This source core contains no teacher signal,
    participant embedding, Liu graph, or lesion switch.
    """

    def __init__(self, config: Mapping[str, object]):
        super().__init__()
        for name in ("outputsize", "inputsize", "hs", "bs"):
            if name not in config:
                raise KeyError("Must provide missing key in config: " + name)

        hidden_size = int(config["hs"])
        input_size = int(config["inputsize"])
        output_size = int(config["outputsize"])
        self.GG = dict(config)
        self.activ = torch.tanh

        # Construction order is part of seeded parity with the readable source.
        self.i2h = nn.Linear(input_size, hidden_size)
        self.w = nn.Parameter(
            (1.0 / np.sqrt(hidden_size))
            * (2.0 * torch.rand(hidden_size, hidden_size) - 1.0),
            requires_grad=True,
        )
        self.alpha = nn.Parameter(
            0.01 * (2.0 * torch.rand(hidden_size, hidden_size) - 1.0),
            requires_grad=True,
        )
        self.etaet = nn.Parameter(0.7 * torch.ones(1), requires_grad=True)
        self.DAmult = nn.Parameter(torch.ones(1), requires_grad=True)
        self.h2DA = nn.Linear(hidden_size, 2)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.h2v = nn.Linear(hidden_size, 1)

    @property
    def hidden_size(self) -> int:
        return int(self.GG["hs"])

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: torch.Tensor,
        et: torch.Tensor,
        pw: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch_size = inputs.shape[0]
        hidden_size = self.hidden_size
        if not (pw.shape[0] == hidden.shape[0] == et.shape[0] == batch_size):
            raise ValueError("inputs, hidden, eligibility, and plastic weights disagree")

        hactiv = self.activ(
            self.i2h(inputs).view(batch_size, hidden_size, 1)
            + torch.matmul(
                self.w + torch.mul(self.alpha, pw),
                hidden.view(batch_size, hidden_size, 1),
            )
        ).view(batch_size, hidden_size)

        activout = self.h2o(hactiv)
        valueout = self.h2v(hactiv)
        daout2 = torch.tanh(self.h2DA(hactiv))
        daout = self.DAmult * (daout2[:, 0] - daout2[:, 1])[:, None]

        next_pw = torch.clamp(
            pw + daout.view(batch_size, 1, 1) * et,
            min=-50.0,
            max=50.0,
        )
        deltaet = torch.bmm(
            hactiv.view(batch_size, hidden_size, 1),
            hidden.view(batch_size, 1, hidden_size),
        )
        deltaet = torch.tanh(deltaet)
        next_et = (1 - self.etaet) * et + self.etaet * deltaet

        return activout, valueout, daout, hactiv, next_et, next_pw

    def initialZeroET(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            self.hidden_size,
            self.hidden_size,
            device=self.w.device,
            dtype=self.w.dtype,
            requires_grad=False,
        )

    def initialZeroPlasticWeights(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            self.hidden_size,
            self.hidden_size,
            device=self.w.device,
            dtype=self.w.dtype,
            requires_grad=False,
        )

    def initialZeroState(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            self.hidden_size,
            device=self.w.device,
            dtype=self.w.dtype,
            requires_grad=False,
        )

    # Readable aliases for new callers; public checkpoint keys are unaffected.
    initial_zero_eligibility = initialZeroET
    initial_zero_plastic_weights = initialZeroPlasticWeights
    initial_zero_state = initialZeroState
