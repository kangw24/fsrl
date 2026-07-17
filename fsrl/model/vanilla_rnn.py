"""Vanilla RNN baseline：与 RetroModulRNN 结构相同但关闭可塑性。"""

import numpy as np
import torch
import torch.nn as nn

from fsrl.device import DEVICE


class VanillaRNN(nn.Module):
    """无 plastic weights 的 RNN，用于对比可塑性的作用。"""

    def __init__(self, config):
        super().__init__()
        for paramname in ["outputsize", "inputsize", "hs", "bs"]:
            if paramname not in config:
                raise KeyError("Must provide missing key in config: " + paramname)

        nbda = 2
        self.GG = config
        self.bs = config["bs"]
        self.pw_init_std = config.get("pw_init_std", 0.0)
        self.subject_embedding_dim = config.get("subject_embedding_dim", 0)
        self.activ = torch.tanh
        self.i2h = nn.Linear(config["inputsize"], config["hs"]).to(DEVICE)
        self.w = nn.Parameter(
            (
                (1.0 / np.sqrt(config["hs"]))
                * (2.0 * torch.rand(config["hs"], config["hs"]) - 1.0)
            ).to(DEVICE),
            requires_grad=True,
        )
        # alpha 固定为 0，等价于关闭 pw
        self.alpha = nn.Parameter(
            torch.zeros(config["hs"], config["hs"]).to(DEVICE), requires_grad=False
        )
        self.etaet = nn.Parameter(
            (0.7 * torch.ones(1)).to(DEVICE), requires_grad=True
        )
        self.DAmult = nn.Parameter(
            (1.0 * torch.ones(1)).to(DEVICE), requires_grad=True
        )
        self.h2DA = nn.Linear(config["hs"], nbda).to(DEVICE)
        self.h2o = nn.Linear(config["hs"], config["outputsize"]).to(DEVICE)
        self.h2v = nn.Linear(config["hs"], 1).to(DEVICE)

        if self.subject_embedding_dim > 0:
            self.subject_embedding = nn.Embedding(
                self.bs, self.subject_embedding_dim
            ).to(DEVICE)
            self.subj2h = nn.Linear(
                self.subject_embedding_dim, config["hs"]
            ).to(DEVICE)

    def forward(
        self,
        inputs,
        hidden,
        et,
        pw,
        teacher_da=None,
        *,
        plastic_write_enabled=True,
        eligibility_update_enabled=True,
    ):
        """Advance one step using only the ordinary recurrent state.

        The plastic-write keyword is accepted for a shared MetaSign episode
        interface but intentionally has no effect.  Eligibility can still be
        switched off for schedule-matched diagnostics.
        """
        batch_size = inputs.shape[0]
        hidden_size = self.GG["hs"]
        assert pw.shape[0] == hidden.shape[0] == et.shape[0] == batch_size

        i2h_out = self.i2h(inputs).view(batch_size, hidden_size, 1)
        if self.subject_embedding_dim > 0:
            subj_idx = torch.arange(batch_size, device=DEVICE) % self.bs
            subj_emb = self.subject_embedding(subj_idx)
            if self.GG.get("zero_subject_embedding", False):
                subj_emb = torch.zeros_like(subj_emb)
            subj_proj = self.subj2h(subj_emb).view(
                batch_size, hidden_size, 1
            )
            i2h_out = i2h_out + subj_proj

        hactiv = self.activ(
            i2h_out
            + torch.matmul(
                (self.w + torch.mul(self.alpha, pw)),
                hidden.view(batch_size, hidden_size, 1),
            )
        ).view(batch_size, hidden_size)

        activout = self.h2o(hactiv)
        valueout = self.h2v(hactiv)

        daout2 = torch.tanh(self.h2DA(hactiv))
        if teacher_da is not None:
            daout = teacher_da
        else:
            daout = self.DAmult * (daout2[:, 0] - daout2[:, 1])[:, None]

        # Vanilla baseline does not update fast weights.  Returning the input
        # tensor unchanged keeps the common interface without hidden state drift.

        deltaet = torch.bmm(
            hactiv.view(batch_size, hidden_size, 1),
            hidden.view(batch_size, 1, hidden_size),
        )
        deltaet = torch.tanh(deltaet)
        if eligibility_update_enabled:
            et = (1 - self.etaet) * et + self.etaet * deltaet

        return activout, valueout, daout, hactiv, et, pw

    def initialZeroET(self, batch_size):
        return torch.zeros(
            batch_size, self.GG["hs"], self.GG["hs"], requires_grad=False
        ).to(DEVICE)

    def initialZeroPlasticWeights(self, batch_size):
        if self.pw_init_std > 0.0:
            return torch.normal(
                0.0,
                self.pw_init_std,
                (batch_size, self.GG["hs"], self.GG["hs"]),
                requires_grad=False,
            ).to(DEVICE)
        return torch.zeros(
            batch_size, self.GG["hs"], self.GG["hs"], requires_grad=False
        ).to(DEVICE)

    def initialZeroState(self, batch_size):
        return torch.zeros(batch_size, self.GG["hs"], requires_grad=False).to(DEVICE)
