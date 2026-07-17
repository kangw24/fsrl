import numpy as np
import torch
import torch.nn as nn

from fsrl.device import DEVICE


class RetroModulRNN(nn.Module):
    """RNN with neuromodulated recurrent plasticity.

    ``et`` is the Hebbian eligibility trace. ``pw`` is the within-episode
    plastic recurrent weight matrix. Only ``pw`` changes during an episode.
    """

    def __init__(self, config):
        super().__init__()
        for paramname in ["outputsize", "inputsize", "hs", "bs"]:
            if paramname not in config:
                raise KeyError("Must provide missing key in config: " + paramname)

        nbda = 2
        self.GG = config
        self.bs = config["bs"]
        self.pw_init_std = config.get("pw_init_std", 0.0)
        self.use_plasticity = config.get("use_plasticity", True)
        self.subject_embedding_dim = config.get("subject_embedding_dim", 0)
        self.activ = torch.tanh
        self.i2h = torch.nn.Linear(config["inputsize"], config["hs"]).to(DEVICE)
        self.w = torch.nn.Parameter(
            (
                (1.0 / np.sqrt(config["hs"]))
                * (2.0 * torch.rand(config["hs"], config["hs"]) - 1.0)
            ).to(DEVICE),
            requires_grad=True,
        )
        self.alpha = torch.nn.Parameter(
            (0.01 * (2.0 * torch.rand(config["hs"], config["hs"]) - 1.0)).to(DEVICE),
            requires_grad=True,
        )
        self.etaet = torch.nn.Parameter(
            (0.7 * torch.ones(1)).to(DEVICE), requires_grad=True
        )
        self.DAmult = torch.nn.Parameter(
            (1.0 * torch.ones(1)).to(DEVICE), requires_grad=True
        )
        self.h2DA = torch.nn.Linear(config["hs"], nbda).to(DEVICE)
        self.h2o = torch.nn.Linear(config["hs"], config["outputsize"]).to(DEVICE)
        self.h2v = torch.nn.Linear(config["hs"], 1).to(DEVICE)

        if self.subject_embedding_dim > 0:
            self.subject_embedding = torch.nn.Embedding(
                self.bs, self.subject_embedding_dim
            ).to(DEVICE)
            self.subj2h = torch.nn.Linear(
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
        """Advance one recurrent step.

        ``plastic_write_enabled=False`` makes the plastic matrix read-only for
        this step.  The default preserves the legacy runner.  A read-only mode
        is needed for no-feedback query trials: they may read the state formed
        during support, but must not silently learn from their own presentation.

        ``eligibility_update_enabled`` is exposed separately so eligibility can
        be lesioned without conflating that intervention with plastic readout.
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

        plasticity_active = self.use_plasticity and not self.GG.get(
            "lesion_retro_plasticity", False
        )
        plastic_weights = self.alpha * pw if plasticity_active else 0.0
        hactiv = self.activ(
            i2h_out
            + torch.matmul(
                (self.w + plastic_weights),
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

        if not plasticity_active:
            daout = torch.zeros_like(daout)

        if plastic_write_enabled:
            pw = pw + daout.view(batch_size, 1, 1) * et
            pw = torch.clamp(pw, min=-50.0, max=50.0)

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
