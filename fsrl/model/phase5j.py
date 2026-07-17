"""Minimal neural process core for the Phase 5j mechanism tests.

The module intentionally does *not* know about item indices, global ranks,
support graphs, Hodge decompositions, or participant identities.  It receives
only the two stimulus vectors that are physically present on a step, an
observed signed relation during learning, and an event marker.

The update follows the differentiable neuromodulated-plasticity family used by
Miconi and Kay (2025): a recurrent network has shared slow weights and an
episode-local plastic recurrent matrix.  A learned modulatory signal stamps a
decaying Hebbian eligibility trace into the plastic matrix.  Hidden activity
and eligibility are reset at each trial; plastic weights persist within an
episode and are reset between episodes.

This architecture merely *permits* active reinstatement.  A trained run is not
called an active-replay solution unless independent neural probes show that a
previously seen, currently absent item is reinstated and causal lesions remove
the predicted list-linking/reassembly advantage.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import nn


class ProcessEvent(IntEnum):
    """Events visible to the network; item identities are never included."""

    OBSERVE_RELATION = 0
    BLANK = 1
    QUERY = 2


@dataclass(frozen=True)
class Phase5jConfig:
    stimulus_dim: int = 15
    hidden_dim: int = 64
    plastic_clip: float = 50.0
    eligibility_init: float = 0.7
    modulation_scale_init: float = 1.0
    relation_timing: str = "immediate"

    def __post_init__(self) -> None:
        if self.relation_timing not in {"immediate", "delayed"}:
            raise ValueError("relation_timing must be 'immediate' or 'delayed'")

    @property
    def input_dim(self) -> int:
        # left stimulus, right stimulus, signed relation, three event bits,
        # and a constant bias channel.
        return 2 * self.stimulus_dim + 1 + len(ProcessEvent) + 1


@dataclass
class Phase5jState:
    """Fast state.  It is never a parameter and must not cross episodes."""

    hidden: torch.Tensor
    eligibility: torch.Tensor
    plastic_weights: torch.Tensor


@dataclass
class Phase5jStep:
    logits: torch.Tensor
    hidden: torch.Tensor
    modulation: torch.Tensor
    plastic_delta: torch.Tensor


@dataclass
class Phase5jTrialTrace:
    event_inputs: list[torch.Tensor]
    steps: list[Phase5jStep]

    @property
    def blank_hidden(self) -> list[torch.Tensor]:
        return [step.hidden for step in self.steps[1:]]


class Phase5jPlasticRNN(nn.Module):
    """Shared slow network with a neuromodulated within-episode fast matrix."""

    def __init__(self, config: Phase5jConfig):
        super().__init__()
        self.config = config
        h = config.hidden_dim

        self.input_to_hidden = nn.Linear(config.input_dim, h)
        self.slow_recurrent = nn.Parameter(torch.empty(h, h))
        self.plasticity_coeff = nn.Parameter(torch.empty(h, h))
        self.hidden_to_modulation = nn.Linear(h, 2)
        self.hidden_to_choice = nn.Linear(h, 2)
        self.eligibility_logit = nn.Parameter(
            torch.logit(torch.tensor(float(config.eligibility_init)))
        )
        self.modulation_scale = nn.Parameter(
            torch.tensor(float(config.modulation_scale_init))
        )

        nn.init.uniform_(self.slow_recurrent, -h ** -0.5, h ** -0.5)
        nn.init.uniform_(self.plasticity_coeff, -0.01, 0.01)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Phase5jState:
        """Return a zero fast state for a genuinely new virtual participant."""
        reference = self.slow_recurrent
        device = reference.device if device is None else device
        dtype = reference.dtype if dtype is None else dtype
        h = self.config.hidden_dim
        return Phase5jState(
            hidden=torch.zeros(batch_size, h, device=device, dtype=dtype),
            eligibility=torch.zeros(batch_size, h, h, device=device, dtype=dtype),
            plastic_weights=torch.zeros(
                batch_size, h, h, device=device, dtype=dtype
            ),
        )

    @staticmethod
    def begin_trial(state: Phase5jState) -> Phase5jState:
        """Reset neural activity/eligibility while retaining episode memory."""
        return Phase5jState(
            hidden=torch.zeros_like(state.hidden),
            eligibility=torch.zeros_like(state.eligibility),
            plastic_weights=state.plastic_weights,
        )

    def event_input(
        self,
        event: ProcessEvent,
        *,
        batch_size: int | None = None,
        left: torch.Tensor | None = None,
        right: torch.Tensor | None = None,
        relation: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Construct the only information that may enter the neural process.

        ``relation`` is legal only for ``OBSERVE_RELATION``.  Query and blank
        events are rejected if they carry outcome information.
        """
        if (left is None) != (right is None):
            raise ValueError("left and right stimuli must be supplied together")
        if left is not None:
            if left.shape != right.shape or left.ndim != 2:
                raise ValueError("stimuli must have matching [batch, stimulus_dim] shape")
            if left.shape[1] != self.config.stimulus_dim:
                raise ValueError("stimulus dimensionality does not match config")
            batch_size = left.shape[0]
            device, dtype = left.device, left.dtype
        else:
            if batch_size is None:
                raise ValueError("batch_size is required for an event without stimuli")
            device, dtype = self.slow_recurrent.device, self.slow_recurrent.dtype

        if event != ProcessEvent.OBSERVE_RELATION and relation is not None:
            relation_tensor = torch.as_tensor(relation)
            if bool(torch.any(relation_tensor != 0)):
                raise ValueError("blank/query events cannot carry a relation outcome")

        x = torch.zeros(batch_size, self.config.input_dim, device=device, dtype=dtype)
        d = self.config.stimulus_dim
        if left is not None:
            x[:, :d] = left
            x[:, d : 2 * d] = right
        if relation is not None:
            rel = torch.as_tensor(relation, device=device, dtype=dtype)
            if rel.ndim == 0:
                rel = rel.expand(batch_size)
            if rel.shape != (batch_size,):
                raise ValueError("relation must be scalar or [batch]")
            x[:, 2 * d] = rel
        x[:, 2 * d + 1 + int(event)] = 1.0
        x[:, -1] = 1.0
        return x

    def step(
        self,
        inputs: torch.Tensor,
        state: Phase5jState,
        *,
        plasticity_enabled: bool = True,
        recurrent_dynamics_enabled: bool = True,
        modulation_enabled: bool = True,
    ) -> tuple[Phase5jStep, Phase5jState]:
        """Advance one neural step without mutating the incoming fast state."""
        if inputs.ndim != 2 or inputs.shape[1] != self.config.input_dim:
            raise ValueError("inputs must have shape [batch, config.input_dim]")
        batch_size = inputs.shape[0]
        if state.hidden.shape[0] != batch_size:
            raise ValueError("state and inputs use different batch sizes")

        if recurrent_dynamics_enabled:
            recurrent = self.slow_recurrent.unsqueeze(0) + (
                self.plasticity_coeff.unsqueeze(0) * state.plastic_weights
            )
            recurrent_drive = torch.bmm(
                recurrent, state.hidden.unsqueeze(-1)
            ).squeeze(-1)
        else:
            recurrent_drive = torch.zeros_like(state.hidden)

        hidden = torch.tanh(self.input_to_hidden(inputs) + recurrent_drive)
        raw_modulation = torch.tanh(self.hidden_to_modulation(hidden))
        modulation = self.modulation_scale * (
            raw_modulation[:, :1] - raw_modulation[:, 1:2]
        )
        if not modulation_enabled:
            modulation = torch.zeros_like(modulation)

        if plasticity_enabled:
            plastic_delta = modulation[:, :, None] * state.eligibility
            plastic_weights = torch.clamp(
                state.plastic_weights + plastic_delta,
                -self.config.plastic_clip,
                self.config.plastic_clip,
            )
        else:
            plastic_delta = torch.zeros_like(state.plastic_weights)
            plastic_weights = state.plastic_weights

        eta = torch.sigmoid(self.eligibility_logit)
        hebbian = torch.tanh(
            torch.bmm(hidden.unsqueeze(-1), state.hidden.unsqueeze(1))
        )
        eligibility = (1.0 - eta) * state.eligibility + eta * hebbian

        output = Phase5jStep(
            logits=self.hidden_to_choice(hidden),
            hidden=hidden,
            modulation=modulation,
            plastic_delta=plastic_delta,
        )
        new_state = Phase5jState(
            hidden=hidden,
            eligibility=eligibility,
            plastic_weights=plastic_weights,
        )
        return output, new_state

    def observe_relation(
        self,
        state: Phase5jState,
        left: torch.Tensor,
        right: torch.Tensor,
        relation: torch.Tensor,
        *,
        blank_steps: int = 3,
        support_blank_recurrence: bool = True,
        plasticity_enabled: bool = True,
        modulation_enabled: bool = True,
        relation_timing: str | None = None,
    ) -> tuple[Phase5jState, Phase5jTrialTrace]:
        """Encode one observed relation, followed by a fixed blank interval.

        Turning ``support_blank_recurrence`` off leaves the parameters and
        observed-pair plasticity machinery intact, but prevents fast weights
        from driving internally generated blank activity.  It is a passive
        encoding control, not proof that the intact network performs replay.
        """
        relation_timing = self.config.relation_timing if relation_timing is None else relation_timing
        if relation_timing not in {"immediate", "delayed"}:
            raise ValueError("relation_timing must be 'immediate' or 'delayed'")
        minimum_blank_steps = 3 if relation_timing == "delayed" else 2
        if blank_steps < minimum_blank_steps:
            raise ValueError(
                f"{relation_timing} relation timing requires at least "
                f"{minimum_blank_steps} post-observation steps"
            )
        trial_state = self.begin_trial(state)
        event_inputs: list[torch.Tensor] = []
        steps: list[Phase5jStep] = []

        observe_relation = relation if relation_timing == "immediate" else torch.zeros_like(relation)
        observe_input = self.event_input(
            ProcessEvent.OBSERVE_RELATION,
            left=left,
            right=right,
            relation=observe_relation,
        )
        event_inputs.append(observe_input)
        output, trial_state = self.step(
            observe_input,
            trial_state,
            plasticity_enabled=plasticity_enabled,
            modulation_enabled=modulation_enabled,
        )
        steps.append(output)

        if relation_timing == "immediate":
            post_observation_events = [(ProcessEvent.BLANK, None)] * blank_steps
        else:
            # Four-step source-inspired sequence with the default blank_steps=3:
            # pair alone -> internally generated activity -> relation feedback
            # -> delayed plastic update.  The externally visible relation is not
            # hidden from the model; only its neural arrival is delayed.
            post_observation_events = (
                [(ProcessEvent.BLANK, None)]
                + [(ProcessEvent.OBSERVE_RELATION, relation)]
                + [(ProcessEvent.BLANK, None)] * (blank_steps - 2)
            )

        for event, event_relation in post_observation_events:
            blank_input = self.event_input(
                event,
                batch_size=left.shape[0],
                relation=event_relation,
            )
            event_inputs.append(blank_input)
            output, trial_state = self.step(
                blank_input,
                trial_state,
                plasticity_enabled=plasticity_enabled,
                recurrent_dynamics_enabled=support_blank_recurrence,
                modulation_enabled=modulation_enabled,
            )
            steps.append(output)

        return trial_state, Phase5jTrialTrace(event_inputs=event_inputs, steps=steps)

    def query_pair(
        self,
        state: Phase5jState,
        left: torch.Tensor,
        right: torch.Tensor,
        *,
        decision_steps: int = 2,
    ) -> tuple[torch.Tensor, Phase5jTrialTrace]:
        """Read a frozen episode geometry; query events never change it."""
        if decision_steps < 1:
            raise ValueError("decision_steps must be positive")
        trial_state = self.begin_trial(state)
        event_inputs: list[torch.Tensor] = []
        steps: list[Phase5jStep] = []

        query_input = self.event_input(
            ProcessEvent.QUERY, left=left, right=right, relation=None
        )
        event_inputs.append(query_input)
        output, trial_state = self.step(
            query_input, trial_state, plasticity_enabled=False
        )
        steps.append(output)
        for _ in range(decision_steps - 1):
            decision_input = self.event_input(
                ProcessEvent.QUERY, batch_size=left.shape[0]
            )
            event_inputs.append(decision_input)
            output, trial_state = self.step(
                decision_input, trial_state, plasticity_enabled=False
            )
            steps.append(output)

        # The caller receives no updated state: queries cannot write memory.
        return output.logits, Phase5jTrialTrace(
            event_inputs=event_inputs, steps=steps
        )
