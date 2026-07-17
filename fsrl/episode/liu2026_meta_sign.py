"""Sign-only Liu 2026 meta-learning episode for :class:`RetroModulRNN`.

This runner is deliberately separate from the legacy Liu runner.  Every batch
element is an independently randomized virtual subject.  The network sees only
two symbolic cue identities during a trial and, during support, their binary
high/low relation.  Rank distance, the full permutation, query targets and
human choices are never placed in the network input.

The relation is an ordinary support input.  The learned ``h2DA`` pathway maps
the recurrent state to a write modulator; the relation is not hard-coded as the
DA value.  Versioned configuration flags select whether transient recurrent
state is continuous across trials (v1) or reset at each trial while only the
plastic matrix persists (v2).  Query trials branch from the frozen support fast
state and run with plastic writes disabled.  Their synthetic targets are used
only to form the outer-loop meta-learning objective.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from fsrl.device import DEVICE
from fsrl.episode.types import EpisodeRecord
from fsrl.episode.symbolic_record import symbolic_logits_to_episode_record
from fsrl.model.retro_modul_rnn import RetroModulRNN
from fsrl.model.vanilla_rnn import VanillaRNN
from fsrl.task.constants import NUMRESPONSESTEP
from fsrl.task.liu2026 import (
    Liu2026SymbolicSubjectTask,
    SymbolicQueryObservation,
    build_liu2026_symbolic_subject_tasks,
    build_meta_training_symbolic_subject_tasks,
    generate_liu2026_cue_data,
)


@dataclass(frozen=True)
class MetaSignIntervention:
    """Predeclared causal intervention on the MetaSign episode path."""

    support_sign_mode: str = "intact"  # intact | omitted | shuffled
    support_write_enabled: bool = True
    support_eligibility_enabled: bool = True
    query_write_enabled: bool = False

    def validate(self) -> None:
        if self.support_sign_mode not in {"intact", "omitted", "shuffled"}:
            raise ValueError(
                "support_sign_mode must be intact, omitted or shuffled"
            )


@dataclass(frozen=True)
class MetaSignEpisodeResult:
    """Differentiable episode result plus generator-only audit metadata."""

    loss: torch.Tensor
    logits: torch.Tensor
    targets: torch.Tensor
    accuracy: float
    support_hidden: torch.Tensor
    support_eligibility: torch.Tensor
    support_plastic_weights: torch.Tensor
    subject_tasks: tuple[Liu2026SymbolicSubjectTask, ...]
    intervention: MetaSignIntervention
    effective_support_signs: tuple[tuple[int, ...], ...]
    support_write_norms: tuple[tuple[float, ...], ...]


def _model_input(
    config,
    cue_batch: torch.Tensor,
    *,
    relation_sign: torch.Tensor | None = None,
    support_event: bool = False,
    query_response_event: bool = False,
) -> torch.Tensor:
    """Build a minimal legacy-compatible input with every unused field zero.

    Layout after the two cue vectors is: relation sign, bias, support event,
    query-response event, then unused legacy fields.  In particular, no rank
    distance, normalized progress, feedback, target or previous action enters.
    """

    inputs = torch.zeros(cue_batch.shape[0], config.inputsize, device=DEVICE)
    inputs[:, : 2 * config.cs] = cue_batch
    if relation_sign is not None:
        inputs[:, 2 * config.cs] = relation_sign.reshape(-1)
    inputs[:, 2 * config.cs + 1] = 1.0
    if support_event:
        inputs[:, 2 * config.cs + 2] = 1.0
    if query_response_event:
        inputs[:, 2 * config.cs + 3] = 1.0
    return inputs


def _cue_batch(cue_data, observations, cs: int) -> torch.Tensor:
    rows = []
    for subject_index, observation in enumerate(observations):
        rows.append(
            np.concatenate(
                (
                    cue_data[subject_index][observation.left_cue],
                    cue_data[subject_index][observation.right_cue],
                )
            )
        )
    array = np.asarray(rows, dtype="float32")
    if array.shape[1] != 2 * cs:
        raise ValueError(f"expected {2 * cs} cue features, got {array.shape[1]}")
    return torch.as_tensor(array, dtype=torch.float32, device=DEVICE)


def _validate_meta_sign_episode(config, net) -> None:
    if not isinstance(net, (RetroModulRNN, VanillaRNN)):
        raise TypeError("MetaSign runner requires RetroModulRNN or VanillaRNN")
    if config.nbcues != 8:
        raise ValueError("the Liu 2026 symbolic task requires eight cues")
    minimum_support_steps = (
        3 if config.meta_sign_reset_support_trial_state else 2
    )
    if config.support_triallen < minimum_support_steps:
        raise ValueError(
            "support_triallen must allow pair formation, eligibility formation "
            "and a later write when support trial state is reset"
        )
    if config.query_triallen <= NUMRESPONSESTEP:
        raise ValueError("query_triallen does not include the response step")
    if net.subject_embedding_dim != 0:
        raise ValueError(
            "MetaSign candidates do not use participant IDs or subject embeddings"
        )


def run_meta_sign_episode(
    config,
    net: RetroModulRNN | VanillaRNN,
    *,
    rng=None,
    query_blocks: int | None = None,
    task_regime: str = "liu_eval",
    intervention: MetaSignIntervention | None = None,
    mirror_queries: bool = False,
) -> MetaSignEpisodeResult:
    """Run one differentiable sign-only meta-learning episode.

    The caller owns optimization.  Calling ``result.loss.backward()`` propagates
    through the support-time construction of the episode-local plastic state.
    No backward call occurs inside this function.
    """

    _validate_meta_sign_episode(config, net)
    if intervention is None:
        intervention = MetaSignIntervention()
    intervention.validate()
    if rng is None:
        rng = np.random
    if task_regime == "liu_eval":
        if query_blocks is None:
            query_blocks = int(config.query_blocks)
        tasks = build_liu2026_symbolic_subject_tasks(
            n_subjects=config.bs,
            rng=rng,
            nbcues=config.nbcues,
            support_pairs=config.support_pairs,
            support_blocks=config.support_blocks,
            query_blocks=query_blocks,
            randomize_true_rank=True,
        )
    elif task_regime == "meta_train":
        if query_blocks is not None:
            raise ValueError("query_blocks is specific to the Liu evaluation regime")
        tasks = build_meta_training_symbolic_subject_tasks(
            n_subjects=config.bs,
            rng=rng,
            min_nbcues=config.meta_train_min_cues,
            max_nbcues=config.meta_train_max_cues,
            min_support_blocks=config.meta_train_min_support_blocks,
            max_support_blocks=config.meta_train_max_support_blocks,
            max_support_edges=config.meta_train_max_support_edges,
            min_query_fraction=config.meta_train_min_query_fraction,
        )
    else:
        raise ValueError(f"unknown MetaSign task_regime: {task_regime!r}")
    cue_data = generate_liu2026_cue_data(config, rng=rng)

    hidden = net.initialZeroState(config.bs)
    eligibility = net.initialZeroET(config.bs)
    plastic_weights = net.initialZeroPlasticWeights(config.bs)

    num_support_trials = len(tasks[0].support_trials)
    if any(len(task.support_trials) != num_support_trials for task in tasks):
        raise RuntimeError("virtual subjects have inconsistent support schedules")

    effective_support_signs = np.asarray(
        [
            [task.support_trials[trial_index].observation.sign for task in tasks]
            for trial_index in range(num_support_trials)
        ],
        dtype=np.int64,
    )
    if intervention.support_sign_mode == "omitted":
        effective_support_signs.fill(0)
    elif intervention.support_sign_mode == "shuffled":
        for subject_index in range(config.bs):
            effective_support_signs[:, subject_index] = rng.permutation(
                effective_support_signs[:, subject_index]
            )

    support_write_norms = []
    for trial_index in range(num_support_trials):
        if config.meta_sign_reset_support_trial_state:
            hidden = net.initialZeroState(config.bs)
            eligibility = net.initialZeroET(config.bs)
        plastic_weights_before_trial = plastic_weights
        observations = [task.support_trials[trial_index].observation for task in tasks]
        cues = _cue_batch(cue_data, observations, config.cs)
        signs = torch.as_tensor(
            effective_support_signs[trial_index, :, None],
            dtype=torch.float32,
            device=DEVICE,
        )

        for step_index in range(config.support_triallen):
            # The visible pair and relation first form an eligibility trace.
            # Only the final computation may use learned h2DA to write it.
            step_cues = cues if step_index == 0 else torch.zeros_like(cues)
            inputs = _model_input(
                config,
                step_cues,
                relation_sign=signs,
                support_event=True,
            )
            _, _, _, hidden, eligibility, plastic_weights = net(
                inputs,
                hidden,
                eligibility,
                plastic_weights,
                teacher_da=None,
                plastic_write_enabled=(
                    intervention.support_write_enabled
                    and step_index == config.support_triallen - 1
                ),
                eligibility_update_enabled=(
                    intervention.support_eligibility_enabled
                ),
            )
        write_norm = (
            (plastic_weights - plastic_weights_before_trial)
            .reshape(config.bs, -1)
            .norm(dim=1)
            .detach()
            .cpu()
        )
        support_write_norms.append(
            tuple(float(value) for value in write_norm.tolist())
        )

    support_hidden = hidden
    support_eligibility = eligibility
    support_plastic_weights = plastic_weights

    num_query_trials = len(tasks[0].query_trials)
    if any(len(task.query_trials) != num_query_trials for task in tasks):
        raise RuntimeError("virtual subjects have inconsistent query schedules")

    all_logits = []
    all_targets = []
    for trial_index in range(num_query_trials):
        query_trials = [task.query_trials[trial_index] for task in tasks]
        observations = [trial.observation for trial in query_trials]
        if mirror_queries:
            observations = [
                SymbolicQueryObservation(
                    left_cue=observation.right_cue,
                    right_cue=observation.left_cue,
                )
                for observation in observations
            ]
        cues = _cue_batch(cue_data, observations, config.cs)
        targets = torch.as_tensor(
            [
                1 - trial.correct_choice
                if mirror_queries
                else trial.correct_choice
                for trial in query_trials
            ],
            dtype=torch.long,
            device=DEVICE,
        )

        # Each no-feedback query is an independent read branch from the same
        # support endpoint.  Its transient hidden/eligibility changes are
        # discarded and it cannot modify the plastic matrix.
        if config.meta_sign_reset_query_state:
            query_hidden = net.initialZeroState(config.bs)
            query_eligibility = net.initialZeroET(config.bs)
        else:
            query_hidden = support_hidden
            query_eligibility = support_eligibility
        query_plastic_weights = support_plastic_weights
        decision_logits = None
        for step_index in range(config.query_triallen):
            step_cues = cues if step_index == 0 else torch.zeros_like(cues)
            inputs = _model_input(
                config,
                step_cues,
                relation_sign=None,
                support_event=False,
                query_response_event=step_index == NUMRESPONSESTEP,
            )
            (
                activout,
                _,
                _,
                query_hidden,
                query_eligibility,
                query_plastic_weights,
            ) = net(
                inputs,
                query_hidden,
                query_eligibility,
                query_plastic_weights,
                teacher_da=None,
                plastic_write_enabled=intervention.query_write_enabled,
            )
            if step_index == NUMRESPONSESTEP:
                decision_logits = activout

        if decision_logits is None:
            raise RuntimeError("query produced no decision logits")
        all_logits.append(decision_logits)
        all_targets.append(targets)

    logits = torch.stack(all_logits, dim=0)
    targets = torch.stack(all_targets, dim=0)
    loss = F.cross_entropy(logits.reshape(-1, 2), targets.reshape(-1))
    accuracy = float(
        (logits.argmax(dim=-1) == targets).float().mean().detach().cpu()
    )
    return MetaSignEpisodeResult(
        loss=loss,
        logits=logits,
        targets=targets,
        accuracy=accuracy,
        support_hidden=support_hidden,
        support_eligibility=support_eligibility,
        support_plastic_weights=support_plastic_weights,
        subject_tasks=tasks,
        intervention=intervention,
        effective_support_signs=tuple(
            tuple(int(value) for value in row)
            for row in effective_support_signs
        ),
        support_write_norms=tuple(support_write_norms),
    )


def run_meta_sign_eval_episode(
    config,
    net: RetroModulRNN | VanillaRNN,
    *,
    rng=None,
) -> EpisodeRecord:
    """Generate a no-feedback virtual cohort record from a frozen model."""

    if rng is None:
        rng = np.random
    with torch.no_grad():
        result = run_meta_sign_episode(
            config, net, rng=rng, task_regime="liu_eval"
        )
    return symbolic_logits_to_episode_record(
        config,
        subject_tasks=result.subject_tasks,
        logits=result.logits,
        rng=rng,
        provenance={
            "runner": config.meta_sign_candidate_name,
            "support_observation": "symbol_pair_plus_sign",
            "query_fast_state": "read_only",
            "support_trial_state": (
                "reset_hidden_and_eligibility"
                if config.meta_sign_reset_support_trial_state
                else "continuous_hidden_and_eligibility"
            ),
            "query_transient_state": (
                "reset"
                if config.meta_sign_reset_query_state
                else "support_endpoint"
            ),
        },
    )
