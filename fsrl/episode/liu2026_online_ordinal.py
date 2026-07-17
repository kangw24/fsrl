"""Sign-only Liu/meta episode for online ordinal prediction-error learning."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from fsrl.episode.symbolic_record import symbolic_logits_to_episode_record
from fsrl.episode.types import EpisodeRecord
from fsrl.model.online_ordinal import OnlineOrdinalPredictionError
from fsrl.task.liu2026 import (
    Liu2026SymbolicSubjectTask,
    build_liu2026_symbolic_subject_tasks,
    build_meta_training_symbolic_subject_tasks,
)


@dataclass(frozen=True)
class OnlineOrdinalEpisodeResult:
    loss: torch.Tensor
    logits: torch.Tensor
    targets: torch.Tensor
    accuracy: float
    score_state: torch.Tensor
    support_prediction_errors: torch.Tensor
    subject_tasks: tuple[Liu2026SymbolicSubjectTask, ...]


def run_online_ordinal_episode(
    config,
    net: OnlineOrdinalPredictionError,
    *,
    rng=None,
    query_blocks: int | None = None,
    task_regime: str = "liu_eval",
    mirror_queries: bool = False,
) -> OnlineOrdinalEpisodeResult:
    """Run one differentiable episode whose query phase is read-only."""

    if not isinstance(net, OnlineOrdinalPredictionError):
        raise TypeError("online ordinal runner requires OnlineOrdinalPredictionError")
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
        raise ValueError(f"unknown online ordinal task_regime: {task_regime!r}")

    state = net.initial_state(config.bs)
    errors = []
    num_support_trials = len(tasks[0].support_trials)
    for trial_index in range(num_support_trials):
        observations = [task.support_trials[trial_index].observation for task in tasks]
        left = torch.as_tensor(
            [observation.left_cue for observation in observations],
            dtype=torch.long,
            device=state.device,
        )
        right = torch.as_tensor(
            [observation.right_cue for observation in observations],
            dtype=torch.long,
            device=state.device,
        )
        sign = torch.as_tensor(
            [observation.sign for observation in observations],
            dtype=state.dtype,
            device=state.device,
        )
        state, error = net.update_support(state, left, right, sign)
        errors.append(error)

    all_logits = []
    all_targets = []
    num_query_trials = len(tasks[0].query_trials)
    for trial_index in range(num_query_trials):
        trials = [task.query_trials[trial_index] for task in tasks]
        left_values = [trial.observation.left_cue for trial in trials]
        right_values = [trial.observation.right_cue for trial in trials]
        targets = [trial.correct_choice for trial in trials]
        if mirror_queries:
            left_values, right_values = right_values, left_values
            targets = [1 - target for target in targets]
        left = torch.as_tensor(left_values, dtype=torch.long, device=state.device)
        right = torch.as_tensor(right_values, dtype=torch.long, device=state.device)
        all_logits.append(net.query_logits(state, left, right))
        all_targets.append(
            torch.as_tensor(targets, dtype=torch.long, device=state.device)
        )

    logits = torch.stack(all_logits, dim=0)
    targets = torch.stack(all_targets, dim=0)
    loss = F.cross_entropy(logits.reshape(-1, 2), targets.reshape(-1))
    accuracy = float(
        (logits.argmax(dim=-1) == targets).float().mean().detach().cpu()
    )
    return OnlineOrdinalEpisodeResult(
        loss=loss,
        logits=logits,
        targets=targets,
        accuracy=accuracy,
        score_state=state,
        support_prediction_errors=torch.stack(errors, dim=0),
        subject_tasks=tasks,
    )


def run_online_ordinal_eval_episode(
    config,
    net: OnlineOrdinalPredictionError,
    *,
    rng=None,
) -> EpisodeRecord:
    if rng is None:
        rng = np.random
    with torch.no_grad():
        result = run_online_ordinal_episode(
            config, net, rng=rng, task_regime="liu_eval"
        )
    return symbolic_logits_to_episode_record(
        config,
        subject_tasks=result.subject_tasks,
        logits=result.logits,
        rng=rng,
        provenance={
            "runner": "online_ordinal_prediction_error_v1",
            "support_observation": "symbol_pair_plus_sign",
            "support_update": "shared_scalar_prediction_error",
            "query_fast_state": "read_only",
        },
    )
