"""Shared conversion from symbolic cohort logits to an analysis record."""

import numpy as np
import torch

from fsrl.episode.types import EpisodeRecord, TestResponse


def symbolic_logits_to_episode_record(
    config,
    *,
    subject_tasks,
    logits: torch.Tensor,
    rng,
    provenance: dict,
) -> EpisodeRecord:
    """Sample choices and retain each subject's independent rank metadata."""

    probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    responses = []
    correct_values = []
    adjacent_values = []
    nonadjacent_values = []
    for trial_index in range(logits.shape[0]):
        for subject_index, task in enumerate(subject_tasks):
            trial = task.query_trials[trial_index]
            observation = trial.observation
            p_right = float(probabilities[trial_index, subject_index, 1])
            action = 1 if float(rng.random()) < p_right else 0
            selected_cue = (
                observation.left_cue if action == 0 else observation.right_cue
            )
            pair = tuple(sorted((observation.left_cue, observation.right_cue)))
            correct = action == trial.correct_choice
            confidence = float(
                abs(probabilities[trial_index, subject_index, 0] - p_right)
            )
            responses.append(
                TestResponse(
                    block_id=trial.block_index,
                    pair=pair,
                    action=action,
                    prefers_i_over_j=selected_cue == pair[0],
                    correct=correct,
                    batch_index=subject_index,
                    confidence=confidence,
                    rt_proxy=float(-np.log(confidence + 1e-10)),
                )
            )
            correct_values.append(float(correct))
            if trial.position_pair[1] - trial.position_pair[0] == 1:
                adjacent_values.append(float(correct))
            else:
                nonadjacent_values.append(float(correct))

    first_task = subject_tasks[0]
    subject_support_pairs = {}
    for subject_index, task in enumerate(subject_tasks):
        subject_support_pairs[subject_index] = sorted(
            {
                tuple(
                    sorted(
                        (
                            trial.observation.left_cue,
                            trial.observation.right_cue,
                        )
                    )
                )
                for trial in task.support_trials
            }
        )

    return EpisodeRecord(
        nbcues=config.nbcues,
        supervision_set=subject_support_pairs[0],
        query_set=sorted(
            {
                tuple(
                    sorted(
                        (
                            trial.observation.left_cue,
                            trial.observation.right_cue,
                        )
                    )
                )
                for trial in first_task.query_trials
            }
        ),
        test_responses=responses,
        true_rank=list(first_task.true_rank),
        subject_true_ranks={
            task.subject_index: list(task.true_rank) for task in subject_tasks
        },
        support_order=[
            (
                trial.observation.left_cue,
                trial.observation.right_cue,
                trial.observation.sign,
            )
            for trial in first_task.support_trials
        ],
        subject_support_pairs=subject_support_pairs,
        provenance=dict(provenance),
        test_perf=float(np.mean(correct_values)),
        test_perf_adjacent=float(np.mean(adjacent_values)),
        test_perf_nonadjacent=float(np.mean(nonadjacent_values)),
    )
