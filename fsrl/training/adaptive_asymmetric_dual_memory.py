"""Path diagnostics for AdaptiveAsymmetricDualMemory-v1.

The routines here manipulate only synthetic sign feedback.  They are designed
to distinguish online item updating from visible-boundary replay and to expose
the directional prediction of winner-biased learning under an exactly paired
up/down schedule.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

from fsrl.model.adaptive_asymmetric_dual_memory import (
    AdaptiveAsymmetricDualMemory,
    AdaptiveAsymmetricDualMemoryState,
    AdaptiveAsymmetricIntervention,
)
from fsrl.model.dual_route_constructive_rank import ConsolidationEvent
from fsrl.training.dcr_changepoint import (
    SyntheticChangepointResult,
    SyntheticChangepointTask,
    run_synthetic_changepoint_episode,
)


@dataclass
class BoundaryReplayResult:
    loss: torch.Tensor
    prequential_loss: torch.Tensor
    block_end_loss: torch.Tensor
    pre_support_logits: torch.Tensor
    block_end_logits: torch.Tensor
    final_state: AdaptiveAsymmetricDualMemoryState


@dataclass
class UpdateTimingComparison:
    online: SyntheticChangepointResult
    boundary_replay: BoundaryReplayResult
    prequential_logit_mean_absolute_difference: float
    block_end_logit_max_absolute_error: float


@dataclass
class PairedDirectionResult:
    moved_preference: torch.Tensor
    up_reversal_latency: int
    down_reversal_latency: int
    up_adaptation_gain: float
    down_adaptation_gain: float
    mirror_error: float


@dataclass
class DownstreamDirectionResult:
    nonanchor_correct_probability: torch.Tensor
    up_change: float
    down_change: float
    post_interaction: float


def _query_block(
    model: AdaptiveAsymmetricDualMemory,
    state: AdaptiveAsymmetricDualMemoryState,
    task: SyntheticChangepointTask,
    block: int,
    intervention: AdaptiveAsymmetricIntervention,
) -> torch.Tensor:
    before = tuple(tensor.clone() for tensor in state.tensors())
    logits = []
    for query in range(task.query_left.shape[1]):
        logits.append(
            model.query_pair(
                state,
                task.cue_features,
                task.query_left[block, query],
                task.query_right[block, query],
                intervention=intervention,
            ).logits
        )
    for old, current in zip(before, state.tensors(), strict=True):
        if not torch.equal(old, current):
            raise RuntimeError("read-only boundary-replay query changed fast state")
    return torch.stack(logits)


def run_aadm_boundary_replay_episode(
    model: AdaptiveAsymmetricDualMemory,
    task: SyntheticChangepointTask,
    *,
    intervention: AdaptiveAsymmetricIntervention | None = None,
    stochastic: bool = False,
) -> BoundaryReplayResult:
    """Keep pair writes online but replay item-value updates at visible boundaries.

    The replay receives the identical sign multiset in the identical order.
    Process noise is deliberately disallowed so chronology is the only changed
    factor.  The model never receives a hidden block number.
    """

    if stochastic:
        raise ValueError("matched boundary replay requires stochastic=False")
    if intervention is None:
        intervention = AdaptiveAsymmetricIntervention()
    state = model.initial_state(task.batch_size)
    local_only_write = replace(intervention, item_update_enabled=False)
    item_only_replay = replace(intervention, local_update_enabled=False)
    pre_support_logits = []
    block_end_logits = []

    for block in range(task.n_blocks):
        trial_logits = []
        replay_buffer = []
        for trial in range(task.support_left.shape[1]):
            trial_logits.append(
                model.query_pair(
                    state,
                    task.cue_features,
                    task.support_left[block, trial],
                    task.support_right[block, trial],
                    intervention=intervention,
                ).logits
            )
            state, _ = model.observe_relation(
                state,
                task.cue_features,
                task.support_left[block, trial],
                task.support_right[block, trial],
                task.support_sign[block, trial],
                intervention=local_only_write,
                stochastic=False,
            )
            replay_buffer.append(
                (
                    task.support_left[block, trial],
                    task.support_right[block, trial],
                    task.support_sign[block, trial],
                )
            )
        pre_support_logits.append(torch.stack(trial_logits))

        if intervention.item_update_enabled:
            for left, right, sign in replay_buffer:
                feedback_count = state.update_count
                state, _ = model.observe_relation(
                    state,
                    task.cue_features,
                    left,
                    right,
                    sign,
                    intervention=item_only_replay,
                    stochastic=False,
                )
                # Replaying a stored feedback is not a second experienced trial.
                state = replace(state, update_count=feedback_count)
        state, _ = model.consolidate(
            state, ConsolidationEvent.VISIBLE_BOUNDARY, stochastic=False
        )
        block_end_logits.append(
            _query_block(model, state, task, block, intervention)
        )

    stacked_pre = torch.stack(pre_support_logits)
    stacked_end = torch.stack(block_end_logits)
    pre_loss = F.cross_entropy(
        stacked_pre.reshape(-1, 2), task.support_target.reshape(-1)
    )
    end_loss = F.cross_entropy(
        stacked_end.reshape(-1, 2), task.query_target.reshape(-1)
    )
    return BoundaryReplayResult(
        loss=0.5 * pre_loss + 0.5 * end_loss,
        prequential_loss=pre_loss,
        block_end_loss=end_loss,
        pre_support_logits=stacked_pre,
        block_end_logits=stacked_end,
        final_state=state,
    )


def compare_aadm_update_timing(
    model: AdaptiveAsymmetricDualMemory,
    task: SyntheticChangepointTask,
    *,
    intervention: AdaptiveAsymmetricIntervention | None = None,
) -> UpdateTimingComparison:
    if intervention is None:
        intervention = AdaptiveAsymmetricIntervention()
    online = run_synthetic_changepoint_episode(
        model, task, intervention=intervention, stochastic=False
    )
    replay = run_aadm_boundary_replay_episode(
        model, task, intervention=intervention, stochastic=False
    )
    return UpdateTimingComparison(
        online=online,
        boundary_replay=replay,
        prequential_logit_mean_absolute_difference=float(
            (online.pre_support_logits - replay.pre_support_logits)
            .abs()
            .mean()
            .detach()
            .cpu()
        ),
        block_end_logit_max_absolute_error=float(
            (online.block_end_logits - replay.block_end_logits)
            .abs()
            .max()
            .detach()
            .cpu()
        ),
    )


def _moved_preference(
    model: AdaptiveAsymmetricDualMemory,
    state: AdaptiveAsymmetricDualMemoryState,
    cue_features: torch.Tensor,
    moved_cue: int,
    intervention: AdaptiveAsymmetricIntervention,
) -> torch.Tensor:
    batch_size = state.item_values.shape[0]
    device = state.item_values.device
    probabilities = []
    for other in range(model.config.max_items):
        if other == moved_cue:
            continue
        probabilities.append(
            model.query_pair(
                state,
                cue_features,
                torch.full((batch_size,), moved_cue, dtype=torch.long, device=device),
                torch.full((batch_size,), other, dtype=torch.long, device=device),
                intervention=intervention,
            ).probability_first
        )
    return torch.stack(probabilities).mean(dim=0)


def run_paired_direction_diagnostic(
    model: AdaptiveAsymmetricDualMemory,
    *,
    intervention: AdaptiveAsymmetricIntervention | None = None,
    pre_repeats: int = 4,
    post_repeats: int = 8,
) -> PairedDirectionResult:
    """Run sign-inverted, trial-order-paired upward/downward anchor moves.

    Subject 0 learns 0>1>...>m and then sees m>0 (the old loser moves
    upward).  Subject 1 receives the exact sign inverse, so m is initially the
    winner and later loses to 0.  Pair identity, side, and order are identical.
    A symmetric update must therefore remain a probability mirror.
    """

    if pre_repeats <= 0 or post_repeats <= 0:
        raise ValueError("pre_repeats and post_repeats must be positive")
    if intervention is None:
        intervention = AdaptiveAsymmetricIntervention()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    n_items = model.config.max_items
    moved_cue = n_items - 1
    cue_features = torch.zeros(
        2, n_items, model.config.cue_dim, device=device, dtype=dtype
    )
    state = model.initial_state(2)
    paired_sign = torch.tensor([1.0, -1.0], device=device, dtype=dtype)

    def observe(left: int, right: int) -> None:
        nonlocal state
        cues_left = torch.full((2,), left, dtype=torch.long, device=device)
        cues_right = torch.full((2,), right, dtype=torch.long, device=device)
        state, _ = model.observe_relation(
            state,
            cue_features,
            cues_left,
            cues_right,
            paired_sign,
            intervention=intervention,
            stochastic=False,
        )

    for _ in range(pre_repeats):
        for left in range(n_items - 1):
            observe(left, left + 1)

    preference = [
        _moved_preference(
            model, state, cue_features, moved_cue, intervention
        )
    ]
    post_pairs = [(moved_cue, 0), *[(left, left + 1) for left in range(n_items - 2)]]
    for _ in range(post_repeats):
        for left, right in post_pairs:
            observe(left, right)
            preference.append(
                _moved_preference(
                    model, state, cue_features, moved_cue, intervention
                )
            )
    trajectory = torch.stack(preference)
    up_crossing = torch.nonzero(trajectory[1:, 0] >= 0.5, as_tuple=False)
    down_crossing = torch.nonzero(trajectory[1:, 1] <= 0.5, as_tuple=False)
    missing = int(trajectory.shape[0])
    up_latency = int(up_crossing[0, 0]) + 1 if len(up_crossing) else missing
    down_latency = int(down_crossing[0, 0]) + 1 if len(down_crossing) else missing
    return PairedDirectionResult(
        moved_preference=trajectory,
        up_reversal_latency=up_latency,
        down_reversal_latency=down_latency,
        up_adaptation_gain=float((trajectory[-1, 0] - trajectory[0, 0]).detach().cpu()),
        down_adaptation_gain=float((trajectory[0, 1] - trajectory[-1, 1]).detach().cpu()),
        mirror_error=float(
            (trajectory[:, 0] + trajectory[:, 1] - 1.0)
            .abs()
            .max()
            .detach()
            .cpu()
        ),
    )


def run_graham_spitzer_downstream_diagnostic(
    model: AdaptiveAsymmetricDualMemory,
    *,
    intervention: AdaptiveAsymmetricIntervention | None = None,
    pre_repeats: int = 4,
    post_repeats: int = 8,
) -> DownstreamDirectionResult:
    """Measure the published up/down effect on unchanged non-anchor TI pairs.

    Both synthetic groups learn the identical seven-item hierarchy before the
    change.  Afterwards both receive the new relation 0>6; the up group no
    longer receives 0<1 feedback, while the down group no longer receives 5<6
    feedback.  Queries exclude both moved anchors and adjacent pairs, matching
    the downstream non-anchor TI construct rather than anchor reversal time.
    Trial order alternates forward/reverse within each phase to prevent a
    single arbitrary edge ordering from defining the interaction.
    """

    if model.config.max_items < 7:
        raise ValueError("Graham-Spitzer diagnostic requires at least seven cues")
    if pre_repeats <= 0 or post_repeats <= 0:
        raise ValueError("pre_repeats and post_repeats must be positive")
    if intervention is None:
        intervention = AdaptiveAsymmetricIntervention()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    cue_features = torch.zeros(
        2,
        model.config.max_items,
        model.config.cue_dim,
        device=device,
        dtype=dtype,
    )
    state = model.initial_state(2)

    def observe(
        up_relation: tuple[int, int, float],
        down_relation: tuple[int, int, float],
    ) -> None:
        nonlocal state
        state, _ = model.observe_relation(
            state,
            cue_features,
            torch.tensor(
                [up_relation[0], down_relation[0]], device=device, dtype=torch.long
            ),
            torch.tensor(
                [up_relation[1], down_relation[1]], device=device, dtype=torch.long
            ),
            torch.tensor(
                [up_relation[2], down_relation[2]], device=device, dtype=dtype
            ),
            intervention=intervention,
            stochastic=False,
        )

    def downstream_probability() -> torch.Tensor:
        probabilities = []
        for low in range(1, 6):
            for high in range(low + 2, 6):
                trace = model.query_pair(
                    state,
                    cue_features,
                    torch.full((2,), low, device=device, dtype=torch.long),
                    torch.full((2,), high, device=device, dtype=torch.long),
                    intervention=intervention,
                )
                probabilities.append(1.0 - trace.probability_first)
        return torch.stack(probabilities).mean(dim=0)

    pre_edges = [(cue, cue + 1, -1.0) for cue in range(6)]
    for repeat in range(pre_repeats):
        ordered = pre_edges if repeat % 2 == 0 else list(reversed(pre_edges))
        for relation in ordered:
            observe(relation, relation)

    trajectory = [downstream_probability()]
    up_edges = [(cue, cue + 1, -1.0) for cue in range(1, 6)]
    down_edges = [(cue, cue + 1, -1.0) for cue in range(5)]
    new_relation = (0, 6, 1.0)
    for repeat in range(post_repeats):
        observe(new_relation, new_relation)
        ordered_up = up_edges if repeat % 2 == 0 else list(reversed(up_edges))
        ordered_down = (
            down_edges if repeat % 2 == 0 else list(reversed(down_edges))
        )
        for up_relation, down_relation in zip(
            ordered_up, ordered_down, strict=True
        ):
            observe(up_relation, down_relation)
        trajectory.append(downstream_probability())

    probability = torch.stack(trajectory)
    up_change = probability[-1, 0] - probability[0, 0]
    down_change = probability[-1, 1] - probability[0, 1]
    return DownstreamDirectionResult(
        nonanchor_correct_probability=probability,
        up_change=float(up_change.detach().cpu()),
        down_change=float(down_change.detach().cpu()),
        post_interaction=float(
            (probability[-1, 0] - probability[-1, 1]).detach().cpu()
        ),
    )
