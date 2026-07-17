"""Synthetic prequential changepoint tasks for DCR-v1.

These tasks are mechanism-identification environments, not human-data fits.
Each virtual subject receives only cue identities and binary high/low feedback.
The correct choice is used after a read-only query to form the outer loss and
is never passed to the model state update.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import torch
import torch.nn.functional as F

from fsrl.model.dual_route_constructive_rank import (
    ConsolidationEvent,
    DualRouteConstructiveRank,
    DualRouteConstructiveRankState,
    DualRouteIntervention,
    EncodingTrace,
    QueryTrace,
)


BoundaryTiming = Literal["visible", "after_probe"]


@dataclass(frozen=True)
class SyntheticChangepointTask:
    """A population batch with independently timed anchor reversals."""

    cue_features: torch.Tensor
    initial_order: torch.Tensor
    changed_order: torch.Tensor
    moved_cue: torch.Tensor
    change_block: torch.Tensor
    support_left: torch.Tensor
    support_right: torch.Tensor
    support_sign: torch.Tensor
    support_target: torch.Tensor
    support_is_first_change: torch.Tensor
    query_left: torch.Tensor
    query_right: torch.Tensor
    query_target: torch.Tensor
    query_is_direct: torch.Tensor
    query_is_incident: torch.Tensor
    query_is_conflict_incident: torch.Tensor
    conflict_pair: torch.Tensor
    n_items: int

    @property
    def batch_size(self) -> int:
        return int(self.cue_features.shape[0])

    @property
    def n_blocks(self) -> int:
        return int(self.support_left.shape[0])


@dataclass
class SyntheticChangepointResult:
    loss: torch.Tensor
    prequential_loss: torch.Tensor
    block_end_loss: torch.Tensor
    postchange_prequential_loss: torch.Tensor
    direct_loss: torch.Tensor
    indirect_loss: torch.Tensor
    incident_loss: torch.Tensor
    disjoint_loss: torch.Tensor
    prequential_accuracy: float
    block_end_accuracy: float
    direct_accuracy: float
    indirect_accuracy: float
    incident_accuracy: float
    disjoint_accuracy: float
    conflict_incident_uncertainty_change: float
    conflict_disjoint_uncertainty_change: float
    conflict_incident_target_probability_change: float
    conflict_disjoint_target_probability_change: float
    final_state: DualRouteConstructiveRankState
    support_traces: list[EncodingTrace]
    pre_support_logits: torch.Tensor
    block_start_traces: list[list[QueryTrace]]
    after_first_support_traces: list[list[QueryTrace]]
    block_end_logits: torch.Tensor


def _randperm(n: int, generator: torch.Generator) -> list[int]:
    return torch.randperm(n, generator=generator, device="cpu").tolist()


def _move_anchor(order: list[int], move_up: bool) -> tuple[list[int], int, tuple[int, int]]:
    if move_up:
        moved = order[-1]
        old_neighbor = order[-2]
        changed = [moved, *order[:-1]]
    else:
        moved = order[0]
        old_neighbor = order[1]
        changed = [*order[1:], moved]
    return changed, moved, tuple(sorted((moved, old_neighbor)))


def _position(order: list[int]) -> dict[int, int]:
    return {cue: position for position, cue in enumerate(order)}


def _orient_pair(
    pair: tuple[int, int],
    positions: dict[int, int],
    generator: torch.Generator,
) -> tuple[int, int, int, int]:
    first, second = pair
    swap = bool(torch.randint(0, 2, (1,), generator=generator, device="cpu"))
    left, right = (second, first) if swap else (first, second)
    sign = 1 if positions[left] < positions[right] else -1
    target = 0 if sign == 1 else 1
    return left, right, sign, target


def sample_synthetic_changepoint_task(
    model: DualRouteConstructiveRank,
    *,
    batch_size: int,
    n_items: int,
    n_blocks: int,
    generator: torch.Generator,
) -> SyntheticChangepointTask:
    """Sample sign-only rank reversals with no fixed change time shortcut.

    One extreme item moves to the opposite extreme.  Its old adjacent pair is
    repeated before the change and presented with the reversed sign as the
    first feedback in the subject's change block.  Change block and direction
    vary independently across subjects.  Every block has the same number of
    support trials: the current adjacent chain plus that diagnostic old pair.
    """

    config = model.config
    if not 4 <= n_items <= config.max_items:
        raise ValueError("changepoint tasks require 4..max_items cues")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if n_blocks < 4:
        raise ValueError("n_blocks must leave pre- and post-change evidence")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    codebook = torch.randn(
        config.max_items,
        config.cue_dim,
        generator=generator,
        device="cpu",
        dtype=dtype,
    ).to(device)
    cue_features = codebook.unsqueeze(0).expand(batch_size, -1, -1).clone()

    initial_orders: list[list[int]] = []
    changed_orders: list[list[int]] = []
    moved_cues: list[int] = []
    old_pairs: list[tuple[int, int]] = []
    change_blocks: list[int] = []
    for _ in range(batch_size):
        initial = _randperm(n_items, generator)
        move_up = bool(torch.randint(0, 2, (1,), generator=generator, device="cpu"))
        changed, moved, old_pair = _move_anchor(initial, move_up)
        # At least one complete pre-change and two complete post-change blocks.
        change_block = int(
            torch.randint(1, n_blocks - 1, (1,), generator=generator, device="cpu")
        )
        initial_orders.append(initial)
        changed_orders.append(changed)
        moved_cues.append(moved)
        old_pairs.append(old_pair)
        change_blocks.append(change_block)

    support_length = n_items
    support: list[list[list[tuple[int, int, int, int, bool]]]] = []
    queries: list[list[list[tuple[int, int, int, bool, bool]]]] = []
    query_pairs = list(combinations(range(n_items), 2))
    for block in range(n_blocks):
        block_support = []
        block_queries = []
        for subject in range(batch_size):
            changed = block >= change_blocks[subject]
            order = changed_orders[subject] if changed else initial_orders[subject]
            positions = _position(order)
            adjacent = [
                tuple(sorted((order[index], order[index + 1])))
                for index in range(n_items - 1)
            ]
            diagnostic_pair = old_pairs[subject]
            edges = [*adjacent, diagnostic_pair]
            if block == change_blocks[subject]:
                remaining = edges.copy()
                remaining.remove(diagnostic_pair)
                ordered_edges = [
                    diagnostic_pair,
                    *[remaining[index] for index in _randperm(len(remaining), generator)],
                ]
            else:
                ordered_edges = [edges[index] for index in _randperm(len(edges), generator)]
            trials = []
            for trial, pair in enumerate(ordered_edges):
                left, right, sign, target = _orient_pair(pair, positions, generator)
                trials.append(
                    (
                        left,
                        right,
                        sign,
                        target,
                        block == change_blocks[subject] and trial == 0,
                    )
                )
            if len(trials) != support_length:
                raise RuntimeError("internal support schedule length mismatch")
            block_support.append(trials)

            direct_pairs = set(adjacent)
            subject_queries = []
            conflict_endpoints = set(old_pairs[subject])
            for pair_index in _randperm(len(query_pairs), generator):
                pair = query_pairs[pair_index]
                left, right, _, target = _orient_pair(pair, positions, generator)
                subject_queries.append(
                    (
                        left,
                        right,
                        target,
                        pair in direct_pairs,
                        moved_cues[subject] in pair,
                        bool(conflict_endpoints.intersection(pair)),
                    )
                )
            block_queries.append(subject_queries)
        support.append(block_support)
        queries.append(block_queries)

    def support_tensor(index: int, tensor_dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(
            [
                [
                    [support[block][subject][trial][index] for subject in range(batch_size)]
                    for trial in range(support_length)
                ]
                for block in range(n_blocks)
            ],
            dtype=tensor_dtype,
            device=device,
        )

    query_length = len(query_pairs)

    def query_tensor(index: int, tensor_dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(
            [
                [
                    [queries[block][subject][trial][index] for subject in range(batch_size)]
                    for trial in range(query_length)
                ]
                for block in range(n_blocks)
            ],
            dtype=tensor_dtype,
            device=device,
        )

    return SyntheticChangepointTask(
        cue_features=cue_features,
        initial_order=torch.tensor(initial_orders, dtype=torch.long, device=device),
        changed_order=torch.tensor(changed_orders, dtype=torch.long, device=device),
        moved_cue=torch.tensor(moved_cues, dtype=torch.long, device=device),
        change_block=torch.tensor(change_blocks, dtype=torch.long, device=device),
        support_left=support_tensor(0, torch.long),
        support_right=support_tensor(1, torch.long),
        support_sign=support_tensor(2, dtype),
        support_target=support_tensor(3, torch.long),
        support_is_first_change=support_tensor(4, torch.bool),
        query_left=query_tensor(0, torch.long),
        query_right=query_tensor(1, torch.long),
        query_target=query_tensor(2, torch.long),
        query_is_direct=query_tensor(3, torch.bool),
        query_is_incident=query_tensor(4, torch.bool),
        query_is_conflict_incident=query_tensor(5, torch.bool),
        conflict_pair=torch.tensor(old_pairs, dtype=torch.long, device=device),
        n_items=n_items,
    )


def _query_all(
    model: DualRouteConstructiveRank,
    state: DualRouteConstructiveRankState,
    task: SyntheticChangepointTask,
    block: int,
    intervention: DualRouteIntervention,
) -> tuple[list[QueryTrace], torch.Tensor]:
    before = tuple(tensor.clone() for tensor in state.tensors())
    traces = []
    for query in range(task.query_left.shape[1]):
        traces.append(
            model.query_pair(
                state,
                task.cue_features,
                task.query_left[block, query],
                task.query_right[block, query],
                intervention=intervention,
            )
        )
    for old, current in zip(before, state.tensors(), strict=True):
        if not torch.equal(old, current):
            raise RuntimeError("read-only changepoint probe changed model state")
    return traces, torch.stack([trace.logits for trace in traces])


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not bool(mask.any()):
        return values.new_tensor(float("nan"))
    return values[mask].mean()


def _masked_accuracy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return float("nan")
    prediction = logits.argmax(dim=-1)
    return float((prediction[mask] == targets[mask]).float().mean().detach().cpu())


def run_synthetic_changepoint_episode(
    model: DualRouteConstructiveRank,
    task: SyntheticChangepointTask,
    *,
    intervention: DualRouteIntervention | None = None,
    boundary_timing: BoundaryTiming = "visible",
    stochastic: bool = True,
    generator: torch.Generator | None = None,
) -> SyntheticChangepointResult:
    """Run pre-feedback choices, feedback writes, and visible-boundary probes."""

    if intervention is None:
        intervention = DualRouteIntervention()
    if boundary_timing not in ("visible", "after_probe"):
        raise ValueError("boundary_timing must be 'visible' or 'after_probe'")
    state = model.initial_state(task.batch_size)
    support_traces: list[EncodingTrace] = []
    pre_support_logits = []
    block_end_logits = []
    block_start_traces: list[list[QueryTrace]] = []
    after_first_support_traces: list[list[QueryTrace]] = []

    for block in range(task.n_blocks):
        start_traces, _ = _query_all(model, state, task, block, intervention)
        block_start_traces.append(start_traces)
        trial_logits = []
        for trial in range(task.support_left.shape[1]):
            prediction = model.query_pair(
                state,
                task.cue_features,
                task.support_left[block, trial],
                task.support_right[block, trial],
                intervention=intervention,
            )
            trial_logits.append(prediction.logits)
            state, encoding = model.observe_relation(
                state,
                task.cue_features,
                task.support_left[block, trial],
                task.support_right[block, trial],
                task.support_sign[block, trial],
                intervention=intervention,
                stochastic=stochastic,
                generator=generator,
            )
            support_traces.append(encoding)
            if trial == 0:
                after_traces, _ = _query_all(model, state, task, block, intervention)
                after_first_support_traces.append(after_traces)
        pre_support_logits.append(torch.stack(trial_logits))

        if boundary_timing == "visible":
            state, _ = model.consolidate(
                state,
                ConsolidationEvent.VISIBLE_BOUNDARY,
                stochastic=stochastic,
                generator=generator,
            )
        _, end_logits = _query_all(model, state, task, block, intervention)
        block_end_logits.append(end_logits)
        if boundary_timing == "after_probe":
            state, _ = model.consolidate(
                state,
                ConsolidationEvent.VISIBLE_BOUNDARY,
                stochastic=stochastic,
                generator=generator,
            )

    stacked_pre = torch.stack(pre_support_logits)
    stacked_end = torch.stack(block_end_logits)
    pre_loss = F.cross_entropy(
        stacked_pre.reshape(-1, 2), task.support_target.reshape(-1), reduction="none"
    ).reshape_as(task.support_target)
    end_loss = F.cross_entropy(
        stacked_end.reshape(-1, 2), task.query_target.reshape(-1), reduction="none"
    ).reshape_as(task.query_target)
    block_indices = torch.arange(task.n_blocks, device=task.change_block.device)[:, None, None]
    post_mask = block_indices >= task.change_block[None, None, :]
    post_mask = post_mask.expand_as(task.support_target)
    direct = task.query_is_direct
    incident = task.query_is_conflict_incident

    incident_uncertainty_changes = []
    disjoint_uncertainty_changes = []
    incident_target_probability_changes = []
    disjoint_target_probability_changes = []
    for subject in range(task.batch_size):
        block = int(task.change_block[subject])
        start_uncertainty = torch.stack(
            [trace.uncertainty[subject] for trace in block_start_traces[block]]
        )
        after_uncertainty = torch.stack(
            [trace.uncertainty[subject] for trace in after_first_support_traces[block]]
        )
        subject_incident = incident[block, :, subject]
        incident_uncertainty_changes.append(
            (after_uncertainty - start_uncertainty)[subject_incident].mean()
        )
        disjoint_uncertainty_changes.append(
            (after_uncertainty - start_uncertainty)[~subject_incident].mean()
        )
        start_probability = torch.stack(
            [trace.probability_first[subject] for trace in block_start_traces[block]]
        )
        after_probability = torch.stack(
            [trace.probability_first[subject] for trace in after_first_support_traces[block]]
        )
        target = task.query_target[block, :, subject]
        start_target_probability = torch.where(target == 0, start_probability, 1.0 - start_probability)
        after_target_probability = torch.where(target == 0, after_probability, 1.0 - after_probability)
        incident_target_probability_changes.append(
            (after_target_probability - start_target_probability)[subject_incident].mean()
        )
        disjoint_target_probability_changes.append(
            (after_target_probability - start_target_probability)[~subject_incident].mean()
        )

    return SyntheticChangepointResult(
        loss=0.5 * pre_loss.mean() + 0.5 * end_loss.mean(),
        prequential_loss=pre_loss.mean(),
        block_end_loss=end_loss.mean(),
        postchange_prequential_loss=_masked_mean(pre_loss, post_mask),
        direct_loss=_masked_mean(end_loss, direct),
        indirect_loss=_masked_mean(end_loss, ~direct),
        incident_loss=_masked_mean(end_loss, incident),
        disjoint_loss=_masked_mean(end_loss, ~incident),
        prequential_accuracy=_masked_accuracy(stacked_pre, task.support_target, torch.ones_like(task.support_target, dtype=torch.bool)),
        block_end_accuracy=_masked_accuracy(stacked_end, task.query_target, torch.ones_like(task.query_target, dtype=torch.bool)),
        direct_accuracy=_masked_accuracy(stacked_end, task.query_target, direct),
        indirect_accuracy=_masked_accuracy(stacked_end, task.query_target, ~direct),
        incident_accuracy=_masked_accuracy(stacked_end, task.query_target, incident),
        disjoint_accuracy=_masked_accuracy(stacked_end, task.query_target, ~incident),
        conflict_incident_uncertainty_change=float(torch.stack(incident_uncertainty_changes).mean().detach().cpu()),
        conflict_disjoint_uncertainty_change=float(torch.stack(disjoint_uncertainty_changes).mean().detach().cpu()),
        conflict_incident_target_probability_change=float(torch.stack(incident_target_probability_changes).mean().detach().cpu()),
        conflict_disjoint_target_probability_change=float(torch.stack(disjoint_target_probability_changes).mean().detach().cpu()),
        final_state=state,
        support_traces=support_traces,
        pre_support_logits=stacked_pre,
        block_start_traces=block_start_traces,
        after_first_support_traces=after_first_support_traces,
        block_end_logits=stacked_end,
    )
