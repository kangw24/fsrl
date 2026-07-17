"""Synthetic-only meta-learning tasks for DualRouteConstructiveRank-v1."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import torch
import torch.nn.functional as F

from fsrl.model.dual_route_constructive_rank import (
    ConsolidationEvent,
    DualRouteConstructiveRank,
    DualRouteConstructiveRankState,
    DualRouteIntervention,
    EncodingTrace,
)


@dataclass(frozen=True)
class SyntheticDualRouteTask:
    cue_features: torch.Tensor
    true_order: torch.Tensor
    support_left: torch.Tensor
    support_right: torch.Tensor
    support_sign: torch.Tensor
    query_left: torch.Tensor
    query_right: torch.Tensor
    query_target: torch.Tensor
    query_is_direct: torch.Tensor
    n_items: int

    @property
    def batch_size(self) -> int:
        return int(self.cue_features.shape[0])


@dataclass
class SyntheticDualRouteResult:
    loss: torch.Tensor
    direct_loss: torch.Tensor
    indirect_loss: torch.Tensor
    accuracy: float
    direct_accuracy: float
    indirect_accuracy: float
    final_state: DualRouteConstructiveRankState
    support_traces: list[EncodingTrace]
    logits: torch.Tensor
    targets: torch.Tensor
    direct_mask: torch.Tensor


def _randperm(n: int, generator: torch.Generator) -> list[int]:
    return torch.randperm(n, generator=generator, device="cpu").tolist()


def _sample_connected_edges(
    n_items: int, edge_count: int, generator: torch.Generator
) -> list[tuple[int, int]]:
    if not n_items - 1 <= edge_count <= n_items * (n_items - 1) // 2:
        raise ValueError("edge_count cannot form the requested connected graph")
    node_order = _randperm(n_items, generator)
    edges: set[tuple[int, int]] = set()
    for offset in range(1, n_items):
        child = node_order[offset]
        parent_offset = int(
            torch.randint(0, offset, (1,), generator=generator, device="cpu")
        )
        parent = node_order[parent_offset]
        edges.add(tuple(sorted((child, parent))))
    remaining = [
        pair for pair in combinations(range(n_items), 2) if pair not in edges
    ]
    for index in _randperm(len(remaining), generator)[: edge_count - len(edges)]:
        edges.add(remaining[index])
    return sorted(edges)


def sample_synthetic_dual_route_task(
    model: DualRouteConstructiveRank,
    *,
    batch_size: int,
    n_items: int,
    edge_count: int,
    support_repeats: int,
    generator: torch.Generator,
) -> SyntheticDualRouteTask:
    """Sample independent rank/graph paths with a shared abstract codebook.

    The target rank is retained by the task object for outer loss only.  It is
    never passed to the model.  Every subject sees the same symbol codebook but
    an independent cue-to-rank mapping, graph, order, and left/right layout.
    """

    config = model.config
    if not 3 <= n_items <= config.max_items:
        raise ValueError("n_items outside model range")
    if batch_size <= 0 or support_repeats <= 0:
        raise ValueError("batch_size and support_repeats must be positive")
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
    true_orders = torch.stack(
        [
            torch.tensor(_randperm(n_items, generator), dtype=torch.long)
            for _ in range(batch_size)
        ]
    ).to(device)

    subject_support: list[list[tuple[int, int, int]]] = []
    subject_direct: list[set[tuple[int, int]]] = []
    for subject in range(batch_size):
        position_edges = _sample_connected_edges(n_items, edge_count, generator)
        cue_edges = []
        direct = set()
        for _ in range(support_repeats):
            for edge_index in _randperm(len(position_edges), generator):
                high_position, low_position = position_edges[edge_index]
                high_cue = int(true_orders[subject, high_position])
                low_cue = int(true_orders[subject, low_position])
                swap = bool(
                    torch.randint(0, 2, (1,), generator=generator, device="cpu")
                )
                if swap:
                    cue_edges.append((low_cue, high_cue, -1))
                else:
                    cue_edges.append((high_cue, low_cue, 1))
                direct.add(tuple(sorted((high_cue, low_cue))))
        subject_support.append(cue_edges)
        subject_direct.append(direct)

    support_length = edge_count * support_repeats
    support_left = torch.tensor(
        [
            [subject_support[b][t][0] for b in range(batch_size)]
            for t in range(support_length)
        ],
        dtype=torch.long,
        device=device,
    )
    support_right = torch.tensor(
        [
            [subject_support[b][t][1] for b in range(batch_size)]
            for t in range(support_length)
        ],
        dtype=torch.long,
        device=device,
    )
    support_sign = torch.tensor(
        [
            [subject_support[b][t][2] for b in range(batch_size)]
            for t in range(support_length)
        ],
        dtype=dtype,
        device=device,
    )

    query_pairs = list(combinations(range(n_items), 2))
    subject_queries: list[list[tuple[int, int, int, bool]]] = []
    for subject in range(batch_size):
        queries = []
        for pair_index in _randperm(len(query_pairs), generator):
            first_position, second_position = query_pairs[pair_index]
            first_cue = int(true_orders[subject, first_position])
            second_cue = int(true_orders[subject, second_position])
            swap = bool(
                torch.randint(0, 2, (1,), generator=generator, device="cpu")
            )
            if swap:
                left, right, target = second_cue, first_cue, 1
            else:
                left, right, target = first_cue, second_cue, 0
            queries.append(
                (
                    left,
                    right,
                    target,
                    tuple(sorted((left, right))) in subject_direct[subject],
                )
            )
        subject_queries.append(queries)
    query_length = len(query_pairs)

    def query_tensor(index: int, dtype_: torch.dtype) -> torch.Tensor:
        return torch.tensor(
            [
                [subject_queries[b][t][index] for b in range(batch_size)]
                for t in range(query_length)
            ],
            dtype=dtype_,
            device=device,
        )

    return SyntheticDualRouteTask(
        cue_features=cue_features,
        true_order=true_orders,
        support_left=support_left,
        support_right=support_right,
        support_sign=support_sign,
        query_left=query_tensor(0, torch.long),
        query_right=query_tensor(1, torch.long),
        query_target=query_tensor(2, torch.long),
        query_is_direct=query_tensor(3, torch.bool),
        n_items=n_items,
    )


def _masked_accuracy(
    predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> float:
    if not bool(mask.any()):
        return float("nan")
    return float((predictions[mask] == targets[mask]).float().mean().detach().cpu())


def run_synthetic_dual_route_episode(
    model: DualRouteConstructiveRank,
    task: SyntheticDualRouteTask,
    *,
    intervention: DualRouteIntervention | None = None,
    stochastic: bool = True,
    generator: torch.Generator | None = None,
) -> SyntheticDualRouteResult:
    """Run support, legal phase consolidation, and read-only outer queries."""

    if intervention is None:
        intervention = DualRouteIntervention()
    state = model.initial_state(task.batch_size)
    support_traces = []
    for trial in range(task.support_left.shape[0]):
        state, trace = model.observe_relation(
            state,
            task.cue_features,
            task.support_left[trial],
            task.support_right[trial],
            task.support_sign[trial],
            intervention=intervention,
            stochastic=stochastic,
            generator=generator,
        )
        support_traces.append(trace)
    state, _ = model.consolidate(
        state,
        ConsolidationEvent.PHASE_TRANSITION,
        stochastic=stochastic,
        generator=generator,
    )
    frozen_state = tuple(tensor.clone() for tensor in state.tensors())

    logits = []
    for trial in range(task.query_left.shape[0]):
        trace = model.query_pair(
            state,
            task.cue_features,
            task.query_left[trial],
            task.query_right[trial],
            intervention=intervention,
        )
        logits.append(trace.logits)
    for before, after in zip(frozen_state, state.tensors(), strict=True):
        if not torch.equal(before, after):
            raise RuntimeError("query changed supposedly frozen fast state")
    stacked_logits = torch.stack(logits)
    per_query_loss = F.cross_entropy(
        stacked_logits.reshape(-1, 2),
        task.query_target.reshape(-1),
        reduction="none",
    ).reshape_as(task.query_target)
    loss = per_query_loss.mean()
    predictions = stacked_logits.argmax(dim=-1)
    correct = predictions == task.query_target
    direct = task.query_is_direct
    return SyntheticDualRouteResult(
        loss=loss,
        direct_loss=per_query_loss[direct].mean(),
        indirect_loss=per_query_loss[torch.logical_not(direct)].mean(),
        accuracy=float(correct.float().mean().detach().cpu()),
        direct_accuracy=_masked_accuracy(predictions, task.query_target, direct),
        indirect_accuracy=_masked_accuracy(
            predictions, task.query_target, torch.logical_not(direct)
        ),
        final_state=state,
        support_traces=support_traces,
        logits=stacked_logits,
        targets=task.query_target,
        direct_mask=direct,
    )
