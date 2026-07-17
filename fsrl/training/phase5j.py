"""Synthetic meta-training task family for the Phase 5j process core.

The generator deliberately varies item identities, rank permutations, item
count, support graph, trial order, and left/right presentation.  It does not
contain the fixed Liu support graph or any human aggregate statistic.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from fsrl.model.phase5j import (
    Phase5jPlasticRNN,
    Phase5jState,
    Phase5jTrialTrace,
)


@dataclass(frozen=True)
class SyntheticRelationTrial:
    left_index: torch.Tensor
    right_index: torch.Tensor
    relation: torch.Tensor


@dataclass(frozen=True)
class SyntheticQueryTrial:
    left_index: torch.Tensor
    right_index: torch.Tensor
    target: torch.Tensor
    is_direct: bool


@dataclass(frozen=True)
class SyntheticRankEpisode:
    stimuli: torch.Tensor
    rank_order: torch.Tensor
    support_trials: tuple[SyntheticRelationTrial, ...]
    query_trials: tuple[SyntheticQueryTrial, ...]
    support_position_edges: tuple[tuple[int, int], ...]

    @property
    def batch_size(self) -> int:
        return int(self.stimuli.shape[0])


@dataclass
class Phase5jEpisodeResult:
    loss: torch.Tensor
    accuracy: float
    direct_accuracy: float
    indirect_accuracy: float
    support_state: Phase5jState
    support_traces: list[Phase5jTrialTrace]
    logits: torch.Tensor
    targets: torch.Tensor
    direct_mask: torch.Tensor


def _cpu_randperm(n: int, generator: torch.Generator) -> torch.Tensor:
    return torch.randperm(n, generator=generator, device="cpu")


def _sample_nonadjacent_edges(
    n_items: int, count: int, generator: torch.Generator
) -> list[tuple[int, int]]:
    candidates = [
        (i, j)
        for i in range(n_items)
        for j in range(i + 2, n_items)
    ]
    if not candidates or count <= 0:
        return []
    order = _cpu_randperm(len(candidates), generator).tolist()
    return [candidates[k] for k in order[: min(count, len(candidates))]]


def sample_synthetic_rank_episode(
    *,
    batch_size: int,
    n_items: int,
    stimulus_dim: int,
    generator: torch.Generator,
    device: torch.device | str = "cpu",
    support_repeats: int = 2,
    extra_edge_count: int = 2,
) -> SyntheticRankEpisode:
    """Sample an identifiable but non-Liu comparison graph.

    Every graph contains the adjacent chain so the full order is identifiable;
    a random subset of non-adjacent edges changes graph density and topology.
    The same position-level graph is used across the batch, but each batch
    element has independent stimuli, rank-to-label mapping, and left/right
    presentation.
    """
    if n_items < 3:
        raise ValueError("n_items must be at least 3")
    if batch_size <= 0 or support_repeats <= 0:
        raise ValueError("batch_size and support_repeats must be positive")
    device = torch.device(device)

    stimuli = (
        torch.randint(
            0,
            2,
            (batch_size, n_items, stimulus_dim),
            generator=generator,
            device="cpu",
        ).float()
        * 2.0
        - 1.0
    ).to(device)
    rank_order = torch.stack(
        [_cpu_randperm(n_items, generator) for _ in range(batch_size)]
    ).to(device)

    edges = [(i, i + 1) for i in range(n_items - 1)]
    edges.extend(_sample_nonadjacent_edges(n_items, extra_edge_count, generator))
    # Preserve unique edges if the sampling scheme changes in the future.
    edges = list(dict.fromkeys(edges))

    support_trials: list[SyntheticRelationTrial] = []
    for _ in range(support_repeats):
        for edge_index in _cpu_randperm(len(edges), generator).tolist():
            strong_pos, weak_pos = edges[edge_index]
            strong_item = rank_order[:, strong_pos]
            weak_item = rank_order[:, weak_pos]
            swap = torch.randint(
                0, 2, (batch_size,), generator=generator, device="cpu"
            ).bool().to(device)
            left_index = torch.where(swap, weak_item, strong_item)
            right_index = torch.where(swap, strong_item, weak_item)
            distance = float(weak_pos - strong_pos) / float(n_items - 1)
            relation = torch.where(
                swap,
                torch.full((batch_size,), -distance, device=device),
                torch.full((batch_size,), distance, device=device),
            )
            support_trials.append(
                SyntheticRelationTrial(left_index, right_index, relation)
            )

    edge_set = set(edges)
    query_edges = [
        (i, j) for i in range(n_items) for j in range(i + 1, n_items)
    ]
    query_trials: list[SyntheticQueryTrial] = []
    for edge_index in _cpu_randperm(len(query_edges), generator).tolist():
        strong_pos, weak_pos = query_edges[edge_index]
        strong_item = rank_order[:, strong_pos]
        weak_item = rank_order[:, weak_pos]
        swap = torch.randint(
            0, 2, (batch_size,), generator=generator, device="cpu"
        ).bool().to(device)
        left_index = torch.where(swap, weak_item, strong_item)
        right_index = torch.where(swap, strong_item, weak_item)
        target = swap.long()  # 0 = choose left; 1 = choose right
        query_trials.append(
            SyntheticQueryTrial(
                left_index,
                right_index,
                target,
                is_direct=(strong_pos, weak_pos) in edge_set,
            )
        )

    return SyntheticRankEpisode(
        stimuli=stimuli,
        rank_order=rank_order,
        support_trials=tuple(support_trials),
        query_trials=tuple(query_trials),
        support_position_edges=tuple(edges),
    )


def sample_list_linking_episode_pair(
    *,
    batch_size: int,
    n_items: int,
    stimulus_dim: int,
    generator: torch.Generator,
    device: torch.device | str = "cpu",
    within_list_repeats: int = 2,
    bridge_repeats: int = 1,
) -> tuple[SyntheticRankEpisode, SyntheticRankEpisode]:
    """Return matched pre/post-bridge episodes with the same stimuli and probes.

    The two halves are separately ordered during the shared prefix.  The post
    episode adds only the boundary relation between the halves.  Cross-list
    query pairs are identical before and after the bridge.
    """
    if n_items < 6 or n_items % 2:
        raise ValueError("list linking requires an even n_items >= 6")
    if within_list_repeats <= 0 or bridge_repeats <= 0:
        raise ValueError("repeat counts must be positive")
    device = torch.device(device)
    half = n_items // 2

    stimuli = (
        torch.randint(
            0,
            2,
            (batch_size, n_items, stimulus_dim),
            generator=generator,
            device="cpu",
        ).float()
        * 2.0
        - 1.0
    ).to(device)
    rank_order = torch.stack(
        [_cpu_randperm(n_items, generator) for _ in range(batch_size)]
    ).to(device)

    within_edges = (
        [(i, i + 1) for i in range(half - 1)]
        + [(i, i + 1) for i in range(half, n_items - 1)]
    )

    def relation_trial(strong_pos: int, weak_pos: int) -> SyntheticRelationTrial:
        strong_item = rank_order[:, strong_pos]
        weak_item = rank_order[:, weak_pos]
        swap = torch.randint(
            0, 2, (batch_size,), generator=generator, device="cpu"
        ).bool().to(device)
        distance = float(weak_pos - strong_pos) / float(n_items - 1)
        return SyntheticRelationTrial(
            left_index=torch.where(swap, weak_item, strong_item),
            right_index=torch.where(swap, strong_item, weak_item),
            relation=torch.where(
                swap,
                torch.full((batch_size,), -distance, device=device),
                torch.full((batch_size,), distance, device=device),
            ),
        )

    prefix: list[SyntheticRelationTrial] = []
    for _ in range(within_list_repeats):
        for edge_index in _cpu_randperm(len(within_edges), generator).tolist():
            prefix.append(relation_trial(*within_edges[edge_index]))

    bridge = (half - 1, half)
    post_support = list(prefix)
    post_support.extend(relation_trial(*bridge) for _ in range(bridge_repeats))

    query_trials: list[SyntheticQueryTrial] = []
    cross_edges = [(i, j) for i in range(half) for j in range(half, n_items)]
    for edge_index in _cpu_randperm(len(cross_edges), generator).tolist():
        strong_pos, weak_pos = cross_edges[edge_index]
        strong_item = rank_order[:, strong_pos]
        weak_item = rank_order[:, weak_pos]
        swap = torch.randint(
            0, 2, (batch_size,), generator=generator, device="cpu"
        ).bool().to(device)
        query_trials.append(
            SyntheticQueryTrial(
                left_index=torch.where(swap, weak_item, strong_item),
                right_index=torch.where(swap, strong_item, weak_item),
                target=swap.long(),
                is_direct=False,
            )
        )

    common = {
        "stimuli": stimuli,
        "rank_order": rank_order,
        "query_trials": tuple(query_trials),
    }
    pre = SyntheticRankEpisode(
        **common,
        support_trials=tuple(prefix),
        support_position_edges=tuple(within_edges),
    )
    post = SyntheticRankEpisode(
        **common,
        support_trials=tuple(post_support),
        support_position_edges=tuple(within_edges + [bridge]),
    )
    return pre, post


def gather_stimuli(stimuli: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Gather [batch] item labels without exposing labels to the model."""
    if stimuli.ndim != 3 or index.shape != (stimuli.shape[0],):
        raise ValueError("expected stimuli [batch, item, dim] and index [batch]")
    batch = torch.arange(stimuli.shape[0], device=stimuli.device)
    return stimuli[batch, index]


def run_synthetic_rank_episode(
    model: Phase5jPlasticRNN,
    episode: SyntheticRankEpisode,
    *,
    support_blank_recurrence: bool = True,
    plasticity_enabled: bool = True,
    modulation_enabled: bool = True,
    reset_fast_before_query: bool = False,
    blank_steps: int = 3,
    decision_steps: int = 2,
    record_support_traces: bool = False,
) -> Phase5jEpisodeResult:
    """Run support then frozen queries under an explicit causal condition."""
    state = model.initial_state(
        episode.batch_size,
        device=episode.stimuli.device,
        dtype=episode.stimuli.dtype,
    )
    support_traces: list[Phase5jTrialTrace] = []
    for trial in episode.support_trials:
        left = gather_stimuli(episode.stimuli, trial.left_index)
        right = gather_stimuli(episode.stimuli, trial.right_index)
        state, trace = model.observe_relation(
            state,
            left,
            right,
            trial.relation,
            blank_steps=blank_steps,
            support_blank_recurrence=support_blank_recurrence,
            plasticity_enabled=plasticity_enabled,
            modulation_enabled=modulation_enabled,
        )
        if record_support_traces:
            support_traces.append(trace)

    if reset_fast_before_query:
        state = model.initial_state(
            episode.batch_size,
            device=episode.stimuli.device,
            dtype=episode.stimuli.dtype,
        )
    support_state = state

    logits: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    direct: list[torch.Tensor] = []
    for trial in episode.query_trials:
        left = gather_stimuli(episode.stimuli, trial.left_index)
        right = gather_stimuli(episode.stimuli, trial.right_index)
        pair_logits, _ = model.query_pair(
            support_state, left, right, decision_steps=decision_steps
        )
        logits.append(pair_logits)
        targets.append(trial.target)
        direct.append(
            torch.full(
                (episode.batch_size,),
                trial.is_direct,
                dtype=torch.bool,
                device=episode.stimuli.device,
            )
        )

    stacked_logits = torch.stack(logits, dim=1)
    stacked_targets = torch.stack(targets, dim=1)
    direct_mask = torch.stack(direct, dim=1)
    loss = F.cross_entropy(
        stacked_logits.reshape(-1, 2), stacked_targets.reshape(-1)
    )
    correct = stacked_logits.argmax(dim=-1).eq(stacked_targets)

    def masked_mean(mask: torch.Tensor) -> float:
        if not bool(mask.any()):
            return float("nan")
        return float(correct[mask].float().mean().detach().cpu())

    return Phase5jEpisodeResult(
        loss=loss,
        accuracy=float(correct.float().mean().detach().cpu()),
        direct_accuracy=masked_mean(direct_mask),
        indirect_accuracy=masked_mean(~direct_mask),
        support_state=support_state,
        support_traces=support_traces,
        logits=stacked_logits,
        targets=stacked_targets,
        direct_mask=direct_mask,
    )
