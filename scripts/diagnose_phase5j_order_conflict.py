"""Frozen causal characterization of Phase 5j order and conflict sensitivity.

This diagnostic holds the stimuli, latent order, support graph, evidence
multiset, and left/right presentations fixed.  It changes only chronology.
The order conditions use four repeats of an adjacent chain:

* forward, reverse, and interleaved contain every edge once at each seven-trial
  block boundary, so their prefix states have identical evidence multisets;
* blocked groups all four repetitions of an edge and is an intentionally
  different learning-history control with the same final multiset.

The conflict conditions replace one correct observation of the middle edge by
the same contradictory observation either in block 1 or block 4.  Early and
late conflict therefore have the same final signed evidence.  Any final
difference is chronology dependence, which a commutative accumulator cannot
produce.

Outputs are predictions of already-frozen synthetic checkpoints.  They are
not evidence that human learners share the model's path dependence; new human
prefix/conflict data are required for that claim.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fsrl.cli.liu2026 import _source_tree_sha256
from fsrl.model.phase5j import Phase5jConfig, Phase5jPlasticRNN, Phase5jState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-items", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--graph-type", choices=("chain", "liu"), default="chain")
    parser.add_argument(
        "--liu-config",
        type=Path,
        default=Path(
            "configs/archive/2026-07-16_legacy_research/phase5_retro/"
            "retro_process_phase5g_fixed_weight_no_proj.yaml"
        ),
    )
    parser.add_argument("--seed", type=int, default=17011)
    parser.add_argument("--blank-steps", type=int, default=3)
    parser.add_argument("--decision-steps", type=int, default=2)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def make_orders(
    n_edges: int, repeats: int, generator: torch.Generator
) -> dict[str, list[tuple[int, int]]]:
    """Return (repeat, edge) schedules; event payloads remain tied to these IDs."""
    forward = [(repeat, edge) for repeat in range(repeats) for edge in range(n_edges)]
    reverse = [
        (repeat, edge)
        for repeat in range(repeats)
        for edge in reversed(range(n_edges))
    ]
    interleaved = []
    for repeat in range(repeats):
        permutation = torch.randperm(n_edges, generator=generator).tolist()
        interleaved.extend((repeat, edge) for edge in permutation)
    blocked = [(repeat, edge) for edge in range(n_edges) for repeat in range(repeats)]
    return {
        "forward": forward,
        "reverse": reverse,
        "interleaved": interleaved,
        "blocked": blocked,
    }


def load_position_edges(
    graph_type: str, n_items: int, liu_config: Path
) -> tuple[list[tuple[int, int]], int]:
    """Return strong-to-weak position edges and the preregistered conflict edge."""
    if graph_type == "chain":
        edges = [(position, position + 1) for position in range(n_items - 1)]
        return edges, (n_items - 1) // 2
    if n_items != 8:
        raise ValueError("the exact Liu graph requires n_items=8")
    raw = yaml.safe_load(liu_config.read_text(encoding="utf-8"))
    task = raw.get("task", raw)
    published = task["support_pairs"]
    # Published coordinates are weak-to-strong (A..H); model diagnostics use
    # position 0 as strongest. Preserve config order while reversing the axis.
    edges = [
        (n_items - 1 - max(int(a), int(b)), n_items - 1 - min(int(a), int(b)))
        for a, b in published
    ]
    if len(edges) != 8 or len(set(edges)) != 8:
        raise ValueError("unexpected Liu support graph")
    # D-F (published positions 3,5; internal 2,4) is chosen before seeing this
    # diagnostic because it is a middle-spanning, non-extreme relation.
    target_edge = edges.index((2, 4))
    return edges, target_edge


def validate_schedules(
    schedules: dict[str, list[tuple[int, int]]], n_edges: int, repeats: int
) -> None:
    expected = Counter((repeat, edge) for repeat in range(repeats) for edge in range(n_edges))
    for name, schedule in schedules.items():
        if Counter(schedule) != expected:
            raise AssertionError(f"{name} does not preserve the final event multiset")
    for block in range(repeats):
        lo, hi = block * n_edges, (block + 1) * n_edges
        expected_prefix = Counter(
            (repeat, edge)
            for repeat in range(block + 1)
            for edge in range(n_edges)
        )
        for name in ("forward", "reverse", "interleaved"):
            if Counter(schedules[name][:hi]) != expected_prefix:
                raise AssertionError(
                    f"{name} block {block + 1} does not share the evidence prefix"
                )


def make_episode_payloads(
    *,
    batch_size: int,
    n_items: int,
    repeats: int,
    position_edges: list[tuple[int, int]],
    target_edge_index: int,
    stimulus_dim: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[tuple[int, int], tuple[torch.Tensor, ...]]]:
    """Create frozen stimuli/ranks and side-balanced event payloads.

    The middle edge uses the same side orientation across repetitions so the
    early-versus-late conflict contrast changes timing but not sensory side.
    Other sides vary by occurrence, while following an event when it moves.
    """
    stimuli = (
        torch.randint(
            0,
            2,
            (batch_size, n_items, stimulus_dim),
            generator=generator,
        ).float()
        * 2.0
        - 1.0
    ).to(device)
    rank_order = torch.stack(
        [torch.randperm(n_items, generator=generator) for _ in range(batch_size)]
    ).to(device)
    target_swap = torch.randint(
        0, 2, (batch_size,), generator=generator
    ).bool().to(device)
    payloads = {}
    for repeat in range(repeats):
        for edge_index, (strong_position, weak_position) in enumerate(position_edges):
            strong = rank_order[:, strong_position]
            weak = rank_order[:, weak_position]
            if edge_index == target_edge_index:
                swap = target_swap
            else:
                swap = torch.randint(
                    0, 2, (batch_size,), generator=generator
                ).bool().to(device)
            left_index = torch.where(swap, weak, strong)
            right_index = torch.where(swap, strong, weak)
            relation = torch.where(
                swap,
                torch.full(
                    (batch_size,),
                    -float(weak_position - strong_position) / (n_items - 1),
                    device=device,
                ),
                torch.full(
                    (batch_size,),
                    float(weak_position - strong_position) / (n_items - 1),
                    device=device,
                ),
            )
            payloads[(repeat, edge_index)] = (left_index, right_index, relation)
    return stimuli, rank_order, payloads


def gather(stimuli: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(stimuli.shape[0], device=stimuli.device)
    return stimuli[batch, indices]


def query_geometry(
    model: Phase5jPlasticRNN,
    state: Phase5jState,
    stimuli: torch.Tensor,
    rank_order: torch.Tensor,
    *,
    decision_steps: int,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Return side-averaged P(correct) in rank-position coordinates."""
    probabilities = []
    pairs = []
    for strong_pos in range(rank_order.shape[1]):
        for weak_pos in range(strong_pos + 1, rank_order.shape[1]):
            strong = gather(stimuli, rank_order[:, strong_pos])
            weak = gather(stimuli, rank_order[:, weak_pos])
            logits_lr, _ = model.query_pair(
                state, strong, weak, decision_steps=decision_steps
            )
            logits_rl, _ = model.query_pair(
                state, weak, strong, decision_steps=decision_steps
            )
            p_correct = 0.5 * (
                torch.softmax(logits_lr, dim=-1)[:, 0]
                + torch.softmax(logits_rl, dim=-1)[:, 1]
            )
            probabilities.append(p_correct)
            pairs.append((strong_pos, weak_pos))
    return torch.stack(probabilities, dim=1), pairs


def encode_schedule(
    model: Phase5jPlasticRNN,
    stimuli: torch.Tensor,
    rank_order: torch.Tensor,
    payloads: dict[tuple[int, int], tuple[torch.Tensor, ...]],
    schedule: list[tuple[int, int]],
    *,
    conflict_event: tuple[int, int] | None,
    blank_recurrence: bool,
    blank_steps: int,
    decision_steps: int,
    block_size: int,
) -> tuple[list[torch.Tensor], list[Phase5jState], list[tuple[int, int]]]:
    state = model.initial_state(
        stimuli.shape[0], device=stimuli.device, dtype=stimuli.dtype
    )
    prefix_geometry = []
    prefix_states = []
    query_pairs = []
    for trial_index, event_id in enumerate(schedule):
        left_index, right_index, relation = payloads[event_id]
        if event_id == conflict_event:
            relation = -relation
        state, _ = model.observe_relation(
            state,
            gather(stimuli, left_index),
            gather(stimuli, right_index),
            relation,
            blank_steps=blank_steps,
            support_blank_recurrence=blank_recurrence,
        )
        if (trial_index + 1) % block_size == 0:
            geometry, query_pairs = query_geometry(
                model,
                state,
                stimuli,
                rank_order,
                decision_steps=decision_steps,
            )
            prefix_geometry.append(geometry)
            prefix_states.append(state)
    return prefix_geometry, prefix_states, query_pairs


def pair_masks(
    pairs: list[tuple[int, int]], conflict_edge: int
) -> dict[str, np.ndarray]:
    left_endpoint, right_endpoint = conflict_edge, conflict_edge + 1
    exact, incident, cross_nonincident, same_side = [], [], [], []
    near_same_side, far_same_side = [], []
    for i, j in pairs:
        is_exact = (i, j) == (left_endpoint, right_endpoint)
        is_incident = (i in {left_endpoint, right_endpoint} or j in {left_endpoint, right_endpoint}) and not is_exact
        crosses = i <= left_endpoint and j >= right_endpoint
        is_cross_nonincident = crosses and not is_exact and not is_incident
        is_same_side = j <= left_endpoint or i >= right_endpoint
        graph_distance = min(
            abs(i - left_endpoint),
            abs(i - right_endpoint),
            abs(j - left_endpoint),
            abs(j - right_endpoint),
        )
        exact.append(is_exact)
        incident.append(is_incident)
        cross_nonincident.append(is_cross_nonincident)
        same_side.append(is_same_side)
        near_same_side.append(is_same_side and graph_distance <= 1)
        far_same_side.append(is_same_side and graph_distance >= 2)
    return {
        "exact_conflict_pair": np.asarray(exact),
        "incident_pairs": np.asarray(incident),
        "cross_cut_nonincident_pairs": np.asarray(cross_nonincident),
        "all_same_side_pairs": np.asarray(same_side),
        "near_same_side_pairs": np.asarray(near_same_side),
        "far_same_side_pairs": np.asarray(far_same_side),
    }


def graph_pair_masks(
    pairs: list[tuple[int, int]],
    position_edges: list[tuple[int, int]],
    target_edge_index: int,
) -> dict[str, np.ndarray]:
    """Graph-theoretic locality groups for a non-chain support graph."""
    n_items = 1 + max(max(edge) for edge in position_edges)
    target = position_edges[target_edge_index]

    def distances(drop_target: bool) -> np.ndarray:
        result = np.full((n_items, n_items), np.inf)
        np.fill_diagonal(result, 0.0)
        for index, (i, j) in enumerate(position_edges):
            if drop_target and index == target_edge_index:
                continue
            result[i, j] = result[j, i] = 1.0
        for k in range(n_items):
            result = np.minimum(result, result[:, k, None] + result[None, k, :])
        return result

    full = distances(False)
    dropped = distances(True)
    exact, incident, dependent, independent = [], [], [], []
    for pair in pairs:
        i, j = pair
        is_exact = pair == target
        is_incident = (i in target or j in target) and not is_exact
        is_dependent = dropped[i, j] > full[i, j] and not is_exact and not is_incident
        exact.append(is_exact)
        incident.append(is_incident)
        dependent.append(is_dependent)
        independent.append(not is_exact and not is_incident and not is_dependent)
    return {
        "exact_conflict_pair": np.asarray(exact),
        "incident_pairs": np.asarray(incident),
        "edge_dependent_nonincident_pairs": np.asarray(dependent),
        "edge_independent_nonincident_pairs": np.asarray(independent),
    }


def geometry_summary(probabilities: torch.Tensor) -> dict[str, float]:
    return {
        "mean_correct_probability": float(probabilities.mean().cpu()),
        "argmax_accuracy": float((probabilities >= 0.5).float().mean().cpu()),
    }


def fast_state_cosine_distance(a: Phase5jState, b: Phase5jState) -> float:
    a_flat = a.plastic_weights.flatten(1)
    b_flat = b.plastic_weights.flatten(1)
    similarity = torch.nn.functional.cosine_similarity(a_flat, b_flat, dim=1)
    return float((1.0 - similarity).mean().cpu())


def analytic_leaky_chain_baseline(
    *,
    n_items: int,
    repeats: int,
    decay: float,
    temperature: float = 1.0,
    position_edges: list[tuple[int, int]] | None = None,
    target_edge_index: int | None = None,
) -> dict:
    """Unfitted recency baseline with one independent trace per chain edge.

    The baseline has no hidden representation, replay, or learned parameters.
    It demonstrates that stronger late-conflict effects are not diagnostic of
    active construction: exponential trace decay is already sufficient.
    """
    chain = position_edges is None
    position_edges = (
        [(position, position + 1) for position in range(n_items - 1)]
        if position_edges is None
        else position_edges
    )
    n_edges = len(position_edges)
    target_edge_index = (
        (n_items - 1) // 2 if target_edge_index is None else target_edge_index
    )
    schedules = {
        "forward": [
            (repeat, edge)
            for repeat in range(repeats)
            for edge in range(n_edges)
        ],
        "reverse": [
            (repeat, edge)
            for repeat in range(repeats)
            for edge in reversed(range(n_edges))
        ],
    }
    pairs = [(i, j) for i in range(n_items) for j in range(i + 1, n_items)]

    incidence = np.zeros((n_edges, n_items), dtype=float)
    for edge_index, (strong, weak) in enumerate(position_edges):
        incidence[edge_index, strong] = 1.0
        incidence[edge_index, weak] = -1.0
    augmented = np.concatenate([incidence, np.ones((1, n_items))], axis=0)

    def encode(schedule, conflict_event=None):
        traces = np.zeros(n_edges, dtype=float)
        prefixes = []
        for trial_index, event_id in enumerate(schedule):
            traces *= decay
            sign = -1.0 if event_id == conflict_event else 1.0
            strong, weak = position_edges[event_id[1]]
            traces[event_id[1]] += (
                sign * float(weak - strong) / float(n_items - 1)
            )
            if (trial_index + 1) % n_edges == 0:
                scores = np.linalg.lstsq(
                    augmented,
                    np.concatenate([traces, np.asarray([0.0])]),
                    rcond=None,
                )[0]
                logits = np.asarray(
                    [temperature * (scores[i] - scores[j]) for i, j in pairs]
                )
                prefixes.append(1.0 / (1.0 + np.exp(-logits)))
        return prefixes

    forward = encode(schedules["forward"])
    reverse = encode(schedules["reverse"])
    early = encode(schedules["forward"], (0, target_edge_index))
    late = encode(schedules["forward"], (repeats - 1, target_edge_index))
    masks = (
        pair_masks(pairs, position_edges[target_edge_index][0])
        if chain
        else graph_pair_masks(pairs, position_edges, target_edge_index)
    )
    final_groups = {}
    for name, mask in masks.items():
        final_groups[name] = {
            "early_minus_baseline": float((early[-1][mask] - forward[-1][mask]).mean()),
            "late_minus_baseline": float((late[-1][mask] - forward[-1][mask]).mean()),
            "early_minus_late": float((early[-1][mask] - late[-1][mask]).mean()),
        }
    return {
        "decay": decay,
        "fitted_to_neural_or_human_data": False,
        "forward_reverse_final_probability_mae": float(
            np.abs(forward[-1] - reverse[-1]).mean()
        ),
        "final_conflict_probability_effects": final_groups,
    }


def aggregate_runs(runs: list[dict]) -> dict:
    conditions = list(runs[0]["curves"])
    n_blocks = len(runs[0]["curves"][conditions[0]])
    curves = {}
    for condition in conditions:
        curves[condition] = []
        for block in range(n_blocks):
            values = [run["curves"][condition][block] for run in runs]
            curves[condition].append(
                {
                    "block": block + 1,
                    "trials_seen": values[0]["trials_seen"],
                    "mean_correct_probability": _mean(
                        [value["mean_correct_probability"] for value in values]
                    ),
                    "argmax_accuracy": _mean(
                        [value["argmax_accuracy"] for value in values]
                    ),
                }
            )

    order_effect = []
    for block in range(n_blocks):
        order_effect.append(
            {
                "block": block + 1,
                "forward_reverse_mean_abs_probability_difference": _mean(
                    [run["order_effect"][block]["probability_mae"] for run in runs]
                ),
                "forward_reverse_fast_state_cosine_distance": _mean(
                    [run["order_effect"][block]["fast_cosine_distance"] for run in runs]
                ),
            }
        )

    group_names = list(runs[0]["conflict_effects"][0]["groups"])
    conflict_effects = []
    for block in range(n_blocks):
        groups = {}
        for group in group_names:
            groups[group] = {}
            for contrast in (
                "early_minus_baseline",
                "late_minus_baseline",
                "early_minus_late",
            ):
                groups[group][contrast] = _mean(
                    [
                        run["conflict_effects"][block]["groups"][group][contrast]
                        for run in runs
                    ]
                )
        conflict_effects.append({"block": block + 1, "groups": groups})
    pair_probabilities = {}
    for condition in conditions:
        pair_probabilities[condition] = []
        for block in range(n_blocks):
            episode_profiles = np.asarray(
                [run["pair_probabilities"][condition][block] for run in runs],
                dtype=float,
            )
            pair_probabilities[condition].append(
                {
                    "block": block + 1,
                    "mean_correct_probability_by_pair": episode_profiles.mean(axis=0).tolist(),
                }
            )
    return {
        "curves": curves,
        "same_multiset_order_effect": order_effect,
        "conflict_probability_effects": conflict_effects,
        "query_pairs_rank_positions": runs[0]["query_pairs_rank_positions"],
        "pair_probability_profiles": pair_probabilities,
    }


def run_condition(
    model: Phase5jPlasticRNN,
    *,
    args: argparse.Namespace,
    training_config: dict,
    blank_recurrence: bool,
    position_edges: list[tuple[int, int]],
    target_edge_index: int,
    graph_type: str,
) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    runs = []
    n_edges = len(position_edges)
    for _episode in range(args.episodes):
        schedules = make_orders(n_edges, args.repeats, generator)
        validate_schedules(schedules, n_edges, args.repeats)
        stimuli, rank_order, payloads = make_episode_payloads(
            batch_size=args.batch_size,
            n_items=args.n_items,
            repeats=args.repeats,
            stimulus_dim=int(training_config["stimulus_dim"]),
            position_edges=position_edges,
            target_edge_index=target_edge_index,
            generator=generator,
            device=next(model.parameters()).device,
        )
        encoded = {}
        for name, schedule in schedules.items():
            geometry, states, pairs = encode_schedule(
                model,
                stimuli,
                rank_order,
                payloads,
                schedule,
                conflict_event=None,
                blank_recurrence=blank_recurrence,
                blank_steps=args.blank_steps,
                decision_steps=args.decision_steps,
                block_size=n_edges,
            )
            encoded[name] = (geometry, states)

        baseline_geometry = encoded["forward"][0]
        early_geometry, _, conflict_pairs = encode_schedule(
            model,
            stimuli,
            rank_order,
            payloads,
            schedules["forward"],
            conflict_event=(0, target_edge_index),
            blank_recurrence=blank_recurrence,
            blank_steps=args.blank_steps,
            decision_steps=args.decision_steps,
            block_size=n_edges,
        )
        late_geometry, _, _ = encode_schedule(
            model,
            stimuli,
            rank_order,
            payloads,
            schedules["forward"],
            conflict_event=(args.repeats - 1, target_edge_index),
            blank_recurrence=blank_recurrence,
            blank_steps=args.blank_steps,
            decision_steps=args.decision_steps,
            block_size=n_edges,
        )
        if pairs != conflict_pairs:
            raise AssertionError("query pair order changed between conditions")
        masks = (
            pair_masks(pairs, position_edges[target_edge_index][0])
            if graph_type == "chain"
            else graph_pair_masks(pairs, position_edges, target_edge_index)
        )

        curves = {}
        for name, (geometry, _) in encoded.items():
            curves[name] = []
            for block, probabilities in enumerate(geometry):
                curves[name].append(
                    {
                        "trials_seen": (block + 1) * n_edges,
                        **geometry_summary(probabilities),
                    }
                )
        curves["conflict_early"] = [
            {"trials_seen": (block + 1) * n_edges, **geometry_summary(probabilities)}
            for block, probabilities in enumerate(early_geometry)
        ]
        curves["conflict_late"] = [
            {"trials_seen": (block + 1) * n_edges, **geometry_summary(probabilities)}
            for block, probabilities in enumerate(late_geometry)
        ]

        order_effect = []
        for block in range(args.repeats):
            forward = encoded["forward"][0][block]
            reverse = encoded["reverse"][0][block]
            order_effect.append(
                {
                    "probability_mae": float((forward - reverse).abs().mean().cpu()),
                    "fast_cosine_distance": fast_state_cosine_distance(
                        encoded["forward"][1][block], encoded["reverse"][1][block]
                    ),
                }
            )

        conflict_effects = []
        for block in range(args.repeats):
            groups = {}
            for group, mask in masks.items():
                baseline = baseline_geometry[block][:, mask]
                early = early_geometry[block][:, mask]
                late = late_geometry[block][:, mask]
                groups[group] = {
                    "early_minus_baseline": float((early - baseline).mean().cpu()),
                    "late_minus_baseline": float((late - baseline).mean().cpu()),
                    "early_minus_late": float((early - late).mean().cpu()),
                }
            conflict_effects.append({"groups": groups})
        pair_probabilities = {
            name: [probability.mean(dim=0).cpu().tolist() for probability in geometry]
            for name, (geometry, _) in encoded.items()
        }
        pair_probabilities["conflict_early"] = [
            probability.mean(dim=0).cpu().tolist() for probability in early_geometry
        ]
        pair_probabilities["conflict_late"] = [
            probability.mean(dim=0).cpu().tolist() for probability in late_geometry
        ]
        runs.append(
            {
                "curves": curves,
                "order_effect": order_effect,
                "conflict_effects": conflict_effects,
                "query_pairs_rank_positions": [list(pair) for pair in pairs],
                "pair_probabilities": pair_probabilities,
            }
        )
    return aggregate_runs(runs)


def main() -> None:
    args = parse_args()
    if args.n_items < 6:
        raise ValueError("n_items must be at least 6 for local/far conflict groups")
    if args.repeats < 2:
        raise ValueError("repeats must be at least 2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.training_dir / "net.dat"
    training_config_path = args.training_dir / "config.json"
    training_config = json.loads(training_config_path.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    model = Phase5jPlasticRNN(
        Phase5jConfig(
            stimulus_dim=int(training_config["stimulus_dim"]),
            hidden_dim=int(training_config["hidden_dim"]),
            relation_timing=training_config.get("relation_timing", "immediate"),
        )
    ).to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model.eval()

    position_edges, target_edge_index = load_position_edges(
        args.graph_type, args.n_items, args.liu_config
    )

    with torch.no_grad():
        intact = run_condition(
            model,
            args=args,
            training_config=training_config,
            blank_recurrence=True,
            position_edges=position_edges,
            target_edge_index=target_edge_index,
            graph_type=args.graph_type,
        )
        recurrence_lesion = run_condition(
            model,
            args=args,
            training_config=training_config,
            blank_recurrence=False,
            position_edges=position_edges,
            target_edge_index=target_edge_index,
            graph_type=args.graph_type,
        )

    report = {
        "status": "frozen model prediction; not human path evidence",
        "checkpoint_sha256": sha256(checkpoint_path),
        "training_config_sha256": sha256(training_config_path),
        "source_tree_sha256": _source_tree_sha256(PROJECT_ROOT),
        "design": {
            "episodes": args.episodes,
            "batch_size": args.batch_size,
            "n_items": args.n_items,
            "support_graph": args.graph_type,
            "support_position_edges": [list(edge) for edge in position_edges],
            "repeats": args.repeats,
            "same_stimuli_rank_graph_evidence_and_sides_across_order_conditions": True,
            "forward_reverse_interleaved_same_multiset_at_each_block_boundary": True,
            "early_late_conflict_same_final_signed_evidence": True,
            "conflict_edge_index": target_edge_index,
            "conflict_edge_in_rank_positions": list(position_edges[target_edge_index]),
            "relation_timing": training_config.get("relation_timing", "immediate"),
            "human_data_used": False,
        },
        "commutative_accumulator_reference": {
            "forward_reverse_final_difference": 0.0,
            "early_late_conflict_final_difference": 0.0,
            "reason": "both contrasts have identical final signed evidence multisets",
        },
        "unfitted_leaky_chain_references": [
            analytic_leaky_chain_baseline(
                n_items=args.n_items,
                repeats=args.repeats,
                decay=decay,
                position_edges=(None if args.graph_type == "chain" else position_edges),
                target_edge_index=target_edge_index,
            )
            for decay in (0.5, 0.8, 0.95, 1.0)
        ],
        "intact": intact,
        "support_blank_recurrence_off": recurrence_lesion,
        "interpretation": {
            "chronology_test": (
                "Nonzero forward/reverse differences at matched block boundaries or "
                "nonzero early/late final differences establish model chronology dependence."
            ),
            "locality_test": (
                "Conflict effects are separated into the observed edge, endpoint-sharing "
                "pairs, cross-cut transitive pairs, and same-side controls."
            ),
            "claim_limit": (
                "Chronology dependence is necessary path evidence inside the model but is "
                "not sufficient for cognitive alignment. Humans must be tested on the same "
                "manipulations, and model recovery must show the design separates candidates."
            ),
            "recency_warning": (
                "A stronger late-conflict effect is also generated by the unfitted leaky "
                "chain baseline, which has no replay or constructed global representation."
            ),
        },
    }
    report_path = args.output_dir / "order_conflict_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
