"""Post-bridge offline-recurrence diagnostic for the public M&K core.

This module intentionally lives outside ``source/`` so the pinned source
parity tree is unchanged.  It reuses the exact source segment implementation,
then inserts deterministic all-zero recurrent steps after bridge training and
before cloned exhaustive queries.  Query hidden and eligibility state are
still reset by the original source trial code, so only changes retained in
episode fast weights can affect query behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from fsrl.miconi_kay.diagnostics.protocol import (
    POSTBRIDGE_BLANK_COUNTS,
    POSTBRIDGE_PRIMARY_BLANK_COUNT,
)
from fsrl.miconi_kay.source.config import MiconiKaySourceConfig
from fsrl.miconi_kay.source.evaluation import (
    LinkCondition,
    _advance,
    _clone_evaluation_state,
    _initial_state,
    _run_segment,
    _tensor_sha256,
)
from fsrl.miconi_kay.source.model import MiconiKayRetroModulRNN
from fsrl.miconi_kay.source.task import generate_cue_data


BlankMode = Literal["full", "freeze_fast_write"]


@dataclass
class PostBridgeRecurrenceResult:
    condition: str
    blank_mode: BlankMode
    blank_steps: int
    canonical_pairs: np.ndarray
    sampled_correct: np.ndarray
    correct_choice_probability: np.ndarray
    fast_weight_delta_l2: np.ndarray
    protocol_audit: dict[str, object]


def diagnostics_tree_sha256() -> str:
    """Hash only files that define this diagnostic protocol.

    The historical function name is retained for report compatibility.  A
    protocol-specific file set prevents an unrelated diagnostic added to the
    same package from invalidating completed reports.
    """

    directory = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    paths = [directory / "protocol.py", directory / "postbridge_recurrence.py"]
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _apply_blank_recurrence(
    model: MiconiKayRetroModulRNN,
    state,
    *,
    blank_steps: int,
    mode: BlankMode,
):
    if blank_steps < 0:
        raise ValueError("blank_steps must be non-negative")
    if mode not in {"full", "freeze_fast_write"}:
        raise ValueError(f"unknown blank mode: {mode}")
    result = _clone_evaluation_state(state)
    before = result.plastic_weights.detach().clone()
    blank = torch.zeros(
        before.shape[0],
        int(model.GG["inputsize"]),
        device=model.w.device,
        dtype=model.w.dtype,
    )
    lesion = "none" if mode == "full" else "no_fast_weight_updates"
    for _ in range(blank_steps):
        _, _, result = _advance(model, blank, result, lesion)
    delta = (result.plastic_weights - before).flatten(start_dim=1).norm(dim=1)
    return result, delta.detach().cpu().numpy().astype("float32")


def evaluate_postbridge_recurrence(
    model: MiconiKayRetroModulRNN,
    config: MiconiKaySourceConfig,
    *,
    link_condition: LinkCondition,
) -> dict[str, PostBridgeRecurrenceResult]:
    """Evaluate the locked blank-dose curve from one shared bridge state."""

    config.validate()
    item_count = config.max_items
    if config.min_items != 8 or item_count != 8:
        raise ValueError("post-bridge recurrence requires exactly 8 items")
    if config.trial_steps != 4 or config.response_step != 1:
        raise ValueError("post-bridge recurrence requires the source trial timing")
    model.eval()
    cue_data = generate_cue_data(config, item_count)
    state = _initial_state(model, config.batch_size)
    midpoint = item_count // 2
    with torch.no_grad():
        first = _run_segment(
            model,
            config,
            cue_data,
            state,
            allowed_items=range(0, midpoint),
            train_trials=10,
            test_trials=0,
            lesion="none",
        )
        second = _run_segment(
            model,
            config,
            cue_data,
            first.state,
            allowed_items=range(midpoint, item_count),
            train_trials=10,
            test_trials=0,
            lesion="none",
        )
        bridge_trials = 4 if link_condition == "true_bridge" else 1
        final_total_trials = bridge_trials + 1
        bridge = _run_segment(
            model,
            config,
            cue_data,
            second.state,
            allowed_items=range(item_count),
            train_trials=bridge_trials,
            test_trials=0,
            lesion="none",
            link_condition=link_condition,
            episode_total_trials=final_total_trials,
        )

        query_numpy_state = np.random.get_state()
        query_torch_cpu_state = torch.get_rng_state()
        device = model.w.device
        query_torch_device_state = (
            torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        )
        canonical_pairs = np.asarray(
            [
                (left, right)
                for left in range(midpoint)
                for right in range(midpoint, item_count)
            ],
            dtype="int16",
        )
        designs: list[tuple[BlankMode, int]] = [
            ("full", count) for count in POSTBRIDGE_BLANK_COUNTS
        ] + [("freeze_fast_write", POSTBRIDGE_PRIMARY_BLANK_COUNT)]
        results: dict[str, PostBridgeRecurrenceResult] = {}
        for mode, blank_steps in designs:
            postblank, fast_delta = _apply_blank_recurrence(
                model,
                bridge.state,
                blank_steps=blank_steps,
                mode=mode,
            )
            correct_columns: list[np.ndarray] = []
            probability_columns: list[np.ndarray] = []
            forced_pair_violations = 0
            for left, right in canonical_pairs:
                np.random.set_state(query_numpy_state)
                torch.set_rng_state(query_torch_cpu_state)
                if query_torch_device_state is not None:
                    torch.cuda.set_rng_state(query_torch_device_state, device)
                query = _run_segment(
                    model,
                    config,
                    cue_data,
                    _clone_evaluation_state(postblank),
                    allowed_items=range(item_count),
                    train_trials=0,
                    test_trials=1,
                    lesion="none",
                    forced_test_pair=(int(left), int(right)),
                    pre_segment_blank_steps=0,
                    episode_total_trials=final_total_trials,
                    episode_step_offset=bridge_trials * config.trial_steps,
                )
                forced_pair_violations += sum(
                    set(map(int, pair)) != {int(left), int(right)}
                    for pair in query.test_pairs[:, 0, :]
                )
                correct_columns.append(query.test_correct[:, 0].copy())
                probability_columns.append(
                    query.test_correct_choice_probability[:, 0].copy()
                )
            key = f"{mode}:{blank_steps}"
            results[key] = PostBridgeRecurrenceResult(
                condition=f"linked_list_{link_condition}_postbridge_recurrence",
                blank_mode=mode,
                blank_steps=blank_steps,
                canonical_pairs=canonical_pairs.copy(),
                sampled_correct=np.stack(correct_columns, axis=1).astype(
                    "int8", copy=False
                ),
                correct_choice_probability=np.stack(
                    probability_columns, axis=1
                ).astype("float32", copy=False),
                fast_weight_delta_l2=fast_delta,
                protocol_audit={
                    "component_segment_pair_sha256": [
                        first.all_pairs_sha256,
                        second.all_pairs_sha256,
                    ],
                    "bridge_training_pair_sha256": bridge.all_pairs_sha256,
                    "bridge_training_trials": bridge_trials,
                    "post_bridge_fast_weight_sha256": _tensor_sha256(
                        bridge.state.plastic_weights
                    ),
                    "post_blank_fast_weight_sha256": _tensor_sha256(
                        postblank.plastic_weights
                    ),
                    "blank_input_is_all_zero": True,
                    "blank_steps": blank_steps,
                    "blank_mode": mode,
                    "query_time_channel_held_at_original_value": True,
                    "query_hidden_and_eligibility_reset_by_source_trial": True,
                    "query_count_per_episode": int(canonical_pairs.shape[0]),
                    "query_common_random_numbers": True,
                    "forced_query_pair_violations": int(forced_pair_violations),
                    "cue_data_sha256": hashlib.sha256(
                        np.asarray(cue_data, dtype="float32").tobytes()
                    ).hexdigest(),
                },
            )
    return results
