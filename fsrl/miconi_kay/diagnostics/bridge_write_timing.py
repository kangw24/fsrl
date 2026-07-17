"""Exhaustive source evaluation under fixed bridge write-timing lesions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from fsrl.miconi_kay.diagnostics.write_timing_protocol import WRITE_TIMING_MODES
from fsrl.miconi_kay.source.config import MiconiKaySourceConfig
from fsrl.miconi_kay.source.evaluation import (
    FastStateLesion,
    LinkCondition,
    _clone_evaluation_state,
    _initial_state,
    _run_segment,
    _tensor_sha256,
)
from fsrl.miconi_kay.source.model import MiconiKayRetroModulRNN
from fsrl.miconi_kay.source.task import generate_cue_data


WriteTimingMode = Literal["baseline", "block_reward_step", "block_delay_step"]


@dataclass
class BridgeWriteTimingResult:
    condition: str
    mode: WriteTimingMode
    canonical_pairs: np.ndarray
    sampled_correct: np.ndarray
    correct_choice_probability: np.ndarray
    protocol_audit: dict[str, object]


def write_timing_tree_sha256() -> str:
    """Hash only the files that define the write-timing diagnostic."""

    directory = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in (
        directory / "write_timing_protocol.py",
        directory / "bridge_write_timing.py",
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _lesion_for_mode(mode: WriteTimingMode) -> FastStateLesion:
    if mode == "baseline":
        return "none"
    if mode == "block_reward_step":
        return "block_bridge_reward_step_updates"
    if mode == "block_delay_step":
        return "block_bridge_delay_step_updates"
    raise ValueError(f"unknown write-timing mode: {mode}")


def evaluate_bridge_write_timing(
    model: MiconiKayRetroModulRNN,
    config: MiconiKaySourceConfig,
    *,
    link_condition: LinkCondition,
) -> dict[str, BridgeWriteTimingResult]:
    """Run all registered lesions with common schedules and query randomness."""

    config.validate()
    if config.min_items != 8 or config.max_items != 8:
        raise ValueError("bridge write-timing evaluation requires exactly 8 items")
    if config.trial_steps != 4 or config.response_step != 1:
        raise ValueError("bridge write-timing evaluation requires source timing")
    model.eval()
    initial_numpy = np.random.get_state()
    initial_torch_cpu = torch.get_rng_state()
    device = model.w.device
    initial_torch_device = (
        torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    )
    results: dict[str, BridgeWriteTimingResult] = {}
    with torch.no_grad():
        for mode_text in WRITE_TIMING_MODES:
            mode: WriteTimingMode = mode_text  # type: ignore[assignment]
            np.random.set_state(initial_numpy)
            torch.set_rng_state(initial_torch_cpu)
            if initial_torch_device is not None:
                torch.cuda.set_rng_state(initial_torch_device, device)
            cue_data = generate_cue_data(config, 8)
            state = _initial_state(model, config.batch_size)
            first = _run_segment(
                model,
                config,
                cue_data,
                state,
                allowed_items=range(0, 4),
                train_trials=10,
                test_trials=0,
                lesion="none",
            )
            second = _run_segment(
                model,
                config,
                cue_data,
                first.state,
                allowed_items=range(4, 8),
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
                allowed_items=range(8),
                train_trials=bridge_trials,
                test_trials=0,
                lesion=_lesion_for_mode(mode),
                link_condition=link_condition,
                episode_total_trials=final_total_trials,
            )
            query_numpy = np.random.get_state()
            query_torch_cpu = torch.get_rng_state()
            query_torch_device = (
                torch.cuda.get_rng_state(device) if device.type == "cuda" else None
            )
            canonical_pairs = np.asarray(
                [(left, right) for left in range(4) for right in range(4, 8)],
                dtype="int16",
            )
            correct_columns: list[np.ndarray] = []
            probability_columns: list[np.ndarray] = []
            forced_pair_violations = 0
            for left, right in canonical_pairs:
                np.random.set_state(query_numpy)
                torch.set_rng_state(query_torch_cpu)
                if query_torch_device is not None:
                    torch.cuda.set_rng_state(query_torch_device, device)
                query = _run_segment(
                    model,
                    config,
                    cue_data,
                    _clone_evaluation_state(bridge.state),
                    allowed_items=range(8),
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
            results[mode] = BridgeWriteTimingResult(
                condition=f"linked_list_{link_condition}_write_timing",
                mode=mode,
                canonical_pairs=canonical_pairs,
                sampled_correct=np.stack(correct_columns, axis=1).astype(
                    "int8", copy=False
                ),
                correct_choice_probability=np.stack(
                    probability_columns, axis=1
                ).astype("float32", copy=False),
                protocol_audit={
                    "component_segment_pair_sha256": [
                        first.all_pairs_sha256,
                        second.all_pairs_sha256,
                    ],
                    "bridge_training_pair_sha256": bridge.all_pairs_sha256,
                    "bridge_training_trials": bridge_trials,
                    "bridge_write_timing_mode": mode,
                    "bridge_lesion": _lesion_for_mode(mode),
                    "post_bridge_fast_weight_sha256": _tensor_sha256(
                        bridge.state.plastic_weights
                    ),
                    "query_count_per_episode": 16,
                    "query_common_random_numbers": True,
                    "forced_query_pair_violations": int(forced_pair_violations),
                },
            )
    return results
