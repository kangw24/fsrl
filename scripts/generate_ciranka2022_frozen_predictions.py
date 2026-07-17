"""Generate choice-blind prequential predictions for Ciranka Exp.4.

Only the schedule split is read. Human choices and RT are not accepted as
arguments and cannot enter model state or prediction generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fsrl.model.leaky_rank_accumulator import LeakyRankAccumulator
from fsrl.model.online_ordinal import OnlineOrdinalPredictionError


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def model_config() -> SimpleNamespace:
    return SimpleNamespace(
        nbcues=8,
        online_learning_rate_init=0.5,
        online_retention_init=0.98,
        online_logit_scale_init=1.0,
        leaky_retention_init=0.9,
        leaky_logit_scale_init=1.0,
    )


def load_models(
    online_checkpoint: Path, leaky_checkpoint: Path
) -> tuple[OnlineOrdinalPredictionError, LeakyRankAccumulator]:
    config = model_config()
    online = OnlineOrdinalPredictionError(config)
    leaky = LeakyRankAccumulator(config)
    online.load_state_dict(
        torch.load(online_checkpoint, map_location="cpu", weights_only=True),
        strict=True,
    )
    leaky.load_state_dict(
        torch.load(leaky_checkpoint, map_location="cpu", weights_only=True),
        strict=True,
    )
    online.eval()
    leaky.eval()
    return online, leaky


def cue_tensor(cue: int) -> torch.Tensor:
    return torch.tensor([cue], dtype=torch.long)


def p_choose_first(model, state: torch.Tensor, left: int, right: int) -> float:
    logits = model.query_logits(state, cue_tensor(left), cue_tensor(right))
    return float(torch.softmax(logits, dim=1)[0, 0])


def update_model(model, state: torch.Tensor, row: dict) -> torch.Tensor:
    sign = row.get("sign_after_trial")
    if sign is None:
        return state
    left = cue_tensor(int(row["left_cue"]))
    right = cue_tensor(int(row["right_cue"]))
    sign_tensor = torch.tensor([float(sign)], dtype=torch.float32)
    result = model.update_support(state, left, right, sign_tensor)
    return result[0] if isinstance(result, tuple) else result


def validate_schedule(rows: list[dict]) -> dict[int, list[dict]]:
    by_subject: dict[int, list[dict]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()
    for row in rows:
        subject = int(row["subject_index"])
        trial = int(row["trial_index"])
        key = (subject, trial)
        if key in seen:
            raise ValueError(f"duplicate schedule key {key}")
        seen.add(key)
        if int(row["left_cue"]) not in range(8) or int(row["right_cue"]) not in range(8):
            raise ValueError("schedule cue outside 0..7")
        if row["left_cue"] == row["right_cue"]:
            raise ValueError("schedule pair contains the same cue twice")
        if row.get("sign_after_trial") not in (-1, 1, None):
            raise ValueError("sign_after_trial must be -1, +1, or null")
        by_subject[subject].append(row)
    for subject_rows in by_subject.values():
        subject_rows.sort(key=lambda item: int(item["trial_index"]))
        if len(subject_rows) != 420:
            raise ValueError("every included subject requires 420 schedule rows")
        if [int(row["trial_index"]) for row in subject_rows] != list(range(420)):
            raise ValueError("subject trial indices must be contiguous 0..419")
    return dict(sorted(by_subject.items()))


@torch.no_grad()
def generate_predictions(
    by_subject: dict[int, list[dict]],
    online: OnlineOrdinalPredictionError,
    leaky: LeakyRankAccumulator,
) -> list[dict]:
    predictions: list[dict] = []
    for subject, subject_rows in by_subject.items():
        online_state = online.initial_state(1)
        leaky_state = leaky.initial_state(1)
        block_end_state = online.initial_state(1)
        for block_index in range(6):
            block_rows = [
                row
                for row in subject_rows
                if int(row["block_index"]) == block_index
            ]
            pending_updates: list[dict] = []
            for row in block_rows:
                left = int(row["left_cue"])
                right = int(row["right_cue"])
                predictions.append(
                    {
                        "subject_index": subject,
                        "block_index": block_index,
                        "slot_index": int(row["slot_index"]),
                        "trial_index": int(row["trial_index"]),
                        "online_p_choose_first": p_choose_first(
                            online, online_state, left, right
                        ),
                        "leaky_p_choose_first": p_choose_first(
                            leaky, leaky_state, left, right
                        ),
                        "support_off_p_choose_first": 0.5,
                        "block_end_p_choose_first": p_choose_first(
                            online, block_end_state, left, right
                        ),
                    }
                )
                online_state = update_model(online, online_state, row)
                leaky_state = update_model(leaky, leaky_state, row)
                if row.get("sign_after_trial") is not None:
                    pending_updates.append(row)
            for row in pending_updates:
                block_end_state = update_model(online, block_end_state, row)
    return predictions


@torch.no_grad()
def active_probabilities(
    model,
    subject_rows: list[dict],
    mapping: list[int] | None = None,
) -> list[float]:
    state = model.initial_state(1)
    probabilities: list[float] = []
    for original in subject_rows:
        row = dict(original)
        if mapping is not None:
            row["left_cue"] = mapping[int(row["left_cue"])]
            row["right_cue"] = mapping[int(row["right_cue"])]
        probabilities.append(
            p_choose_first(
                model, state, int(row["left_cue"]), int(row["right_cue"])
            )
        )
        state = update_model(model, state, row)
    return probabilities


def max_abs_difference(first: list[float], second: list[float]) -> float:
    return max(abs(a - b) for a, b in zip(first, second, strict=True))


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate frozen choice-blind prequential predictions."
    )
    parser.add_argument(
        "--schedule",
        type=Path,
        default=Path("outputs/ciranka2022_exp4_audit/schedule.jsonl"),
    )
    parser.add_argument(
        "--online-checkpoint",
        type=Path,
        default=Path(
            "outputs/online_ordinal_v1_learnability_audit_ep200_seed42/"
            "online_ordinal_prediction_error_v1_ep200.pt"
        ),
    )
    parser.add_argument(
        "--leaky-checkpoint",
        type=Path,
        default=Path(
            "outputs/online_ordinal_v1_learnability_audit_ep200_seed42/"
            "leaky_rank_accumulator_meta_sign_v1_ep200.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ciranka2022_exp4_frozen_predictions"),
    )
    return parser


def main() -> None:
    args = create_parser().parse_args()
    schedule_rows = read_jsonl(args.schedule)
    by_subject = validate_schedule(schedule_rows)
    if len(by_subject) != 49:
        raise ValueError(f"schedule contains {len(by_subject)} subjects, expected 49")
    online, leaky = load_models(args.online_checkpoint, args.leaky_checkpoint)

    predictions = generate_predictions(by_subject, online, leaky)
    first_subject_rows = next(iter(by_subject.values()))
    mapping = [3, 0, 7, 1, 6, 2, 5, 4]
    online_mapping_error = max_abs_difference(
        active_probabilities(online, first_subject_rows),
        active_probabilities(online, first_subject_rows, mapping),
    )
    leaky_mapping_error = max_abs_difference(
        active_probabilities(leaky, first_subject_rows),
        active_probabilities(leaky, first_subject_rows, mapping),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    write_jsonl(predictions_path, predictions)
    report = {
        "protocol": {
            "generator": "ciranka2022_exp4_choice_blind_prequential_v1",
            "human_choices_or_rt_read": False,
            "schedule_only": True,
            "trial_order": "predict first; relation update only after deterministic feedback",
            "query_or_no_feedback_write": False,
            "participant_parameters": False,
            "parameter_fitting_on_ciranka": False,
            "decoded_sign_advantage": (
                "model receives the correct relation after feedback and does not implement "
                "human action-reward credit assignment"
            ),
        },
        "inputs": {
            "schedule": str(args.schedule.resolve()),
            "schedule_sha256": sha256(args.schedule),
            "online_checkpoint": str(args.online_checkpoint.resolve()),
            "online_checkpoint_sha256": sha256(args.online_checkpoint),
            "leaky_checkpoint": str(args.leaky_checkpoint.resolve()),
            "leaky_checkpoint_sha256": sha256(args.leaky_checkpoint),
        },
        "frozen_parameters": {
            "online_learning_rate": float(online.learning_rate),
            "online_retention": float(online.retention),
            "online_logit_scale": float(online.logit_scale),
            "leaky_retention": float(leaky.retention),
            "leaky_logit_scale": float(leaky.logit_scale),
        },
        "mapping_audit": {
            "mapping": mapping,
            "online_max_probability_error": online_mapping_error,
            "leaky_max_probability_error": leaky_mapping_error,
            "threshold": 1e-7,
            "pass": online_mapping_error <= 1e-7 and leaky_mapping_error <= 1e-7,
        },
        "output": {
            "subjects": len(by_subject),
            "rows": len(predictions),
            "predictions": str(predictions_path.resolve()),
            "predictions_sha256": sha256(predictions_path),
        },
    }
    report_path = args.output_dir / "prediction_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
