"""Audit and split Ciranka et al. (2022) Experiment 4 data.

The public MATLAB matrix is separated into a schedule file and a choice file.
Downstream model prediction reads the schedule only; human choices and RT stay
in a physically separate artifact for the later agent-side comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.stats import binomtest


EXPECTED_SHA256 = "c8f3c8d1dd2111495d24fbddbde4f2e3e084455e51165b87809bce2eebd19c7a"
EXPECTED_SHAPE = (70, 8, 6, 60)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def validate_matrix(matrix: np.ndarray) -> dict:
    if matrix.shape != EXPECTED_SHAPE:
        raise ValueError(f"unexpected behav_result_mat shape {matrix.shape}")

    stimuli = matrix[:, 0:2, :, :]
    if not np.isfinite(stimuli).all():
        raise ValueError("stimulus columns must be complete")
    if set(np.unique(stimuli).tolist()) != set(range(1, 9)):
        raise ValueError("stimulus tokens must cover exactly 1..8")

    choices = matrix[:, 2, :, :]
    finite_choice = np.isfinite(choices)
    if not set(np.unique(choices[finite_choice]).tolist()).issubset({1.0, 2.0}):
        raise ValueError("valid choices must be side 1 or 2")

    feedback = matrix[:, 3, :, :]
    if not np.isfinite(feedback).all():
        raise ValueError("feedback column must be complete")
    if not set(np.unique(feedback).tolist()).issubset({0.0, 1.0, 2.0, 3.0}):
        raise ValueError("feedback must use public codes 0/1/2/3")
    if np.any((feedback == 2.0) & finite_choice):
        raise ValueError("feedback=2 must occur only on missing-response rows")
    if np.any((~finite_choice) & (feedback != 2.0)):
        raise ValueError("every missing response must carry feedback=2")

    chosen_value = matrix[:, 4, :, :]
    expected_chosen = np.where(
        choices == 1.0, matrix[:, 0, :, :], matrix[:, 1, :, :]
    )
    if not np.allclose(
        chosen_value[finite_choice], expected_chosen[finite_choice]
    ):
        raise ValueError("chosen-value column disagrees with choice side")

    rt = matrix[:, 5, :, :]
    if not np.isfinite(rt[finite_choice]).all() or np.any(rt[finite_choice] <= 0):
        raise ValueError("valid responses require a positive finite RT")

    return {
        "valid_choice_rows": int(finite_choice.sum()),
        "missing_choice_rows": int((~finite_choice).sum()),
        "feedback_rows": int(np.isin(feedback, [0.0, 1.0]).sum()),
        "no_feedback_rows": int((feedback == 3.0).sum()),
        "missing_feedback_sentinel_rows": int((feedback == 2.0).sum()),
        "rt_min": float(rt[finite_choice].min()),
        "rt_median": float(np.median(rt[finite_choice])),
        "rt_max": float(rt[finite_choice].max()),
    }


def included_subjects(matrix: np.ndarray) -> tuple[list[int], list[dict]]:
    """Replicate the public R rule: blocks 5--6, binomial p < .01."""

    included: list[int] = []
    profiles: list[dict] = []
    for subject_zero in range(matrix.shape[3]):
        late_correct: list[int] = []
        for block_zero in (4, 5):
            block = matrix[:, :, block_zero, subject_zero]
            valid = np.isfinite(block[:, 2])
            selected = block[valid, 4]
            correct_value = np.maximum(block[valid, 0], block[valid, 1])
            late_correct.extend((selected == correct_value).astype(int).tolist())
        successes = int(sum(late_correct))
        trials = len(late_correct)
        p_value = float(
            binomtest(successes, trials, 0.5, alternative="greater").pvalue
        )
        subject_index = subject_zero + 1
        keep = p_value < 0.01
        profiles.append(
            {
                "subject_index": subject_index,
                "late_valid_trials": trials,
                "late_accuracy": successes / trials,
                "one_sided_binomial_p": p_value,
                "included": keep,
            }
        )
        if keep:
            included.append(subject_index)
    return included, profiles


def split_rows(
    matrix: np.ndarray, included: list[int]
) -> tuple[list[dict], list[dict]]:
    schedule_rows: list[dict] = []
    choice_rows: list[dict] = []
    for subject_index in included:
        subject_zero = subject_index - 1
        for block_zero in range(matrix.shape[2]):
            for slot_zero in range(matrix.shape[0]):
                row = matrix[slot_zero, :, block_zero, subject_zero]
                left = int(row[0]) - 1
                right = int(row[1]) - 1
                feedback_code = int(row[3])
                if feedback_code in (0, 1):
                    feedback_kind = "relation"
                    sign_after_trial = 1 if left > right else -1
                elif feedback_code == 3:
                    feedback_kind = "none"
                    sign_after_trial = None
                elif feedback_code == 2:
                    feedback_kind = "missing_response"
                    sign_after_trial = None
                else:  # guarded by validate_matrix
                    raise AssertionError(feedback_code)
                trial_index = block_zero * matrix.shape[0] + slot_zero
                schedule_rows.append(
                    {
                        "subject_index": subject_index,
                        "block_index": block_zero,
                        "slot_index": slot_zero,
                        "trial_index": trial_index,
                        "left_cue": left,
                        "right_cue": right,
                        "feedback_kind": feedback_kind,
                        "sign_after_trial": sign_after_trial,
                    }
                )

                if not np.isfinite(row[2]):
                    continue
                chosen_side = int(row[2]) - 1
                correct_side = 0 if left > right else 1
                choice_rows.append(
                    {
                        "subject_index": subject_index,
                        "block_index": block_zero,
                        "slot_index": slot_zero,
                        "trial_index": trial_index,
                        "chosen_side": chosen_side,
                        "correct_side": correct_side,
                        "correct": chosen_side == correct_side,
                        "condition": (
                            "feedback" if feedback_code in (0, 1) else "no_feedback"
                        ),
                        "rt_seconds": float(row[5]),
                    }
                )
    return schedule_rows, choice_rows


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit and split the public Ciranka 2022 Exp.4 matrix."
    )
    parser.add_argument(
        "mat",
        type=Path,
        default=Path("outputs/external_data/ciranka2022/exp4.mat"),
        nargs="?",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ciranka2022_exp4_audit"),
    )
    return parser


def main() -> None:
    args = create_parser().parse_args()
    source_hash = sha256(args.mat)
    if source_hash != EXPECTED_SHA256:
        raise ValueError(f"source SHA-256 mismatch: {source_hash}")
    payload = loadmat(args.mat)
    if "behav_result_mat" not in payload:
        raise ValueError("MAT file lacks behav_result_mat")
    matrix = np.asarray(payload["behav_result_mat"], dtype=np.float64)
    profile = validate_matrix(matrix)
    included, subject_profiles = included_subjects(matrix)
    if len(included) != 49:
        raise ValueError(f"official inclusion rule yielded {len(included)}, expected 49")
    schedule_rows, choice_rows = split_rows(matrix, included)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    schedule_path = args.output_dir / "schedule.jsonl"
    choices_path = args.output_dir / "choices.jsonl"
    write_jsonl(schedule_path, schedule_rows)
    write_jsonl(choices_path, choice_rows)

    report = {
        "source": {
            "citation": "Ciranka et al. (2022), Nature Human Behaviour",
            "doi": "10.1038/s41562-021-01263-w",
            "repository": "https://doi.org/10.5281/zenodo.5561411",
            "official_path": "data/combined_mat/exp4.mat",
            "local_path": str(args.mat.resolve()),
            "sha256": source_hash,
        },
        "matrix": {"shape": list(matrix.shape), **profile},
        "inclusion": {
            "rule": "blocks 5-6 one-sided binomial test above chance, p < 0.01",
            "raw_subjects": matrix.shape[3],
            "included_subjects": len(included),
            "included_subject_indices": included,
            "subject_profiles": subject_profiles,
        },
        "split_firewall": {
            "schedule_contains_human_choices_or_rt": False,
            "schedule_rows": len(schedule_rows),
            "schedule_path": str(schedule_path.resolve()),
            "schedule_sha256": sha256(schedule_path),
            "choices_rows": len(choice_rows),
            "choices_path": str(choices_path.resolve()),
            "choices_sha256": sha256(choices_path),
        },
        "limitations": [
            "This action-before-feedback task is not Liu passive observation learning.",
            "The sign-only model receives the decoded correct relation after feedback and does not implement action-reward credit assignment.",
            "Passing this audit cannot establish a Liu mechanism.",
        ],
    }
    report_path = args.output_dir / "data_quality_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "included_subjects": len(included),
                "schedule_rows": len(schedule_rows),
                "choices_rows": len(choice_rows),
                "schedule_sha256": report["split_firewall"]["schedule_sha256"],
                "choices_sha256": report["split_firewall"]["choices_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
