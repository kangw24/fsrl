"""Validated loader for the public Liu et al. (2026) behavioral CSV files.

The OSF release contains test-phase choices only.  It does not contain
learning-phase presentation order, response time, or confidence.  This module
keeps that absence explicit so downstream code cannot silently treat a model
proxy as a measured human variable.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from fsrl.episode.types import EpisodeRecord, TestResponse


LIU2026_EXPECTED_COLUMNS = (
    "id",
    "trial",
    "block",
    "film_choose_index",
    "film_index_1",
    "film_index_2",
    "r_or_w",
)

# Published A--H learning pairs, expressed in the public CSV's low-to-high
# ordinal coding and converted to zero-based indices.
LIU2026_SUPPORT_PAIRS = (
    (0, 5),
    (1, 2),
    (1, 4),
    (2, 6),
    (3, 5),
    (3, 6),
    (4, 7),
    (0, 7),
)


class Liu2026DataError(ValueError):
    """Raised when a public behavior file violates its declared grain."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_pair(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


def _read_rows(path: Path) -> tuple[list[dict[str, int]], tuple[str, ...]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        missing = [column for column in LIU2026_EXPECTED_COLUMNS if column not in columns]
        if missing:
            raise Liu2026DataError(f"{path}: missing required columns {missing}")

        rows: list[dict[str, int]] = []
        for line_number, raw in enumerate(reader, start=2):
            if any(raw.get(column, "") in (None, "") for column in LIU2026_EXPECTED_COLUMNS):
                raise Liu2026DataError(
                    f"{path}:{line_number}: null/empty value in a required column"
                )
            try:
                row = {column: int(raw[column]) for column in LIU2026_EXPECTED_COLUMNS}
            except (TypeError, ValueError) as exc:
                raise Liu2026DataError(
                    f"{path}:{line_number}: required values must be integers"
                ) from exc
            rows.append(row)
    if not rows:
        raise Liu2026DataError(f"{path}: no data rows")
    return rows, columns


def audit_liu2026_behavior_csv(path: str | Path) -> dict:
    """Validate row grain and domain constraints, returning a compact profile."""
    source = Path(path).resolve()
    rows, columns = _read_rows(source)

    grain_keys: set[tuple[int, int, int]] = set()
    exact_rows: set[tuple[int, ...]] = set()
    subjects: dict[int, list[dict[str, int]]] = defaultdict(list)
    pair_counts: Counter[tuple[int, int]] = Counter()
    correctness_mismatches = 0

    for row in rows:
        subject = row["id"]
        trial = row["trial"]
        block = row["block"]
        cue_1 = row["film_index_1"]
        cue_2 = row["film_index_2"]
        choice = row["film_choose_index"]
        correct = row["r_or_w"]

        if subject <= 0:
            raise Liu2026DataError(f"{source}: subject id must be positive")
        if block not in range(1, 11):
            raise Liu2026DataError(f"{source}: block outside 1..10: {block}")
        if trial not in range(1, 29):
            raise Liu2026DataError(f"{source}: trial outside 1..28: {trial}")
        if cue_1 not in range(1, 9) or cue_2 not in range(1, 9) or cue_1 == cue_2:
            raise Liu2026DataError(
                f"{source}: invalid film pair ({cue_1}, {cue_2})"
            )
        if choice not in (cue_1, cue_2):
            raise Liu2026DataError(
                f"{source}: chosen film {choice} is not in ({cue_1}, {cue_2})"
            )
        if correct not in (0, 1):
            raise Liu2026DataError(f"{source}: r_or_w must be 0 or 1")

        grain = (subject, block, trial)
        if grain in grain_keys:
            raise Liu2026DataError(f"{source}: duplicate subject/block/trial {grain}")
        grain_keys.add(grain)

        exact = tuple(row[column] for column in LIU2026_EXPECTED_COLUMNS)
        if exact in exact_rows:
            raise Liu2026DataError(f"{source}: exact duplicate row {exact}")
        exact_rows.add(exact)

        # The public release is already expressed in low-to-high ordinal film
        # indices; choosing the larger index is therefore the correct response.
        expected_correct = int(choice == max(cue_1, cue_2))
        correctness_mismatches += int(correct != expected_correct)
        pair_counts[_canonical_pair(cue_1, cue_2)] += 1
        subjects[subject].append(row)

    if correctness_mismatches:
        raise Liu2026DataError(
            f"{source}: {correctness_mismatches} r_or_w values disagree with choice"
        )

    all_pairs = {
        (i, j) for i in range(1, 9) for j in range(i + 1, 9)
    }
    for subject, subject_rows in subjects.items():
        if len(subject_rows) != 280:
            raise Liu2026DataError(
                f"{source}: subject {subject} has {len(subject_rows)} rows, expected 280"
            )
        for block in range(1, 11):
            block_rows = [row for row in subject_rows if row["block"] == block]
            block_pairs = {
                _canonical_pair(row["film_index_1"], row["film_index_2"])
                for row in block_rows
            }
            block_trials = {row["trial"] for row in block_rows}
            if len(block_rows) != 28 or block_pairs != all_pairs or block_trials != set(range(1, 29)):
                raise Liu2026DataError(
                    f"{source}: subject {subject}, block {block} is not one complete 28-pair block"
                )

    extra_columns = [column for column in columns if column not in LIU2026_EXPECTED_COLUMNS]
    return {
        "path": str(source),
        "sha256": _sha256(source),
        "rows": len(rows),
        "columns": list(columns),
        "extra_columns": extra_columns,
        "n_subjects": len(subjects),
        "subject_ids": sorted(subjects),
        "rows_per_subject": 280,
        "blocks_per_subject": 10,
        "pairs_per_block": 28,
        "repetitions_per_pair": 10,
        "duplicate_grain_rows": 0,
        "exact_duplicate_rows": 0,
        "required_null_rows": 0,
        "correctness_mismatches": 0,
        "available_measures": {
            "choice": True,
            "correctness": True,
            "response_time": False,
            "confidence": False,
            "learning_order": False,
            "learning_phase_response": False,
        },
    }


def _iter_validated_rows(paths: Iterable[str | Path]):
    for path in paths:
        source = Path(path).resolve()
        audit_liu2026_behavior_csv(source)
        rows, _ = _read_rows(source)
        yield source, rows


def build_liu2026_human_episode(paths: Iterable[str | Path]) -> EpisodeRecord:
    """Convert validated public choices to the repository analysis contract."""
    responses: list[TestResponse] = []
    provenance_files = []
    seen_subject_ids: set[int] = set()

    for source, rows in _iter_validated_rows(paths):
        provenance_files.append({"path": str(source), "sha256": _sha256(source)})
        file_subjects = {row["id"] for row in rows}
        overlap = seen_subject_ids & file_subjects
        if overlap:
            raise Liu2026DataError(
                f"subject ids occur in more than one cohort file: {sorted(overlap)}"
            )
        seen_subject_ids.update(file_subjects)

        for row in rows:
            cue_1 = row["film_index_1"] - 1
            cue_2 = row["film_index_2"] - 1
            chosen = row["film_choose_index"] - 1
            pair = _canonical_pair(cue_1, cue_2)
            responses.append(
                TestResponse(
                    block_id=row["block"] - 1,
                    pair=pair,
                    action=0 if chosen == cue_1 else 1,
                    prefers_i_over_j=chosen == pair[0],
                    correct=bool(row["r_or_w"]),
                    batch_index=row["id"],
                    confidence=None,
                    rt_proxy=None,
                )
            )

    if not provenance_files:
        raise Liu2026DataError("at least one behavior CSV is required")

    return EpisodeRecord(
        nbcues=8,
        supervision_set=list(LIU2026_SUPPORT_PAIRS),
        query_set=[(i, j) for i in range(8) for j in range(i + 1, 8)],
        test_responses=responses,
        # Public film indices increase from low to high; analysis expects a
        # strongest-to-weakest permutation.
        true_rank=list(range(7, -1, -1)),
        provenance={
            "source": "Liu et al. (2026) OSF gya95",
            "files": provenance_files,
            "n_subjects": len(seen_subject_ids),
            "measured_response_time": False,
            "measured_confidence": False,
            "learning_order_available": False,
        },
    )
