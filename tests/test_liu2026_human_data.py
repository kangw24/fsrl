import csv
import tempfile
import unittest
from pathlib import Path

from fsrl.data.liu2026_human import (
    LIU2026_EXPECTED_COLUMNS,
    LIU2026_SUPPORT_PAIRS,
    Liu2026DataError,
    audit_liu2026_behavior_csv,
    build_liu2026_human_episode,
)


def _write_complete_subject(path: Path, subject_id: int = 1, corrupt_choice=False):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LIU2026_EXPECTED_COLUMNS)
        writer.writeheader()
        for block in range(1, 11):
            trial = 0
            for cue_1 in range(1, 9):
                for cue_2 in range(cue_1 + 1, 9):
                    trial += 1
                    choice = cue_2
                    correct = 1
                    if corrupt_choice and block == 1 and trial == 1:
                        correct = 0
                    writer.writerow(
                        {
                            "id": subject_id,
                            "trial": trial,
                            "block": block,
                            "film_choose_index": choice,
                            "film_index_1": cue_1,
                            "film_index_2": cue_2,
                            "r_or_w": correct,
                        }
                    )


class Liu2026HumanDataTests(unittest.TestCase):
    def test_complete_public_grain_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "behavior.csv"
            _write_complete_subject(path)
            audit = audit_liu2026_behavior_csv(path)
            self.assertEqual(audit["rows"], 280)
            self.assertEqual(audit["n_subjects"], 1)
            self.assertFalse(audit["available_measures"]["response_time"])

    def test_correctness_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "behavior.csv"
            _write_complete_subject(path, corrupt_choice=True)
            with self.assertRaisesRegex(Liu2026DataError, "r_or_w"):
                audit_liu2026_behavior_csv(path)

    def test_episode_uses_published_pair_set_and_strong_to_weak_rank(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "behavior.csv"
            _write_complete_subject(path)
            record = build_liu2026_human_episode([path])
            self.assertEqual(record.supervision_set, list(LIU2026_SUPPORT_PAIRS))
            self.assertEqual(record.true_rank, list(range(7, -1, -1)))
            self.assertEqual(len(record.test_responses), 280)
            self.assertTrue(all(response.correct for response in record.test_responses))


if __name__ == "__main__":
    unittest.main()
