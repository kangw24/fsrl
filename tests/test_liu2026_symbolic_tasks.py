import unittest
from collections import Counter
from dataclasses import fields
from itertools import combinations

import numpy as np

from fsrl.task.liu2026 import (
    DEFAULT_LIU2026_SUPPORT_PAIRS,
    SymbolicQueryObservation,
    SymbolicSupportObservation,
    build_liu2026_symbolic_subject_tasks,
    build_meta_training_symbolic_subject_tasks,
)


class Liu2026SymbolicSubjectTaskTests(unittest.TestCase):
    def setUp(self):
        self.tasks = build_liu2026_symbolic_subject_tasks(
            n_subjects=12, rng=np.random.default_rng(2026)
        )

    def test_each_subject_has_an_independent_rank_permutation(self):
        ranks = [task.true_rank for task in self.tasks]
        self.assertTrue(all(sorted(rank) == list(range(8)) for rank in ranks))
        self.assertGreater(len(set(ranks)), 1)

    def test_model_facing_support_is_sign_only(self):
        self.assertEqual(
            [field.name for field in fields(SymbolicSupportObservation)],
            ["left_cue", "right_cue", "sign"],
        )
        self.assertTrue(
            all(
                trial.observation.sign in (-1, 1)
                for task in self.tasks
                for trial in task.support_trials
            )
        )

    def test_model_facing_query_has_no_sign_target_or_feedback(self):
        self.assertEqual(
            [field.name for field in fields(SymbolicQueryObservation)],
            ["left_cue", "right_cue"],
        )

    def test_support_schedule_is_four_independent_blocks(self):
        expected = set(DEFAULT_LIU2026_SUPPORT_PAIRS)
        for task in self.tasks:
            self.assertEqual(len(task.support_trials), 32)
            for block_index in range(4):
                pairs = [
                    trial.position_pair
                    for trial in task.support_trials
                    if trial.block_index == block_index
                ]
                self.assertEqual(set(pairs), expected)
                self.assertEqual(Counter(pairs), Counter({pair: 1 for pair in expected}))

    def test_query_schedule_is_ten_complete_blocks(self):
        expected = set(combinations(range(8), 2))
        for task in self.tasks:
            self.assertEqual(len(task.query_trials), 280)
            for block_index in range(10):
                pairs = {
                    trial.position_pair
                    for trial in task.query_trials
                    if trial.block_index == block_index
                }
                self.assertEqual(pairs, expected)

    def test_support_sign_matches_each_subjects_own_rank(self):
        for task in self.tasks:
            cue_to_rank = {cue: position for position, cue in enumerate(task.true_rank)}
            for trial in task.support_trials:
                observation = trial.observation
                left_is_stronger = (
                    cue_to_rank[observation.left_cue]
                    < cue_to_rank[observation.right_cue]
                )
                self.assertEqual(observation.sign, 1 if left_is_stronger else -1)

    def test_query_target_matches_each_subjects_own_rank(self):
        for task in self.tasks:
            cue_to_rank = {cue: position for position, cue in enumerate(task.true_rank)}
            for trial in task.query_trials:
                observation = trial.observation
                left_is_stronger = (
                    cue_to_rank[observation.left_cue]
                    < cue_to_rank[observation.right_cue]
                )
                self.assertEqual(trial.correct_choice, 0 if left_is_stronger else 1)

    def test_left_right_orientation_is_not_fixed(self):
        orientations = {
            trial.observation.sign
            for task in self.tasks
            for trial in task.support_trials
        }
        self.assertEqual(orientations, {-1, 1})

    def test_fixed_seed_is_deterministic(self):
        first = build_liu2026_symbolic_subject_tasks(
            n_subjects=3, rng=np.random.default_rng(99)
        )
        second = build_liu2026_symbolic_subject_tasks(
            n_subjects=3, rng=np.random.default_rng(99)
        )
        self.assertEqual(first, second)


class MetaTrainingTaskDistributionTests(unittest.TestCase):
    def test_distribution_varies_structure_without_distance_fields(self):
        cue_counts = set()
        support_trial_counts = set()
        query_trial_counts = set()
        support_graphs = set()
        for seed in range(20):
            tasks = build_meta_training_symbolic_subject_tasks(
                n_subjects=4,
                rng=np.random.RandomState(seed),
            )
            cue_counts.add(len(tasks[0].true_rank))
            support_trial_counts.add(len(tasks[0].support_trials))
            query_trial_counts.add(len(tasks[0].query_trials))
            support_graphs.update(
                frozenset(trial.position_pair for trial in task.support_trials)
                for task in tasks
            )
            self.assertTrue(
                all(
                    set(task.true_rank) == set(range(len(task.true_rank)))
                    for task in tasks
                )
            )

        self.assertGreater(len(cue_counts), 1)
        self.assertGreater(len(support_trial_counts), 1)
        self.assertGreater(len(query_trial_counts), 1)
        self.assertGreater(len(support_graphs), 20)
        self.assertEqual(
            [field.name for field in fields(SymbolicSupportObservation)],
            ["left_cue", "right_cue", "sign"],
        )

    def test_each_sampled_support_graph_covers_every_active_item(self):
        tasks = build_meta_training_symbolic_subject_tasks(
            n_subjects=12,
            rng=np.random.RandomState(77),
        )
        for task in tasks:
            covered = {
                position
                for trial in task.support_trials
                for position in trial.position_pair
            }
            self.assertEqual(covered, set(range(len(task.true_rank))))


if __name__ == "__main__":
    unittest.main()
