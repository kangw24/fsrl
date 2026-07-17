"""Boundary and smoke tests for ConstructiveGlobalRank-v1 (CGR-v1).

These tests guard the preregistered input boundary (sign-only, no magnitude /
true_rank / human data) and the constructive output (a self-consistent total
order).  They do NOT assert any human-matching result -- that is a separate
falsification-only step per README section 8.
"""

import unittest

import numpy as np

from fsrl.model.constructive_global_rank import (
    BetaQControl,
    ConstructiveGlobalRank,
    ConstructiveGlobalRankMemoryConstrained,
    ConstructiveGlobalRankOnlineMemory,
    QLearningControl,
    SignOnlyHodgeControl,
)
from fsrl.task.liu2026 import build_liu2026_symbolic_subject_tasks


def _one_task(seed=7):
    rng = np.random.default_rng(seed)
    tasks = build_liu2026_symbolic_subject_tasks(
        n_subjects=1,
        rng=rng,
        nbcues=8,
        randomize_true_rank=True,
        include_magnitude=True,
    )
    return tasks[0]


class ConstructiveGlobalRankBoundaryTests(unittest.TestCase):
    def test_committed_order_is_a_total_order(self):
        task = _one_task()
        model = ConstructiveGlobalRank(8)
        run = model.run_subject(task, np.random.default_rng(1))
        self.assertEqual(sorted(run.order_strong_to_weak), list(range(8)))
        self.assertEqual(len(run.order_strong_to_weak), 8)

    def test_choice_probabilities_are_valid(self):
        task = _one_task()
        run = ConstructiveGlobalRank(8).run_subject(task, np.random.default_rng(2))
        self.assertEqual(len(run.choice_prob), 28)
        for (i, j), p in run.choice_prob.items():
            self.assertTrue(i < j)
            self.assertTrue(0.0 <= p <= 1.0)

    def test_model_does_not_read_true_rank_or_magnitude(self):
        # The candidate's run_subject only inspects support_trials.observation
        # (left/right cue + sign); it must not depend on position_pair or
        # true_rank.  Erase that metadata and confirm identical output.
        task = _one_task()
        model = ConstructiveGlobalRank(8)
        ref = model.run_subject(task, np.random.default_rng(3))

        from fsrl.task.liu2026 import SymbolicSupportTrial, SymbolicSupportObservation

        stripped = type(task)(
            subject_index=task.subject_index,
            true_rank=tuple(range(8)),  # deliberately wrong; must not matter
            support_trials=tuple(
                SymbolicSupportTrial(
                    observation=SymbolicSupportObservation(
                        left_cue=s.observation.left_cue,
                        right_cue=s.observation.right_cue,
                        sign=s.observation.sign,
                    ),
                    block_index=0,
                    position_pair=(0, 0),  # deliberately erased magnitude
                )
                for s in task.support_trials
            ),
            query_trials=task.query_trials,
        )
        out = model.run_subject(stripped, np.random.default_rng(3))
        self.assertEqual(out.order_strong_to_weak, ref.order_strong_to_weak)

    def test_classical_controls_produce_total_orders(self):
        task = _one_task()
        for cls in (QLearningControl, SignOnlyHodgeControl, BetaQControl):
            run = cls(8).run_subject(task, np.random.default_rng(4))
            self.assertEqual(sorted(run.order_strong_to_weak), list(range(8)))


class ConstructiveGlobalRankV31Tests(unittest.TestCase):
    def test_v31_returns_total_order_and_valid_probabilities(self):
        task = _one_task()
        run = ConstructiveGlobalRankMemoryConstrained(8).run_subject(
            task, np.random.default_rng(11)
        )
        self.assertEqual(sorted(run.order_strong_to_weak), list(range(8)))
        self.assertEqual(len(run.choice_prob), 28)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in run.choice_prob.values()))

    def test_perfect_memory_order_respects_every_support_sign(self):
        task = _one_task()
        run = ConstructiveGlobalRankMemoryConstrained(
            8, encoding_probability=1.0
        ).run_subject(task, np.random.default_rng(12))
        rank = {cue: i for i, cue in enumerate(run.order_strong_to_weak)}
        for trial in task.support_trials:
            obs = trial.observation
            high, low = (
                (obs.left_cue, obs.right_cue)
                if obs.sign >= 0
                else (obs.right_cue, obs.left_cue)
            )
            self.assertLess(rank[high], rank[low])

    def test_support_presentation_order_is_not_a_source_of_variation(self):
        task = _one_task()
        reversed_task = type(task)(
            subject_index=task.subject_index,
            true_rank=tuple(reversed(task.true_rank)),  # must not be read
            support_trials=tuple(reversed(task.support_trials)),
            query_trials=task.query_trials,
        )
        model = ConstructiveGlobalRankMemoryConstrained(8)
        ref = model.run_subject(task, np.random.default_rng(13))
        reordered = model.run_subject(reversed_task, np.random.default_rng(13))
        self.assertEqual(ref.order_strong_to_weak, reordered.order_strong_to_weak)
        self.assertEqual(ref.choice_prob, reordered.choice_prob)

    def test_v31_does_not_read_query_trials(self):
        task = _one_task()
        stripped_task = type(task)(
            subject_index=999,
            true_rank=tuple(range(8)),
            support_trials=task.support_trials,
            query_trials=tuple(),
        )
        model = ConstructiveGlobalRankMemoryConstrained(8)
        ref = model.run_subject(task, np.random.default_rng(14))
        stripped = model.run_subject(stripped_task, np.random.default_rng(14))
        self.assertEqual(ref.order_strong_to_weak, stripped.order_strong_to_weak)
        self.assertEqual(ref.choice_prob, stripped.choice_prob)

    def test_v31_parameter_boundaries(self):
        with self.assertRaises(ValueError):
            ConstructiveGlobalRankMemoryConstrained(1)
        with self.assertRaises(ValueError):
            ConstructiveGlobalRankMemoryConstrained(8, sigma_a=-0.1)
        with self.assertRaises(ValueError):
            ConstructiveGlobalRankMemoryConstrained(8, encoding_probability=1.1)
        with self.assertRaises(ValueError):
            ConstructiveGlobalRankMemoryConstrained(8, beta=-1.0)


class ConstructiveGlobalRankV32Tests(unittest.TestCase):
    def test_online_episode_has_one_state_per_support_prefix(self):
        task = _one_task()
        model = ConstructiveGlobalRankOnlineMemory(8)
        run, state = model.run_subject_with_state(task, np.random.default_rng(21))
        self.assertEqual(state.support_steps, len(task.support_trials))
        self.assertEqual(len(state.order_trajectory), len(task.support_trials) + 1)
        self.assertEqual(
            len(state.encoded_relation_trajectory), len(task.support_trials) + 1
        )
        self.assertEqual(len(state.block_trajectory), len(task.support_trials) + 1)
        self.assertEqual(run.order_strong_to_weak, state.order_strong_to_weak)

    def test_each_support_step_updates_the_inspectable_prefix_state(self):
        task = _one_task()
        model = ConstructiveGlobalRankOnlineMemory(8, encoding_probability=1.0)
        state = model.initialize_state(np.random.default_rng(22))
        for step, trial in enumerate(task.support_trials, start=1):
            model.support_step(state, trial)
            self.assertEqual(state.support_steps, step)
            self.assertEqual(len(state.order_trajectory), step + 1)
            self.assertEqual(state.block_trajectory[-1], trial.block_index)
        self.assertGreater(np.linalg.norm(state.axis), 0.0)

    def test_perfect_online_memory_respects_all_seen_support_relations(self):
        task = _one_task()
        model = ConstructiveGlobalRankOnlineMemory(8, encoding_probability=1.0)
        _run, state = model.run_subject_with_state(task, np.random.default_rng(23))
        rank = {cue: index for index, cue in enumerate(state.order_strong_to_weak)}
        for high, low in state.relation_direction.values():
            self.assertLess(rank[high], rank[low])
        self.assertEqual(len(state.encoded_counts), 8)
        self.assertTrue(all(count == 4 for count in state.encoded_counts.values()))

    def test_final_state_is_invariant_to_support_presentation_order(self):
        task = _one_task()
        reordered_task = type(task)(
            subject_index=task.subject_index,
            true_rank=tuple(reversed(task.true_rank)),  # must not be read
            support_trials=tuple(reversed(task.support_trials)),
            query_trials=tuple(),
        )
        model = ConstructiveGlobalRankOnlineMemory(8)
        ref_run, ref_state = model.run_subject_with_state(
            task, np.random.default_rng(24)
        )
        reordered_run, reordered_state = model.run_subject_with_state(
            reordered_task, np.random.default_rng(24)
        )
        self.assertEqual(ref_run.order_strong_to_weak, reordered_run.order_strong_to_weak)
        self.assertEqual(ref_run.choice_prob, reordered_run.choice_prob)
        self.assertEqual(ref_state.encoded_counts, reordered_state.encoded_counts)
        np.testing.assert_allclose(ref_state.axis, reordered_state.axis, atol=1e-10)

    def test_query_is_read_only(self):
        task = _one_task()
        model = ConstructiveGlobalRankOnlineMemory(8)
        _run, state = model.run_subject_with_state(task, np.random.default_rng(25))
        before = (
            state.support_steps,
            state.axis.copy(),
            state.precision_inverse.copy(),
            list(state.order_strong_to_weak),
            list(state.order_trajectory),
            dict(state.encoded_counts),
        )
        probability = model.query_step(state, task.query_trials[0])
        self.assertTrue(0.0 <= probability <= 1.0)
        self.assertEqual(state.support_steps, before[0])
        np.testing.assert_array_equal(state.axis, before[1])
        np.testing.assert_array_equal(state.precision_inverse, before[2])
        self.assertEqual(state.order_strong_to_weak, before[3])
        self.assertEqual(state.order_trajectory, before[4])
        self.assertEqual(state.encoded_counts, before[5])

    def test_online_model_does_not_read_query_trials_or_targets(self):
        task = _one_task()
        stripped_task = type(task)(
            subject_index=1001,
            true_rank=tuple(range(8)),
            support_trials=task.support_trials,
            query_trials=tuple(),
        )
        model = ConstructiveGlobalRankOnlineMemory(8)
        reference = model.run_subject(task, np.random.default_rng(26))
        stripped = model.run_subject(stripped_task, np.random.default_rng(26))
        self.assertEqual(reference.order_strong_to_weak, stripped.order_strong_to_weak)
        self.assertEqual(reference.choice_prob, stripped.choice_prob)

    def test_v32_reads_displayed_magnitude_not_private_position_metadata(self):
        from fsrl.task.liu2026 import SymbolicSupportTrial

        task = _one_task()
        metadata_corrupted = type(task)(
            subject_index=task.subject_index,
            true_rank=tuple(reversed(task.true_rank)),
            support_trials=tuple(
                SymbolicSupportTrial(
                    observation=trial.observation,
                    block_index=trial.block_index,
                    position_pair=(0, 0),
                )
                for trial in task.support_trials
            ),
            query_trials=tuple(),
        )
        model = ConstructiveGlobalRankOnlineMemory(8)
        reference = model.run_subject(task, np.random.default_rng(27))
        corrupted = model.run_subject(metadata_corrupted, np.random.default_rng(27))
        self.assertEqual(reference.order_strong_to_weak, corrupted.order_strong_to_weak)
        self.assertEqual(reference.choice_prob, corrupted.choice_prob)

    def test_v32_rejects_missing_model_visible_magnitude(self):
        from fsrl.task.liu2026 import SymbolicSupportObservation, SymbolicSupportTrial

        task = _one_task()
        original = task.support_trials[0]
        obs = original.observation
        missing = SymbolicSupportTrial(
            observation=SymbolicSupportObservation(
                left_cue=obs.left_cue,
                right_cue=obs.right_cue,
                sign=obs.sign,
            ),
            block_index=original.block_index,
            position_pair=original.position_pair,
        )
        model = ConstructiveGlobalRankOnlineMemory(8)
        state = model.initialize_state(np.random.default_rng(28))
        with self.assertRaises(ValueError):
            model.support_step(state, missing)

    def test_online_endpoint_matches_batch_solve_given_same_latents(self):
        task = _one_task()
        model = ConstructiveGlobalRankOnlineMemory(8)
        _run, state = model.run_subject_with_state(task, np.random.default_rng(29))
        summaries = {
            key: (high, low, magnitude)
            for key, high, low, magnitude, _repetitions in model._group_support(task)
        }
        recalled = []
        for key, count in sorted(state.encoded_counts.items()):
            high, low, magnitude = summaries[key]
            recalled.append((high, low, magnitude, count))
        batch_axis = model._fit_recalled_axis(recalled)
        batch_order = model._preferred_linear_extension(
            batch_axis + state.anchor, recalled
        )
        np.testing.assert_allclose(state.axis, batch_axis, atol=1e-9)
        self.assertEqual(state.order_strong_to_weak, batch_order)


if __name__ == "__main__":
    unittest.main()
