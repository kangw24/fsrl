"""Behavioral boundary and numerical audit tests for CGR 3.2."""

import copy
import unittest

import numpy as np

from cgr_v3_2 import CGR32, SupportObservation, SupportTrial, build_subject_tasks


def one_task(seed: int = 7):
    return build_subject_tasks(
        n_subjects=1,
        rng=np.random.default_rng(seed),
    )[0]


class CGR32Tests(unittest.TestCase):
    def test_task_matches_experimental_trial_counts(self):
        task = one_task()
        self.assertEqual(len(task.support_trials), 32)
        self.assertEqual(len(task.query_trials), 280)
        self.assertEqual(len({t.position_pair for t in task.support_trials}), 8)
        self.assertEqual(len({t.position_pair for t in task.query_trials}), 28)

    def test_one_inspectable_state_for_every_support_prefix(self):
        task = one_task()
        model = CGR32(8)
        run, state = model.run_subject_with_state(task, np.random.default_rng(21))
        expected = len(task.support_trials) + 1
        self.assertEqual(state.support_steps, 32)
        self.assertEqual(len(state.order_trajectory), expected)
        self.assertEqual(len(state.encoded_relation_trajectory), expected)
        self.assertEqual(len(state.block_trajectory), expected)
        self.assertEqual(run.order_strong_to_weak, state.order_strong_to_weak)

    def test_every_encoded_step_updates_online_axis(self):
        task = one_task()
        model = CGR32(8, encoding_probability=1.0)
        state = model.initialize_state(np.random.default_rng(22))
        for step, trial in enumerate(task.support_trials, start=1):
            previous_axis = state.axis.copy()
            model.support_step(state, trial)
            self.assertEqual(state.support_steps, step)
            self.assertEqual(state.block_trajectory[-1], trial.block_index)
            if step == 1:
                self.assertFalse(np.array_equal(previous_axis, state.axis))
        self.assertGreater(np.linalg.norm(state.axis), 0.0)

    def test_perfect_encoding_respects_all_support_directions(self):
        task = one_task()
        model = CGR32(8, encoding_probability=1.0)
        _, state = model.run_subject_with_state(task, np.random.default_rng(23))
        rank = {cue: index for index, cue in enumerate(state.order_strong_to_weak)}
        for high, low in state.relation_direction.values():
            self.assertLess(rank[high], rank[low])
        self.assertEqual(len(state.encoded_counts), 8)
        self.assertTrue(all(count == 4 for count in state.encoded_counts.values()))

    def test_final_state_is_invariant_to_support_order(self):
        task = one_task()
        reordered = type(task)(
            subject_index=task.subject_index,
            true_rank=tuple(reversed(task.true_rank)),
            support_trials=tuple(reversed(task.support_trials)),
            query_trials=tuple(),
        )
        model = CGR32(8)
        run_a, state_a = model.run_subject_with_state(
            task, np.random.default_rng(24)
        )
        run_b, state_b = model.run_subject_with_state(
            reordered, np.random.default_rng(24)
        )
        self.assertEqual(run_a.order_strong_to_weak, run_b.order_strong_to_weak)
        self.assertEqual(run_a.choice_prob, run_b.choice_prob)
        self.assertEqual(state_a.encoded_counts, state_b.encoded_counts)
        np.testing.assert_allclose(state_a.axis, state_b.axis, atol=1e-10)

    def test_query_is_read_only(self):
        task = one_task()
        model = CGR32(8)
        _, state = model.run_subject_with_state(task, np.random.default_rng(25))
        before = copy.deepcopy(state)
        probability = model.query_step(state, task.query_trials[0])
        self.assertTrue(0.0 <= probability <= 1.0)
        self.assertEqual(state.support_steps, before.support_steps)
        np.testing.assert_array_equal(state.axis, before.axis)
        np.testing.assert_array_equal(
            state.precision_inverse, before.precision_inverse
        )
        self.assertEqual(state.order_trajectory, before.order_trajectory)
        self.assertEqual(state.encoded_counts, before.encoded_counts)

    def test_model_does_not_read_query_targets_or_private_metadata(self):
        task = one_task()
        private_data_changed = type(task)(
            subject_index=999,
            true_rank=tuple(reversed(task.true_rank)),
            support_trials=tuple(
                SupportTrial(trial.observation, trial.block_index, (0, 0))
                for trial in task.support_trials
            ),
            query_trials=tuple(),
        )
        model = CGR32(8)
        reference = model.run_subject(task, np.random.default_rng(26))
        changed = model.run_subject(
            private_data_changed, np.random.default_rng(26)
        )
        self.assertEqual(reference, changed)

    def test_missing_visible_magnitude_is_rejected(self):
        class SignOnlyObservation:
            left_cue = 0
            right_cue = 1
            sign = 1

        trial = SupportTrial(SignOnlyObservation(), 0, (0, 1))
        model = CGR32(8)
        state = model.initialize_state(np.random.default_rng(27))
        with self.assertRaises(ValueError):
            model.support_step(state, trial)

    def test_online_endpoint_matches_direct_ridge_solution(self):
        task = one_task()
        model = CGR32(8)
        _, state = model.run_subject_with_state(task, np.random.default_rng(29))
        direct_axis = model.batch_axis_for_audit(state)
        np.testing.assert_allclose(state.axis, direct_axis, atol=1e-9)

    def test_parameter_boundaries(self):
        with self.assertRaises(ValueError):
            CGR32(1)
        with self.assertRaises(ValueError):
            CGR32(8, sigma_a=-0.1)
        with self.assertRaises(ValueError):
            CGR32(8, encoding_probability=1.1)
        with self.assertRaises(ValueError):
            CGR32(8, beta=-1.0)
        with self.assertRaises(ValueError):
            CGR32(8, ridge=0.0)


if __name__ == "__main__":
    unittest.main()
