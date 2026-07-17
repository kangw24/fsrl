import unittest
from contextlib import redirect_stderr
import io

import numpy as np
import torch
import yaml

from fsrl.analysis.error_dynamics import _compare_static_vs_drift
from fsrl.analysis.hodge import detect_circular_triads
from fsrl.analysis.liu_effects import fit_beta_profile
from fsrl.analysis.matrix import (
    pair_accuracy_from_response_matrix,
    split_supervised_unsupervised,
)
from fsrl.analysis.mode_support_consistency import support_consistency_score
from fsrl.cli.liu2026 import (
    _validate_yaml_schema,
    create_parser,
    merge_episode_records,
)
from fsrl.config import Liu2026Config
from fsrl.episode.liu2026 import _position_pairs_to_canonical_cue_pairs
from fsrl.episode.types import EpisodeRecord, TestResponse
from fsrl.analysis.per_subject import analyze_per_subject_rankings
from fsrl.model.permutation_dist import _permutation_to_scores
from fsrl.model.bayesian_rank_posterior import BayesianRankPosterior
from fsrl.utils.linear_extensions import generate_linear_extensions
from fsrl.task.liu2026 import (
    build_liu2026_support_set,
    generate_liu2026_cue_data,
)


POSITION_PAIRS = ((0, 1), (0, 2), (1, 2))


def _cue_pair(true_rank, position_pair):
    pos_i, pos_j = position_pair
    cue_i, cue_j = true_rank[pos_i], true_rank[pos_j]
    return (min(cue_i, cue_j), max(cue_i, cue_j))


def _response(true_rank, position_pair, correct, batch_index=0):
    strong_cue = true_rank[position_pair[0]]
    pair = _cue_pair(true_rank, position_pair)
    prefers_first_cue = pair[0] == strong_cue if correct else pair[0] != strong_cue
    return TestResponse(
        block_id=0,
        pair=pair,
        action=0,
        prefers_i_over_j=prefers_first_cue,
        correct=correct,
        batch_index=batch_index,
        confidence=0.75,
        rt_proxy=0.25,
    )


def _record(true_rank, supervision_position=(0, 2)):
    correctness = {(0, 1): True, (0, 2): True, (1, 2): False}
    return EpisodeRecord(
        nbcues=3,
        supervision_set=[_cue_pair(true_rank, supervision_position)],
        query_set=[
            (true_rank[pos_i], true_rank[pos_j])
            for pos_i, pos_j in POSITION_PAIRS
        ],
        test_responses=[
            _response(true_rank, pair, correctness[pair])
            for pair in POSITION_PAIRS
        ],
        true_rank=list(true_rank),
        subject_true_ranks={0: list(true_rank)},
        subject_modes={0: 0},
        support_order=[
            (
                true_rank[supervision_position[0]],
                true_rank[supervision_position[1]],
                1,
            )
        ],
        support_dropped_pairs=[_cue_pair(true_rank, (1, 2))],
        subject_support_pairs={0: [_cue_pair(true_rank, supervision_position)]},
        test_perf=2 / 3,
    )


class AnalysisAlignmentTests(unittest.TestCase):
    def test_support_consistency_uses_reference_rank_not_tuple_order(self):
        # Public Liu pairs are stored canonically as (low, high), while ranks
        # are strongest-to-weakest. Pair tuple order must not reverse evidence.
        self.assertEqual(
            support_consistency_score([2, 1, 0], [(0, 2)], [2, 1, 0]),
            1.0,
        )
        self.assertEqual(
            support_consistency_score([0, 1, 2], [(0, 2)], [2, 1, 0]),
            0.0,
        )

    def test_paper_beta_boundary_rule_is_explicit_and_reproducible(self):
        values = [0.0, 0.0, 0.1, 0.9, 1.0, 1.0]
        paper = fit_beta_profile(values, boundary_rule="paper_clip")
        sensitivity = fit_beta_profile(
            values, proportion_denominator=10, boundary_rule="binomial_midpoint"
        )
        self.assertEqual(paper["boundary_rule"], "clip_0.01_0.99")
        self.assertEqual(sensitivity["boundary_rule"], "binomial_midpoint")
        self.assertNotAlmostEqual(paper["alpha"], sensitivity["alpha"])

    def test_tied_pair_is_not_called_a_deterministic_global_ranking(self):
        responses = [
            TestResponse(0, (0, 1), 0, True, True, 0),
            TestResponse(1, (0, 1), 0, False, False, 0),
            # 1 > 2 > 0; resolving the 0/1 tie as 0 > 1 creates a cycle.
            TestResponse(0, (0, 2), 0, False, False, 0),
            TestResponse(0, (1, 2), 0, True, True, 0),
        ]
        result = analyze_per_subject_rankings(
            responses,
            nbcues=3,
            true_rank=[0, 1, 2],
            exclude_all_pairs_majority_correct=False,
        )
        subject = result["per_subject_rankings"][0]
        self.assertEqual(subject["n_tied_pairs"], 1)
        self.assertEqual(subject["n_strict_circular_triads"], 0)
        self.assertEqual(subject["n_circular_triads"], 1)
        self.assertFalse(subject["globally_self_consistent"])

    def test_config_schema_rejects_unknown_nested_field(self):
        parser = create_parser()
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                _validate_yaml_schema(
                    parser,
                    yaml.safe_load("model:\n  temperature: 0.5\n"),
                )

    def test_analytic_control_rejects_training_and_hidden_noise(self):
        config = Liu2026Config(
            analytic_pairwise_control=True,
            use_pairwise_preference_accumulator=True,
            pairwise_preference_fixed_weight=1.0,
            rank_adjustment_scale=0.0,
            num_rank_modes=1,
            subject_embedding_dim=0,
            rank_score_noise_std=0.1,
            pairwise_choice_temperature=0.5,
        )
        with self.assertRaisesRegex(ValueError, "rank_score_noise_std"):
            config.validate(mode="analysis")
        config.rank_score_noise_std = 0.0
        with self.assertRaisesRegex(ValueError, "no trainable behavioral path"):
            config.validate(mode="train")

    def test_cue_identities_are_shared_across_virtual_subjects(self):
        config = Liu2026Config(nbcues=4, cs=6, bs=3)
        cues = generate_liu2026_cue_data(config, np.random.RandomState(9))
        self.assertTrue(np.array_equal(cues[0], cues[1]))
        self.assertTrue(np.array_equal(cues[1], cues[2]))

    def test_published_support_positions_use_low_to_high_axis(self):
        config = Liu2026Config(support_blocks=1)
        # With cue IDs chosen to equal the public low-to-high ordinal labels,
        # the emitted pair set must match the paper exactly.
        public_rank_strong_to_weak = list(range(7, -1, -1))
        trials = build_liu2026_support_set(
            config,
            public_rank_strong_to_weak,
            rng=np.random.RandomState(1),
        )
        emitted = {(min(i, j), max(i, j)) for i, j, _ in trials}
        self.assertEqual(emitted, set(config.support_pairs))
        self.assertTrue(all(sign == -1.0 for _, _, sign in trials))

    def test_circular_triad_reads_reverse_edges_from_upper_triangle(self):
        # 0 > 1, 1 > 2, 2 > 0.  R[2,0] is represented by R[0,2] < .5.
        response_matrix = np.array(
            [
                [0.5, 1.0, 0.0],
                [0.5, 0.5, 1.0],
                [0.5, 0.5, 0.5],
            ]
        )
        self.assertEqual(detect_circular_triads(response_matrix, 3), [[0, 1, 2]])

    def test_transitive_tournament_has_no_circular_triad(self):
        response_matrix = np.array(
            [
                [0.5, 1.0, 1.0],
                [0.5, 0.5, 1.0],
                [0.5, 0.5, 0.5],
            ]
        )
        self.assertEqual(detect_circular_triads(response_matrix, 3), [])

    def test_drift_comparison_uses_complexity_penalty(self):
        comparison = _compare_static_vs_drift(
            np.array([0.5, 0.6, 0.55, 0.45, 0.5, 0.4, 0.52, 0.48, 0.58, 0.42])
        )
        self.assertIsNotNone(comparison)
        self.assertFalse(comparison["drift_better"])
        self.assertIn("aicc_static", comparison)
        self.assertIn("aicc_drift", comparison)

    def test_permutation_to_scores_indexes_scores_by_item(self):
        scores = _permutation_to_scores(torch.tensor([[2, 0, 1]])).tolist()
        self.assertEqual(scores, [[1.0, 0.0, 2.0]])

    def test_linear_extensions_never_inject_invalid_identity(self):
        extensions = generate_linear_extensions(
            3, [(1, 0)], max_extensions=20, seed=7
        )
        self.assertTrue(extensions)
        for extension in extensions:
            self.assertLess(extension.index(1), extension.index(0))

    def test_bayesian_hypotheses_can_reverse_a_configured_support_edge(self):
        posterior = BayesianRankPosterior(
            3, [(0, 1)], max_extensions=6, seed=7
        )
        self.assertEqual(len(posterior.extensions), 6)
        posterior.compute_posterior(
            center_perm=torch.tensor([0, 1, 2], device=posterior._scores.device),
            phi=0.0,
            pairs=[(0, 1)],
            outcomes=[-1.0],
            tau=0.1,
        )
        weights = torch.softmax(posterior._log_weights, dim=0)
        mean_scores = (weights[:, None] * posterior._scores).sum(dim=0)
        self.assertGreater(float(mean_scores[1]), float(mean_scores[0]))

    def test_merge_aligns_randomized_cue_ids_to_ordinal_positions(self):
        merged = merge_episode_records(
            [_record([0, 1, 2]), _record([2, 0, 1])]
        )

        self.assertEqual(merged.true_rank, [0, 1, 2])
        self.assertEqual(merged.supervision_set, [(0, 2)])
        self.assertEqual(merged.query_set, list(POSITION_PAIRS))
        self.assertEqual(merged.support_order, [(0, 2, 1)])
        self.assertEqual(merged.support_dropped_pairs, [(1, 2)])
        self.assertEqual(
            merged.subject_true_ranks, {0: [0, 1, 2], 1: [0, 1, 2]}
        )
        self.assertEqual(merged.subject_modes, {0: 0, 1: 0})
        self.assertEqual(
            merged.subject_support_pairs, {0: [(0, 2)], 1: [(0, 2)]}
        )

        by_pair = {}
        for response in merged.test_responses:
            by_pair.setdefault(response.pair, []).append(response)
        self.assertEqual(set(by_pair), set(POSITION_PAIRS))
        self.assertTrue(all(len(responses) == 2 for responses in by_pair.values()))
        self.assertTrue(all(r.prefers_i_over_j for r in by_pair[(0, 1)]))
        self.assertTrue(all(r.prefers_i_over_j for r in by_pair[(0, 2)]))
        self.assertTrue(all(not r.prefers_i_over_j for r in by_pair[(1, 2)]))

        acc_s, acc_u, n_s, n_u = split_supervised_unsupervised(
            merged.test_responses, merged.supervision_set, merged.query_set
        )
        self.assertEqual((acc_s, acc_u, n_s, n_u), (1.0, 0.5, 1, 2))
        self.assertAlmostEqual(merged.test_perf, 2 / 3)

    def test_merge_rejects_different_position_level_designs(self):
        first = _record([0, 1, 2], supervision_position=(0, 2))
        second = _record([2, 0, 1], supervision_position=(0, 1))

        with self.assertRaisesRegex(ValueError, "ordinal alignment"):
            merge_episode_records([first, second])

    def test_merge_rejects_invalid_true_rank(self):
        record = _record([0, 1, 2])
        record.true_rank = [0, 0, 2]

        with self.assertRaisesRegex(ValueError, "Invalid true_rank"):
            merge_episode_records([record])

    def test_pair_accuracy_does_not_treat_lower_triangle_as_data(self):
        response_matrix = [
            [0.5, 0.8, 0.3],
            [0.5, 0.5, 0.6],
            [0.5, 0.5, 0.5],
        ]
        self.assertEqual(
            pair_accuracy_from_response_matrix(response_matrix, [0, 1, 2]),
            {(0, 1): 0.8, (0, 2): 0.3, (1, 2): 0.6},
        )
        reversed_accuracy = pair_accuracy_from_response_matrix(
            response_matrix, [2, 1, 0]
        )
        self.assertAlmostEqual(reversed_accuracy[(0, 1)], 0.2)
        self.assertAlmostEqual(reversed_accuracy[(0, 2)], 0.7)
        self.assertAlmostEqual(reversed_accuracy[(1, 2)], 0.4)

    def test_position_pair_metadata_is_mapped_before_cue_comparison(self):
        self.assertEqual(
            _position_pairs_to_canonical_cue_pairs([(0, 2)], [2, 0, 1]),
            {(1, 2)},
        )


if __name__ == "__main__":
    unittest.main()
