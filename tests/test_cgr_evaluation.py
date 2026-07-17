import unittest
from itertools import combinations

import numpy as np

from fsrl.analysis.cgr_evaluation import (
    ALL_PAIRS,
    compute_endpoint_metrics,
    probability_left_from_run,
    report_provenance,
    simulate_query_chronology,
)
from fsrl.model.constructive_global_rank import ModelRun
from fsrl.task.liu2026 import build_liu2026_symbolic_subject_tasks


class _RecordingChanceModel:
    def __init__(self, n_items=8):
        self.n_items = n_items
        self.model_draws = []

    def run_subject(self, task, rng):
        self.model_draws.append(float(rng.random()))
        return ModelRun(
            order_strong_to_weak=list(range(self.n_items)),
            choice_prob={pair: 0.5 for pair in combinations(range(self.n_items), 2)},
        )


class CGREvaluationContractTests(unittest.TestCase):
    def test_endpoint_metrics_on_perfect_transitive_cohort(self):
        cohort = {
            subject: {pair: 1.0 for pair in ALL_PAIRS}
            for subject in range(3)
        }
        result = compute_endpoint_metrics(cohort)
        self.assertEqual(result["overall_accuracy"], 1.0)
        self.assertEqual(result["learned_accuracy"], 1.0)
        self.assertEqual(result["unlearned_accuracy"], 1.0)
        self.assertEqual(result["self_consistency_mean"], 1.0)
        self.assertEqual(result["circular_triads"], 0)

    def test_query_chronology_isolates_model_and_choice_rngs(self):
        tasks = build_liu2026_symbolic_subject_tasks(
            n_subjects=4,
            rng=np.random.default_rng(101),
            include_magnitude=True,
        )
        model = _RecordingChanceModel()
        result = simulate_query_chronology(
            model,
            tasks,
            model_seed=202,
            choice_seed=303,
        )
        expected_model_draws = list(np.random.default_rng(202).random(4))
        np.testing.assert_allclose(model.model_draws, expected_model_draws)
        self.assertEqual(set(result), {0, 1, 2, 3})
        self.assertTrue(all(set(pairs) == set(ALL_PAIRS) for pairs in result.values()))

    def test_probability_left_is_side_complementary(self):
        run = ModelRun(
            order_strong_to_weak=[0, 1],
            choice_prob={(0, 1): 0.8},
        )
        self.assertAlmostEqual(probability_left_from_run(run, 0, 1), 0.8)
        self.assertAlmostEqual(probability_left_from_run(run, 1, 0), 0.2)

    def test_report_provenance_has_explicit_rng_and_protocol_fields(self):
        provenance = report_provenance(
            status="retrospective",
            evaluator="query_chronology_isolated_rng",
            task_seed=1,
            model_seed=2,
            choice_seed=3,
            lapse=0.08,
            parameters={"beta": 12.0},
        )
        self.assertEqual(provenance["task_seed"], 1)
        self.assertEqual(provenance["model_seed"], 2)
        self.assertEqual(provenance["choice_seed"], 3)
        self.assertEqual(provenance["evaluator"], "query_chronology_isolated_rng")
        self.assertEqual(provenance["parameters"], {"beta": 12.0})


if __name__ == "__main__":
    unittest.main()
