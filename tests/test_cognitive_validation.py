from __future__ import annotations

from math import log
import unittest

import numpy as np

from fsrl.cognitive_validation.heldout import (
    ChoicePanel,
    evaluate_independent_binary_prior,
    evaluate_particle_prior,
    make_uniform_permutation_prior,
    particle_prior_log_marginal,
    temperature_scale_probabilities,
)


class CognitiveValidationTests(unittest.TestCase):
    def _panel(self) -> ChoicePanel:
        pairs = np.asarray([(0, 1), (0, 2), (1, 2)], dtype="int16")
        choices = np.ones((2, 10, 3), dtype="int8")
        choices[1, :, 1] = 0
        return ChoicePanel(
            subject_ids=np.asarray([11, 12], dtype="int32"),
            canonical_pairs=pairs,
            chose_higher=choices,
            provenance={},
        )

    def test_chance_has_exact_conditional_log_score(self):
        panel = self._panel()
        result = evaluate_particle_prior(
            panel,
            np.full((1, 3), 0.5),
            candidate="chance",
            conditioning_blocks=(1, 2, 3, 4, 5),
            scored_blocks=(6, 7, 8, 9, 10),
        )
        np.testing.assert_allclose(result.subject_log_predictive, -15 * log(2.0))
        np.testing.assert_allclose(result.subject_posterior_ess, 1.0)

    def test_shared_particles_are_conditioned_without_subject_lookup(self):
        panel = self._panel()
        particles = np.asarray(
            [[0.95, 0.95, 0.95], [0.95, 0.05, 0.95]], dtype="float64"
        )
        result = evaluate_particle_prior(
            panel,
            particles,
            candidate="two_rank_states",
            conditioning_blocks=(1, 2, 3, 4, 5),
            scored_blocks=(6, 7, 8, 9, 10),
        )
        self.assertGreater(result.subject_accuracy.mean(), 0.99)
        self.assertTrue(np.all(result.subject_posterior_ess < 1.01))

    def test_independent_binary_prior_is_an_exact_pairwise_ceiling(self):
        panel = self._panel()
        result = evaluate_independent_binary_prior(
            panel,
            lapse=0.05,
            conditioning_blocks=(1, 2, 3, 4, 5),
            scored_blocks=(6, 7, 8, 9, 10),
        )
        self.assertGreater(result.summary()["mean_bits_per_choice_over_chance"], 0.5)
        self.assertIsNone(result.subject_posterior_ess)

    def test_uniform_permutation_prior_enumerates_every_ranking(self):
        pairs, probabilities = make_uniform_permutation_prior(4, lapse=0.05)
        self.assertEqual(pairs.shape, (6, 2))
        self.assertEqual(probabilities.shape, (24, 6))
        np.testing.assert_allclose(
            np.unique(probabilities), np.asarray([0.05, 0.95]), atol=1e-7
        )

    def test_temperature_scaling_softens_without_changing_direction(self):
        probabilities = np.asarray([[0.001, 0.25, 0.75, 0.999]])
        softened = temperature_scale_probabilities(probabilities, 4.0)
        self.assertTrue(np.all(softened[0, :2] < 0.5))
        self.assertTrue(np.all(softened[0, 2:] > 0.5))
        self.assertTrue(np.all(np.abs(softened - 0.5) < np.abs(probabilities - 0.5)))

    def test_particle_marginal_prefers_matching_prior(self):
        panel = self._panel()
        matching = np.asarray([[0.95, 0.95, 0.95], [0.95, 0.05, 0.95]])
        chance = np.full((1, 3), 0.5)
        self.assertGreater(
            particle_prior_log_marginal(
                panel, matching, blocks=(1, 2, 3, 4, 5)
            ),
            particle_prior_log_marginal(
                panel, chance, blocks=(1, 2, 3, 4, 5)
            ),
        )


if __name__ == "__main__":
    unittest.main()
