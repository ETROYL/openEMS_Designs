"""Tests for the discrete Gaussian-process surrogate and EI acquisition."""
import unittest

import numpy as np

from optimizer.surrogate import GaussianProcessSurrogate, expected_improvement


class SurrogateTests(unittest.TestCase):
    def test_gp_interpolates_training_data_approximately(self):
        x = np.array([[0], [2], [4], [6], [8]])
        y = np.array([4.0, 1.0, 0.0, 1.0, 4.0])
        model = GaussianProcessSurrogate(length_scale=0.3, noise=1e-8)
        model.fit(x, y, levels=(9,))
        prediction = model.predict(x)
        self.assertTrue(np.allclose(prediction.mean, y, atol=1e-3))
        self.assertTrue(np.all(prediction.std < 0.01))

    def test_predictions_have_finite_mean_and_nonnegative_uncertainty(self):
        model = GaussianProcessSurrogate().fit(
            [(0, 0), (2, 0), (0, 2), (2, 2)],
            [2.0, 1.0, 1.0, 2.0],
            levels=(3, 3),
        )
        prediction = model.predict([(1, 1), (0, 1)])
        self.assertTrue(np.all(np.isfinite(prediction.mean)))
        self.assertTrue(np.all(np.isfinite(prediction.std)))
        self.assertTrue(np.all(prediction.std >= 0))

    def test_expected_improvement_prefers_promising_or_uncertain_points(self):
        ei = expected_improvement(
            mean=np.array([0.5, 1.5]),
            std=np.array([0.1, 1.0]),
            best_objective=1.0,
            xi=0.0,
        )
        self.assertGreater(ei[0], 0.0)
        self.assertGreater(ei[1], 0.0)

    def test_candidate_ranking_is_discrete_and_excludes_observed(self):
        model = GaussianProcessSurrogate(length_scale=0.3).fit(
            [(0,), (2,), (4,), (6,), (8,)],
            [4.0, 1.0, 0.0, 1.0, 4.0],
            levels=(9,),
        )
        ranked = model.rank_candidates(
            [(1,), (3,), (5,), (7,), (4,)],
            best_objective=0.0,
            observed={(4,)},
        )
        self.assertEqual(len(ranked), 4)
        self.assertNotIn((4,), [item[0] for item in ranked])
        self.assertTrue(all(isinstance(item[0][0], int) for item in ranked))
        self.assertEqual(ranked, sorted(ranked, key=lambda item: item[1], reverse=True))

    def test_rejects_continuous_candidate_coordinates(self):
        model = GaussianProcessSurrogate().fit([(0,), (2,)], [1.0, 0.0], levels=(3,))
        with self.assertRaises(ValueError):
            model.rank_candidates([(0.5,)], best_objective=0.0)


if __name__ == "__main__":
    unittest.main()
