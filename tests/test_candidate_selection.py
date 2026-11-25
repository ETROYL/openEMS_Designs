"""Tests for discrete Sobol candidate-pool generation and EI selection."""
import unittest

from optimizer import (
    DiscreteParameterSpace,
    GaussianProcessSurrogate,
    ParameterSpaceError,
    discrete_candidate_pool,
    select_next_candidates,
)


class CandidateSelectionTests(unittest.TestCase):
    def setUp(self):
        self.space = DiscreteParameterSpace({
            "x": {"type": "int", "start": 0, "stop": 19},
            "y": {"type": "int", "start": 0, "stop": 9},
        })

    def test_pool_is_unique_legal_and_reproducible(self):
        excluded = {(0, 0), (19, 9)}
        first = discrete_candidate_pool(self.space, 80, excluded=excluded, seed=17)
        second = discrete_candidate_pool(self.space, 80, excluded=excluded, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))
        self.assertEqual(len(first), 80)
        self.assertTrue(all(0 <= x <= 19 and 0 <= y <= 9 for x, y in first))
        self.assertTrue(set(first).isdisjoint(excluded))

    def test_pool_reports_exhausted_space(self):
        tiny = DiscreteParameterSpace({"x": {"type": "int", "start": 0, "stop": 2}})
        with self.assertRaises(ParameterSpaceError):
            discrete_candidate_pool(tiny, 2, excluded={(0,), (1,)})

    def test_pool_rejects_non_integer_exclusions(self):
        with self.assertRaises(ParameterSpaceError):
            discrete_candidate_pool(self.space, 3, excluded={(0.5, 1)})

    def test_selector_returns_top_ei_candidates_and_excludes_observed(self):
        observed = {(0, 0), (2, 0), (0, 2), (2, 2)}
        model = GaussianProcessSurrogate(length_scale=0.3).fit(
            list(observed), [2.0, 1.0, 1.0, 2.0], levels=(3, 3)
        )
        space = DiscreteParameterSpace({
            "x": {"type": "int", "start": 0, "stop": 2},
            "y": {"type": "int", "start": 0, "stop": 2},
        })
        selected = select_next_candidates(
            space, model, best_objective=1.0, observed=observed,
            n_candidates=2, pool_size=5, seed=3,
        )
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(point not in observed for point, _ in selected))
        self.assertGreaterEqual(selected[0][1], selected[1][1])

    def test_selector_rejects_invalid_pool_size(self):
        model = GaussianProcessSurrogate().fit([(0,), (2,)], [1.0, 0.0], levels=(3,))
        tiny = DiscreteParameterSpace({"x": {"type": "int", "start": 0, "stop": 2}})
        with self.assertRaises(ParameterSpaceError):
            select_next_candidates(tiny, model, 0.0, {(0,), (2,)}, pool_size=0)


if __name__ == "__main__":
    unittest.main()
