"""Tests for selectable objective functions and combined-objective weights."""
import unittest

import numpy as np

from objectives import (
    maximize_bandwidth_in_band,
    maximize_impedance_smoothness,
    minimize_max_s11_in_band,
    minimize_s11_in_band,
    multi_objective_combined,
)


class ObjectiveFunctionTests(unittest.TestCase):
    def setUp(self):
        self.results = {
            "freq": np.linspace(1.0e9, 2.0e9, 5),
            "s11": np.array([0.10 + 0.02j, 0.08 + 0.01j, 0.06 + 0.01j, 0.09 + 0.02j, 0.12 + 0.01j]),
            "s11_db": np.array([-19.5, -21.8, -24.3, -20.7, -18.3]),
            "z_in": np.array([45 + 5j, 48 + 8j, 51 + 4j, 53 + 7j, 49 + 3j]),
        }

    def test_available_objectives_return_finite_costs(self):
        functions = (
            maximize_bandwidth_in_band,
            minimize_max_s11_in_band,
            minimize_s11_in_band,
            maximize_impedance_smoothness,
        )
        for function in functions:
            with self.subTest(objective=function.__name__):
                self.assertTrue(np.isfinite(function(self.results)))

    def test_combined_objective_default_weights_preserve_existing_formula(self):
        expected = (
            minimize_s11_in_band(self.results)
            + 0.0053 * maximize_impedance_smoothness(self.results)
        )
        self.assertAlmostEqual(multi_objective_combined(self.results), expected)

    def test_combined_objective_uses_configured_weights(self):
        expected = (
            2.5 * minimize_s11_in_band(self.results)
            + 0.02 * maximize_impedance_smoothness(self.results)
        )
        actual = multi_objective_combined(
            self.results, weight_s11=2.5, weight_smooth=0.02
        )
        self.assertAlmostEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
