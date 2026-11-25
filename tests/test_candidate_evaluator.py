import os
import tempfile
import unittest

import numpy as np

from optimizer.evaluator import CandidateEvaluator
from optimizer.parameter_space import DiscreteParameterSpace


class CandidateEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.temp_dir.name, "design.yaml")
        with open(self.config_path, "w", encoding="utf-8") as stream:
            stream.write("variables: {}\nsimulation: {}\n")
        self.space = DiscreteParameterSpace({
            "radius": {"type": "float", "start": 0.10, "stop": 0.20, "step": 0.01},
            "gap": {"type": "float", "start": 0.001, "stop": 0.003, "step": 0.001},
        })

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_evaluates_candidate_and_returns_metrics(self):
        calls = []

        def runner(config_file, parameters, trial_dir):
            calls.append((config_file, parameters, trial_dir))
            return {
                "freq": np.array([1e8, 2e8, 3e8]),
                "s11_db": np.array([-12.0, -18.0, -9.0]),
                "z_in": np.array([70 + 2j, 75 + 1j, 80 + 0j]),
            }

        def objective(results, f_min=None, f_max=None):
            self.assertIsNone(f_min)
            self.assertIsNone(f_max)
            return float(np.max(results["s11_db"]))

        evaluator = CandidateEvaluator(
            self.config_path, os.path.join(self.temp_dir.name, "runs"),
            self.space, runner, objective,
        )
        result = evaluator.evaluate((3, 1), evaluation_id=7)

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.parameters, {"radius": 0.13, "gap": 0.002})
        self.assertEqual(result.objective, -9.0)
        self.assertEqual(result.metrics["max_s11_db"], -9.0)
        self.assertEqual(result.metrics["frequency_points"], 3.0)
        self.assertEqual(len(calls), 1)
        self.assertTrue(result.fingerprint)
        self.assertEqual(result.to_dict()["indices"], [3, 1])

    def test_same_geometry_has_stable_fingerprint(self):
        runner = lambda *_: {"s11_db": np.array([-10.0]), "freq": np.array([1.0])}
        objective = lambda *_args, **_kwargs: 1.0
        evaluator = CandidateEvaluator(
            self.config_path, os.path.join(self.temp_dir.name, "runs"),
            self.space, runner, objective,
        )
        first = evaluator.evaluate((2, 1))
        second = evaluator.evaluate((2, 1))
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertNotEqual(first.evaluation_dir, second.evaluation_dir)

    def test_simulation_failure_is_recorded_not_hidden(self):
        def broken_runner(*_args):
            raise RuntimeError("solver failed")

        evaluator = CandidateEvaluator(
            self.config_path, os.path.join(self.temp_dir.name, "runs"),
            self.space, broken_runner, lambda *_args, **_kwargs: 0.0,
            failed_penalty=12345.0,
        )
        result = evaluator.evaluate((0, 0))
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.objective, 12345.0)
        self.assertIn("solver failed", result.error)


if __name__ == "__main__":
    unittest.main()
