import os
from unittest.mock import patch
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

    def test_fingerprint_includes_plan_and_normalizes_empty_settings(self):
        plan_path = os.path.join(self.temp_dir.name, "characterization.json")
        with open(plan_path, "w", encoding="utf-8") as stream:
            stream.write('{"plan_version": 1}')
        runner = lambda *_args: {}
        objective = lambda *_args, **_kwargs: 0.0
        first = CandidateEvaluator(
            self.config_path, os.path.join(self.temp_dir.name, "runs"),
            self.space, runner, objective,
            interval_plan_path=plan_path,
            evaluation_settings=None,
        )
        same = CandidateEvaluator(
            self.config_path, os.path.join(self.temp_dir.name, "runs"),
            self.space, runner, objective,
            interval_plan_path=plan_path,
            evaluation_settings={},
        )
        self.assertEqual(first.fingerprint((1, 1)), same.fingerprint((1, 1)))
        with open(plan_path, "w", encoding="utf-8") as stream:
            stream.write('{"plan_version": 1, "changed": true}')
        changed = CandidateEvaluator(
            self.config_path, os.path.join(self.temp_dir.name, "runs"),
            self.space, runner, objective,
            interval_plan_path=plan_path,
            evaluation_settings={},
        )
        self.assertNotEqual(first.fingerprint((1, 1)), changed.fingerprint((1, 1)))

    def test_existing_pipeline_uses_fixed_plan_and_candidate_variables(self):
        from multiband_fdtd import FrequencyBand

        with open(self.config_path, "w", encoding="utf-8") as stream:
            stream.write(
                "variables: {}\\n"
                "simulation:\\n"
                "  fstart: 0.1\\n"
                "  fstop: 1.0\\n"
                "  n_pts: 101\\n"
                "  end_criteria_db: -18.0\\n"
                "  max_timesteps: 25000\\n"
            )
        plan_path = os.path.join(self.temp_dir.name, "characterization.json")
        with open(plan_path, "w", encoding="utf-8") as stream:
            stream.write('{"plan_version": 1}')
        fixed_bands = [
            FrequencyBand(0.1, 0.6, 51),
            FrequencyBand(0.5, 1.0, 51),
        ]
        simulation_result = {
            "freq": np.array([0.1e9, 1.0e9]),
            "s11_db": np.array([-12.0, -18.0]),
            "z_in": np.array([50.0 + 0j, 55.0 + 0j]),
        }
        with (
            patch("multiband_fdtd.load_interval_plan", return_value=fixed_bands) as load_plan,
            patch("multiband_fdtd.run_multiband", return_value=simulation_result) as run_multi,
            patch("objectives.minimize_s11_in_band", return_value=0.25),
            patch.dict(os.environ, {"EM_THREADS": "3"}),
        ):
            evaluator = CandidateEvaluator.from_existing_pipeline(
                self.config_path,
                os.path.join(self.temp_dir.name, "runs"),
                self.space,
                objective_name="minimize_s11_in_band",
                interval_plan_path=plan_path,
            )
            result = evaluator.evaluate((3, 1), evaluation_id=9)

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.objective, 0.25)
        load_plan.assert_called_once_with(
            plan_path, n_points=101, config_path=self.config_path
        )
        kwargs = run_multi.call_args.kwargs
        self.assertEqual(kwargs["bands"], fixed_bands)
        self.assertEqual(kwargs["active_vars"], {"radius": 0.13, "gap": 0.002})
        self.assertEqual(kwargs["frequency_range_ghz"], (0.1, 1.0))
        self.assertEqual(kwargs["n_points"], 101)
        self.assertEqual(kwargs["end_criteria_db"], -18.0)
        self.assertEqual(kwargs["max_timesteps"], 25000)
        self.assertEqual(kwargs["threads"], 3)

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
