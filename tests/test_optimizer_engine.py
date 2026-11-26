"""Tests for the end-to-end discrete optimizer orchestration."""
import os
import tempfile
import unittest

from optimizer import (
    CandidateEvaluator,
    DiscreteParameterSpace,
    DiscreteSurrogateOptimizer,
    EvaluationStore,
)


class DiscreteSurrogateOptimizerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.config = os.path.join(self.temp.name, "design.yaml")
        with open(self.config, "w", encoding="utf-8") as stream:
            stream.write("variables: {x: {start: 0, stop: 8, step: 1}}\n")
        self.space = DiscreteParameterSpace({"x": {"start": 0, "stop": 8, "step": 1}})
        self.calls = []

        def simulate(_config, parameters, output_dir):
            self.calls.append(parameters["x"])
            return {"score": float((parameters["x"] - 3) ** 2)}

        def objective(results, f_min=None, f_max=None):
            return results["score"]

        self.evaluator = CandidateEvaluator(
            self.config, os.path.join(self.temp.name, "runs"), self.space,
            simulate, objective,
        )
        self.database = os.path.join(self.temp.name, "history.sqlite")
        self.store = EvaluationStore(self.database)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_runs_initial_design_and_surrogate_steps_within_budget(self):
        optimizer = DiscreteSurrogateOptimizer(
            self.space, self.evaluator, self.store,
            initial_samples=3, pool_size=8, min_surrogate_points=3, seed=7,
        )
        summary = optimizer.run(budget=5)
        self.assertEqual(len(summary.evaluations), 5)
        self.assertEqual(self.store.count(), 5)
        self.assertEqual(len(self.calls), 5)
        self.assertIsNotNone(summary.best)
        self.assertLessEqual(summary.best.objective, max(r.objective for r in summary.evaluations))

    def test_resume_uses_history_and_never_repeats_simulation(self):
        optimizer = DiscreteSurrogateOptimizer(
            self.space, self.evaluator, self.store,
            initial_samples=3, pool_size=8, min_surrogate_points=3, seed=11,
        )
        first = optimizer.run(budget=3)
        self.assertEqual(len(first.evaluations), 3)
        second = optimizer.run(budget=2)
        self.assertEqual(len(second.evaluations), 2)
        self.assertEqual(self.store.count(), 5)
        self.assertEqual(len(self.calls), 5)
        self.assertEqual(len(set(self.calls)), 5)

    def test_existing_fingerprint_is_skipped_before_simulation(self):
        point = (2,)
        result = self.evaluator.evaluate(point, evaluation_id="preexisting")
        self.store.save_result(result)
        before = len(self.calls)
        optimizer = DiscreteSurrogateOptimizer(
            self.space, self.evaluator, self.store,
            initial_samples=2, pool_size=8, seed=3,
        )
        summary = optimizer.run(budget=2)
        self.assertEqual(len(summary.evaluations), 2)
        self.assertEqual(len(self.calls) - before, 2)
        self.assertTrue(self.store.contains(self.evaluator.fingerprint(point)))

    def test_progress_callback_and_cooperative_stop(self):
        optimizer = DiscreteSurrogateOptimizer(
            self.space, self.evaluator, self.store,
            initial_samples=4, pool_size=8, seed=13,
        )
        events = []
        stop = {"requested": False}

        def report(result, completed, budget):
            events.append((result.status, completed, budget))
            stop["requested"] = True

        summary = optimizer.run(
            budget=6,
            progress_callback=report,
            stop_requested=lambda: stop["requested"],
        )
        self.assertEqual(len(summary.evaluations), 1)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][1:], (1, 6))
        self.assertEqual(summary.stop_reason, "stop requested")
        self.assertEqual(self.store.count(), 1)

    def test_zero_budget_does_not_evaluate(self):
        optimizer = DiscreteSurrogateOptimizer(self.space, self.evaluator, self.store)
        summary = optimizer.run(budget=0)
        self.assertEqual(summary.evaluations, [])
        self.assertEqual(self.store.count(), 0)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
