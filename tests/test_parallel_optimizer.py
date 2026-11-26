"""Tests for process-batched discrete optimization scheduling."""
import os
import tempfile
import unittest
from concurrent.futures import Future
from unittest.mock import patch

from optimizer import (
    CandidateEvaluator,
    DiscreteParameterSpace,
    DiscreteSurrogateOptimizer,
    EvaluationResult,
    EvaluationStore,
)


class _InlineProcessPool:
    """Executor test double that records submissions without spawning processes."""
    instances = []

    def __init__(self, max_workers, **_kwargs):
        self.max_workers = max_workers
        self.max_in_flight = 0
        self.submitted = 0
        self.__class__.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def submit(self, function, task):
        self.submitted += 1
        self.max_in_flight = max(self.max_in_flight, self.submitted)
        future = Future()
        try:
            future.set_result(function(task))
        except Exception as exc:
            future.set_exception(exc)
        return future


class ParallelOptimizerSchedulingTests(unittest.TestCase):
    def test_initial_design_is_dispatched_as_worker_sized_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            config = os.path.join(temp, "design.yaml")
            with open(config, "w", encoding="utf-8") as stream:
                stream.write("variables: {x: {start: 0, stop: 20, step: 1}}\\n")
            space = DiscreteParameterSpace({"x": {"start": 0, "stop": 20, "step": 1}})
            evaluator = CandidateEvaluator(
                config, os.path.join(temp, "runs"), space,
                simulation_runner=lambda *_args: {},
                objective_function=lambda *_args, **_kwargs: 0.0,
            )
            database = os.path.join(temp, "history.sqlite")
            with EvaluationStore(database) as store:
                optimizer = DiscreteSurrogateOptimizer(
                    space, evaluator, store, initial_samples=4, pool_size=16, seed=3,
                )

                def fake_worker(task):
                    indices, evaluation_id = task
                    fingerprint = evaluator.fingerprint(indices)
                    return EvaluationResult(
                        fingerprint=fingerprint,
                        indices=indices,
                        parameters=space.to_values(indices),
                        status="ok",
                        objective=float(indices[0] ** 2),
                        evaluation_dir=os.path.join(temp, f"eval_{evaluation_id}"),
                        elapsed_seconds=0.01,
                        metrics={},
                    )

                _InlineProcessPool.instances.clear()
                with patch("optimizer.engine.ProcessPoolExecutor", _InlineProcessPool), \\
                     patch("optimizer.engine.evaluate_fdtd_candidate", fake_worker):
                    summary = optimizer.run_parallel(
                        4, workers=4, threads_per_worker=2,
                        objective_name="minimize_s11_in_band",
                    )

                pool = _InlineProcessPool.instances[-1]
                self.assertEqual(pool.max_workers, 4)
                self.assertEqual(pool.submitted, 4)
                self.assertEqual(len(summary.evaluations), 4)
                self.assertEqual(store.count(), 4)
                self.assertEqual(len({item.indices for item in store.list_results()}), 4)


if __name__ == "__main__":
    unittest.main()
