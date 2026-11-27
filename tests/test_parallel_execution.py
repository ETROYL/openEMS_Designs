import os
import unittest
from unittest.mock import patch

from optimizer.parallel_execution import initialize_fdtd_worker


class ParallelWorkerInitializationTests(unittest.TestCase):
    def test_passes_fixed_plan_to_worker_evaluator_after_thread_setup(self):
        with patch.dict(os.environ, {}, clear=False):
            with patch(
                "optimizer.evaluator.CandidateEvaluator.from_existing_pipeline"
            ) as create_evaluator:
                initialize_fdtd_worker(
                    "design.yaml",
                    "runs",
                    object(),
                    "minimize_s11_in_band",
                    {"s11": 1.0},
                    3,
                    {"fstart": 0.1, "fstop": 1.0},
                    "/run/characterization.json",
                )
                self.assertEqual(os.environ["EM_THREADS"], "3")
                self.assertEqual(os.environ["OMP_NUM_THREADS"], "3")
                create_evaluator.assert_called_once()
                kwargs = create_evaluator.call_args.kwargs
                self.assertEqual(kwargs["evaluation_settings"], {
                    "fstart": 0.1, "fstop": 1.0,
                })
                self.assertEqual(
                    kwargs["interval_plan_path"], "/run/characterization.json"
                )


if __name__ == "__main__":
    unittest.main()
