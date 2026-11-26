"""Unit tests for the Optuna adapter's SQLite history restoration."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

from optimizer import (
    CandidateEvaluator,
    DiscreteParameterSpace,
    EvaluationResult,
    EvaluationStore,
    OptunaTPEOptimizer,
)


@unittest.skipUnless(importlib.util.find_spec("optuna"), "Optuna is an optional dependency")
class OptunaHistoryTests(unittest.TestCase):
    def test_existing_sqlite_evaluations_restore_as_completed_trials(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.yaml"
            config.write_text("variables: {}\n", encoding="utf-8")
            space = DiscreteParameterSpace({
                "arm_length": {
                    "type": "float", "start": 0.10, "stop": 0.30,
                    "step": 0.10, "optimize": True,
                }
            })
            evaluator = CandidateEvaluator(
                str(config), str(root / "runs"), space,
                simulation_runner=lambda *_args: {},
                objective_function=lambda *_args, **_kwargs: 0.0,
            )
            with EvaluationStore(root / "evaluations.sqlite") as store:
                store.save_result(EvaluationResult(
                    fingerprint=evaluator.fingerprint((1,)),
                    indices=(1,),
                    parameters={"arm_length": 0.2},
                    status="ok",
                    objective=0.375,
                    evaluation_dir=str(root / "runs" / "eval_1"),
                    elapsed_seconds=1.0,
                    metrics={},
                ))
                optimizer = OptunaTPEOptimizer(space, evaluator, store, seed=7)
                _optuna, study, _distributions = optimizer._restore_study()

                self.assertEqual(len(study.trials), 1)
                self.assertEqual(study.trials[0].state.name, "COMPLETE")
                self.assertAlmostEqual(study.trials[0].value, 0.375)
                self.assertEqual(study.trials[0].params, {"arm_length": 1})


if __name__ == "__main__":
    unittest.main()
