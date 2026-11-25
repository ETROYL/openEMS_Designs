"""Tests for persistent evaluation history."""
import tempfile
import unittest
from pathlib import Path

from optimizer import EvaluationResult, EvaluationStore


def result(fingerprint="abc", objective=2.5, status="ok"):
    return EvaluationResult(
        fingerprint=fingerprint,
        indices=(1, 2),
        parameters={"length": 0.2, "gap": 0.003},
        status=status,
        objective=objective,
        evaluation_dir="/tmp/eval_1",
        elapsed_seconds=12.3,
        metrics={"min_s11_db": -18.2},
        error="simulation failed" if status == "failed" else None,
    )


class EvaluationStoreTests(unittest.TestCase):
    def test_round_trip_and_persistence(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "history.sqlite"
            with EvaluationStore(database) as store:
                self.assertTrue(store.save_result(result()))
                self.assertEqual(store.get("abc").to_dict(), result().to_dict())
            with EvaluationStore(database) as reopened:
                self.assertTrue(reopened.contains("abc"))
                self.assertEqual(reopened.count(), 1)

    def test_duplicate_is_not_overwritten_by_default(self):
        with EvaluationStore(":memory:") as store:
            self.assertTrue(store.save_result(result(objective=2.5)))
            self.assertFalse(store.save_result(result(objective=1.0)))
            self.assertEqual(store.get("abc").objective, 2.5)

    def test_explicit_replace_supports_retry(self):
        with EvaluationStore(":memory:") as store:
            store.save_result(result(status="failed", objective=1e6))
            self.assertTrue(store.save_result(result(objective=1.2), replace=True))
            self.assertEqual(store.get("abc").status, "ok")
            self.assertEqual(store.get("abc").objective, 1.2)

    def test_failed_results_are_persisted_but_not_best(self):
        with EvaluationStore(":memory:") as store:
            store.save_result(result("bad", status="failed", objective=1e6))
            self.assertEqual(store.count("failed"), 1)
            self.assertIsNone(store.best())

    def test_best_ignores_failed_and_selects_minimum(self):
        with EvaluationStore(":memory:") as store:
            store.save_result(result("a", objective=3.0))
            store.save_result(result("b", objective=1.5))
            store.save_result(result("c", objective=-100, status="failed"))
            self.assertEqual(store.best().fingerprint, "b")

    def test_invalid_status_filter_is_rejected(self):
        with EvaluationStore(":memory:") as store:
            with self.assertRaises(ValueError):
                store.list_results("pending")


if __name__ == "__main__":
    unittest.main()
