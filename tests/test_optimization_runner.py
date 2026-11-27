import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gui.optimization_runner import _prepare_interval_plan
from multiband_fdtd import FrequencyBand


class IntervalPlanRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        self.config = self.root / "design.yaml"
        self.config.write_text("variables: {}\nsimulation: {}\n", encoding="utf-8")
        self.plan_path = self.run_dir / "interval_preparation" / "characterization.json"
        self.bands = [
            FrequencyBand(0.1, 0.6, 168),
            FrequencyBand(0.5, 1.0, 168),
        ]

    def tearDown(self):
        self.temp.cleanup()

    def test_prepares_once_and_persists_plan_for_resume(self):
        settings = {}
        prepared = {
            "manifest_path": str(self.plan_path),
            "selected_bands": self.bands,
        }
        with patch(
            "multiband_fdtd.characterize_and_select_intervals",
            return_value=prepared,
        ) as characterize:
            path, bands, created = _prepare_interval_plan(
                self.config, self.run_dir, settings, (0.1, 1.0), 301
            )
        characterize.assert_called_once()
        self.assertTrue(created)
        self.assertEqual(path, str(self.plan_path.resolve()))
        self.assertEqual(bands, self.bands)
        saved = json.loads((self.run_dir / "run_settings.json").read_text())
        self.assertEqual(saved["interval_plan_path"], path)
        self.assertEqual(saved["interval_plan_frequency_range_ghz"], [0.1, 1.0])

        with (
            patch("multiband_fdtd.characterize_and_select_intervals") as characterize_again,
            patch("multiband_fdtd.load_interval_plan", return_value=self.bands) as load_plan,
        ):
            path_again, bands_again, created_again = _prepare_interval_plan(
                self.config, self.run_dir, saved, (0.1, 1.0), 301
            )
        characterize_again.assert_not_called()
        load_plan.assert_called_once_with(
            path, n_points=301, config_path=self.config
        )
        self.assertFalse(created_again)
        self.assertEqual(path_again, path)
        self.assertEqual(bands_again, self.bands)

    def test_refuses_to_prepare_new_plan_for_existing_history(self):
        (self.run_dir / "evaluations.sqlite").write_bytes(b"")
        with patch("multiband_fdtd.characterize_and_select_intervals") as characterize:
            with self.assertRaisesRegex(ValueError, "already has evaluation history"):
                _prepare_interval_plan(
                    self.config, self.run_dir, {}, (0.1, 1.0), 301
                )
        characterize.assert_not_called()

    def test_rejects_frequency_range_change_on_resume(self):
        settings = {
            "interval_plan_path": str(self.plan_path),
            "interval_plan_frequency_range_ghz": [0.1, 1.0],
        }
        with patch("multiband_fdtd.load_interval_plan") as load_plan:
            with self.assertRaisesRegex(ValueError, "frequency range"):
                _prepare_interval_plan(
                    self.config, self.run_dir, settings, (0.2, 1.0), 301
                )
        load_plan.assert_not_called()


if __name__ == "__main__":
    unittest.main()
