import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gui.optimization_runner import _prepare_interval_plan
from multiband_fdtd import load_interval_plan


class IntervalPlanRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        self.config = self.root / "design.yaml"
        self.config.write_text("variables: {}\\nsimulation: {}\\n", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_prepares_equal_width_plan_and_persists_it_for_resume(self):
        settings = {}
        path, bands, created = _prepare_interval_plan(
            self.config, self.run_dir, settings, (0.1, 1.0), 301,
            stop_requested=lambda: False,
        )
        self.assertTrue(created)
        self.assertTrue(Path(path).is_file())
        self.assertEqual(len(bands), 2)
        self.assertEqual(sum(band.n_pts for band in bands), 301)
        self.assertAlmostEqual(bands[0].fstart_ghz, 0.1)
        self.assertAlmostEqual(bands[0].fstop_ghz, 0.55)
        self.assertAlmostEqual(bands[1].fstart_ghz, 0.55)
        self.assertAlmostEqual(bands[1].fstop_ghz, 1.0)
        self.assertTrue(all(band.fstop_ghz - band.fstart_ghz <= 0.5
                            for band in bands))

        saved = json.loads((self.run_dir / "run_settings.json").read_text())
        self.assertEqual(saved["interval_plan_path"], path)
        self.assertEqual(saved["interval_plan_frequency_range_ghz"], [0.1, 1.0])
        manifest = json.loads(Path(path).read_text())
        self.assertEqual(
            manifest["candidate_generation"]["sampling_mode"],
            "equal_total_points",
        )

        reloaded = load_interval_plan(path, n_points=301, config_path=self.config)
        self.assertEqual(reloaded, bands)
        path_again, bands_again, created_again = _prepare_interval_plan(
            self.config, self.run_dir, saved, (0.1, 1.0), 301
        )
        self.assertFalse(created_again)
        self.assertEqual(path_again, path)
        self.assertEqual(bands_again, bands)

    def test_refuses_to_prepare_new_plan_for_existing_history(self):
        (self.run_dir / "evaluations.sqlite").write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "already has evaluation history"):
            _prepare_interval_plan(
                self.config, self.run_dir, {}, (0.1, 1.0), 301
            )

    def test_rejects_frequency_range_change_on_resume(self):
        settings = {
            "interval_plan_path": str(self.run_dir / "missing.json"),
            "interval_plan_frequency_range_ghz": [0.1, 1.0],
        }
        with self.assertRaisesRegex(ValueError, "frequency range"):
            _prepare_interval_plan(
                self.config, self.run_dir, settings, (0.2, 1.0), 301
            )


if __name__ == "__main__":
    unittest.main()
