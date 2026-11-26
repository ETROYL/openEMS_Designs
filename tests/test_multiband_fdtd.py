import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from multiband_fdtd.engine import BandResult, FrequencyBand, reconstruct_response
from multiband_fdtd.planner import BandCost, select_fastest_band_sequence
from multiband_fdtd.characterization import characterize_frequency_intervals


class MultiBandPlannerTests(unittest.TestCase):
    def test_selects_fastest_measured_sequence(self):
        candidates = [
            BandCost(0.1, 0.6, 4.0),
            BandCost(0.5, 1.5, 3.0),
            BandCost(1.4, 3.0, 4.0),
            BandCost(0.1, 1.5, 12.0),
        ]
        selected = select_fastest_band_sequence(
            candidates, (0.1, 3.0), minimum_overlap_ghz=0.1
        )
        self.assertEqual(
            [(x.fstart_ghz, x.fstop_ghz) for x in selected],
            [(0.1, 0.6), (0.5, 1.5), (1.4, 3.0)],
        )

    def test_rejects_gap(self):
        candidates = [BandCost(0.1, 0.5, 2.0), BandCost(0.6, 1.0, 2.0)]
        with self.assertRaises(ValueError):
            select_fastest_band_sequence(candidates, (0.1, 1.0))

    def test_reconstructs_overlapping_complex_responses(self):
        bands = []
        for low, high in ((0.1, 0.6), (0.5, 1.0)):
            freq = np.linspace(low, high, 101) * 1e9
            response = 0.3 * np.exp(1j * 2 * np.pi * freq / 1e9)
            bands.append(BandResult(
                FrequencyBand(low, high, 101), freq, response,
                np.full(freq.shape, 75.0 + 0j), 1.0, None, "test",
            ))
        result = reconstruct_response(bands, (0.1, 1.0), 181)
        self.assertEqual(len(result["freq"]), 181)
        self.assertTrue(np.all(np.isfinite(result["s11"])))
        self.assertIn("s11_raw", result)
        self.assertTrue(np.allclose(result["s11"], result["s11_raw"]))
        self.assertTrue(np.allclose(np.abs(result["s11"]), 0.3, atol=1e-3))


class CharacterizationTests(unittest.TestCase):
    @patch("multiband_fdtd.characterization._run_band")
    def test_records_reference_interval_measurements(self, run_band):
        class FakeResult:
            def __init__(self, runtime, cells, output):
                self.runtime_seconds = runtime
                self.cells = cells
                self.output_dir = str(output)

        run_band.side_effect = [
            FakeResult(1.25, 12000, "pilot_001"),
            FakeResult(2.50, 18000, "pilot_002"),
        ]
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "design.yaml"
            config.write_text("variables: {}\n", encoding="utf-8")
            output = Path(temp) / "measurements"
            result = characterize_frequency_intervals(
                config, output,
                [FrequencyBand(0.1, 0.6, 41), FrequencyBand(0.5, 1.0, 41)],
                threads=2,
            )
            self.assertEqual(len(result["records"]), 2)
            self.assertEqual(result["records"][0].runtime_seconds, 1.25)
            self.assertEqual(result["records"][1].mesh_cells, 18000)
            self.assertTrue(Path(result["manifest_path"]).is_file())
            saved = json.loads(Path(result["manifest_path"]).read_text())
            self.assertEqual(len(saved["intervals"]), 2)
            self.assertEqual(saved["threads"], 2)
            self.assertEqual(run_band.call_count, 2)

    def test_rejects_empty_candidate_list(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "design.yaml"
            config.write_text("variables: {}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                characterize_frequency_intervals(config, Path(temp) / "out", [])


if __name__ == "__main__":
    unittest.main()
