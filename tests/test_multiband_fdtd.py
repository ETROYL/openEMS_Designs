import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from multiband_fdtd.engine import BandResult, FrequencyBand, reconstruct_response
from multiband_fdtd.planner import BandCost, select_fastest_band_sequence
from multiband_fdtd.characterization import (
    characterize_frequency_intervals, generate_candidate_bands,
    characterize_and_select_intervals,
)


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


class CandidateGenerationTests(unittest.TestCase):
    def test_generates_grid_candidates_and_full_range(self):
        bands = generate_candidate_bands((0.1, 1.1), interval_widths_ghz=(0.5, 1.0), boundary_step_ghz=0.25, n_pts=41)
        pairs = {(b.fstart_ghz, b.fstop_ghz) for b in bands}
        self.assertIn((0.1, 1.1), pairs)
        self.assertIn((0.35, 0.85), pairs)
        self.assertTrue(all(b.n_pts == 41 for b in bands))
        self.assertTrue(all(0.1 <= b.fstart_ghz < b.fstop_ghz <= 1.1 for b in bands))

    def test_candidate_limit_and_full_range(self):
        bands = generate_candidate_bands((0.1, 3.0), interval_widths_ghz=(0.5, 1.0, 1.5), boundary_step_ghz=0.1, max_candidates=7)
        self.assertLessEqual(len(bands), 7)
        self.assertIn((0.1, 3.0), {(b.fstart_ghz, b.fstop_ghz) for b in bands})

    def test_single_candidate_limit(self):
        bands = generate_candidate_bands((0.1, 1.0), max_candidates=1)
        self.assertEqual(len(bands), 1)
        self.assertEqual((bands[0].fstart_ghz, bands[0].fstop_ghz), (0.1, 1.0))

    def test_rejects_invalid_generation_settings(self):
        with self.assertRaises(ValueError):
            generate_candidate_bands((0.0, 1.0))
        with self.assertRaises(ValueError):
            generate_candidate_bands((0.1, 1.0), interval_widths_ghz=())
        with self.assertRaises(ValueError):
            generate_candidate_bands((0.1, 1.0), max_candidates=0)


class IntegratedPlanningTests(unittest.TestCase):
    @patch("multiband_fdtd.characterization.characterize_frequency_intervals")
    def test_measures_candidates_selects_and_persists_plan(self, characterize):
        with tempfile.TemporaryDirectory() as temp:
            manifest_path = Path(temp) / "characterization.json"
            manifest = {"intervals": []}
            def fake_characterize(config, output, bands, **kwargs):
                records = [
                    BandCost(
                        band.fstart_ghz, band.fstop_ghz,
                        1.0 + (band.fstop_ghz - band.fstart_ghz),
                        1000, None,
                    )
                    for band in bands
                ]
                return {
                    "records": records,
                    "manifest": manifest,
                    "manifest_path": str(manifest_path),
                    "output_dir": str(temp),
                }
            characterize.side_effect = fake_characterize
            result = characterize_and_select_intervals(
                "design.yaml", Path(temp) / "unused",
                (0.1, 1.1),
                interval_widths_ghz=(0.5, 1.0),
                boundary_step_ghz=0.25,
                n_pts=31,
                max_candidates=20,
                minimum_overlap_ghz=0.1,
            )
            self.assertGreater(len(result["candidates"]), 0)
            self.assertTrue(result["selected"])
            self.assertEqual(
                result["selection"]["frequency_range_ghz"], [0.1, 1.1]
            )
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertIn("selection", saved)
            self.assertEqual(
                len(saved["selection"]["selected_intervals"]),
                len(result["selected"]),
            )
            characterize.assert_called_once()


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
