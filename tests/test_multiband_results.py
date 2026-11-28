"""Tests for reconstructed broadband result analysis."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from multiband_fdtd.results import (
    _calculate_efficiencies, _draw_radiation_pattern, _power_to_dbi,
    analyze_broadband_csv, calculate_far_field, process_results,
    wrapped_phase_degrees,
  )


class BroadbandResultsTests(unittest.TestCase):
    def _write_csv(self, path: Path) -> None:
        columns = [
            "frequency_GHz", "s11_raw_real", "s11_raw_imag",
            "s11_smoothed_real", "s11_smoothed_imag", "s11_smoothed_dB",
            "z_in_raw_real", "z_in_raw_imag",
            "z_in_smoothed_real", "z_in_smoothed_imag",
        ]
        rows = [
            [1.0, 0.1, 0.0, 0.1, 0.0, -5.0, 40, 2, 40, 2],
            [2.0, 0.03, 0.0, 0.03, 0.0, -30.0, 49, -1, 49, -1],
            [3.0, 0.05, 0.0, 0.05, 0.0, -26.0, 51, 3, 51, 3],
            [4.0, 0.2, 0.0, 0.2, 0.0, -8.0, 60, 5, 60, 5],
            [5.0, 0.02, 0.0, 0.02, 0.0, -34.0, 50, 0, 50, 0],
            [6.0, 0.2, 0.0, 0.2, 0.0, -10.0, 55, 2, 55, 2],
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)

    def test_analyzes_matching_bands_resonances_and_impedance(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "response.csv"
            self._write_csv(csv_path)
            result = analyze_broadband_csv(csv_path)

        self.assertEqual(result["sample_count"], 6)
        self.assertEqual(result["minimum_s11_db"], -34.0)
        self.assertEqual(result["minimum_s11_frequency_ghz"], 5.0)
        self.assertAlmostEqual(result["impedance_at_minimum_s11_ohm"]["real"], 50.0)
        self.assertEqual(len(result["bands_below_minus_10_db"]), 2)
        self.assertEqual(len(result["bands_below_minus_20_db"]), 2)
        self.assertAlmostEqual(result["total_bandwidth_below_minus_10_db_ghz"], 2.0)
        self.assertEqual(len(result["resonances"]), 2)

    def test_rejects_non_monotonic_frequency_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "response.csv"
            self._write_csv(csv_path)
            text = csv_path.read_text(encoding="utf-8").replace(
                "3.0,0.05", "1.5,0.05"
            )
            csv_path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                analyze_broadband_csv(csv_path)

    def test_rejects_missing_required_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "response.csv"
            csv_path.write_text(
                "frequency_GHz,s11_smoothed_dB\n1,-20\n2,-30\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing columns"):
                analyze_broadband_csv(csv_path)

    def test_generates_metrics_and_plot(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "response.csv"
            output_dir = Path(tmp) / "report"
            self._write_csv(csv_path)
            result = process_results(csv_path, output_dir)
            self.assertTrue((output_dir / "multiband_metrics.json").is_file())
            self.assertTrue((output_dir / "multiband_results.png").is_file())
            self.assertTrue((output_dir / "s11_magnitude_db.png").is_file())
            self.assertTrue((output_dir / "integrated_overview.png").is_file())
            catalog = json.loads((output_dir / "plot_catalog.json").read_text())
            self.assertEqual(catalog[0]["label"], "Integrated overview (2×4)")
            self.assertEqual(catalog[1]["label"], "|S11| magnitude (dB)")
            self.assertIn("wrapped ±180°", catalog[5]["label"])
            self.assertTrue((output_dir / "s11_phase.png").is_file())
            self.assertEqual(result["sample_count"], 6)

    def test_efficiencies_are_derived_from_simulated_power(self):
        values = _calculate_efficiencies(prad=6.0, p_acc=8.0, p_inc=10.0)
        self.assertAlmostEqual(values["radiation"], 0.75)
        self.assertAlmostEqual(values["mismatch"], 0.8)
        self.assertAlmostEqual(values["total"], 0.6)
        self.assertAlmostEqual(
            values["radiation"] * values["mismatch"], values["total"]
        )

    def test_efficiency_power_balance_rejects_invalid_power(self):
        with self.assertRaisesRegex(ValueError, "power balance"):
            _calculate_efficiencies(prad=1.0, p_acc=0.0, p_inc=2.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            _calculate_efficiencies(prad=float("nan"), p_acc=1.0, p_inc=2.0)

    def test_linear_gain_conversion_uses_power_decibels(self):
        self.assertAlmostEqual(_power_to_dbi(10.0), 10.0)
        self.assertAlmostEqual(_power_to_dbi(1.0), 0.0)

    def test_s11_phase_is_wrapped_to_principal_degrees(self):
        import numpy as np

        phase = np.deg2rad([170.0, 190.0, -190.0, 540.0])
        values = np.exp(1j * phase)
        wrapped = wrapped_phase_degrees(values)
        self.assertTrue(np.all(wrapped >= -180.0))
        self.assertTrue(np.all(wrapped <= 180.0))
        np.testing.assert_allclose(wrapped[:3], [170.0, -170.0, 170.0], atol=1e-10)
        self.assertAlmostEqual(abs(wrapped[3]), 180.0, places=10)

    def test_far_field_analysis_rejects_runs_without_recorded_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            simulation_dir = Path(tmp) / "simulation"
            simulation_dir.mkdir()
            (simulation_dir / "multiband_run.json").write_text(
                json.dumps({"far_field_recorded": False}), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "not recorded"):
                calculate_far_field(
                    simulation_dir / "unused.csv",
                    simulation_dir,
                    simulation_dir,
                )


    def test_radiation_pattern_uses_supported_polar_axis_limits(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure = plt.figure()
        axis = figure.add_subplot(111, projection="polar")
        _draw_radiation_pattern(axis, {
            "radiation_pattern_theta_deg": [-180.0, 0.0, 180.0],
            "radiation_pattern_e_plane_db": [-20.0, 0.0, -20.0],
            "radiation_pattern_h_plane_db": [-15.0, -3.0, -15.0],
            "radiation_pattern_frequency_ghz": 2.5,
        })
        self.assertEqual(tuple(axis.get_ylim()), (-30.0, 0.0))
        self.assertEqual(list(axis.get_yticks()), [-30, -20, -10, 0])
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
