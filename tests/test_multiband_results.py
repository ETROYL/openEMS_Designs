"""Tests for reconstructed broadband result analysis."""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from multiband_fdtd.results import analyze_broadband_csv, process_results


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
            self.assertEqual(result["sample_count"], 6)


if __name__ == "__main__":
    unittest.main()
