import unittest

import numpy as np

from multiband_fdtd.engine import BandResult, FrequencyBand, reconstruct_response
from multiband_fdtd.planner import BandCost, select_fastest_band_sequence


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


if __name__ == "__main__":
    unittest.main()
