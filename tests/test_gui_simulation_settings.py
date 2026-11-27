import unittest

from gui.simulation_settings import format_simulation_settings


class SimulationSettingsValidationTests(unittest.TestCase):
    def test_valid_settings_are_formatted(self):
        text = """simulation:\n  fstart: 0.1\n  fstop: 3.0\n  n_pts: 301\n  end_criteria_db: -40\n  max_timesteps: 1000000\n"""
        result = format_simulation_settings(text)
        self.assertIn("Frequency: 0.1–3 GHz", result)
        self.assertIn("Frequency points: 301", result)
        self.assertIn("Energy decay: -40 dB", result)
        self.assertIn("Maximum timesteps: 1000000", result)

    def test_rejects_non_numeric_frequency_typo(self):
        text = "simulation:\n  fstart: 0f.1\n  fstop: 3.0\n"
        with self.assertRaisesRegex(ValueError, "fstart"):
            format_simulation_settings(text)

    def test_rejects_non_increasing_frequency_range(self):
        text = "simulation:\n  fstart: 3.0\n  fstop: 0.1\n"
        with self.assertRaisesRegex(ValueError, "increasing"):
            format_simulation_settings(text)

    def test_rejects_invalid_simulation_block(self):
        with self.assertRaisesRegex(ValueError, "simulation"):
            format_simulation_settings("simulation: 0.1\n")


if __name__ == "__main__":
    unittest.main()
