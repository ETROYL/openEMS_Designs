import unittest

from gui.simulation_settings import format_simulation_settings, parse_simulation_settings


class SimulationSettingsValidationTests(unittest.TestCase):
    def test_valid_settings_are_formatted(self):
        text = """simulation:
  fstart: 0.1
  fstop: 3.0
  n_pts: 301
  end_criteria_db: -40
  max_timesteps: 1000000
"""
        result = format_simulation_settings(text)
        self.assertIn("Frequency: 0.1–3 GHz", result)
        self.assertIn("Frequency points: 301", result)
        self.assertIn("Energy decay: -40 dB", result)
        self.assertIn("Maximum timesteps: 1000000", result)
        self.assertIn("\\n", result)

    def test_legacy_design_settings_are_formatted(self):
        text = """simulation:
  unit: 1.0
  fstart: 0.3
  fstop: 1.0
  n_pts: 301
  end_criteria: 1e-2
  max_timesteps: 50000
  target_s11_db: -15.0
"""
        result = format_simulation_settings(text)
        self.assertIn("Frequency: 0.3–1 GHz", result)
        self.assertIn("Frequency points: 301", result)
        self.assertIn("legacy linear threshold 0.01", result)
        self.assertIn("Maximum timesteps: 50000", result)

    def test_frequency_edit_is_not_blocked_by_unrelated_solver_field(self):
        text = """simulation:
  fstart: 0.25
  fstop: 2.5
  n_pts: not-yet-entered
  end_criteria: 1e-2
  max_timesteps: 50000
"""
        result = format_simulation_settings(text)
        self.assertIn("Frequency: 0.25–2.5 GHz", result)
        self.assertIn("Frequency points: not-yet-entered", result)

    def test_rejects_non_numeric_frequency_typo(self):
        text = "simulation:\\n  fstart: 0f.1\\n  fstop: 3.0\\n"
        with self.assertRaisesRegex(ValueError, "fstart"):
            format_simulation_settings(text)

    def test_rejects_non_increasing_frequency_range(self):
        text = "simulation:\\n  fstart: 3.0\\n  fstop: 0.1\\n"
        with self.assertRaisesRegex(ValueError, "increasing"):
            format_simulation_settings(text)

    def test_rejects_invalid_simulation_block(self):
        with self.assertRaisesRegex(ValueError, "simulation"):
            format_simulation_settings("simulation: 0.1\\n")

    def test_optimization_parser_accepts_integer_values_written_as_decimal_strings(self):
        settings = parse_simulation_settings(
            "simulation:\\n  fstart: 0.1\\n  fstop: 1.0\\n"
            "  n_pts: '301.0'\\n  max_timesteps: '50000.0'\\n"
            "  end_criteria: 1e-2\\n"
        )
        self.assertEqual(settings["n_pts"], 301)
        self.assertEqual(settings["max_timesteps"], 50000)


if __name__ == "__main__":
    unittest.main()
