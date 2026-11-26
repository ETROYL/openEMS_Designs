"""Tests for shared simulation settings and runtime optimization overrides."""
import math
import unittest

from design import get_simulation_settings


class SimulationSettingsTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "simulation": {
                "fstart": 0.1,
                "fstop": 1.0,
                "n_pts": 301,
                "end_criteria_db": -40.0,
                "max_timesteps": 500000,
            },
            # Legacy YAML overrides must not affect the new GUI-only workflow.
            "optimization": {
                "evaluation": {
                    "fstart": 0.2,
                    "fstop": 0.8,
                    "n_pts": 51,
                    "end_criteria_db": -15.0,
                }
            },
        }

    def test_default_profile_uses_yaml_simulation_settings_only(self):
        settings = get_simulation_settings(self.config, "simulation")
        self.assertEqual(settings["fstart"], 0.1)
        self.assertEqual(settings["fstop"], 1.0)
        self.assertEqual(settings["n_pts"], 301)
        self.assertAlmostEqual(settings["end_criteria"], 1e-4)

    def test_runtime_overrides_are_applied_without_mutating_yaml(self):
        overrides = {
            "fstart": 0.15,
            "fstop": 0.9,
            "n_pts": 61,
            "end_criteria_db": -15.0,
            "max_timesteps": 40000,
        }
        settings = get_simulation_settings(
            self.config, "simulation", overrides=overrides
        )
        self.assertEqual(settings["fstart"], 0.15)
        self.assertEqual(settings["fstop"], 0.9)
        self.assertEqual(settings["n_pts"], 61)
        self.assertEqual(settings["max_timesteps"], 40000)
        self.assertTrue(math.isclose(settings["end_criteria"], 10 ** -1.5))
        self.assertEqual(self.config["simulation"]["fstart"], 0.1)
        self.assertEqual(self.config["simulation"]["end_criteria_db"], -40.0)

    def test_invalid_nonnegative_decay_is_rejected(self):
        with self.assertRaises(ValueError):
            get_simulation_settings(
                self.config, "simulation", overrides={"end_criteria_db": 0.0}
            )


if __name__ == "__main__":
    unittest.main()
