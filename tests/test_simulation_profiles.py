"""Tests for independent standalone-simulation and optimization settings."""
import unittest

from design import get_simulation_settings


class SimulationProfileTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "simulation": {
                "unit": 1.0,
                "fstart": 0.1,
                "fstop": 1.0,
                "n_pts": 301,
                "end_criteria": 1e-4,
                "max_timesteps": 500000,
            },
            "optimization": {
                "evaluation": {
                    "fstart": 0.2,
                    "fstop": 0.8,
                    "n_pts": 51,
                    "end_criteria": 1e-2,
                    "max_timesteps": 40000,
                }
            },
        }

    def test_simulation_profile_uses_high_fidelity_settings(self):
        settings = get_simulation_settings(self.config, "simulation")
        self.assertEqual(settings["fstart"], 0.1)
        self.assertEqual(settings["fstop"], 1.0)
        self.assertEqual(settings["n_pts"], 301)
        self.assertEqual(settings["end_criteria"], 1e-4)
        self.assertEqual(settings["max_timesteps"], 500000)

    def test_optimization_profile_overrides_only_evaluation_settings(self):
        settings = get_simulation_settings(self.config, "optimization")
        self.assertEqual(settings["fstart"], 0.2)
        self.assertEqual(settings["fstop"], 0.8)
        self.assertEqual(settings["n_pts"], 51)
        self.assertEqual(settings["end_criteria"], 1e-2)
        self.assertEqual(settings["max_timesteps"], 40000)
        self.assertEqual(settings["unit"], 1.0)

    def test_missing_optimization_overrides_fall_back_to_simulation(self):
        config = {"simulation": self.config["simulation"], "optimization": {}}
        settings = get_simulation_settings(config, "optimization")
        self.assertEqual(settings, self.config["simulation"])

    def test_profiles_return_independent_copies(self):
        settings = get_simulation_settings(self.config, "optimization")
        settings["fstart"] = 0.5
        self.assertEqual(self.config["simulation"]["fstart"], 0.1)
        self.assertEqual(self.config["optimization"]["evaluation"]["fstart"], 0.2)

    def test_unknown_profile_is_rejected(self):
        with self.assertRaises(ValueError):
            get_simulation_settings(self.config, "final")


if __name__ == "__main__":
    unittest.main()
