import unittest

from gui.optimization_compatibility import find_incompatible_settings


class ContinueCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.run = {
            "use_simulation_parameters": False,
            "use_multiband": True,
            "simulation_settings": {
                "fstart": 0.3, "fstop": 1.0, "n_pts": 301,
                "end_criteria": "1e-2", "max_timesteps": 200000,
            },
            "evaluation_settings": {
                "fstart": 0.3, "fstop": 1.0, "n_pts": 301,
                "end_criteria_db": -20.0, "max_timesteps": 200000,
            },
        }
        self.current_simulation = {
            "fstart": 0.3, "fstop": 1.0, "n_pts": 301,
            "end_criteria": 1e-2, "max_timesteps": 200000,
        }
        self.current_override = dict(self.run["evaluation_settings"])

    def check(self, **kwargs):
        args = {
            "current_simulation_settings": self.current_simulation,
            "current_evaluation_settings": self.current_override,
            "current_use_multiband": True,
        }
        args.update(kwargs)
        return find_incompatible_settings(self.run, **args)

    def test_matching_override_and_engine_are_compatible(self):
        self.assertEqual(self.check(), [])

    def test_frequency_range_changes_are_incompatible(self):
        changed = dict(self.current_override, fstart=0.4)
        self.assertIn("start frequency", self.check(current_evaluation_settings=changed))

    def test_frequency_resolution_changes_are_incompatible(self):
        changed = dict(self.current_override, n_pts=501)
        self.assertIn("number of frequency points", self.check(current_evaluation_settings=changed))

    def test_energy_decay_changes_are_incompatible(self):
        changed = dict(self.current_override, end_criteria_db=-30.0)
        self.assertIn("energy-decay criterion", self.check(current_evaluation_settings=changed))

    def test_maximum_timesteps_changes_are_incompatible(self):
        changed = dict(self.current_override, max_timesteps=100000)
        self.assertIn("maximum timesteps", self.check(current_evaluation_settings=changed))

    def test_engine_changes_are_incompatible(self):
        self.assertIn("simulation engine", self.check(current_use_multiband=False))

    def test_yaml_mode_uses_saved_and_current_yaml_settings(self):
        run = dict(self.run, use_simulation_parameters=True)
        self.assertEqual(
            find_incompatible_settings(
                run,
                current_simulation_settings=self.current_simulation,
                current_evaluation_settings=None,
                current_use_multiband=True,
            ),
            [],
        )

    def test_equivalent_linear_and_db_decay_values_match(self):
        run = dict(self.run, use_simulation_parameters=True)
        current = dict(self.current_simulation, end_criteria_db=-20.0)
        current.pop("end_criteria")
        self.assertEqual(
            find_incompatible_settings(
                run,
                current_simulation_settings=current,
                current_evaluation_settings=None,
                current_use_multiband=True,
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
