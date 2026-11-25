import unittest
from optimizer.parameter_space import DiscreteParameterSpace, ParameterSpaceError
from optimizer.initial_sampling import sobol_initial_design

class DiscreteParameterSpaceTests(unittest.TestCase):
    def test_disk_example_grid_cardinality_and_round_trip(self):
        space = DiscreteParameterSpace({
            "disk_radius": {"type": "float", "start": 0.15, "stop": 0.55, "step": 0.01, "default": 0.20},
            "feed_gap": {"type": "float", "start": 0.001, "stop": 0.005, "step": 0.001, "default": 0.001},
            "port_width": {"type": "float", "start": 0.001, "stop": 0.020, "step": 0.001, "default": 0.004},
        })
        self.assertEqual(space.levels(), (41, 5, 20))
        self.assertEqual(space.cardinality, 4100)
        values = space.to_values((5, 2, 3))
        self.assertEqual(values, {"disk_radius": 0.20, "feed_gap": 0.003, "port_width": 0.004})
        self.assertEqual(space.to_indices(values), (5, 2, 3))

    def test_bowtie_example_off_grid_stop_and_out_of_range_preview_default(self):
        space = DiscreteParameterSpace({
            "arm_length": {"type": "float", "start": 0.20, "stop": 0.35, "step": 0.02, "default": 0.18},
            "arm_width": {"type": "float", "start": 0.10, "stop": 0.35, "step": 0.02, "default": 0.18},
            "flare_width": {"type": "float", "start": 0.01, "stop": 0.20, "step": 0.01, "default": 0.10},
            "flare_length": {"type": "float", "start": 0.005, "stop": 0.08, "step": 0.005, "default": 0.02},
            "feed_gap": {"type": "float", "start": 0.002, "stop": 0.010, "step": 0.002, "default": 0.002},
            "feed_width": {"type": "float", "start": 0.006, "stop": 0.03, "step": 0.002, "default": 0.015},
        })
        self.assertEqual(space.levels(), (8, 13, 20, 16, 5, 13))
        self.assertEqual(space.parameters[0].value_at(7), 0.34)
        self.assertEqual(space.to_values((0, 0, 0, 0, 0, 0))["arm_length"], 0.20)

    def test_fixed_and_non_optimized_variables_are_preserved(self):
        space = DiscreteParameterSpace({
            "fixed": {"type": "float", "start": 0.002, "stop": 0.002, "default": 0.002},
            "disabled": {"type": "float", "start": 1, "stop": 3, "step": 1, "default": 2, "optimize": False},
            "literal": 7,
            "active": {"type": "int", "start": 1, "stop": 5, "step": 2, "default": 1},
        })
        self.assertEqual(space.names, ("active",))
        self.assertEqual(space.to_values((1,)), {"fixed": 0.002, "disabled": 2, "literal": 7, "active": 3})

    def test_non_aligned_stop_is_not_included(self):
        space = DiscreteParameterSpace({"x": {"type": "float", "start": 0.0, "stop": 1.0, "step": 0.3}})
        self.assertEqual(space.parameters[0].value_at(3), 0.9)
        with self.assertRaises(ParameterSpaceError):
            space.parameters[0].index_of(1.0)

    def test_invalid_float_without_step_is_rejected(self):
        with self.assertRaises(ParameterSpaceError):
            DiscreteParameterSpace({"x": {"type": "float", "start": 0.0, "stop": 1.0}})

    def test_sobol_design_is_unique_and_reproducible(self):
        space = DiscreteParameterSpace({
            "x": {"type": "int", "start": 0, "stop": 9},
            "y": {"type": "int", "start": 0, "stop": 4},
        })
        first = sobol_initial_design(space, 20, seed=7)
        self.assertEqual(first, sobol_initial_design(space, 20, seed=7))
        self.assertEqual(len(first), len(set(first)))
        self.assertEqual(len(first), 20)

if __name__ == "__main__":
    unittest.main()
