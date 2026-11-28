import unittest

import numpy as np

from design import NF2FF_MIN_GRID_LINES, _add_nf2ff_mesh_support_lines


class FakeMesh:
    def __init__(self):
        self.lines = {}

    def AddLine(self, axis, values):
        self.lines[axis] = np.asarray(values, dtype=float)


class NF2FFMeshSupportTests(unittest.TestCase):
    def test_adds_minimum_mesh_lines_on_all_axes(self):
        mesh = FakeMesh()
        bounds = ((-1.0, 1.0), (-0.5, 0.5), (-0.02, 0.02))

        _add_nf2ff_mesh_support_lines(mesh, bounds)

        self.assertEqual(set(mesh.lines), {"x", "y", "z"})
        for axis, (lower, upper) in zip(("x", "y", "z"), bounds):
            lines = mesh.lines[axis]
            self.assertEqual(len(lines), NF2FF_MIN_GRID_LINES)
            self.assertAlmostEqual(lines[0], lower)
            self.assertAlmostEqual(lines[-1], upper)
            self.assertTrue(np.all(np.diff(lines) > 0))

    def test_rejects_too_few_support_lines(self):
        mesh = FakeMesh()
        with self.assertRaises(ValueError):
            _add_nf2ff_mesh_support_lines(
                mesh, ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                minimum_lines=19,
            )



if __name__ == "__main__":
    unittest.main()
