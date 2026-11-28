import unittest

import numpy as np

from design import NF2FF_MIN_GRID_LINES, _add_nf2ff_mesh_support_lines


class FakeMesh:
    def __init__(self):
        self.lines = {}

    def AddLine(self, axis, values):
        existing = self.lines.get(axis, np.array([], dtype=float))
        self.lines[axis] = np.unique(np.concatenate(
            (existing, np.asarray(values, dtype=float).reshape(-1))
        ))

    def GetLines(self, axis):
        return self.lines.get(axis, np.array([], dtype=float))


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


    def test_subdivision_avoids_near_coincident_geometry_lines(self):
        mesh = FakeMesh()
        geometry_y = np.array([
            -0.200000, -0.170000, -0.100000, -0.011000,
            -0.002665, -0.001000, 0.001000, 0.002665,
            0.011000, 0.100000, 0.170000, 0.200000,
        ])
        mesh.AddLine("y", geometry_y)

        _add_nf2ff_mesh_support_lines(
            mesh,
            ((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2)),
        )

        y_lines = mesh.GetLines("y")
        self.assertGreaterEqual(y_lines.size, NF2FF_MIN_GRID_LINES)
        self.assertTrue(np.all(np.diff(y_lines) > 0))
        self.assertGreater(float(np.min(np.diff(y_lines))), 1e-4)
        for fixed_line in geometry_y:
            self.assertTrue(np.any(y_lines == fixed_line))

    def test_does_not_add_lines_when_existing_mesh_is_sufficient(self):
        mesh = FakeMesh()
        original = np.linspace(-1.0, 1.0, NF2FF_MIN_GRID_LINES)
        mesh.AddLine("x", original)
        mesh.AddLine("y", original)
        mesh.AddLine("z", original)

        _add_nf2ff_mesh_support_lines(
            mesh, ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
        )

        for axis in ("x", "y", "z"):
            np.testing.assert_array_equal(mesh.GetLines(axis), original)

    def test_rejects_too_few_support_lines(self):
        mesh = FakeMesh()
        with self.assertRaises(ValueError):
            _add_nf2ff_mesh_support_lines(
                mesh, ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
                minimum_lines=19,
            )



if __name__ == "__main__":
    unittest.main()
