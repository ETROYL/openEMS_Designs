import unittest

import numpy as np

from multiband_fdtd.mesh_diagnostic import collect_mesh_report


class FakeGrid:
    def __init__(self):
        self.lines = {
            "x": np.array([-1.0, -0.5, 0.0, 1.0]),
            "y": np.array([-2.0, 0.0, 2.0]),
            "z": np.array([0.0, 0.25, 0.5]),
        }

    def GetLines(self, axis):
        return self.lines[axis]


class FakeCSX:
    def GetGrid(self):
        return FakeGrid()


class MeshDiagnosticReportTests(unittest.TestCase):
    def test_reports_axis_mesh_statistics_and_total_cells(self):
        report = collect_mesh_report(FakeCSX())

        self.assertEqual(report["axes"]["x"]["line_count"], 4)
        self.assertEqual(report["axes"]["x"]["cell_count"], 3)
        self.assertAlmostEqual(report["axes"]["x"]["minimum_spacing"], 0.5)
        self.assertAlmostEqual(report["axes"]["y"]["minimum_spacing"], 2.0)
        self.assertAlmostEqual(report["axes"]["z"]["minimum_spacing"], 0.25)
        self.assertEqual(report["total_cells"], 3 * 2 * 2)

    def test_rejects_axis_with_fewer_than_two_lines(self):
        class InvalidGrid(FakeGrid):
            def GetLines(self, axis):
                if axis == "z":
                    return np.array([0.0])
                return super().GetLines(axis)

        class InvalidCSX:
            def GetGrid(self):
                return InvalidGrid()

        with self.assertRaisesRegex(RuntimeError, "z-axis"):
            collect_mesh_report(InvalidCSX())


if __name__ == "__main__":
    unittest.main()
