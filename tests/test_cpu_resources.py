"""Tests for optimization CPU allocation validation."""
import unittest
from unittest.mock import patch

from gui.cpu_resources import detect_cpu_resources, validate_allocation


class CpuResourceTests(unittest.TestCase):
    def test_detect_cpu_resources_returns_positive_capacity(self):
        resources = detect_cpu_resources()
        self.assertGreaterEqual(resources["physical_cores"], 1)
        self.assertGreaterEqual(resources["logical_threads"], resources["physical_cores"])

    @patch("gui.cpu_resources.detect_cpu_resources", return_value={
        "physical_cores": 8, "logical_threads": 16
    })
    def test_valid_manual_allocation(self, _mock_resources):
        validate_allocation(4, 3)

    @patch("gui.cpu_resources.detect_cpu_resources", return_value={
        "physical_cores": 8, "logical_threads": 16
    })
    def test_rejects_too_many_workers(self, _mock_resources):
        with self.assertRaisesRegex(ValueError, "physical CPU cores"):
            validate_allocation(9, 1)

    @patch("gui.cpu_resources.detect_cpu_resources", return_value={
        "physical_cores": 8, "logical_threads": 16
    })
    def test_rejects_too_many_threads_per_worker(self, _mock_resources):
        with self.assertRaisesRegex(ValueError, "logical CPUs"):
            validate_allocation(1, 17)

    @patch("gui.cpu_resources.detect_cpu_resources", return_value={
        "physical_cores": 8, "logical_threads": 16
    })
    def test_accepts_allocation_with_one_cpu_reserved(self, _mock_resources):
        validate_allocation(7, 2)

    @patch("gui.cpu_resources.detect_cpu_resources", return_value={
        "physical_cores": 8, "logical_threads": 16
    })
    def test_rejects_allocation_consuming_reserved_cpu(self, _mock_resources):
        with self.assertRaisesRegex(ValueError, "usable logical CPU budget"):
            validate_allocation(8, 2)

    @patch("gui.cpu_resources.detect_cpu_resources", return_value={
        "physical_cores": 8, "logical_threads": 16
    })
    def test_rejects_total_thread_overcommit(self, _mock_resources):
        with self.assertRaisesRegex(ValueError, "usable logical CPU budget"):
            validate_allocation(5, 4)


if __name__ == "__main__":
    unittest.main()
