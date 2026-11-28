import unittest
from unittest.mock import patch

from multiband_fdtd.resources import (
    allocate_band_resources, candidate_workers_for_allocation,
)


class BandResourceAllocationTests(unittest.TestCase):
    def test_sixteen_logical_cpu_examples(self):
        expected = {
            1: (1, 4),
            2: (2, 4),
            3: (3, 4),
            4: (4, 4),
            5: (5, 2),
            6: (6, 2),
            7: (7, 2),
            8: (7, 2),
            12: (7, 2),
        }
        for interval_count, allocation in expected.items():
            with self.subTest(interval_count=interval_count):
                result = allocate_band_resources(interval_count, logical_cpus=16)
                self.assertEqual(
                    (result.workers, result.threads_per_worker), allocation
                )
                self.assertLessEqual(result.total_threads, 16)

    def test_adapts_to_smaller_cpu_affinity(self):
        result = allocate_band_resources(6, logical_cpus=8)
        self.assertEqual((result.workers, result.threads_per_worker), (4, 2))
        self.assertLessEqual(result.total_threads, 8)

    def test_one_interval_uses_available_capacity_up_to_four_threads(self):
        self.assertEqual(
            allocate_band_resources(1, logical_cpus=2).threads_per_worker, 2
        )
        self.assertEqual(
            allocate_band_resources(1, logical_cpus=4).threads_per_worker, 4
        )

    def test_requires_two_logical_cpus(self):
        with self.assertRaisesRegex(ValueError, "At least 2 logical CPUs"):
            allocate_band_resources(1, logical_cpus=1)

    def test_candidate_concurrency_shares_the_interval_cpu_budget(self):
        cases = {
            1: 4,   # 1 interval × 4 threads -> 4 candidates
            2: 2,   # 2 intervals × 4 threads -> 2 candidates
            3: 1,   # 3 intervals × 4 threads -> 1 candidate
            4: 1,   # 4 intervals × 4 threads -> 1 candidate
            5: 1,   # 5 intervals × 2 threads -> 1 candidate
            7: 1,
            9: 1,
        }
        for interval_count, expected_workers in cases.items():
            with self.subTest(interval_count=interval_count):
                allocation = allocate_band_resources(
                    interval_count, logical_cpus=16
                )
                self.assertEqual(
                    candidate_workers_for_allocation(allocation),
                    expected_workers,
                )

    def test_detects_logical_cpu_affinity(self):
        with patch(
            "multiband_fdtd.resources.available_logical_cpus", return_value=16
        ):
            result = allocate_band_resources(5)
        self.assertEqual((result.workers, result.threads_per_worker), (5, 2))


if __name__ == "__main__":
    unittest.main()
