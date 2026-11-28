"""CPU allocation for concurrent frequency-interval simulations."""
from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class BandAllocation:
    workers: int
    threads_per_worker: int
    logical_cpus: int

    @property
    def total_threads(self) -> int:
        return self.workers * self.threads_per_worker


def available_logical_cpus() -> int:
    """Return logical CPUs available to this process, respecting affinity."""
    try:
        count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        count = os.cpu_count() or 1
    return max(1, int(count))


def allocate_band_resources(
    interval_count: int,
    logical_cpus: int | None = None,
    *,
    max_workers: int = 7,
    min_threads: int = 2,
    max_threads: int = 4,
) -> BandAllocation:
    """Allocate interval workers using the interval-first execution policy.

    One to four intervals receive up to four threads each. Five or more
    intervals use two threads each and up to seven concurrent workers.
    Additional intervals are queued by the caller in later batches.
    """
    if (isinstance(interval_count, bool) or not isinstance(interval_count, int)
            or interval_count < 1):
        raise ValueError("interval_count must be a positive integer.")
    if max_workers < 1 or min_threads < 1 or max_threads < min_threads:
        raise ValueError("Invalid worker or thread limits.")
    capacity = available_logical_cpus() if logical_cpus is None else int(logical_cpus)
    if capacity < min_threads:
        raise ValueError(
            f"At least {min_threads} logical CPUs are required for an FDTD interval; "
            f"only {capacity} are available."
        )

    if interval_count <= 4:
        workers = min(interval_count, capacity // min_threads)
        threads = min(max_threads, capacity // workers)
    else:
        workers = min(interval_count, max_workers, capacity // min_threads)
        threads = min_threads

    if workers < 1 or threads < min_threads:
        raise ValueError("The available CPU capacity cannot satisfy the FDTD thread minimum.")
    return BandAllocation(workers, threads, capacity)


def candidate_workers_for_allocation(allocation: BandAllocation) -> int:
    """Return safe concurrent candidate count for identical interval pools."""
    return max(1, allocation.logical_cpus // allocation.total_threads)
