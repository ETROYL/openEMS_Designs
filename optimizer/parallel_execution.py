"""Process-local FDTD evaluator used by the parallel discrete optimizer.

Each spawned worker imports openEMS only after its CPU-thread environment has
been configured. The parent process remains the sole SQLite writer.
"""
from __future__ import annotations

import os
from typing import Any

_WORKER_EVALUATOR = None


def initialize_fdtd_worker(
    config_file: str,
    output_dir: str,
    parameter_space: Any,
    objective_name: str,
    objective_weights: dict[str, float],
    threads_per_worker: int,
    evaluation_settings: dict[str, Any] | None = None,
) -> None:
    """Configure native thread pools, then construct one evaluator per process."""
    global _WORKER_EVALUATOR

    threads = str(int(threads_per_worker))
    os.environ["EM_THREADS"] = threads
    os.environ["OMP_NUM_THREADS"] = threads
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

    # Import the openEMS pipeline only after EM_THREADS is set. optimization.py
    # imports parallel.NUM_THREADS at module load time.
    from .evaluator import CandidateEvaluator

    _WORKER_EVALUATOR = CandidateEvaluator.from_existing_pipeline(
        config_file,
        output_dir,
        parameter_space,
        objective_name=objective_name,
        objective_weights=objective_weights,
        evaluation_settings=evaluation_settings,
    )


def evaluate_fdtd_candidate(task: tuple[tuple[int, ...], int]):
    """Evaluate one candidate in the current worker process."""
    if _WORKER_EVALUATOR is None:
        raise RuntimeError("FDTD worker was not initialized")
    indices, evaluation_id = task
    return _WORKER_EVALUATOR.evaluate(indices, evaluation_id=evaluation_id)
