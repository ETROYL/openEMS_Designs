"""Sequential and process-parallel surrogate-assisted discrete optimization."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import multiprocessing
import os
from typing import Callable
from .candidate_selection import discrete_candidate_pool, select_next_candidates
from .evaluator import CandidateEvaluator, EvaluationResult
from .evaluation_store import EvaluationStore
from .initial_sampling import sobol_initial_design
from .parameter_space import DiscreteParameterSpace, ParameterSpaceError
from .surrogate import GaussianProcessSurrogate
from .parallel_execution import initialize_fdtd_worker, evaluate_fdtd_candidate


@dataclass
class OptimizationSummary:
    """Results and counters for one optimizer invocation."""
    evaluations: list[EvaluationResult]
    best: EvaluationResult | None
    total_stored: int
    successful_stored: int
    failed_stored: int
    stop_reason: str


class DiscreteSurrogateOptimizer:
    """Sobol + GP/EI search that resumes from SQLite history.

    ``run`` provides sequential evaluation; ``run_parallel`` dispatches
    batches to spawned FDTD worker processes. Both reuse SQLite history.
    Failed attempts remain excluded from selection but never train the GP.
    """

    def __init__(
        self,
        space: DiscreteParameterSpace,
        evaluator: CandidateEvaluator,
        store: EvaluationStore,
        *,
        initial_samples: int = 12,
        pool_size: int = 4096,
        min_surrogate_points: int = 4,
        xi: float = 0.01,
        seed: int | None = 42,
        length_scale: float = 0.25,
    ) -> None:
        if evaluator.parameter_space is not space:
            raise ValueError("evaluator and optimizer must use the same parameter-space instance")
        if initial_samples <= 0 or pool_size <= 0 or min_surrogate_points < 2:
            raise ValueError("initial_samples and pool_size must be positive; min_surrogate_points >= 2")
        if xi < 0:
            raise ValueError("xi must be non-negative")
        self.space = space
        self.evaluator = evaluator
        self.store = store
        self.initial_samples = int(initial_samples)
        self.pool_size = int(pool_size)
        self.min_surrogate_points = int(min_surrogate_points)
        self.xi = float(xi)
        self.seed = seed
        self.length_scale = float(length_scale)

    def _evaluate_if_new(self, point: tuple[int, ...], evaluation_id: int) -> EvaluationResult | None:
        fingerprint = self.evaluator.fingerprint(point)
        if self.store.contains(fingerprint):
            return None
        result = self.evaluator.evaluate(point, evaluation_id=evaluation_id)
        self.store.save_result(result)
        return result

    def run(
        self,
        budget: int,
        *,
        progress_callback: Callable[[EvaluationResult, int, int], None] | None = None,
        stop_requested: Callable[[], bool] | None = None,
    ) -> OptimizationSummary:
        """Perform at most budget new evaluations, reporting each completed attempt.

        Cancellation is cooperative: an in-progress FDTD evaluation is allowed to
        finish, then no further candidate is launched.
        """
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            raise ParameterSpaceError("budget must be a non-negative integer")
        new_results: list[EvaluationResult] = []

        def cancelled() -> bool:
            return bool(stop_requested and stop_requested())

        def record(result: EvaluationResult | None) -> None:
            if result is not None:
                new_results.append(result)
                if progress_callback is not None:
                    progress_callback(result, len(new_results), budget)

        if budget == 0:
            return self._summary(new_results, "zero budget")

        # A fresh database starts with a space-filling design. On restart,
        # existing records are reused and the optimizer immediately resumes.
        if self.store.count() == 0:
            n_initial = min(self.initial_samples, budget, self.space.cardinality)
            initial = sobol_initial_design(self.space, n_initial, seed=self.seed)
            for point in initial:
                if cancelled():
                    return self._summary(new_results, "stop requested")
                result = self._evaluate_if_new(point, len(new_results) + 1)
                record(result)
            if len(new_results) >= budget:
                return self._summary(new_results, "evaluation budget reached")

        stop_reason = "evaluation budget reached"
        while len(new_results) < budget:
            if cancelled():
                stop_reason = "stop requested"
                break
            all_results = self.store.list_results()
            observed = {result.indices for result in all_results}
            available = self.space.cardinality - len(observed)
            if available <= 0:
                stop_reason = "discrete search space exhausted"
                break

            successful = [result for result in all_results if result.status == "ok"]
            remaining = budget - len(new_results)
            n_pool = min(self.pool_size, available)

            if len(successful) >= self.min_surrogate_points:
                model = GaussianProcessSurrogate(length_scale=self.length_scale)
                model.fit(
                    [result.indices for result in successful],
                    [result.objective for result in successful],
                    levels=self.space.levels(),
                )
                ranked = select_next_candidates(
                    self.space,
                    model,
                    best_objective=min(result.objective for result in successful),
                    observed=observed,
                    n_candidates=1,
                    pool_size=n_pool,
                    xi=self.xi,
                    seed=None if self.seed is None else self.seed + self.store.count(),
                )
                candidates = [ranked[0][0]] if ranked else []
            else:
                candidates = discrete_candidate_pool(
                    self.space,
                    n_candidates=min(remaining, n_pool),
                    excluded=observed,
                    seed=None if self.seed is None else self.seed + self.store.count(),
                )

            if not candidates:
                stop_reason = "no unexplored candidates available"
                break
            for point in candidates:
                if cancelled():
                    stop_reason = "stop requested"
                    break
                result = self._evaluate_if_new(point, len(new_results) + 1)
                record(result)
                if len(new_results) >= budget:
                    break

        return self._summary(new_results, stop_reason)

    def run_parallel(
        self,
        budget: int,
        *,
        workers: int,
        threads_per_worker: int,
        objective_name: str,
        objective_weights: dict[str, float] | None = None,
        progress_callback: Callable[[EvaluationResult, int, int], None] | None = None,
        stop_requested: Callable[[], bool] | None = None,
    ) -> OptimizationSummary:
        """Evaluate candidate batches in spawned worker processes.

        The surrogate is refitted after each batch. Results are collected and
        persisted by the parent process, keeping SQLite single-writer.
        """
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            raise ParameterSpaceError("budget must be a non-negative integer")
        if workers < 1 or threads_per_worker < 1:
            raise ValueError("workers and threads_per_worker must be positive")
        if budget == 0:
            return self._summary([], "zero budget")

        new_results: list[EvaluationResult] = []

        def cancelled() -> bool:
            return bool(stop_requested and stop_requested())

        def evaluate_batch(executor, points: list[tuple[int, ...]]) -> bool:
            tasks = [
                (point, self.store.count() + offset + 1)
                for offset, point in enumerate(points)
            ]
            futures = {
                executor.submit(evaluate_fdtd_candidate, task): task
                for task in tasks
            }
            for future in as_completed(futures):
                result = future.result()
                self.store.save_result(result)
                new_results.append(result)
                if progress_callback is not None:
                    progress_callback(result, len(new_results), budget)
            return not cancelled()

        # Spawned children inherit these settings before importing NumPy or
        # the native openEMS bindings. The worker initializer also applies
        # EM_THREADS explicitly before importing optimization.py.
        os.environ["EM_THREADS"] = str(int(threads_per_worker))
        os.environ["OMP_NUM_THREADS"] = str(int(threads_per_worker))
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

        stop_reason = "evaluation budget reached"
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=initialize_fdtd_worker,
            initargs=(
                self.evaluator.config_file,
                self.evaluator.output_dir,
                self.space,
                objective_name,
                dict(objective_weights or {}),
                int(threads_per_worker),
            ),
        ) as executor:
            if self.store.count() == 0:
                initial_count = min(self.initial_samples, budget, self.space.cardinality)
                initial = sobol_initial_design(self.space, initial_count, seed=self.seed)
                for start in range(0, len(initial), workers):
                    if cancelled():
                        stop_reason = "stop requested"
                        break
                    batch = initial[start:start + workers]
                    if not evaluate_batch(executor, batch):
                        stop_reason = "stop requested"
                        break
                    if len(new_results) >= budget:
                        break

            while stop_reason == "evaluation budget reached" and len(new_results) < budget:
                if cancelled():
                    stop_reason = "stop requested"
                    break

                all_results = self.store.list_results()
                observed = {result.indices for result in all_results}
                available = self.space.cardinality - len(observed)
                if available <= 0:
                    stop_reason = "discrete search space exhausted"
                    break

                batch_count = min(workers, budget - len(new_results), available)
                successful = [result for result in all_results if result.status == "ok"]
                n_pool = min(self.pool_size, available)

                if len(successful) >= self.min_surrogate_points:
                    model = GaussianProcessSurrogate(length_scale=self.length_scale)
                    model.fit(
                        [result.indices for result in successful],
                        [result.objective for result in successful],
                        levels=self.space.levels(),
                    )
                    ranked = select_next_candidates(
                        self.space,
                        model,
                        best_objective=min(result.objective for result in successful),
                        observed=observed,
                        n_candidates=batch_count,
                        pool_size=max(batch_count, n_pool),
                        xi=self.xi,
                        seed=None if self.seed is None else self.seed + self.store.count(),
                    )
                    candidates = [point for point, _score in ranked]
                else:
                    candidates = discrete_candidate_pool(
                        self.space,
                        n_candidates=batch_count,
                        excluded=observed,
                        seed=None if self.seed is None else self.seed + self.store.count(),
                    )

                if not candidates:
                    stop_reason = "no unexplored candidates available"
                    break
                if not evaluate_batch(executor, candidates):
                    stop_reason = "stop requested"
                    break

        return self._summary(new_results, stop_reason)

    def _summary(self, results: list[EvaluationResult], reason: str) -> OptimizationSummary:
        return OptimizationSummary(
            evaluations=results,
            best=self.store.best(),
            total_stored=self.store.count(),
            successful_stored=self.store.count("ok"),
            failed_stored=self.store.count("failed"),
            stop_reason=reason,
        )
