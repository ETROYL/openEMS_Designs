"""Optuna TPE adapter for the shared discrete FDTD evaluation pipeline.

The adapter operates on integer lattice indices, so every proposal maps to an
exact legal YAML design. SQLite remains the source of truth for evaluations;
Optuna's in-memory study is reconstructed from that history on every launch.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
from typing import Callable

from .engine import OptimizationSummary
from .evaluator import CandidateEvaluator, EvaluationResult
from .evaluation_store import EvaluationStore
from .parallel_execution import initialize_fdtd_worker, evaluate_fdtd_candidate
from .parameter_space import DiscreteParameterSpace, ParameterSpaceError


class OptunaTPEOptimizer:
    """TPE search using the same evaluator, SQLite store and FDTD workers."""

    def __init__(
        self,
        space: DiscreteParameterSpace,
        evaluator: CandidateEvaluator,
        store: EvaluationStore,
        *,
        seed: int | None = 42,
        startup_trials: int = 10,
    ) -> None:
        if evaluator.parameter_space is not space:
            raise ValueError("evaluator and optimizer must use the same parameter-space instance")
        if startup_trials < 1:
            raise ValueError("startup_trials must be positive")
        self.space = space
        self.evaluator = evaluator
        self.store = store
        self.seed = seed
        self.startup_trials = int(startup_trials)

    def _restore_study(self):
        try:
            import optuna
            from optuna.distributions import IntDistribution
            from optuna.trial import create_trial
        except ImportError as exc:
            raise RuntimeError(
                "Optuna is not installed. Install the GUI requirements with: "
                "pip install -r requirements-gui.txt"
            ) from exc

        sampler = optuna.samplers.TPESampler(
            seed=self.seed,
            n_startup_trials=self.startup_trials,
        )
        study = optuna.create_study(direction="minimize", sampler=sampler)
        distributions = {
            parameter.name: IntDistribution(0, parameter.levels - 1)
            for parameter in self.space.parameters
        }
        # Rebuild Optuna's observation history from the canonical SQLite store.
        for result in self.store.list_results():
            params = {
                name: int(index)
                for name, index in zip(self.space.names, result.indices)
            }
            study.add_trial(create_trial(
                params=params,
                distributions=distributions,
                value=float(result.objective),
            ))
        return optuna, study, distributions

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
        """Run at most budget new FDTD evaluations, in parallel batches."""
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            raise ParameterSpaceError("budget must be a non-negative integer")
        if workers < 1 or threads_per_worker < 1:
            raise ValueError("workers and threads_per_worker must be positive")
        if budget == 0:
            return self._summary([], "zero budget")

        optuna, study, distributions = self._restore_study()
        from optuna.trial import TrialState

        new_results: list[EvaluationResult] = []
        stop_reason = "evaluation budget reached"

        def cancelled() -> bool:
            return bool(stop_requested and stop_requested())

        # Match the existing process-pool thread isolation used by ETROYL Discrete.
        os.environ["EM_THREADS"] = str(int(threads_per_worker))
        os.environ["OMP_NUM_THREADS"] = str(int(threads_per_worker))
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

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
            while len(new_results) < budget:
                if cancelled():
                    stop_reason = "stop requested"
                    break

                remaining = min(workers, budget - len(new_results))
                observed = {result.indices for result in self.store.list_results()}
                pending: set[tuple[int, ...]] = set()
                trials: list[tuple[object, tuple[int, ...]]] = []
                attempts = 0

                # Ask a batch before evaluating it. Duplicate proposals are
                # rejected so an FDTD geometry is never evaluated twice.
                while len(trials) < remaining and attempts < remaining * 25:
                    attempts += 1
                    trial = study.ask()
                    indices = tuple(
                        int(trial.suggest_int(name, 0, levels - 1))
                        for name, levels in zip(self.space.names, self.space.levels())
                    )
                    fingerprint = self.evaluator.fingerprint(indices)
                    if indices in observed or indices in pending or self.store.contains(fingerprint):
                        study.tell(trial, state=TrialState.FAIL)
                        continue
                    pending.add(indices)
                    trials.append((trial, indices))

                if not trials:
                    stop_reason = (
                        "discrete search space exhausted"
                        if len(observed) >= self.space.cardinality
                        else "no unexplored candidates available"
                    )
                    break

                tasks = [
                    ((indices), self.store.count() + offset + 1)
                    for offset, (_trial, indices) in enumerate(trials)
                ]
                futures = {
                    executor.submit(evaluate_fdtd_candidate, task): (trial, indices)
                    for task, (trial, indices) in zip(tasks, trials)
                }
                for future in as_completed(futures):
                    trial, _indices = futures[future]
                    result = future.result()
                    self.store.save_result(result)
                    # Failed FDTD evaluations receive the evaluator's penalty,
                    # matching the existing minimization convention.
                    study.tell(trial, float(result.objective))
                    new_results.append(result)
                    if progress_callback is not None:
                        progress_callback(result, len(new_results), budget)

                if cancelled():
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
