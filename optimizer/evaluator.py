"""Adapter between discrete candidates and the repository's FDTD/objective pipeline.

The evaluator deliberately accepts injected simulation/objective callables so
its orchestration can be unit-tested without importing openEMS or launching FDTD.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
import re
import time
from typing import Any, Callable, Mapping, Sequence

from .parameter_space import DiscreteParameterSpace


SimulationRunner = Callable[[str, Mapping[str, Any], str], Mapping[str, Any]]
ObjectiveFunction = Callable[..., float]


@dataclass
class EvaluationResult:
    """Serializable result for one attempted geometry evaluation."""

    fingerprint: str
    indices: tuple[int, ...]
    parameters: dict[str, Any]
    status: str
    objective: float
    evaluation_dir: str
    elapsed_seconds: float
    metrics: dict[str, float]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["indices"] = list(self.indices)
        return result


def _stable_value(value: Any) -> Any:
    """Normalize values for deterministic JSON fingerprints."""
    if isinstance(value, float):
        return format(value, ".15g")
    if isinstance(value, Mapping):
        return {str(k): _stable_value(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_stable_value(v) for v in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return repr(value)


def _result_metrics(results: Mapping[str, Any]) -> dict[str, float]:
    """Extract compact diagnostic metrics without requiring a particular objective."""
    import numpy as np

    metrics: dict[str, float] = {}
    for source_key, metric_name, reducer in (
        ("s11_db", "min_s11_db", np.min),
        ("s11_db", "max_s11_db", np.max),
        ("s11_db", "mean_s11_db", np.mean),
        ("z_in", "mean_abs_zin_ohm", lambda x: np.mean(np.abs(x))),
    ):
        if source_key not in results:
            continue
        values = np.asarray(results[source_key])
        if values.size == 0:
            continue
        scalar = float(reducer(values))
        if np.isfinite(scalar):
            metrics[metric_name] = scalar

    if "freq" in results:
        freq = np.asarray(results["freq"])
        if freq.size:
            metrics["frequency_start_hz"] = float(freq[0])
            metrics["frequency_stop_hz"] = float(freq[-1])
            metrics["frequency_points"] = float(freq.size)
    return metrics


class CandidateEvaluator:
    """Evaluate legal discrete candidates through an existing simulation runner.

    A failed simulation is recorded as a failed result with a configurable
    penalty; it is never confused with a successful objective value.
    """

    def __init__(
        self,
        config_file: str,
        output_dir: str,
        parameter_space: DiscreteParameterSpace,
        simulation_runner: SimulationRunner,
        objective_function: ObjectiveFunction,
        failed_penalty: float = 1e6,
    ) -> None:
        self.config_file = os.path.abspath(config_file)
        self.output_dir = os.path.abspath(output_dir)
        self.parameter_space = parameter_space
        self.simulation_runner = simulation_runner
        self.objective_function = objective_function
        self.failed_penalty = float(failed_penalty)
        if not os.path.isfile(self.config_file):
            raise FileNotFoundError(self.config_file)
        if not callable(simulation_runner):
            raise TypeError("simulation_runner must be callable")
        if not callable(objective_function):
            raise TypeError("objective_function must be callable")

    @classmethod
    def from_existing_pipeline(
        cls,
        config_file: str,
        output_dir: str,
        parameter_space: DiscreteParameterSpace,
        failed_penalty: float = 1e6,
        objective_name: str = "multi_objective_combined",
        objective_weights: Mapping[str, float] | None = None,
    ) -> "CandidateEvaluator":
        """Bind to the current simulation pipeline and a selectable objective.

        Imports are lazy so unit tests and parameter-space tooling do not need
        the openEMS Python bindings installed.
        """
        from optimization import run_simulation_instance
        from objectives import (
            maximize_bandwidth_in_band,
            maximize_impedance_smoothness,
            minimize_max_s11_in_band,
            minimize_s11_in_band,
            multi_objective_combined,
        )

        objective_functions = {
            "maximize_bandwidth_in_band": maximize_bandwidth_in_band,
            "minimize_max_s11_in_band": minimize_max_s11_in_band,
            "minimize_s11_in_band": minimize_s11_in_band,
            "maximize_impedance_smoothness": maximize_impedance_smoothness,
        }
        if objective_name == "multi_objective_combined":
            weights = dict(objective_weights or {})
            objective_function = lambda results, f_min=None, f_max=None: multi_objective_combined(
                results,
                f_min=f_min,
                f_max=f_max,
                weight_s11=float(weights.get("s11", 1.0)),
                weight_smooth=float(weights.get("smoothness", 0.0053)),
            )
        elif objective_name in objective_functions:
            objective_function = objective_functions[objective_name]
        else:
            raise ValueError(f"Unknown objective function: {objective_name}")

        return cls(
            config_file=config_file,
            output_dir=output_dir,
            parameter_space=parameter_space,
            simulation_runner=run_simulation_instance,
            objective_function=objective_function,
            failed_penalty=failed_penalty,
        )

    def fingerprint(self, indices: Sequence[int]) -> str:
        """Return the stable config/geometry identity without running a simulation."""
        index_tuple = tuple(int(index) for index in indices)
        parameters = self.parameter_space.to_values(index_tuple)
        with open(self.config_file, "rb") as stream:
            config_digest = hashlib.sha256(stream.read()).hexdigest()
        identity = {
            "config_sha256": config_digest,
            "indices": list(index_tuple),
            "parameters": _stable_value(parameters),
        }
        return hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def evaluate(
        self,
        indices: Sequence[int],
        evaluation_id: int | str | None = None,
    ) -> EvaluationResult:
        """Run one candidate and return its objective, diagnostics and status."""
        index_tuple = tuple(int(index) for index in indices)
        parameters = self.parameter_space.to_values(index_tuple)
        fingerprint = self.fingerprint(index_tuple)

        if evaluation_id is None:
            directory_name = f"eval_{time.time_ns()}_{fingerprint[:10]}"
        else:
            safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(evaluation_id)).strip("._")
            if not safe_id:
                raise ValueError("evaluation_id must contain at least one safe character")
            directory_name = f"eval_{safe_id}_{fingerprint[:10]}"
        trial_dir = os.path.join(self.output_dir, directory_name)

        started = time.perf_counter()
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            results = self.simulation_runner(self.config_file, parameters, trial_dir)
            objective = float(self.objective_function(results, f_min=None, f_max=None))
            if not (objective == objective and abs(objective) != float("inf")):
                raise ValueError(f"Objective returned a non-finite value: {objective!r}")
            return EvaluationResult(
                fingerprint=fingerprint,
                indices=index_tuple,
                parameters=parameters,
                status="ok",
                objective=objective,
                evaluation_dir=trial_dir,
                elapsed_seconds=time.perf_counter() - started,
                metrics=_result_metrics(results),
            )
        except Exception as exc:
            return EvaluationResult(
                fingerprint=fingerprint,
                indices=index_tuple,
                parameters=parameters,
                status="failed",
                objective=self.failed_penalty,
                evaluation_dir=trial_dir,
                elapsed_seconds=time.perf_counter() - started,
                metrics={},
                error=f"{type(exc).__name__}: {exc}",
            )
