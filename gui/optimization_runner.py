"""GUI adapter for the ETROYL discrete surrogate optimizer."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil

from gui.cpu_resources import detect_cpu_resources, validate_allocation
from multiband_fdtd.resources import allocate_band_resources
from pathlib import Path

import yaml

from design import get_simulation_settings, load_config
from optimizer import (
    CandidateEvaluator,
    DiscreteParameterSpace,
    DiscreteSurrogateOptimizer,
    EvaluationStore,
    OptunaTPEOptimizer,
)




def _prepare_interval_plan(
    config_path: Path,
    run_dir: Path,
    run_settings: dict,
    frequency_range: tuple[float, float],
    output_points: int,
    stop_requested=None,
):
    """Load or create a deterministic equal-width interval plan."""
    from multiband_fdtd import build_equal_width_frequency_bands, load_interval_plan

    saved_plan = run_settings.get("interval_plan_path")
    if saved_plan:
        plan_path = str(Path(saved_plan).expanduser().resolve())
        saved_range = run_settings.get("interval_plan_frequency_range_ghz")
        if saved_range is not None and (
            len(saved_range) != 2
            or not math.isclose(float(saved_range[0]), frequency_range[0], abs_tol=1e-12)
            or not math.isclose(float(saved_range[1]), frequency_range[1], abs_tol=1e-12)
        ):
            raise ValueError(
                "The saved interval plan frequency range does not match the current "
                "optimization settings. Start a new optimization to change the range."
            )
        bands = load_interval_plan(
            plan_path, n_points=output_points, config_path=config_path
        )
        return plan_path, bands, False

    if (run_dir / "evaluations.sqlite").exists():
        raise ValueError(
            "This optimization already has evaluation history but no saved interval "
            "plan. Refusing to prepare a different plan for an existing run."
        )
    if stop_requested is not None and stop_requested():
        raise InterruptedError("Interval planning was interrupted before execution.")

    bands = build_equal_width_frequency_bands(
        frequency_range, output_points, maximum_bandwidth_ghz=0.5
    )
    preparation_dir = run_dir / "interval_preparation"
    preparation_dir.mkdir(parents=True, exist_ok=True)
    plan_path = preparation_dir / "interval_plan.json"
    manifest = {
        "plan_version": 1,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "candidate_generation": {
            "n_pts": int(output_points),
            "sampling_mode": "equal_total_points",
        },
        "selection": {
            "frequency_range_ghz": [float(frequency_range[0]), float(frequency_range[1])],
            "selected_intervals": [
                {
                    "fstart_ghz": float(band.fstart_ghz),
                    "fstop_ghz": float(band.fstop_ghz),
                    "n_pts": int(band.n_pts),
                }
                for band in bands
            ],
        },
    }
    plan_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    run_settings["interval_plan_path"] = str(plan_path.resolve())
    run_settings["interval_plan_frequency_range_ghz"] = list(frequency_range)
    metadata_path = run_dir / "run_settings.json"
    metadata_path.write_text(json.dumps(run_settings, indent=2), encoding="utf-8")
    return str(plan_path.resolve()), bands, True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Immutable YAML snapshot for this optimization")
    parser.add_argument("--run-dir", required=True, help="Dedicated timestamped optimization-run directory")
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--strategy", choices=["etroyl_discrete", "optuna_tpe"], default="etroyl_discrete")
    parser.add_argument("--initial-samples", type=int, default=8)
    parser.add_argument("--pool-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stop-file", required=True)
    parser.add_argument("--pause-file", required=True)
    parser.add_argument("--objective", choices=[
        "maximize_bandwidth_in_band",
        "minimize_max_s11_in_band",
        "minimize_s11_in_band",
        "maximize_impedance_smoothness",
        "multi_objective_combined",
    ], default="multi_objective_combined")
    parser.add_argument("--weight-s11", type=float, default=1.0)
    parser.add_argument("--weight-smoothness", type=float, default=0.0053)
    parser.add_argument("--resource-mode", choices=["default", "manual"], default="default")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    stop_file = Path(args.stop_file).expanduser().resolve()
    pause_file = Path(args.pause_file).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Optimization YAML snapshot not found: {config_path}")
    if args.budget <= 0:
        parser.error("--budget must be greater than zero")
    if args.initial_samples <= 0 or args.pool_size <= 0:
        parser.error("--initial-samples and --pool-size must be greater than zero")
    resources = detect_cpu_resources()
    if args.resource_mode == "default":
        args.workers, args.threads_per_worker = 4, 2
    try:
        validate_allocation(args.workers, args.threads_per_worker)
    except ValueError as exc:
        parser.error(str(exc))

    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    metadata_path = run_dir / "run_settings.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Optimization run settings not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as stream:
        run_settings = json.load(stream)
    evaluation_settings = run_settings.get("evaluation_settings")
    if run_settings.get("use_simulation_parameters", True):
        evaluation_settings = None
    elif not isinstance(evaluation_settings, dict):
        raise ValueError("Optimization-only mode requires saved evaluation settings.")

    interval_plan_path = None
    if bool(run_settings.get("use_multiband", False)):
        resolved_config = load_config(str(config_path))
        effective_settings = get_simulation_settings(
            resolved_config, "optimization", overrides=evaluation_settings
        )
        frequency_range = (
            float(effective_settings["fstart"]),
            float(effective_settings["fstop"]),
        )
        output_points = int(effective_settings.get("n_pts", 301))
        print(
            "Preparing deterministic equal-width intervals "
            "or loading the saved plan.",
            flush=True,
        )
        try:
            interval_plan_path, fixed_bands, created_plan = _prepare_interval_plan(
                config_path, run_dir, run_settings, frequency_range, output_points,
                stop_requested=lambda: stop_file.exists() or pause_file.exists(),
            )
        except InterruptedError:
            if pause_file.exists() and not stop_file.exists():
                print(
                    "Interval planning paused before candidate evaluations started.",
                    flush=True,
                )
                print("OPTIMIZATION_PAUSED=1", flush=True)
                return 0
            if stop_file.exists():
                print(
                    "Interval planning stopped before candidate evaluations started.",
                    flush=True,
                )
                print("OPTIMIZATION_STOPPED=1", flush=True)
                return 0
            raise
        if created_plan:
            print(
                f"Prepared {len(fixed_bands)} fixed interval(s): "
                + ", ".join(
                    f"{band.fstart_ghz:.3f}–{band.fstop_ghz:.3f} GHz"
                    for band in fixed_bands
                ),
                flush=True,
            )
            print(f"Saved interval plan: {interval_plan_path}", flush=True)
        else:
            print(f"Using saved interval plan: {interval_plan_path}", flush=True)
        if pause_file.exists() and not stop_file.exists():
            print("Interval plan is saved; resume will reuse it.", flush=True)
            print("OPTIMIZATION_PAUSED=1", flush=True)
            return 0
        if stop_file.exists():
            print("Interval plan is saved; no candidate evaluations were started.", flush=True)
            print("OPTIMIZATION_STOPPED=1", flush=True)
            return 0

    band_allocation = None
    if interval_plan_path is not None:
        band_allocation = allocate_band_resources(len(fixed_bands))
        # One candidate at a time; its interval simulations consume the shared
        # CPU budget. This prevents nested candidate/interval pools.
        args.workers = 1
        args.threads_per_worker = 1
        print(
            f"Interval execution allocation: {band_allocation.workers} concurrent "
            f"interval(s) × {band_allocation.threads_per_worker} thread(s) "
            f"({band_allocation.logical_cpus} logical CPUs available).",
            flush=True,
        )
        print(
            "Candidate evaluations are serialized so interval workers can use "
            "the CPU budget without nested-pool oversubscription.",
            flush=True,
        )

    space = DiscreteParameterSpace.from_config(config)
    if space.dimension == 0:
        raise ValueError("The YAML contains no active discrete optimization variables.")

    database = run_dir / "evaluations.sqlite"
    evaluation_dir = run_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    print(f"OPTIMIZATION_CONFIG={config_path}", flush=True)
    print(f"OPTIMIZATION_RUN_DIR={run_dir}", flush=True)
    print(f"Active parameters: {space.dimension}", flush=True)
    print(f"Search-space cardinality: {space.cardinality:,}", flush=True)
    for item in space.describe():
        print(
            f"  {item['name']}: {item['levels']} levels, "
            f"{item['start']} .. {item['last_grid_value']} (step {item['step']})",
            flush=True,
        )
    print(f"Evaluation budget: {args.budget}", flush=True)
    print(
        "Evaluation profile: " + (
            "YAML simulation settings" if evaluation_settings is None
            else json.dumps(evaluation_settings, sort_keys=True)
        ),
        flush=True,
    )
    print(
        f"CPU capacity: {resources['physical_cores']} physical cores / "
        f"{resources['logical_threads']} logical CPUs",
        flush=True,
    )
    print(
        f"Validated allocation: {args.workers} worker(s) × "
        f"{args.threads_per_worker} thread(s) per worker ({args.resource_mode})",
        flush=True,
    )
    print(
        f"Active candidate execution: {args.workers} spawned process worker(s), "
        f"{args.threads_per_worker} process thread(s) per worker; "
        f"up to {args.workers} candidates evaluated concurrently.",
        flush=True,
    )
    print(f"SQLite history: {database}", flush=True)

    objective_weights = {"s11": args.weight_s11, "smoothness": args.weight_smoothness}
    # The parent evaluator is used only for deterministic candidate
    # fingerprinting. Native openEMS evaluators are constructed inside each
    # spawned worker after its EM_THREADS setting has been applied.
    evaluator = CandidateEvaluator(
        str(config_path),
        str(evaluation_dir),
        space,
        simulation_runner=lambda *_args: (_ for _ in ()).throw(
            RuntimeError("Simulation must execute in a spawned FDTD worker")
        ),
        objective_function=lambda *_args, **_kwargs: float("nan"),
        interval_plan_path=interval_plan_path,
        evaluation_settings=evaluation_settings,
    )

    with EvaluationStore(database) as store:
        already_completed = store.count()
        remaining_budget = max(0, args.budget - already_completed)
        print(f"Objective: {args.objective}", flush=True)
        if args.objective == "multi_objective_combined":
            print(f"Combined weights: S11={args.weight_s11:g}, impedance smoothness={args.weight_smoothness:g}", flush=True)
        print(f"Evaluations already stored: {already_completed}", flush=True)
        print(f"Additional evaluations requested for this run target: {max(0, args.budget - already_completed)}", flush=True)
        print(f"Total evaluation target: {args.budget}", flush=True)
        if args.strategy == "etroyl_discrete":
            optimizer = DiscreteSurrogateOptimizer(
                space,
                evaluator,
                store,
                initial_samples=args.initial_samples,
                pool_size=args.pool_size,
                seed=None if args.seed < 0 else args.seed,
            )
            strategy_label = "ETROYL Discrete (Sobol + GP + Expected Improvement)"
        else:
            optimizer = OptunaTPEOptimizer(
                space,
                evaluator,
                store,
                seed=None if args.seed < 0 else args.seed,
                startup_trials=args.initial_samples,
            )
            strategy_label = "Optuna TPE"
        print(f"Strategy: {strategy_label}", flush=True)

        # Restore the full sequence when resuming a paused run.
        history = []
        historical_best = None
        for step, previous in enumerate(store.list_results(), start=1):
            if previous.status == "ok":
                try:
                    value = float(previous.objective)
                    if math.isfinite(value):
                        historical_best = value if historical_best is None else min(historical_best, value)
                except (TypeError, ValueError):
                    pass
            history.append({
                "step": step,
                "status": previous.status,
                "objective": previous.objective if previous.status == "ok" else None,
                "best_objective": historical_best,
            })
        print("__OPTIMIZATION_HISTORY__=" + json.dumps(history, allow_nan=False), flush=True)

        def report(result, completed: int, budget: int) -> None:
            best = store.best()
            payload = {
                "completed": store.count(),
                "budget": args.budget,
                "step": store.count(),
                "status": result.status,
                "objective": result.objective if result.status == "ok" else None,
                "elapsed_seconds": result.elapsed_seconds,
                "parameters": result.parameters,
                "best_objective": best.objective if best else None,
                "best_parameters": best.parameters if best else None,
                "successful": store.count("ok"),
                "failed": store.count("failed"),
            }
            print("__OPTIMIZATION_PROGRESS__=" + json.dumps(payload, sort_keys=True), flush=True)
            print(
                f"{result.status.upper():6s} objective={result.objective:.8g} "
                f"elapsed={result.elapsed_seconds:.1f}s parameters={result.parameters}",
                flush=True,
            )
            if result.error:
                print(f"  evaluation error: {result.error}", flush=True)

        summary = optimizer.run_parallel(
            remaining_budget,
            workers=args.workers,
            threads_per_worker=args.threads_per_worker,
            objective_name=args.objective,
            objective_weights=objective_weights,
            progress_callback=report,
            stop_requested=lambda: stop_file.exists() or pause_file.exists(),
            evaluation_settings=evaluation_settings,
            interval_plan_path=interval_plan_path,
        )

        print(f"\n========== {strategy_label.upper()} OPTIMIZATION SUMMARY ==========", flush=True)
        print(f"New FDTD attempts: {len(summary.evaluations)}", flush=True)
        print(f"Total stored: {summary.total_stored}", flush=True)
        print(f"Successful: {summary.successful_stored}", flush=True)
        print(f"Failed: {summary.failed_stored}", flush=True)
        print(f"Stop reason: {summary.stop_reason}", flush=True)
        if pause_file.exists() and not stop_file.exists():
            print("OPTIMIZATION_PAUSED=1", flush=True)
        elif stop_file.exists():
            print("OPTIMIZATION_STOPPED=1", flush=True)

        if summary.best is None:
            print("No successful evaluation is available; no best design was written.", flush=True)
            return 2

        best_config = load_config(str(config_path), active_vars=summary.best.parameters)
        best_path = run_dir / "best_design.yaml"
        with best_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(best_config, stream, sort_keys=False, allow_unicode=True)
        print(f"Best objective: {summary.best.objective:.10g}", flush=True)
        print(f"Best parameters: {summary.best.parameters}", flush=True)
        print(f"BEST_DESIGN_PATH={best_path}", flush=True)
        if (
            not pause_file.exists()
            and not stop_file.exists()
            and summary.stop_reason in (
                "evaluation budget reached",
                "discrete search space exhausted",
                "no unexplored candidates available",
                "zero budget",
            )
        ):
            print("OPTIMIZATION_COMPLETED=1", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
