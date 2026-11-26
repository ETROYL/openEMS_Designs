"""GUI adapter for the ETROYL discrete surrogate optimizer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from design import load_config
from optimizer import (
    CandidateEvaluator,
    DiscreteParameterSpace,
    DiscreteSurrogateOptimizer,
    EvaluationStore,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Immutable YAML snapshot for this optimization")
    parser.add_argument("--run-dir", required=True, help="Dedicated timestamped optimization-run directory")
    parser.add_argument("--budget", type=int, required=True)
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

    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
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
    print(f"SQLite history: {database}", flush=True)

    objective_weights = {"s11": args.weight_s11, "smoothness": args.weight_smoothness}
    evaluator = CandidateEvaluator.from_existing_pipeline(
        str(config_path), str(evaluation_dir), space,
        objective_name=args.objective, objective_weights=objective_weights,
    )

    with EvaluationStore(database) as store:
        already_completed = store.count()
        remaining_budget = max(0, args.budget - already_completed)
        print(f"Objective: {args.objective}", flush=True)
        if args.objective == "multi_objective_combined":
            print(f"Combined weights: S11={args.weight_s11:g}, impedance smoothness={args.weight_smoothness:g}", flush=True)
        print(f"Completed evaluations already stored: {already_completed}/{args.budget}", flush=True)
        print(f"Remaining evaluation budget: {remaining_budget}", flush=True)
        optimizer = DiscreteSurrogateOptimizer(
            space,
            evaluator,
            store,
            initial_samples=args.initial_samples,
            pool_size=args.pool_size,
            seed=None if args.seed < 0 else args.seed,
        )

        def report(result, completed: int, budget: int) -> None:
            best = store.best()
            payload = {
                "completed": store.count(),
                "budget": args.budget,
                "status": result.status,
                "objective": result.objective,
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

        summary = optimizer.run(
            remaining_budget,
            progress_callback=report,
            stop_requested=lambda: stop_file.exists() or pause_file.exists(),
        )

        print("\n========== ETROYL DISCRETE OPTIMIZATION SUMMARY ==========", flush=True)
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
        if not pause_file.exists() and not stop_file.exists() and store.count() >= args.budget:
            print("OPTIMIZATION_COMPLETED=1", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
