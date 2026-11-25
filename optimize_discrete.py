#!/usr/bin/env python3
"""Run the discrete Sobol + Gaussian-process optimizer on a YAML design."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from optimizer import CandidateEvaluator, DiscreteParameterSpace, DiscreteSurrogateOptimizer, EvaluationStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Parametric design YAML")
    parser.add_argument("--budget", type=int, default=30, help="Maximum new FDTD evaluations")
    parser.add_argument("--database", help="SQLite history path (default: <design>_discrete_opt/evaluations.sqlite)")
    parser.add_argument("--output-dir", help="FDTD run directory (default: <design>_discrete_opt/runs)")
    parser.add_argument("--initial-samples", type=int, default=8, help="Sobol evaluations before surrogate-guided search")
    parser.add_argument("--pool-size", type=int, default=4096, help="Discrete candidate pool considered per GP step")
    parser.add_argument("--min-surrogate-points", type=int, default=4, help="Successful observations required before fitting GP")
    parser.add_argument("--seed", type=int, default=42, help="Random seed; use a negative value for nondeterministic sampling")
    parser.add_argument("--xi", type=float, default=0.01, help="Expected-improvement exploration parameter")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        parser.error(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    space = DiscreteParameterSpace.from_config(config)
    if space.dimension == 0:
        parser.error("The YAML file contains no active discrete optimization variables.")

    base_dir = config_path.parent / f"{config_path.stem}_discrete_opt"
    database = Path(args.database).expanduser().resolve() if args.database else base_dir / "evaluations.sqlite"
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else base_dir / "runs"

    evaluator = CandidateEvaluator.from_existing_pipeline(
        str(config_path), str(output_dir), space
    )

    print(f"Design: {config_path}")
    print(f"Active parameters: {space.dimension}")
    print(f"Search-space cardinality: {space.cardinality:,}")
    for item in space.describe():
        print(f"  {item['name']}: {item['levels']} levels, "
              f"{item['start']} .. {item['last_grid_value']} (step {item['step']})")
    print(f"Database: {database}")
    print(f"FDTD output: {output_dir}")
    print(f"New-evaluation budget: {args.budget}\n")

    with EvaluationStore(database) as store:
        optimizer = DiscreteSurrogateOptimizer(
            space,
            evaluator,
            store,
            initial_samples=args.initial_samples,
            pool_size=args.pool_size,
            min_surrogate_points=args.min_surrogate_points,
            xi=args.xi,
            seed=None if args.seed < 0 else args.seed,
        )
        summary = optimizer.run(args.budget)

        print("\n========== DISCRETE OPTIMIZATION SUMMARY ==========")
        for result in summary.evaluations:
            print(f"{result.status:6s} objective={result.objective:.8g} "
                  f"time={result.elapsed_seconds:.1f}s parameters={result.parameters}")
            if result.error:
                print(f"  error: {result.error}")
        print(f"New FDTD attempts: {len(summary.evaluations)}")
        print(f"Total stored: {summary.total_stored}")
        print(f"Successful: {summary.successful_stored}")
        print(f"Failed: {summary.failed_stored}")
        if summary.best:
            print(f"Best objective: {summary.best.objective:.10g}")
            print(f"Best parameters: {summary.best.parameters}")
        else:
            print("No successful evaluation is available yet.")
        print(f"Stop reason: {summary.stop_reason}")


if __name__ == "__main__":
    main()
