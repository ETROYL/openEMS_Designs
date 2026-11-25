#!/usr/bin/env python3
"""Export the best stored discrete candidate as a resolved high-accuracy YAML."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from design import load_config
from optimizer import EvaluationStore


HIGH_ACCURACY_SETTINGS = {
    "end_criteria": 1e-4,
    "max_timesteps": 1_000_000,
    "n_pts": 1024,
    "target_s11_db": -40.0,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Original parametric design YAML")
    parser.add_argument("--database", required=True, help="Optimizer SQLite history")
    parser.add_argument("--output", help="Output YAML (default: <design>_discrete_optimized.yaml)")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    database = Path(args.database).expanduser().resolve()
    if not config_path.is_file():
        parser.error(f"Configuration file not found: {config_path}")
    if not database.is_file():
        parser.error(f"Database not found: {database}")

    with EvaluationStore(database) as store:
        best = store.best()
    if best is None:
        parser.error("No successful evaluation exists in this database.")

    # Resolve all geometry expressions using exactly the stored winning values.
    resolved = load_config(str(config_path), active_vars=best.parameters)
    resolved.pop("variables", None)
    resolved.setdefault("simulation", {}).update(HIGH_ACCURACY_SETTINGS)

    output = Path(args.output).expanduser().resolve() if args.output else (
        config_path.parent / f"{config_path.stem}_discrete_optimized.yaml"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(resolved, stream, sort_keys=False)

    print(f"Best objective: {best.objective:.10g}")
    print(f"Best parameters: {best.parameters}")
    print(f"Saved resolved design: {output}")
    print("The exported YAML uses high-accuracy verification settings; run simulate.py on it separately.")


if __name__ == "__main__":
    main()
