#!/usr/bin/env python3
"""Summarize a discrete optimizer SQLite history and export CSV/plots."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optimizer import EvaluationStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, help="Path to evaluations.sqlite")
    parser.add_argument("--output-dir", help="Report directory (default: alongside database)")
    args = parser.parse_args()

    database = Path(args.database).expanduser().resolve()
    if not database.is_file():
        parser.error(f"Database not found: {database}")
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else database.parent / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    with EvaluationStore(database) as store:
        results = store.list_results()

    if not results:
        print("No evaluations are stored in this database.")
        return

    csv_path = output_dir / "evaluations.csv"
    fields = [
        "evaluation_number", "status", "objective", "elapsed_seconds",
        "parameters", "metrics", "fingerprint", "evaluation_dir", "error",
    ]
    best_so_far = None
    rows = []
    for number, result in enumerate(results, start=1):
        if result.status == "ok":
            best_so_far = result.objective if best_so_far is None else min(best_so_far, result.objective)
        rows.append({
            "evaluation_number": number,
            "status": result.status,
            "objective": result.objective,
            "elapsed_seconds": result.elapsed_seconds,
            "parameters": result.parameters,
            "metrics": result.metrics,
            "fingerprint": result.fingerprint,
            "evaluation_dir": result.evaluation_dir,
            "error": result.error or "",
            "best_so_far": best_so_far,
        })

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + ["best_so_far"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (row[key] if key not in {"parameters", "metrics"} else repr(row[key]))
                             for key in fields + ["best_so_far"]})

    successful = [row for row in rows if row["status"] == "ok"]
    failed = [row for row in rows if row["status"] == "failed"]
    best = min(successful, key=lambda row: row["objective"]) if successful else None

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    x_ok = [row["evaluation_number"] for row in successful]
    y_ok = [row["objective"] for row in successful]
    axes[0].scatter(x_ok, y_ok, label="Objective", alpha=0.8)
    if best:
        axes[0].plot(
            [row["evaluation_number"] for row in rows if row["best_so_far"] is not None],
            [row["best_so_far"] for row in rows if row["best_so_far"] is not None],
            label="Best so far",
        )
    x_fail = [row["evaluation_number"] for row in failed]
    if x_fail:
        axes[0].scatter(x_fail, [0] * len(x_fail), marker="x", label="Failed (positioned at zero)")
    axes[0].set_ylabel("Objective (lower is better)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        [row["evaluation_number"] for row in rows],
        [row["elapsed_seconds"] for row in rows],
        marker="o",
    )
    axes[1].set_xlabel("Stored evaluation number")
    axes[1].set_ylabel("FDTD wall time (s)")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = output_dir / "optimization_history.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    print(f"Evaluations: {len(rows)} ({len(successful)} successful, {len(failed)} failed)")
    if best:
        print(f"Best objective: {best['objective']:.10g}")
        print(f"Best parameters: {best['parameters']}")
    print(f"CSV: {csv_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
