#!/usr/bin/env python3
"""Benchmark one fixed openEMS geometry with several thread allocations.

The parent process launches a fresh Python subprocess for every run so that
parallel.py reads EM_THREADS before optimization.py imports NUM_THREADS.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

RESULT_PREFIX = "__THREAD_BENCHMARK_RESULT__="
MAX_TIMESTEP_WARNING = re.compile(
    r"Max\.?\s+number\s+of\s+timesteps\s+was\s+reached",
    re.IGNORECASE,
)
PARAMETERS = {
    "arm_length": 0.26,
    "root_width": 0.020,
    "tip_width": 0.400,
    "feed_gap": 0.002,
}


def run_worker(args: argparse.Namespace) -> int:
    # This must happen before importing optimization/parallel.
    os.environ["EM_THREADS"] = str(args.threads)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)

    started = time.perf_counter()
    payload: dict[str, object]
    try:
        from optimization import run_simulation_instance
        from objectives import multi_objective_combined

        results = run_simulation_instance(
            str(Path(args.config).resolve()),
            PARAMETERS,
            str(Path(args.run_dir).resolve()),
        )
        objective = float(multi_objective_combined(results, f_min=None, f_max=None))
        payload = {
            "status": "ok",
            "objective": objective,
            "fdtd_seconds": time.perf_counter() - started,
            "parameters": PARAMETERS,
            "threads": args.threads,
        }
    except Exception as exc:
        payload = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "fdtd_seconds": time.perf_counter() - started,
            "parameters": PARAMETERS,
            "threads": args.threads,
        }

    print(RESULT_PREFIX + json.dumps(payload, sort_keys=True), flush=True)
    return 0 if payload["status"] == "ok" else 1


def parse_worker_result(output: str) -> dict[str, object] | None:
    for line in reversed(output.splitlines()):
        if line.startswith(RESULT_PREFIX):
            try:
                return json.loads(line[len(RESULT_PREFIX):])
            except json.JSONDecodeError:
                return None
    return None


def build_run_order(threads: list[int], repeats: int) -> list[tuple[int, int]]:
    # Rotate/reverse the order between rounds to reduce systematic thermal/order bias.
    schedule: list[tuple[int, int]] = []
    for repeat in range(1, repeats + 1):
        if repeat % 3 == 1:
            order = threads
        elif repeat % 3 == 2:
            order = list(reversed(threads))
        else:
            order = threads[1:] + threads[:1]
        schedule.extend((thread_count, repeat) for thread_count in order)
    return schedule


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="designs/dipole_variable_thread_benchmark.yaml",
        help="Benchmark YAML (must retain the same mesh and simulation settings)",
    )
    parser.add_argument("--threads", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", help=argparse.SUPPRESS)
    parser.add_argument("--parameters-json", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        return run_worker(args)

    config = Path(args.config).expanduser().resolve()
    if not config.is_file():
        parser.error(f"Configuration file not found: {config}")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if not args.threads or any(n < 1 for n in args.threads):
        parser.error("Thread counts must all be positive integers")
    if len(set(args.threads)) != len(args.threads):
        parser.error("Thread counts must be unique")

    root = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        config.parent / "thread_benchmark"
    )
    stamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = root / stamp
    results_dir.mkdir(parents=True, exist_ok=False)

    print("openEMS thread-allocation benchmark")
    print(f"Config: {config}")
    print(f"Fixed geometry: {PARAMETERS}")
    print("Simulation settings: end_criteria=-20 dB (1e-2), max_timesteps=200000")
    print(f"Thread counts: {args.threads}; repeats per setting: {args.repeats}")
    print(f"Results directory: {results_dir}")
    print("\nRuns are sequential; do not run other heavy CPU workloads during the test.\n")

    rows: list[dict[str, object]] = []
    schedule = build_run_order(args.threads, args.repeats)

    for run_number, (thread_count, repeat) in enumerate(schedule, start=1):
        run_name = f"threads_{thread_count:02d}_repeat_{repeat:02d}"
        run_dir = results_dir / "runs" / run_name
        run_dir.mkdir(parents=True, exist_ok=False)
        log_path = results_dir / f"{run_name}.log"

        env = os.environ.copy()
        env["EM_THREADS"] = str(thread_count)
        env["OMP_NUM_THREADS"] = str(thread_count)

        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--threads", str(thread_count),
            "--config", str(config),
            "--run-dir", str(run_dir),
        ]

        print(f"[{run_number}/{len(schedule)}] {thread_count} threads, repeat {repeat} ...", flush=True)
        wall_started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=str(Path(__file__).resolve().parent),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        wall_seconds = time.perf_counter() - wall_started
        output = completed.stdout or ""
        log_path.write_text(output, encoding="utf-8")
        worker_result = parse_worker_result(output)

        warning_found = bool(MAX_TIMESTEP_WARNING.search(output))
        if warning_found:
            convergence_status = "max_timesteps_warning"
        elif completed.returncode == 0 and worker_result and worker_result.get("status") == "ok":
            convergence_status = "no_max_timesteps_warning"
        else:
            convergence_status = "unknown_or_failed"

        row: dict[str, object] = {
            "threads": thread_count,
            "repeat": repeat,
            "status": worker_result.get("status", "failed") if worker_result else "failed",
            "objective": worker_result.get("objective", "") if worker_result else "",
            "fdtd_seconds": worker_result.get("fdtd_seconds", "") if worker_result else "",
            "wall_seconds": round(wall_seconds, 3),
            "convergence_status": convergence_status,
            "return_code": completed.returncode,
            "log_file": str(log_path),
            "run_dir": str(run_dir),
            "error": worker_result.get("error", "") if worker_result else "No result marker found; inspect log",
        }
        rows.append(row)
        print(
            f"    status={row['status']} objective={row['objective']} "
            f"FDTD={row['fdtd_seconds']}s wall={wall_seconds:.1f}s "
            f"decay_check={convergence_status}"
        )

    csv_path = results_dir / "thread_benchmark_results.csv"
    columns = [
        "threads", "repeat", "status", "objective", "fdtd_seconds",
        "wall_seconds", "convergence_status", "return_code",
        "log_file", "run_dir", "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print("\n========== THREAD BENCHMARK SUMMARY ==========")
    for thread_count in args.threads:
        valid = [
            row for row in rows
            if row["threads"] == thread_count
            and row["status"] == "ok"
            and isinstance(row["fdtd_seconds"], (int, float))
        ]
        if not valid:
            print(f"{thread_count} threads: no successful runs")
            continue
        times = [float(row["fdtd_seconds"]) for row in valid]
        objectives = [float(row["objective"]) for row in valid]
        print(
            f"{thread_count} threads: {len(valid)}/{args.repeats} successful; "
            f"mean FDTD time={sum(times)/len(times):.2f}s; "
            f"mean objective={sum(objectives)/len(objectives):.6g}"
        )

    print(f"\nCSV: {csv_path}")
    print(f"Logs: {results_dir}")
    print("Note: 'no_max_timesteps_warning' means the max-timestep warning was not found;")
    print("confirm the openEMS log shows the requested -20 dB end criterion was reached.")
    return 0 if all(row["status"] == "ok" for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
