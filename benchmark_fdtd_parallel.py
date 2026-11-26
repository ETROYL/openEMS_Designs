#!/usr/bin/env python3
"""Benchmark concurrent openEMS throughput using independent worker processes.

Each worker delegates to benchmark_fdtd_threads.py, which sets EM_THREADS
before importing optimization.py. Every process gets its own simulation folder.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

RESULT_PREFIX = "__THREAD_BENCHMARK_RESULT__="
ALLOCATIONS = [(4, 3), (4, 2)]  # (number of workers, threads per worker)


def parse_result(output: str) -> dict[str, object] | None:
    for line in reversed(output.splitlines()):
        if line.startswith(RESULT_PREFIX):
            try:
                return json.loads(line[len(RESULT_PREFIX):])
            except json.JSONDecodeError:
                return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="designs/dipole_variable_thread_benchmark.yaml",
        help="Same validated benchmark YAML used by benchmark_fdtd_threads.py",
    )
    parser.add_argument("--repeats", type=int, default=5,
                        help="Number of concurrent batches per allocation (default: 5)")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parent
    worker_script = repo_dir / "benchmark_fdtd_threads.py"
    config = Path(args.config).expanduser().resolve()
    if not worker_script.is_file():
        parser.error(f"Worker script not found: {worker_script}")
    if not config.is_file():
        parser.error(f"Configuration file not found: {config}")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    root = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        config.parent / "parallel_thread_benchmark"
    )
    stamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = root / stamp
    results_dir.mkdir(parents=True, exist_ok=False)

    print("openEMS concurrent-throughput benchmark")
    print(f"Config: {config}")
    print("Fixed geometry: arm_length=0.26, root_width=0.020, tip_width=0.400, feed_gap=0.002")
    print("Simulation settings: end_criteria=1e-2 (-20 dB), max_timesteps=200000, n_pts=301")
    print(f"Concurrent batches per allocation: {args.repeats}")
    print("Allocations: " + "; ".join(f"{workers} workers x {threads} threads" for workers, threads in ALLOCATIONS))
    print(f"Results directory: {results_dir}")
    print("\nEach pair starts together. Avoid other heavy CPU workloads during the test.\n")

    rows: list[dict[str, object]] = []
    batch_number = 0
    total_batches = args.repeats * len(ALLOCATIONS)
    # Alternate allocation order between repeats to reduce systematic thermal bias.
    for repeat in range(1, args.repeats + 1):
        allocations = ALLOCATIONS if repeat % 2 else list(reversed(ALLOCATIONS))
        for workers, threads in allocations:
            batch_number += 1
            batch_name = f"workers_{workers:02d}_threads_{threads:02d}_repeat_{repeat:02d}"
            batch_dir = results_dir / batch_name
            batch_dir.mkdir(parents=True, exist_ok=False)
            print(f"[{batch_number}/{total_batches}] Starting {workers} workers x {threads} threads ...",
                  flush=True)

            processes: list[tuple[int, Path, subprocess.Popen[str]]] = []
            batch_started = time.perf_counter()
            for worker_index in range(1, workers + 1):
                run_dir = batch_dir / f"worker_{worker_index:02d}"
                run_dir.mkdir(parents=True, exist_ok=False)
                log_path = batch_dir / f"worker_{worker_index:02d}.log"
                env = os.environ.copy()
                env["EM_THREADS"] = str(threads)
                env["OMP_NUM_THREADS"] = str(threads)
                command = [
                    sys.executable, str(worker_script),
                    "--worker", "--threads", str(threads),
                    "--config", str(config), "--run-dir", str(run_dir),
                ]
                process = subprocess.Popen(
                    command,
                    cwd=str(repo_dir),
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                processes.append((worker_index, log_path, process))

            batch_results: list[dict[str, object]] = []
            for worker_index, log_path, process in processes:
                output, _ = process.communicate()
                output = output or ""
                log_path.write_text(output, encoding="utf-8")
                result = parse_result(output) or {
                    "status": "failed",
                    "error": "No result marker found; inspect worker log",
                }
                row = {
                    "workers": workers,
                    "threads_per_worker": threads,
                    "repeat": repeat,
                    "worker": worker_index,
                    "status": result.get("status", "failed"),
                    "objective": result.get("objective", ""),
                    "fdtd_seconds": result.get("fdtd_seconds", ""),
                    "return_code": process.returncode,
                    "log_file": str(log_path),
                    "run_dir": str(batch_dir / f"worker_{worker_index:02d}"),
                    "error": result.get("error", ""),
                }
                rows.append(row)
                batch_results.append(row)

            batch_wall = time.perf_counter() - batch_started
            successful = sum(r["status"] == "ok" and r["return_code"] == 0 for r in batch_results)
            print(f"    completed {successful}/{workers}; batch wall={batch_wall:.1f}s")
            for row in batch_results:
                print(f"      worker {row['worker']}: status={row['status']} "
                      f"FDTD={row['fdtd_seconds']}s error={row['error']}")

            # Repeat the batch wall time on each row for straightforward CSV analysis.
            for row in batch_results:
                row["batch_wall_seconds"] = round(batch_wall, 3)

    csv_path = results_dir / "parallel_thread_benchmark_results.csv"
    columns = [
        "workers", "threads_per_worker", "repeat", "worker", "status",
        "objective", "fdtd_seconds", "batch_wall_seconds", "return_code",
        "log_file", "run_dir", "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print("\n========== CONCURRENT THROUGHPUT SUMMARY ==========")
    for workers, threads in ALLOCATIONS:
        batches = []
        for repeat in range(1, args.repeats + 1):
            batch_rows = [r for r in rows if r["workers"] == workers
                          and r["threads_per_worker"] == threads and r["repeat"] == repeat]
            if len(batch_rows) == workers and all(
                r["status"] == "ok" and r["return_code"] == 0 for r in batch_rows
            ):
                batches.append(float(batch_rows[0]["batch_wall_seconds"]))
        if not batches:
            print(f"{workers} workers x {threads} threads: no fully successful batches")
            continue
        mean_wall = sum(batches) / len(batches)
        throughput = workers * 3600.0 / mean_wall
        print(f"{workers} workers x {threads} threads: {len(batches)}/{args.repeats} batches successful; "
              f"mean batch wall={mean_wall:.2f}s; throughput={throughput:.1f} simulations/hour")

    print(f"\nCSV: {csv_path}")
    print(f"Logs and isolated run directories: {results_dir}")
    print("Inspect each worker log to confirm the -20 dB end criterion was reached.")
    return 0 if all(r["status"] == "ok" and r["return_code"] == 0 for r in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
