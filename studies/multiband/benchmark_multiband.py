#!/usr/bin/env python3
"""Run overlapping broadband FDTD windows and compare their complex S11 responses.

The YAML supplies only the design geometry and variable defaults. Frequency,
frequency sampling, energy-decay threshold, timestep limit and thread count are
controlled by this benchmark.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

# Deliberately overlapping windows: the overlap is used to compare complex S11.
DEFAULT_WINDOWS_GHZ = (
    (0.10, 0.55),
    (0.45, 1.55),
    (1.45, 3.00),
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Benchmark overlapping openEMS frequency windows."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root / "studies/multiband/dipole_multiband.yaml",
        help="Parametric YAML design (geometry and defaults).",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--n-pts", type=int, default=101,
                        help="Frequency samples per window, including endpoints.")
    parser.add_argument("--end-criteria-db", type=float, default=-15.0,
                        help="openEMS energy threshold expressed in dB.")
    parser.add_argument("--max-timesteps", type=int, default=200000)
    parser.add_argument(
        "--output-root", type=Path, default=None,
        help="Output directory. Defaults beside the YAML file.",
    )
    return parser.parse_args()


def write_window_config(source_cfg: dict, config_path: Path,
                        fstart: float, fstop: float, n_pts: int,
                        end_criteria: float, max_timesteps: int) -> None:
    cfg = copy.deepcopy(source_cfg)
    cfg.setdefault("simulation", {})
    cfg["simulation"].update({
        "fstart": float(fstart),
        "fstop": float(fstop),
        "n_pts": int(n_pts),
        "end_criteria": float(end_criteria),
        "max_timesteps": int(max_timesteps),
    })
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False)


def run_window(index: int, fstart: float, fstop: float, args: argparse.Namespace,
               source_cfg: dict, run_root: Path, end_criteria: float) -> dict:
    # Set thread limits before importing openEMS / CSXCAD.
    os.environ["OMP_NUM_THREADS"] = str(args.threads)

    from CSXCAD import ContinuousStructure
    from openEMS import openEMS
    from design import build_design, load_config

    tag = f"window_{index:02d}_{fstart:.2f}_{fstop:.2f}GHz"
    sim_dir = run_root / tag
    sim_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_root / "configs" / f"{tag}.yaml"
    write_window_config(
        source_cfg, config_path, fstart, fstop, args.n_pts,
        end_criteria, args.max_timesteps,
    )

    fstart_hz = fstart * 1e9
    fstop_hz = fstop * 1e9
    f0_hz = 0.5 * (fstart_hz + fstop_hz)
    fc_hz = fstop_hz - fstart_hz

    print("\n" + "=" * 76, flush=True)
    print(f"{tag}: {args.n_pts} points, EndCriteria={end_criteria:.9g}, "
          f"max timesteps={args.max_timesteps}, threads={args.threads}", flush=True)
    print(f"Excitation: f0={f0_hz / 1e9:.6g} GHz, "
          f"fc={fc_hz / 1e9:.6g} GHz", flush=True)

    csx = ContinuousStructure()
    fdtd = openEMS(EndCriteria=end_criteria, NrTS=args.max_timesteps)
    fdtd.SetCSX(csx)
    fdtd.SetGaussExcite(f0_hz, fc_hz)
    fdtd.SetBoundaryCond(
        ["PML_8", "PML_8", "PML_8", "PML_8", "PML_8", "PML_8"]
    )
    build_design(csx, fdtd, str(config_path))

    xml_path = sim_dir / "geometry.xml"
    csx.Write2XML(str(xml_path))

    started = time.perf_counter()
    fdtd.Run(str(sim_dir), numThreads=args.threads)
    elapsed = time.perf_counter() - started

    # Rebuild the port object and extract the complex frequency-domain response.
    post_csx = ContinuousStructure()
    post_fdtd = openEMS(EndCriteria=end_criteria, NrTS=args.max_timesteps)
    post_fdtd.SetCSX(post_csx)
    post_fdtd.SetGaussExcite(f0_hz, fc_hz)
    post_port, _ = build_design(
        post_csx, post_fdtd, str(config_path), enable_nf2ff=False
    )
    frequencies_hz = np.linspace(fstart_hz, fstop_hz, args.n_pts)
    post_port.CalcPort(str(sim_dir), frequencies_hz)
    s11 = np.asarray(post_port.uf_ref) / np.asarray(post_port.uf_inc)

    freq_ghz = frequencies_hz / 1e9
    s11_db = 20.0 * np.log10(np.maximum(np.abs(s11), 1e-15))
    phase_deg = np.rad2deg(np.unwrap(np.angle(s11)))

    csv_path = sim_dir / "s11_complex.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "frequency_GHz", "s11_real", "s11_imag",
            "s11_magnitude_dB", "s11_phase_deg",
        ])
        writer.writerows(zip(
            freq_ghz, s11.real, s11.imag, s11_db, phase_deg
        ))

    result = {
        "name": tag,
        "fstart_GHz": fstart,
        "fstop_GHz": fstop,
        "n_pts": args.n_pts,
        "end_criteria_db": args.end_criteria_db,
        "end_criteria_linear": end_criteria,
        "max_timesteps": args.max_timesteps,
        "threads": args.threads,
        "runtime_seconds": elapsed,
        "frequency_GHz": freq_ghz,
        "s11_real": s11.real,
        "s11_imag": s11.imag,
        "s11_db": s11_db,
        "phase_deg": phase_deg,
        "simulation_directory": str(sim_dir),
        "config_file": str(config_path),
    }
    # Keep numeric response in a machine-readable format for later comparison.
    np.savez(
        sim_dir / "s11_complex.npz",
        frequency_GHz=freq_ghz,
        s11=s11,
        s11_db=s11_db,
        phase_deg=phase_deg,
    )
    print(f"Completed {tag} in {elapsed:.2f} s; response saved to {csv_path}",
          flush=True)
    return result


def make_comparison_plot(results: list[dict], output_path: Path) -> None:
    fig, (ax_mag, ax_phase) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True
    )
    for result in results:
        label = (
            f"{result['fstart_GHz']:.2f}–{result['fstop_GHz']:.2f} GHz"
        )
        color = None  # Matplotlib's standard colour cycle assigns a distinct colour.
        ax_mag.plot(
            result["frequency_GHz"], result["s11_db"],
            linewidth=1.8, label=label, color=color,
        )
        ax_phase.plot(
            result["frequency_GHz"], result["phase_deg"],
            linewidth=1.5, label=label, color=color,
        )

    for low, high in ((0.45, 0.55), (1.45, 1.55)):
        ax_mag.axvspan(low, high, alpha=0.12, color="grey")
        ax_phase.axvspan(low, high, alpha=0.12, color="grey")

    ax_mag.set_ylabel("|S11| (dB)")
    ax_mag.set_title("Overlapping FDTD frequency windows — complex S11 comparison")
    ax_mag.grid(True, alpha=0.3)
    ax_mag.legend()
    ax_phase.set_xlabel("Frequency (GHz)")
    ax_phase.set_ylabel("Unwrapped phase (degrees)")
    ax_phase.grid(True, alpha=0.3)
    ax_phase.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Comparison plot saved to: {output_path}", flush=True)
    plt.show()


def main() -> None:
    args = parse_args()
    if args.threads < 1:
        raise SystemExit("--threads must be >= 1")
    if args.n_pts < 2:
        raise SystemExit("--n-pts must be >= 2")
    if args.end_criteria_db >= 0:
        raise SystemExit("--end-criteria-db must be negative")

    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Design YAML not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        source_cfg = yaml.safe_load(stream)

    # openEMS EndCriteria is a linear energy ratio: -15 dB -> 10^(-15/10).
    end_criteria = 10.0 ** (args.end_criteria_db / 10.0)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root else config_path.parent / f"multiband_run_{stamp}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Design: {config_path}", flush=True)
    print(f"Output: {output_root}", flush=True)
    print(f"Energy threshold: {args.end_criteria_db:.2f} dB "
          f"-> {end_criteria:.9g} linear", flush=True)

    results = []
    for index, (fstart, fstop) in enumerate(DEFAULT_WINDOWS_GHZ, start=1):
        results.append(run_window(
            index, fstart, fstop, args, source_cfg, output_root, end_criteria
        ))

    # JSON summary excludes the full response arrays; CSV/NPZ files retain them.
    summary = {
        "design": str(config_path),
        "windows": [
            {key: value for key, value in item.items()
             if key not in {"frequency_GHz", "s11_real", "s11_imag",
                            "s11_db", "phase_deg"}}
            for item in results
        ],
        "total_runtime_seconds": sum(item["runtime_seconds"] for item in results),
        "threads": args.threads,
        "n_pts_per_window": args.n_pts,
        "end_criteria_db": args.end_criteria_db,
        "end_criteria_linear": end_criteria,
        "max_timesteps": args.max_timesteps,
    }
    summary_path = output_root / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Total FDTD runtime: {summary['total_runtime_seconds']:.2f} s",
          flush=True)
    print(f"Summary saved to: {summary_path}", flush=True)
    make_comparison_plot(results, output_root / "s11_window_comparison.png")


if __name__ == "__main__":
    main()
