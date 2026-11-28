"""Fast mesh/NF2FF validation without advancing FDTD timesteps.

Example:
    python -m multiband_fdtd.mesh_diagnostic bowtie.yaml \
        --fstart 2.5 --fstop 3.0 --output mesh_diagnostic_2p5_3p0
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np


def collect_mesh_report(csx):
    """Return per-axis mesh statistics without starting the FDTD solver."""
    grid = csx.GetGrid()
    axes = {}
    cell_counts = []
    for axis in ("x", "y", "z"):
        lines = np.unique(np.asarray(grid.GetLines(axis), dtype=float).reshape(-1))
        if lines.size < 2:
            raise RuntimeError(
                f"The {axis}-axis mesh has fewer than two distinct lines."
            )
        gaps = np.diff(lines)
        axes[axis] = {
            "line_count": int(lines.size),
            "cell_count": int(gaps.size),
            "minimum_spacing": float(np.min(gaps)),
            "maximum_spacing": float(np.max(gaps)),
            "minimum_spacing_location": [
                float(lines[int(np.argmin(gaps))]),
                float(lines[int(np.argmin(gaps)) + 1]),
            ],
        }
        cell_counts.append(int(gaps.size))

    return {
        "axes": axes,
        "total_cells": int(np.prod(cell_counts)),
    }


def run_mesh_diagnostic(config_path, output_dir, fstart_ghz, fstop_ghz,
                        enable_nf2ff=True, active_vars=None):
    """Build the selected design and ask openEMS for setup-only diagnostics."""
    from CSXCAD import ContinuousStructure
    from openEMS import openEMS
    from design import build_design

    config_path = Path(config_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Design configuration not found: {config_path}")
    if not (np.isfinite(fstart_ghz) and np.isfinite(fstop_ghz)
            and 0 < fstart_ghz < fstop_ghz):
        raise ValueError("Frequency range must satisfy 0 < fstart < fstop.")
    output_dir.mkdir(parents=True, exist_ok=False)

    fstart = float(fstart_ghz) * 1e9
    fstop = float(fstop_ghz) * 1e9
    f0 = 0.5 * (fstart + fstop)
    fc = fstop - fstart

    csx = ContinuousStructure()
    # Setup-only mode does not advance the FDTD solver. NrTS=1 and
    # EndCriteria=0 are additional safeguards if the API behavior changes.
    fdtd = openEMS(EndCriteria=0.0, NrTS=1)
    fdtd.SetCSX(csx)
    fdtd.SetGaussExcite(f0, fc)
    fdtd.SetBoundaryCond(["PML_8"] * 6)

    _, nf2ff = build_design(
        csx,
        fdtd,
        config_file=str(config_path),
        active_vars=active_vars,
        enable_nf2ff=enable_nf2ff,
        simulation_settings_override={
            "fstart": float(fstart_ghz),
            "fstop": float(fstop_ghz),
        },
    )

    report = collect_mesh_report(csx)
    report.update({
        "status": "mesh_built",
        "frequency_range_ghz": [float(fstart_ghz), float(fstop_ghz)],
        "nf2ff_requested": bool(enable_nf2ff),
        "nf2ff_created": nf2ff is not None,
        "fdtd_timesteps_executed": 0,
        "solver_mode": "setup_only",
        "configuration": str(config_path),
        "solver_setup": {
            "end_criteria": 0.0,
            "maximum_timesteps_configured": 1,
            "time_stepping_enabled": False,
        },
    })

    csx.Write2XML(str(output_dir / "geometry.xml"))

    print("\nMesh diagnostic summary")
    print("=======================")
    for axis in ("x", "y", "z"):
        item = report["axes"][axis]
        print(
            f"{axis.upper()}: {item['line_count']} lines, "
            f"{item['cell_count']} cells, "
            f"minimum spacing {item['minimum_spacing']:.9g} "
            f"at {item['minimum_spacing_location']}"
        )
    print(f"Total mesh cells: {report['total_cells']:,}")
    print(f"NF2FF box created: {report['nf2ff_created']}")
    print("Starting openEMS setup-only validation (zero FDTD timesteps)...")

    original_cwd = Path.cwd()
    try:
        # openEMS performs its normal mesh/operator setup and emits its
        # native mesh-size and timestep diagnostics, but setup_only=True
        # prevents the time-stepping loop from starting.
        fdtd.Run(
            sim_path=str(output_dir),
            setup_only=True,
            verbose=2,
        )
    finally:
        os.chdir(original_cwd)

    report["status"] = "setup_validated"
    report_path = output_dir / "mesh_diagnostic.json"
    report_path.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"\nDiagnostic report: {report_path}")
    print("Completed without running FDTD timesteps.")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a design mesh and validate openEMS setup without FDTD time stepping."
    )
    parser.add_argument("config", help="Path to the design YAML file.")
    parser.add_argument("--fstart", type=float, default=2.5,
                        help="Lower frequency in GHz (default: 2.5).")
    parser.add_argument("--fstop", type=float, default=3.0,
                        help="Upper frequency in GHz (default: 3.0).")
    parser.add_argument("--output", default="mesh_diagnostic_2p5_3p0",
                        help="New output directory (must not already exist).")
    parser.add_argument("--no-nf2ff", action="store_true",
                        help="Skip NF2FF creation (not recommended for this validation).")
    args = parser.parse_args(argv)

    try:
        run_mesh_diagnostic(
            args.config,
            args.output,
            args.fstart,
            args.fstop,
            enable_nf2ff=not args.no_nf2ff,
        )
    except Exception as exc:
        print(f"Mesh diagnostic failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
