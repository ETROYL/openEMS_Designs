"""Run one standalone openEMS FDTD simulation from a YAML design."""
from __future__ import annotations
import argparse
from pathlib import Path
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from design import build_design, load_config
from parallel import NUM_THREADS


def run_simulation(config_path: Path, output_parent: Path, design_name: str) -> Path:
    cfg = load_config(str(config_path))
    simulation = cfg.get("simulation", {})
    fstart = float(simulation["fstart"]) * 1e9
    fstop = float(simulation["fstop"]) * 1e9
    if fstart <= 0 or fstop <= fstart:
        raise ValueError("simulation.fstart/fstop must define a positive increasing frequency range")
    end_criteria = float(simulation.get("end_criteria", 1e-4))
    max_timesteps = int(simulation.get("max_timesteps", 500000))
    if not 0 < end_criteria < 1:
        raise ValueError("simulation.end_criteria must be between 0 and 1")
    if max_timesteps <= 0:
        raise ValueError("simulation.max_timesteps must be positive")

    output_parent.mkdir(parents=True, exist_ok=True)
    safe_name = Path(design_name).name or "design"
    output_dir = output_parent / f"{safe_name}_fdtd"
    suffix = 1
    while output_dir.exists():
        output_dir = output_parent / f"{safe_name}_fdtd_{suffix:02d}"
        suffix += 1
    output_dir.mkdir()

    f0, fc = (fstart + fstop) / 2.0, fstop - fstart
    print(f"Design: {design_name}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print(f"Frequency range: {fstart / 1e9:g}–{fstop / 1e9:g} GHz", flush=True)
    print(f"End criteria: {end_criteria:g}", flush=True)
    print(f"Maximum timesteps: {max_timesteps}", flush=True)
    print(f"OpenMP threads: {NUM_THREADS}", flush=True)
    print("Building CSXCAD model and mesh…", flush=True)

    csx = ContinuousStructure()
    fdtd = openEMS(EndCriteria=end_criteria, NrTS=max_timesteps)
    fdtd.SetCSX(csx)
    fdtd.SetGaussExcite(f0, fc)
    fdtd.SetBoundaryCond(["PML_8"] * 6)
    build_design(csx, fdtd, config_file=str(config_path), enable_nf2ff=True)

    xml_filename = "geometry.xml"
    csx.Write2XML(str(output_dir / xml_filename))
    print(f"Geometry written: {output_dir / xml_filename}", flush=True)
    print("Starting openEMS FDTD solver…", flush=True)
    fdtd.Run(str(output_dir), xml_filename, verbose=0, numThreads=NUM_THREADS)
    print("FDTD simulation completed successfully.", flush=True)
    print(f"Simulation data: {output_dir}", flush=True)
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output_parent", type=Path)
    parser.add_argument("design_name")
    args = parser.parse_args()
    run_simulation(args.config.resolve(), args.output_parent.resolve(), args.design_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
