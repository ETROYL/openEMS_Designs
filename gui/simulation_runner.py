"""Run one standalone openEMS FDTD simulation from a YAML design."""
from __future__ import annotations
import argparse
from pathlib import Path
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from design import build_design, get_simulation_settings, load_config
from parallel import NUM_THREADS

def _unique_output_dir(output_parent: Path, design_name: str) -> Path:
    output_parent.mkdir(parents=True, exist_ok=True)
    safe_name = Path(design_name).name or "design"
    output_dir = output_parent / f"{safe_name}_fdtd"
    suffix = 1
    while output_dir.exists():
        output_dir = output_parent / f"{safe_name}_fdtd_{suffix:02d}"
        suffix += 1
    return output_dir


def run_multiband_simulation(config_path: Path, output_parent: Path, design_name: str) -> Path:
    """Run an automatically planned multi-band simulation with the accurate profile."""
    cfg = load_config(str(config_path))
    simulation = get_simulation_settings(cfg, "simulation")
    fstart = float(simulation["fstart"])
    fstop = float(simulation["fstop"])
    n_points = int(simulation.get("n_pts", 1024))
    if fstart <= 0 or fstop <= fstart:
        raise ValueError("simulation.fstart/fstop must define a positive increasing frequency range")

    output_dir = _unique_output_dir(output_parent, design_name)
    snapshot_path = output_dir / "design.yaml"
    print(f"Design: {design_name}", flush=True)
    print("Simulation engine: multi-band", flush=True)
    print(f"Simulation profile: accurate (-40 dB, 1,000,000 timesteps, {n_points:,} points)", flush=True)
    print(f"Frequency range: {fstart:g}–{fstop:g} GHz", flush=True)
    print(f"OpenMP threads per band: {NUM_THREADS}", flush=True)
    print("Preparing deterministic equal-width frequency intervals…", flush=True)

    from multiband_fdtd import run_multiband_auto
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = run_multiband_auto(
        config_path=config_path,
        output_dir=output_dir,
        frequency_range_ghz=(fstart, fstop),
        profile="accurate",
        n_points=n_points,
        threads=NUM_THREADS,
        record_far_field=True,
    )
    snapshot_path.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

    figure, s11_axis = plt.subplots(figsize=(12, 7))
    s11_axis.plot(result["frequency_ghz"], result["s11_db"], linewidth=1.8)
    s11_axis.axhline(-10.0, linestyle="--", linewidth=1.0, label="-10 dB")
    s11_axis.axhline(-20.0, linestyle=":", linewidth=1.0, label="-20 dB")
    s11_axis.set_xlabel("Frequency (GHz)")
    s11_axis.set_ylabel("|S11| (dB)")
    s11_axis.set_title("S11 magnitude")
    s11_axis.grid(True, alpha=0.3)
    s11_axis.legend()
    figure.tight_layout()
    figure.savefig(Path(result["output_dir"]) / "s11_plot.png", dpi=180)
    plt.close(figure)

    production_dir = output_dir / "simulation"
    print(f"Output directory: {production_dir}", flush=True)
    print(f"SIMULATION_OUTPUT_DIR={production_dir.resolve()}", flush=True)
    print(f"SIMULATION_CONFIG={snapshot_path.resolve()}", flush=True)
    print(f"Selected intervals: {len(result['bands'])}", flush=True)
    print(f"Total production FDTD runtime: {result['total_runtime_seconds']:.2f} s", flush=True)
    print(f"Broadband response: {production_dir / 'broadband_response.csv'}", flush=True)
    print("Far-field recording is enabled for the production bands; full analysis is available in Results.", flush=True)
    print("Multi-band FDTD simulation completed successfully.", flush=True)
    return output_dir




def run_simulation(config_path: Path, output_parent: Path, design_name: str) -> Path:
    cfg = load_config(str(config_path))
    simulation = get_simulation_settings(cfg, "simulation")
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

    output_dir = _unique_output_dir(output_parent, design_name)
    output_dir.mkdir()
    snapshot_path = output_dir / "design.yaml"
    snapshot_path.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

    f0, fc = (fstart + fstop) / 2.0, fstop - fstart
    print(f"Design: {design_name}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print(f"SIMULATION_OUTPUT_DIR={output_dir.resolve()}", flush=True)
    print(f"SIMULATION_CONFIG={snapshot_path.resolve()}", flush=True)
    print(f"Frequency range: {fstart / 1e9:g}–{fstop / 1e9:g} GHz", flush=True)
    if "end_criteria_db" in simulation:
        print(f"Energy-decay target: {float(simulation['end_criteria_db']):g} dB", flush=True)
    print(f"openEMS EndCriteria (linear energy ratio): {end_criteria:g}", flush=True)
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
    parser.add_argument("--multiband", action="store_true", help="Use automatic multi-band execution with the accurate profile")
    args = parser.parse_args()
    runner = run_multiband_simulation if args.multiband else run_simulation
    runner(args.config.resolve(), args.output_parent.resolve(), args.design_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
