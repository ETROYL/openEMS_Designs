"""GUI-independent multi-band openEMS simulation and response reconstruction."""
from __future__ import annotations

import json
import math
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .resources import allocate_band_resources

import numpy as np
import yaml


@dataclass(frozen=True)
class FrequencyBand:
    fstart_ghz: float
    fstop_ghz: float
    n_pts: int = 101

    def validate(self) -> None:
        if not (math.isfinite(self.fstart_ghz) and math.isfinite(self.fstop_ghz)):
            raise ValueError("Band frequencies must be finite.")
        if self.fstart_ghz <= 0 or self.fstop_ghz <= self.fstart_ghz:
            raise ValueError("Each band must have 0 < fstart < fstop.")
        if self.n_pts < 2:
            raise ValueError("Each band requires at least two frequency points.")



def build_equal_width_frequency_bands(
    frequency_range_ghz: tuple[float, float],
    n_points: int,
    *,
    maximum_bandwidth_ghz: float = 0.5,
    overlap_fraction: float = 0.2,
) -> list[FrequencyBand]:
    """Build a deterministic interval plan with controlled adjacent overlap.

    overlap_fraction is measured against the maximum interval bandwidth.
    The final interval may be narrower than the maximum width; it still
    reaches the requested upper frequency without changing the overlap step.
    n_points defines broadband extraction-grid density; each interval
    receives enough samples to preserve that density independently.
    """
    fstart, fstop = map(float, frequency_range_ghz)
    if not (math.isfinite(fstart) and math.isfinite(fstop)
            and 0 < fstart < fstop):
        raise ValueError("Frequency range must satisfy 0 < fstart < fstop.")
    if not isinstance(n_points, int) or n_points < 2:
        raise ValueError("n_points must be an integer of at least 2.")
    if not math.isfinite(maximum_bandwidth_ghz) or maximum_bandwidth_ghz <= 0:
        raise ValueError("maximum_bandwidth_ghz must be finite and positive.")
    if not math.isfinite(overlap_fraction) or not 0 <= overlap_fraction < 1:
        raise ValueError("overlap_fraction must be finite and in [0, 1).")

    span = fstop - fstart
    width = min(maximum_bandwidth_ghz, span)
    step = width * (1.0 - overlap_fraction)
    starts = [fstart]
    tolerance = 1e-12
    while starts[-1] + width < fstop - tolerance:
        next_start = starts[-1] + step
        if next_start <= starts[-1] + tolerance:
            break
        starts.append(next_start)

    grid_spacing = span / (n_points - 1)
    bands = []
    for low in starts:
        high = min(fstop, low + width)
        points = max(2, int(math.ceil((high - low) / grid_spacing - 1e-12)) + 1)
        bands.append(FrequencyBand(low, high, points))
    return bands


class PilotMeshLimitExceeded(RuntimeError):
    """Raised when a pilot interval is unsafe to simulate."""

    def __init__(self, message: str, cells: int | None = None):
        super().__init__(message)
        self.cells = cells


@dataclass
class BandResult:
    band: FrequencyBand
    frequency_hz: np.ndarray
    s11: np.ndarray
    z_in: np.ndarray
    runtime_seconds: float
    cells: int | None
    output_dir: str


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError("The design YAML root must be a mapping.")
    return config


def _write_band_yaml(source: dict[str, Any], path: Path, band: FrequencyBand,
                     end_criteria: float, max_timesteps: int) -> None:
    config = yaml.safe_load(yaml.safe_dump(source))
    config.setdefault("simulation", {})
    config["simulation"].update({
        "fstart": band.fstart_ghz, "fstop": band.fstop_ghz,
        "n_pts": band.n_pts, "end_criteria": end_criteria,
        "max_timesteps": max_timesteps,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)


def _count_mesh_cells(csx: Any) -> int | None:
    try:
        grid = csx.GetGrid()
        counts = [len(np.asarray(grid.GetLines(axis)).reshape(-1)) - 1
                  for axis in ("x", "y", "z")]
        return int(np.prod(counts))
    except Exception:
        return None


def _run_band(config_path: Path, output_dir: Path, band: FrequencyBand,
              end_criteria: float, max_timesteps: int, threads: int,
              max_mesh_cells: int | None = None,
              active_vars: Mapping[str, Any] | None = None,
              record_far_field: bool = False) -> BandResult:
    os.environ["OMP_NUM_THREADS"] = str(threads)
    from CSXCAD import ContinuousStructure
    from openEMS import openEMS
    from design import build_design

    output_dir.mkdir(parents=True, exist_ok=False)
    band_yaml = output_dir / "design.yaml"
    _write_band_yaml(_read_yaml(config_path), band_yaml, band,
                     end_criteria, max_timesteps)

    fstart = band.fstart_ghz * 1e9
    fstop = band.fstop_ghz * 1e9
    f0, fc = 0.5 * (fstart + fstop), fstop - fstart

    csx = ContinuousStructure()
    fdtd = openEMS(EndCriteria=end_criteria, NrTS=max_timesteps)
    fdtd.SetCSX(csx)
    fdtd.SetGaussExcite(f0, fc)
    fdtd.SetBoundaryCond(["PML_8"] * 6)
    build_design(
        csx, fdtd, config_file=str(band_yaml), active_vars=active_vars,
        enable_nf2ff=record_far_field,
    )
    cells = _count_mesh_cells(csx)
    if max_mesh_cells is not None:
        if cells is None:
            raise PilotMeshLimitExceeded(
                "Mesh cell count could not be determined; pilot skipped."
            )
        if cells > max_mesh_cells:
            raise PilotMeshLimitExceeded(
                f"Mesh contains {cells:,} cells, exceeding the "
                f"{max_mesh_cells:,}-cell pilot limit.",
                cells=cells,
            )
    csx.Write2XML(str(output_dir / "geometry.xml"))

    started = time.perf_counter()
    # The solver may change the process working directory. Keep all later
    # paths stable for callers that use relative paths.
    original_cwd = Path.cwd()
    try:
        # Match the invocation used by the validated study benchmark.
        fdtd.Run(str(output_dir), numThreads=threads)
    finally:
        os.chdir(original_cwd)
    elapsed = time.perf_counter() - started

    # Recreate the port object for frequency-domain extraction.
    post_csx = ContinuousStructure()
    post_fdtd = openEMS(EndCriteria=end_criteria, NrTS=max_timesteps)
    post_fdtd.SetCSX(post_csx)
    post_fdtd.SetGaussExcite(f0, fc)
    port, _ = build_design(
        post_csx, post_fdtd, config_file=str(band_yaml),
        active_vars=active_vars, enable_nf2ff=record_far_field,
    )
    frequency_hz = np.linspace(fstart, fstop, band.n_pts)
    port.CalcPort(str(output_dir), frequency_hz)
    s11 = np.asarray(port.uf_ref) / np.asarray(port.uf_inc)
    z_in = np.asarray(port.uf_tot) / np.asarray(port.if_tot)

    np.savez_compressed(output_dir / "response.npz",
                        frequency_hz=frequency_hz, s11=s11, z_in=z_in)
    metadata = {
        "band": asdict(band), "end_criteria_linear": end_criteria,
        "max_timesteps": max_timesteps, "threads": threads,
        "runtime_seconds": elapsed, "mesh_cells": cells,
        "response_file": "response.npz",
        "far_field_recorded": bool(record_far_field),
    }
    (output_dir / "band_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return BandResult(band, frequency_hz, s11, z_in, elapsed, cells, str(output_dir))


def _smooth_complex(values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return values.copy()
    radius = max(1, int(math.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()

    def smooth(part: np.ndarray) -> np.ndarray:
        padded = np.pad(part, (radius, radius), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    return smooth(values.real) + 1j * smooth(values.imag)


def reconstruct_response(
    band_results: Iterable[BandResult],
    frequency_range_ghz: tuple[float, float],
    n_points: int,
    *,
    smoothing_sigma_points: float = 0.0,
) -> dict[str, np.ndarray]:
    """Blend complex responses across overlaps on a uniform frequency grid."""
    results = sorted(band_results, key=lambda result: result.band.fstart_ghz)
    if not results:
        raise ValueError("At least one band result is required.")
    fmin, fmax = map(float, frequency_range_ghz)
    if not (0 < fmin < fmax) or n_points < 2 or smoothing_sigma_points < 0:
        raise ValueError("Invalid reconstruction range, point count or smoothing.")
    if results[0].band.fstart_ghz > fmin + 1e-12:
        raise ValueError("Bands do not cover the requested lower frequency.")
    if results[-1].band.fstop_ghz < fmax - 1e-12:
        raise ValueError("Bands do not cover the requested upper frequency.")
    for left, right in zip(results, results[1:]):
        if right.band.fstart_ghz > left.band.fstop_ghz + 1e-12:
            raise ValueError("Frequency bands contain a gap.")

    target = np.linspace(fmin * 1e9, fmax * 1e9, n_points)
    s_sum = np.zeros(n_points, dtype=complex)
    z_sum = np.zeros(n_points, dtype=complex)
    w_sum = np.zeros(n_points, dtype=float)

    for index, result in enumerate(results):
        low, high = result.band.fstart_ghz * 1e9, result.band.fstop_ghz * 1e9
        mask = (target >= low - 1e-6) & (target <= high + 1e-6)
        weight = np.ones(n_points, dtype=float)
        if index > 0:
            overlap_low = max(results[index - 1].band.fstart_ghz,
                              result.band.fstart_ghz) * 1e9
            overlap_high = min(results[index - 1].band.fstop_ghz,
                               result.band.fstop_ghz) * 1e9
            if overlap_high > overlap_low:
                t = np.clip((target - overlap_low) / (overlap_high - overlap_low), 0, 1)
                weight *= np.sin(0.5 * np.pi * t) ** 2
        if index + 1 < len(results):
            overlap_low = max(result.band.fstart_ghz,
                              results[index + 1].band.fstart_ghz) * 1e9
            overlap_high = min(result.band.fstop_ghz,
                               results[index + 1].band.fstop_ghz) * 1e9
            if overlap_high > overlap_low:
                t = np.clip((target - overlap_low) / (overlap_high - overlap_low), 0, 1)
                weight *= np.cos(0.5 * np.pi * t) ** 2
        weight *= mask
        s_interp = (np.interp(target, result.frequency_hz, result.s11.real)
                    + 1j * np.interp(target, result.frequency_hz, result.s11.imag))
        z_interp = (np.interp(target, result.frequency_hz, result.z_in.real)
                    + 1j * np.interp(target, result.frequency_hz, result.z_in.imag))
        s_sum += weight * s_interp
        z_sum += weight * z_interp
        w_sum += weight

    if np.any(w_sum <= 0):
        raise ValueError("The bands leave uncovered reconstruction samples.")
    s11_raw = s_sum / w_sum
    z_in_raw = z_sum / w_sum
    s11 = _smooth_complex(s11_raw, smoothing_sigma_points)
    z_in = _smooth_complex(z_in_raw, smoothing_sigma_points)
    return {
        "freq": target,
        "frequency_ghz": target / 1e9,
        "s11_raw": s11_raw,
        "z_in_raw": z_in_raw,
        "s11": s11,
        "s11_db": 20 * np.log10(np.maximum(np.abs(s11), 1e-15)),
        "z_in": z_in,
    }


def _run_band_task(task):
    """Picklable process-pool entry point for one independent frequency interval."""
    (
        index, config_path, band_dir, band, end_criteria, timestep_limit,
        thread_count, variables, far_field,
    ) = task
    thread_value = str(int(thread_count))
    os.environ["EM_THREADS"] = thread_value
    os.environ["OMP_NUM_THREADS"] = thread_value
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    result = _run_band(
        config_path, band_dir, band, end_criteria, timestep_limit,
        thread_count, active_vars=variables, record_far_field=far_field,
    )
    return index, result


def run_multiband(
    config_path: str | Path,
    output_dir: str | Path,
    bands: Iterable[FrequencyBand | tuple],
    *,
    profile: str = "optimization",
    frequency_range_ghz: tuple[float, float] | None = None,
    n_points: int | None = None,
    end_criteria_db: float | None = None,
    max_timesteps: int | None = None,
    threads: int | None = None,
    workers: int | None = None,
    smoothing_sigma_points: float = 0.0,
    active_vars: Mapping[str, Any] | None = None,
    record_far_field: bool = False,
) -> dict[str, Any]:
    """Run multi-band FDTD and return standard freq/S11/impedance arrays.

    Optimization and accurate profiles share the same execution path; they
    differ only in default convergence, timestep and sampling settings.
    """
    profiles = {
        "optimization": (-15.0, 200000, 301),
        "accurate": (-40.0, 1000000, 1024),
    }
    if profile not in profiles:
        raise ValueError("profile must be 'optimization' or 'accurate'.")
    default_db, default_steps, default_points = profiles[profile]
    end_db = default_db if end_criteria_db is None else float(end_criteria_db)
    max_ts = default_steps if max_timesteps is None else int(max_timesteps)
    sample_count = default_points if n_points is None else int(n_points)
    if not math.isfinite(end_db) or end_db >= 0:
        raise ValueError("end_criteria_db must be finite and negative.")
    if max_ts <= 0 or sample_count < 2:
        raise ValueError("Invalid max_timesteps or n_points.")
    if smoothing_sigma_points < 0:
        raise ValueError("smoothing_sigma_points cannot be negative.")
    end_linear = 10.0 ** (end_db / 10.0)

    normalized = [item if isinstance(item, FrequencyBand) else FrequencyBand(*item)
                  for item in bands]
    for band in normalized:
        band.validate()
    normalized.sort(key=lambda band: band.fstart_ghz)
    if not normalized:
        raise ValueError("At least one frequency band is required.")
    for left, right in zip(normalized, normalized[1:]):
        if right.fstart_ghz > left.fstop_ghz + 1e-12:
            raise ValueError("Frequency bands must overlap or touch; gaps are not allowed.")
    requested = frequency_range_ghz or (
        normalized[0].fstart_ghz, normalized[-1].fstop_ghz
    )

    config = Path(config_path).expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "profile": profile, "frequency_range_ghz": list(requested),
        "end_criteria_db": end_db, "end_criteria_linear": end_linear,
        "max_timesteps": max_ts, "threads": threads,
        "smoothing_sigma_points": smoothing_sigma_points,
        "bands": [asdict(band) for band in normalized],
        "far_field_recorded": bool(record_far_field),
    }
    (root / "multiband_run.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    allocation = allocate_band_resources(len(normalized))
    if threads is not None:
        requested_threads = int(threads)
        if requested_threads < 2:
            raise ValueError("Each production FDTD interval requires at least two threads.")
        allocation_threads = min(allocation.threads_per_worker, requested_threads)
    else:
        allocation_threads = allocation.threads_per_worker
    worker_count = allocation.workers if workers is None else int(workers)
    if worker_count < 1 or worker_count > allocation.workers:
        raise ValueError(
            f"Requested interval workers ({worker_count}) exceed the available "
            f"allocation ({allocation.workers})."
        )
    if worker_count * allocation_threads > allocation.logical_cpus:
        raise ValueError("The requested interval allocation exceeds available logical CPUs.")

    manifest.update({
        "interval_workers": worker_count,
        "threads": allocation_threads,
        "threads_per_interval": allocation_threads,
        "logical_cpu_capacity": allocation.logical_cpus,
        "execution_batches": math.ceil(len(normalized) / worker_count),
    })
    (root / "multiband_run.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    results: list[BandResult | None] = [None] * len(normalized)
    tasks = [
        (
            index, config, root / f"band_{index + 1:02d}", band,
            end_linear, max_ts, allocation_threads, active_vars,
            record_far_field,
        )
        for index, band in enumerate(normalized)
    ]

    if worker_count == 1:
        for task in tasks:
            index, result = _run_band_task(task)
            results[index] = result
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
            futures = [executor.submit(_run_band_task, task) for task in tasks]
            for future in as_completed(futures):
                index, result = future.result()
                results[index] = result

    completed_results = [item for item in results if item is not None]
    combined = reconstruct_response(
        completed_results, requested, sample_count,
        smoothing_sigma_points=smoothing_sigma_points,
    )
    np.savez_compressed(root / "broadband_response.npz", **combined)
    np.savetxt(
        root / "broadband_response.csv",
        np.column_stack((combined["frequency_ghz"],
                         combined["s11_raw"].real, combined["s11_raw"].imag,
                         combined["s11"].real, combined["s11"].imag,
                         combined["s11_db"],
                         combined["z_in_raw"].real, combined["z_in_raw"].imag,
                         combined["z_in"].real, combined["z_in"].imag)),
        delimiter=",",
        header=("frequency_GHz,s11_raw_real,s11_raw_imag,s11_smoothed_real,"
                "s11_smoothed_imag,s11_smoothed_dB,z_in_raw_real,z_in_raw_imag,"
                "z_in_smoothed_real,z_in_smoothed_imag"),
        comments="",
    )
    manifest["total_runtime_seconds"] = sum(item.runtime_seconds for item in completed_results)
    manifest["band_results"] = [
        {"runtime_seconds": item.runtime_seconds, "mesh_cells": item.cells,
         "output_dir": item.output_dir} for item in completed_results
    ]
    (root / "multiband_run.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return {
        **combined, "bands": completed_results,
        "total_runtime_seconds": manifest["total_runtime_seconds"],
        "output_dir": str(root), "profile": profile,
        "interval_workers": worker_count,
        "threads_per_interval": allocation_threads,
        "logical_cpu_capacity": allocation.logical_cpus,
    }


def run_multiband_characterized(
    config_path: str | Path,
    output_dir: str | Path,
    frequency_range_ghz: tuple[float, float],
    *,
    profile: str = "optimization",
    interval_widths_ghz: Iterable[float] = (0.5, 1.0, 1.5),
    boundary_step_ghz: float = 0.25,
    pilot_n_pts: int = 41,
    max_candidates: int = 40,
    minimum_overlap_ghz: float = 0.1,
    pilot_end_criteria_db: float = -10.0,
    pilot_max_timesteps: int = 50000,
    pilot_threads: int = 1,
    max_pilot_mesh_cells: int | None = 2_000_000,
    n_points: int | None = None,
    end_criteria_db: float | None = None,
    max_timesteps: int | None = None,
    threads: int | None = None,
    smoothing_sigma_points: float = 0.0,
    plan_cache_dir: str | Path | None = None,
    record_far_field: bool = False,
) -> dict[str, Any]:
    """Prepare an interval plan, then run the requested production profile.

    Pilot measurements and production simulations are written to separate
    subdirectories and use independent solver settings. The selected interval
    sequence is fixed for the production call.
    """
    profiles = {
        "optimization": (-15.0, 200000, 301),
        "accurate": (-40.0, 1000000, 1024),
    }
    if profile not in profiles:
        raise ValueError("profile must be 'optimization' or 'accurate'.")
    default_db, default_steps, default_points = profiles[profile]
    production_db = default_db if end_criteria_db is None else float(end_criteria_db)
    production_steps = default_steps if max_timesteps is None else int(max_timesteps)
    production_points = default_points if n_points is None else int(n_points)
    if not math.isfinite(production_db) or production_db >= 0:
        raise ValueError("end_criteria_db must be finite and negative.")
    if production_steps <= 0 or production_points < 2:
        raise ValueError("Invalid production max_timesteps or n_points.")
    if smoothing_sigma_points < 0:
        raise ValueError("smoothing_sigma_points cannot be negative.")
    if pilot_threads < 1 or pilot_max_timesteps < 1:
        raise ValueError("Pilot max_timesteps and threads must be positive.")
    if (max_pilot_mesh_cells is not None
            and (not isinstance(max_pilot_mesh_cells, int)
                 or max_pilot_mesh_cells < 1)):
        raise ValueError("max_pilot_mesh_cells must be a positive integer or None.")
    if not math.isfinite(pilot_end_criteria_db) or pilot_end_criteria_db >= 0:
        raise ValueError("pilot_end_criteria_db must be finite and negative.")

    # Resolve the source before any solver invocation. openEMS may change
    # the process working directory while running a pilot.
    config = Path(config_path).expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(config)

    root = Path(output_dir).expanduser().resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=False, exist_ok=False)

    # Import locally to avoid a module cycle: characterization uses _run_band.
    from .characterization import characterize_and_select_intervals

    plan = characterize_and_select_intervals(
        config_path=config,
        output_dir=root / "characterization",
        frequency_range_ghz=frequency_range_ghz,
        interval_widths_ghz=interval_widths_ghz,
        boundary_step_ghz=boundary_step_ghz,
        n_pts=pilot_n_pts,
        max_candidates=max_candidates,
        minimum_overlap_ghz=minimum_overlap_ghz,
        end_criteria_db=pilot_end_criteria_db,
        max_timesteps=pilot_max_timesteps,
        threads=pilot_threads,
        max_pilot_mesh_cells=max_pilot_mesh_cells,
        cache_dir=(
            Path(plan_cache_dir).expanduser().resolve()
            if plan_cache_dir is not None
            else Path.home() / ".cache" / "etroyl" / "openems" / "interval-plans"
        ),
    )
    if plan.get("cache_hit"):
        print("Reusing a compatible interval plan from the local cache.", flush=True)
    else:
        print("No compatible cached interval plan; preparing intervals.", flush=True)
    pilot_bands = plan["selected_bands"]
    if not pilot_bands:
        raise RuntimeError("Interval preparation returned an empty selection.")

    # Keep the production extraction grid commensurate with the requested
    # broadband output spacing, rather than reusing the coarser pilot grid.
    fmin, fmax = map(float, frequency_range_ghz)
    broadband_step_ghz = (fmax - fmin) / (production_points - 1)
    selected_bands = [
        FrequencyBand(
            band.fstart_ghz,
            band.fstop_ghz,
            max(2, int(math.ceil(
                (band.fstop_ghz - band.fstart_ghz) / broadband_step_ghz
            )) + 1),
        )
        for band in pilot_bands
    ]

    production = run_multiband(
        config_path=config,
        output_dir=root / "simulation",
        bands=selected_bands,
        profile=profile,
        frequency_range_ghz=frequency_range_ghz,
        n_points=n_points,
        end_criteria_db=end_criteria_db,
        max_timesteps=max_timesteps,
        threads=threads,
        smoothing_sigma_points=smoothing_sigma_points,
        record_far_field=record_far_field,
    )
    orchestration_manifest = {
        "profile": profile,
        "frequency_range_ghz": list(map(float, frequency_range_ghz)),
        "planning_cache_hit": bool(plan.get("cache_hit", False)),
        "characterization_manifest": str(
            Path(plan["manifest_path"]).resolve()
        ),
        "selected_intervals": [
            {
                "fstart_ghz": band.fstart_ghz,
                "fstop_ghz": band.fstop_ghz,
                "n_pts": band.n_pts,
            }
            for band in selected_bands
        ],
        "production_output_dir": production["output_dir"],
        "pilot_settings": {
            "end_criteria_db": float(pilot_end_criteria_db),
            "max_timesteps": int(pilot_max_timesteps),
            "threads": int(pilot_threads),
            "n_pts": int(pilot_n_pts),
            "max_mesh_cells": max_pilot_mesh_cells,
        },
        "production_settings": {
            "end_criteria_db": production_db,
            "max_timesteps": production_steps,
            "n_points": production_points,
            "threads": int(threads),
            "smoothing_sigma_points": float(smoothing_sigma_points),
            "far_field_recorded": bool(record_far_field),
        },
    }
    manifest_path = root / "automatic_run.json"
    manifest_path.write_text(
        json.dumps(orchestration_manifest, indent=2), encoding="utf-8"
    )
    return {
        **production,
        "planning": plan,
        "selected_bands": selected_bands,
        "planning_manifest_path": plan["manifest_path"],
        "automatic_run_manifest_path": str(manifest_path),
        "automatic_output_dir": str(root),
    }


def run_multiband_auto(
    config_path: str | Path,
    output_dir: str | Path,
    frequency_range_ghz: tuple[float, float],
    *,
    profile: str = "optimization",
    n_points: int | None = None,
    end_criteria_db: float | None = None,
    max_timesteps: int | None = None,
    threads: int | None = None,
    workers: int | None = None,
    smoothing_sigma_points: float = 1.5,
    record_far_field: bool = False,
    maximum_bandwidth_ghz: float = 0.5,
    overlap_fraction: float = 0.2,
    **_legacy_planner_options,
) -> dict[str, Any]:
    """Run using a fast, deterministic overlapping interval plan.

    The legacy measured-planning implementation remains available through
    run_multiband_characterized(); normal execution performs no pilot FDTD runs.
    """
    profiles = {
        "optimization": (-15.0, 200000, 301),
        "accurate": (-40.0, 1000000, 1024),
    }
    if profile not in profiles:
        raise ValueError("profile must be 'optimization' or 'accurate'.")
    default_db, default_steps, default_points = profiles[profile]
    sample_count = default_points if n_points is None else int(n_points)
    bands = build_equal_width_frequency_bands(
        frequency_range_ghz, sample_count,
        maximum_bandwidth_ghz=maximum_bandwidth_ghz,
        overlap_fraction=overlap_fraction,
    )
    print(
        f"Prepared {len(bands)} overlapping interval(s) "
        f"(maximum {maximum_bandwidth_ghz:g} GHz, "
        f"{overlap_fraction:.0%} nominal overlap); no pilot simulations required.",
        flush=True,
    )
    root = Path(output_dir).expanduser().resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=False, exist_ok=False)
    production = run_multiband(
        config_path=config_path,
        output_dir=root / "simulation",
        bands=bands,
        profile=profile,
        frequency_range_ghz=frequency_range_ghz,
        n_points=sample_count,
        end_criteria_db=default_db if end_criteria_db is None else end_criteria_db,
        max_timesteps=default_steps if max_timesteps is None else max_timesteps,
        threads=threads,
        workers=workers,
        smoothing_sigma_points=smoothing_sigma_points,
        record_far_field=record_far_field,
    )
    plan_manifest = {
        "profile": profile,
        "frequency_range_ghz": list(map(float, frequency_range_ghz)),
        "selected_intervals": [asdict(band) for band in bands],
        "interval_settings": {
            "maximum_bandwidth_ghz": float(maximum_bandwidth_ghz),
            "overlap_fraction": float(overlap_fraction),
        },
        "production_output_dir": production["output_dir"],
        "production_settings": {
            "n_points": sample_count,
            "end_criteria_db": default_db if end_criteria_db is None else float(end_criteria_db),
            "max_timesteps": default_steps if max_timesteps is None else int(max_timesteps),
            "threads": production.get("threads_per_interval", threads),
            "interval_workers": production.get("interval_workers", workers),
            "logical_cpu_capacity": production.get("logical_cpu_capacity"),
            "smoothing_sigma_points": float(smoothing_sigma_points),
            "far_field_recorded": bool(record_far_field),
        },
    }
    (root / "automatic_run.json").write_text(
        json.dumps(plan_manifest, indent=2), encoding="utf-8"
    )
    return {
        **production,
        "interval_plan": [asdict(band) for band in bands],
        "selected_bands": bands,
        "automatic_output_dir": str(root),
    }
