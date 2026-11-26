"""GUI-independent multi-band openEMS simulation and response reconstruction."""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

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
              end_criteria: float, max_timesteps: int, threads: int) -> BandResult:
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
    build_design(csx, fdtd, config_file=str(band_yaml), enable_nf2ff=False)
    cells = _count_mesh_cells(csx)
    csx.Write2XML(str(output_dir / "geometry.xml"))

    started = time.perf_counter()
    # Match the invocation used by the validated study benchmark.
    fdtd.Run(str(output_dir), numThreads=threads)
    elapsed = time.perf_counter() - started

    # Recreate the port object for frequency-domain extraction.
    post_csx = ContinuousStructure()
    post_fdtd = openEMS(EndCriteria=end_criteria, NrTS=max_timesteps)
    post_fdtd.SetCSX(post_csx)
    post_fdtd.SetGaussExcite(f0, fc)
    port, _ = build_design(
        post_csx, post_fdtd, config_file=str(band_yaml), enable_nf2ff=False
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
    s11 = _smooth_complex(s_sum / w_sum, smoothing_sigma_points)
    z_in = _smooth_complex(z_sum / w_sum, smoothing_sigma_points)
    return {
        "freq": target,
        "frequency_ghz": target / 1e9,
        "s11": s11,
        "s11_db": 20 * np.log10(np.maximum(np.abs(s11), 1e-15)),
        "z_in": z_in,
    }


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
    threads: int = 1,
    smoothing_sigma_points: float = 0.0,
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
    if max_ts <= 0 or sample_count < 2 or threads < 1:
        raise ValueError("Invalid max_timesteps, n_points or threads.")
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
    }
    (root / "multiband_run.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    results = []
    for index, band in enumerate(normalized, start=1):
        results.append(_run_band(
            config, root / f"band_{index:02d}", band,
            end_linear, max_ts, threads,
        ))
    combined = reconstruct_response(
        results, requested, sample_count,
        smoothing_sigma_points=smoothing_sigma_points,
    )
    np.savez_compressed(root / "broadband_response.npz", **combined)
    np.savetxt(
        root / "broadband_response.csv",
        np.column_stack((combined["frequency_ghz"], combined["s11"].real,
                         combined["s11"].imag, combined["s11_db"],
                         combined["z_in"].real, combined["z_in"].imag)),
        delimiter=",",
        header="frequency_GHz,s11_real,s11_imag,s11_dB,z_in_real_ohm,z_in_imag_ohm",
        comments="",
    )
    manifest["total_runtime_seconds"] = sum(item.runtime_seconds for item in results)
    manifest["band_results"] = [
        {"runtime_seconds": item.runtime_seconds, "mesh_cells": item.cells,
         "output_dir": item.output_dir} for item in results
    ]
    (root / "multiband_run.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return {
        **combined, "bands": results,
        "total_runtime_seconds": manifest["total_runtime_seconds"],
        "output_dir": str(root), "profile": profile,
    }
