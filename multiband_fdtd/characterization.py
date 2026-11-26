"""Utilities for collecting interval simulation data."""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Any

from .engine import FrequencyBand, _run_band
from .planner import BandCost


def generate_candidate_bands(
    frequency_range_ghz: tuple[float, float],
    *,
    interval_widths_ghz: Iterable[float] = (0.5, 1.0, 1.5),
    boundary_step_ghz: float = 0.25,
    n_pts: int = 41,
    max_candidates: int = 80,
) -> list[FrequencyBand]:
    """Generate a bounded, deterministic set of intervals for measurement.

    Candidate boundaries are placed on a uniform grid. Requested widths are
    used at each grid position and clipped at the upper frequency limit.
    The complete requested range is always included as a candidate.
    """
    if len(frequency_range_ghz) != 2:
        raise ValueError("frequency_range_ghz must contain two values.")
    fmin, fmax = map(float, frequency_range_ghz)
    if not (math.isfinite(fmin) and math.isfinite(fmax) and 0 < fmin < fmax):
        raise ValueError("The requested range must satisfy 0 < fmin < fmax.")
    if not math.isfinite(boundary_step_ghz) or boundary_step_ghz <= 0:
        raise ValueError("boundary_step_ghz must be finite and positive.")
    if n_pts < 2 or max_candidates < 1:
        raise ValueError("n_pts must be at least 2 and max_candidates positive.")

    widths = sorted({float(width) for width in interval_widths_ghz})
    if not widths or any(not math.isfinite(width) or width <= 0 for width in widths):
        raise ValueError("interval_widths_ghz must contain positive finite widths.")

    span = fmax - fmin
    step_count = int(math.floor(span / boundary_step_ghz))
    starts = [fmin + index * boundary_step_ghz for index in range(step_count + 1)]
    if not math.isclose(starts[-1], fmax, rel_tol=0.0, abs_tol=1e-12):
        starts.append(fmax)

    unique: dict[tuple[float, float], FrequencyBand] = {}
    for start in starts:
        for width in widths:
            stop = min(start + width, fmax)
            if stop - start > 1e-12:
                key = (round(start, 12), round(stop, 12))
                unique[key] = FrequencyBand(start, stop, n_pts)

    full_key = (round(fmin, 12), round(fmax, 12))
    unique[full_key] = FrequencyBand(fmin, fmax, n_pts)
    candidates = sorted(unique.values(), key=lambda band: (
        band.fstart_ghz, band.fstop_ghz
    ))

    if len(candidates) > max_candidates:
        # Evenly retain candidates from the ordered set, while preserving the
        # complete-range candidate needed as a guaranteed coverage fallback.
        indexes = {
            round(index * (len(candidates) - 1) / (max_candidates - 1))
            for index in range(max_candidates)
        } if max_candidates > 1 else set()
        selected = [candidates[index] for index in sorted(indexes)]
        if full_key not in {
            (round(item.fstart_ghz, 12), round(item.fstop_ghz, 12))
            for item in selected
        }:
            selected[-1] = unique[full_key]
        candidates = sorted(selected, key=lambda band: (
            band.fstart_ghz, band.fstop_ghz
        ))

    return candidates


def characterize_frequency_intervals(
    config_path: str | Path,
    output_dir: str | Path,
    candidate_bands: Iterable[FrequencyBand | tuple],
    *,
    end_criteria_db: float = -10.0,
    max_timesteps: int = 50000,
    threads: int = 1,
) -> dict[str, Any]:
    """Run short reference-design simulations for an explicit interval set.

    The YAML design's configured default variable values define the reference
    geometry. Results are measurement records only; they do not alter the
    source design or start a production simulation.
    """
    if not math.isfinite(end_criteria_db) or end_criteria_db >= 0:
        raise ValueError("end_criteria_db must be finite and negative.")
    if max_timesteps < 1 or threads < 1:
        raise ValueError("max_timesteps and threads must be positive.")

    bands = [
        item if isinstance(item, FrequencyBand) else FrequencyBand(*item)
        for item in candidate_bands
    ]
    if not bands:
        raise ValueError("At least one candidate interval is required.")
    for band in bands:
        band.validate()

    config = Path(config_path).expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    end_linear = 10.0 ** (float(end_criteria_db) / 10.0)

    records: list[BandCost] = []
    details = []
    for index, band in enumerate(bands, start=1):
        result = _run_band(
            config, root / f"pilot_{index:03d}", band,
            end_linear, int(max_timesteps), int(threads),
        )
        record = BandCost(
            band.fstart_ghz, band.fstop_ghz,
            result.runtime_seconds, result.cells, None,
        )
        records.append(record)
        details.append({
            **asdict(record),
            "n_pts": band.n_pts,
            "output_dir": result.output_dir,
        })

    manifest = {
        "config_path": str(config),
        "reference_geometry": "YAML configured default variable values",
        "end_criteria_db": float(end_criteria_db),
        "max_timesteps": int(max_timesteps),
        "threads": int(threads),
        "intervals": details,
    }
    manifest_path = root / "characterization.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "records": records,
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "output_dir": str(root),
    }
