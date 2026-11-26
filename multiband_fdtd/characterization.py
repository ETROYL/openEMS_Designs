"""Utilities for collecting interval simulation data."""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Any

from .engine import FrequencyBand, _run_band
from .planner import BandCost


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
