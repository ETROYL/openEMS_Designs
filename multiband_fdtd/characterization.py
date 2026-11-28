"""Utilities for collecting interval simulation data."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable, Any

from .engine import FrequencyBand, PilotMeshLimitExceeded, _run_band
from .planner import BandCost, select_fastest_band_sequence


def generate_candidate_bands(
    frequency_range_ghz: tuple[float, float],
    *,
    interval_widths_ghz: Iterable[float] = (0.5, 1.0, 1.5),
    boundary_step_ghz: float = 0.25,
    n_pts: int = 41,
    max_candidates: int = 40,
) -> list[FrequencyBand]:
    """Generate a bounded, deterministic set of intervals for measurement.

    Candidate boundaries are placed on a uniform grid. Requested widths are
    used at each grid position and clipped at the upper frequency limit.
    The complete requested range is deliberately excluded: pilot runs should
    use bounded intervals rather than a potentially expensive broadband mesh.
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
            if stop - start > 1e-12 and not (
                math.isclose(start, fmin, rel_tol=0.0, abs_tol=1e-12)
                and math.isclose(stop, fmax, rel_tol=0.0, abs_tol=1e-12)
            ):
                key = (round(start, 12), round(stop, 12))
                unique[key] = FrequencyBand(start, stop, n_pts)

    candidates = sorted(unique.values(), key=lambda band: (
        band.fstart_ghz, band.fstop_ghz
    ))

    if len(candidates) > max_candidates:
        # Evenly retain candidates from the ordered set. Keep the low- and
        # high-frequency edge candidates represented without adding a
        # full-range fallback.
        if max_candidates == 1:
            selected = [candidates[0]]
        else:
            indexes = {
                round(index * (len(candidates) - 1) / (max_candidates - 1))
                for index in range(max_candidates)
            }
            selected = [candidates[index] for index in sorted(indexes)]
        candidates = sorted(selected, key=lambda band: (
            band.fstart_ghz, band.fstop_ghz
        ))

    return candidates


def characterize_and_select_intervals(
    config_path: str | Path,
    output_dir: str | Path,
    frequency_range_ghz: tuple[float, float],
    *,
    interval_widths_ghz: Iterable[float] = (0.5, 1.0, 1.5),
    boundary_step_ghz: float = 0.25,
    n_pts: int = 41,
    max_candidates: int = 40,
    minimum_overlap_ghz: float = 0.1,
    end_criteria_db: float = -10.0,
    max_timesteps: int = 50000,
    threads: int = 1,
    max_pilot_mesh_cells: int | None = 2_000_000,
    stop_requested: Callable[[], bool] | None = None,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Measure generated intervals and return a reusable complete interval plan.

    The generated candidate set is bounded by max_candidates. Oversized
    candidates are skipped before the FDTD solver starts. The selected
    sequence, skipped candidates and settings are saved in the manifest.
    """
    if not math.isfinite(minimum_overlap_ghz) or minimum_overlap_ghz < 0:
        raise ValueError("minimum_overlap_ghz must be finite and non-negative.")
    if (max_pilot_mesh_cells is not None
            and (not isinstance(max_pilot_mesh_cells, int)
                 or max_pilot_mesh_cells < 1)):
        raise ValueError("max_pilot_mesh_cells must be a positive integer or None.")
    widths = tuple(float(width) for width in interval_widths_ghz)
    candidates = generate_candidate_bands(
        frequency_range_ghz,
        interval_widths_ghz=widths,
        boundary_step_ghz=boundary_step_ghz,
        n_pts=n_pts,
        max_candidates=max_candidates,
    )

    cache_fingerprint = None
    cache_path = None
    if cache_dir is not None:
        cache_fingerprint = _interval_plan_fingerprint(
            config_path,
            frequency_range_ghz=frequency_range_ghz,
            interval_widths_ghz=widths,
            boundary_step_ghz=boundary_step_ghz,
            n_pts=n_pts,
            max_candidates=max_candidates,
            minimum_overlap_ghz=minimum_overlap_ghz,
            end_criteria_db=end_criteria_db,
            max_timesteps=max_timesteps,
            threads=threads,
            max_pilot_mesh_cells=max_pilot_mesh_cells,
        )
        if cache_fingerprint is not None:
            cache_path = Path(cache_dir).expanduser().resolve() / (
                cache_fingerprint + ".json"
            )
            cached = _load_cached_interval_plan(
                cache_path, config_path=config_path,
                fingerprint=cache_fingerprint, n_pts=n_pts,
            )
            if cached is not None:
                root = Path(output_dir).expanduser().resolve()
                root.mkdir(parents=True, exist_ok=False)
                manifest_path = root / "characterization.json"
                manifest = cached["manifest"]
                manifest["cache_hit"] = True
                manifest["cached_from"] = str(cache_path)
                for interval in manifest.get("intervals", []):
                    interval["output_dir"] = None
                manifest_path.write_text(
                    json.dumps(manifest, indent=2), encoding="utf-8"
                )
                return {
                    "records": cached["records"],
                    "manifest": manifest,
                    "manifest_path": str(manifest_path),
                    "output_dir": str(root),
                    "candidates": candidates,
                    "selected": cached["selected"],
                    "selected_bands": cached["selected_bands"],
                    "selection": cached["selection"],
                    "cache_hit": True,
                }

    result = characterize_frequency_intervals(
        config_path,
        output_dir,
        candidates,
        end_criteria_db=end_criteria_db,
        max_timesteps=max_timesteps,
        threads=threads,
        max_pilot_mesh_cells=max_pilot_mesh_cells,
        stop_requested=stop_requested,
    )
    selected = select_fastest_band_sequence(
        result["records"],
        frequency_range_ghz,
        minimum_overlap_ghz=minimum_overlap_ghz,
    )
    selection = {
        "frequency_range_ghz": list(map(float, frequency_range_ghz)),
        "minimum_overlap_ghz": float(minimum_overlap_ghz),
        "candidate_count": len(candidates),
        "selected_intervals": [
            {
                "fstart_ghz": item.fstart_ghz,
                "fstop_ghz": item.fstop_ghz,
                "runtime_seconds": item.runtime_seconds,
                "mesh_cells": item.mesh_cells,
            }
            for item in selected
        ],
        "total_measured_runtime_seconds": sum(
            item.runtime_seconds for item in selected
        ),
    }
    result["manifest"]["candidate_generation"] = {
        "interval_widths_ghz": sorted(set(widths)),
        "boundary_step_ghz": float(boundary_step_ghz),
        "n_pts": int(n_pts),
        "max_candidates": int(max_candidates),
        "max_pilot_mesh_cells": max_pilot_mesh_cells,
    }
    result["manifest"]["selection"] = selection
    result["manifest"]["cache_hit"] = False
    Path(result["manifest_path"]).write_text(
        json.dumps(result["manifest"], indent=2), encoding="utf-8"
    )
    if cache_path is not None and cache_fingerprint is not None:
        _store_cached_interval_plan(cache_path, result["manifest"])
    candidate_by_pair = {
        (round(item.fstart_ghz, 12), round(item.fstop_ghz, 12)): item
        for item in candidates
    }
    selected_bands = [
        candidate_by_pair[(round(item.fstart_ghz, 12), round(item.fstop_ghz, 12))]
        for item in selected
    ]
    result["candidates"] = candidates
    result["selected"] = selected
    result["selected_bands"] = selected_bands
    result["selection"] = selection
    result["cache_hit"] = False
    return result


def _interval_plan_fingerprint(
    config_path: str | Path,
    *,
    frequency_range_ghz: tuple[float, float],
    interval_widths_ghz: Iterable[float],
    boundary_step_ghz: float,
    n_pts: int,
    max_candidates: int,
    minimum_overlap_ghz: float,
    end_criteria_db: float,
    max_timesteps: int,
    threads: int,
    max_pilot_mesh_cells: int | None,
) -> str | None:
    """Return a cache key, or None when the design source cannot be identified."""
    config = Path(config_path).expanduser().resolve()
    if not config.is_file():
        return None
    try:
        spec = importlib.util.find_spec("design")
        source_path = Path(spec.origin).resolve() if spec and spec.origin else None
        if source_path is None or not source_path.is_file():
            return None
        source_files = {
            "design": source_path,
            "characterization": Path(__file__).resolve(),
            "planner": Path(__file__).with_name("planner.py").resolve(),
            "engine": Path(__file__).with_name("engine.py").resolve(),
        }
        if any(not path.is_file() for path in source_files.values()):
            return None
        source_digests = {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in source_files.items()
        }
    except (ImportError, OSError, ValueError):
        return None

    payload = {
        "cache_schema": 2,
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "source_sha256": source_digests,
        "frequency_range_ghz": list(map(float, frequency_range_ghz)),
        "interval_widths_ghz": sorted(set(map(float, interval_widths_ghz))),
        "boundary_step_ghz": float(boundary_step_ghz),
        "pilot_n_pts": int(n_pts),
        "max_candidates": int(max_candidates),
        "minimum_overlap_ghz": float(minimum_overlap_ghz),
        "pilot_end_criteria_db": float(end_criteria_db),
        "pilot_max_timesteps": int(max_timesteps),
        "pilot_threads": int(threads),
        "max_pilot_mesh_cells": max_pilot_mesh_cells,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load_cached_interval_plan(
    cache_path: Path,
    *,
    config_path: str | Path,
    fingerprint: str,
    n_pts: int,
) -> dict[str, Any] | None:
    """Load a cache entry only when its key and plan structure are valid."""
    if not cache_path.is_file():
        return None
    try:
        manifest = json.loads(cache_path.read_text(encoding="utf-8"))
        if (not isinstance(manifest, dict)
                or manifest.get("cache_fingerprint") != fingerprint
                or manifest.get("plan_version") != 1
                or not isinstance(manifest.get("selection"), dict)):
            return None
        selected_bands = load_interval_plan(
            cache_path, config_path=config_path
        )
        selected = [
            BandCost(
                float(item["fstart_ghz"]),
                float(item["fstop_ghz"]),
                float(item["runtime_seconds"]),
                item.get("mesh_cells"),
                None,
            )
            for item in manifest["selection"]["selected_intervals"]
        ]
        records = [
            BandCost(
                float(item["fstart_ghz"]),
                float(item["fstop_ghz"]),
                float(item["runtime_seconds"]),
                item.get("mesh_cells"),
                None,
            )
            for item in manifest.get("intervals", [])
        ]
        if not selected or not records:
            return None
        if any(band.n_pts != n_pts for band in selected_bands):
            return None
        return {
            "manifest": manifest,
            "selected_bands": selected_bands,
            "selected": selected,
            "records": records,
            "selection": manifest["selection"],
        }
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _store_cached_interval_plan(cache_path: Path, manifest: dict[str, Any]) -> None:
    """Atomically persist a completed plan; partial cache files are never visible."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(manifest)
    payload["cache_fingerprint"] = cache_path.stem
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=cache_path.parent,
            prefix=f".{cache_path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            json.dump(payload, stream, indent=2)
            temporary_path = Path(stream.name)
        os.replace(temporary_path, cache_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()



def load_interval_plan(
    plan_path: str | Path,
    *,
    n_points: int | None = None,
    config_path: str | Path | None = None,
) -> list[FrequencyBand]:
    """Load and validate a previously measured interval plan.

    If n_points is supplied, each interval receives enough extraction samples
    to match the requested uniform broadband output spacing. The interval
    boundaries remain unchanged, so the plan stays frozen across evaluations.
    When config_path is supplied, its content hash must match the design used
    to prepare the plan.
    """
    path = Path(plan_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)

    if not isinstance(manifest, dict) or manifest.get("plan_version") != 1:
        raise ValueError("Unsupported or missing interval plan version.")
    selection = manifest.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("The file does not contain a selected interval plan.")
    frequency_range = selection.get("frequency_range_ghz")
    if (not isinstance(frequency_range, list) or len(frequency_range) != 2):
        raise ValueError("The interval plan has an invalid frequency range.")
    fmin, fmax = map(float, frequency_range)
    if not (math.isfinite(fmin) and math.isfinite(fmax) and 0 < fmin < fmax):
        raise ValueError("The interval plan has an invalid frequency range.")

    intervals = selection.get("selected_intervals")
    if not isinstance(intervals, list) or not intervals:
        raise ValueError("The interval plan contains no selected intervals.")

    if config_path is not None:
        config = Path(config_path).expanduser().resolve()
        if not config.is_file():
            raise FileNotFoundError(config)
        expected = manifest.get("config_sha256")
        actual = hashlib.sha256(config.read_bytes()).hexdigest()
        if not expected or actual != expected:
            raise ValueError(
                "The interval plan was prepared from a different design configuration."
            )

    if n_points is not None and (
        not isinstance(n_points, int) or n_points < 2
    ):
        raise ValueError("n_points must be an integer of at least 2.")

    generation = manifest.get("candidate_generation")
    if generation is not None and not isinstance(generation, dict):
        raise ValueError("The interval plan has invalid candidate-generation settings.")
    pilot_points = (generation or {}).get("n_pts", 41)
    if not isinstance(pilot_points, int) or pilot_points < 2:
        raise ValueError("The interval plan has invalid pilot sampling settings.")
    equal_total_points = (generation or {}).get("sampling_mode") == "equal_total_points"

    bands = []
    for item in intervals:
        try:
            low = float(item["fstart_ghz"])
            high = float(item["fstop_ghz"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("The interval plan contains an invalid interval.") from exc
        if equal_total_points:
            samples = int(item.get("n_pts", 0))
            if samples < 2:
                raise ValueError("The deterministic interval plan has invalid point allocation.")
        else:
            broadband_step = (
                (fmax - fmin) / (n_points - 1) if n_points is not None else None
            )
            samples = (
                max(2, int(math.ceil((high - low) / broadband_step)) + 1)
                if broadband_step is not None else pilot_points
            )
        band = FrequencyBand(low, high, samples)
        band.validate()
        bands.append(band)

    bands.sort(key=lambda band: band.fstart_ghz)
    if bands[0].fstart_ghz > fmin + 1e-12:
        raise ValueError("The interval plan does not cover the lower frequency limit.")
    if bands[-1].fstop_ghz < fmax - 1e-12:
        raise ValueError("The interval plan does not cover the upper frequency limit.")
    for left, right in zip(bands, bands[1:]):
        if right.fstart_ghz > left.fstop_ghz + 1e-12:
            raise ValueError("The interval plan contains a frequency gap.")
    return bands


def characterize_frequency_intervals(
    config_path: str | Path,
    output_dir: str | Path,
    candidate_bands: Iterable[FrequencyBand | tuple],
    *,
    end_criteria_db: float = -10.0,
    max_timesteps: int = 50000,
    threads: int = 1,
    max_pilot_mesh_cells: int | None = 2_000_000,
    stop_requested: Callable[[], bool] | None = None,
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

    if (max_pilot_mesh_cells is not None
            and (not isinstance(max_pilot_mesh_cells, int)
                 or max_pilot_mesh_cells < 1)):
        raise ValueError("max_pilot_mesh_cells must be a positive integer or None.")

    records: list[BandCost] = []
    details = []
    skipped = []
    for index, band in enumerate(bands, start=1):
        if stop_requested is not None and stop_requested():
            raise InterruptedError(
                "Interval preparation interrupted by a stop or pause request."
            )
        try:
            result = _run_band(
                config, root / f"pilot_{index:03d}", band,
                end_linear, int(max_timesteps), int(threads),
                max_mesh_cells=max_pilot_mesh_cells,
            )
        except PilotMeshLimitExceeded as exc:
            skipped.append({
                "fstart_ghz": band.fstart_ghz,
                "fstop_ghz": band.fstop_ghz,
                "n_pts": band.n_pts,
                "mesh_cells": exc.cells,
                "reason": str(exc),
                "output_dir": str(root / f"pilot_{index:03d}"),
            })
            continue
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
    if not records:
        raise RuntimeError(
            "No pilot candidates were simulated successfully. "
            "Increase max_pilot_mesh_cells or review the design mesh."
        )

    config_digest = hashlib.sha256(config.read_bytes()).hexdigest()
    manifest = {
        "plan_version": 1,
        "config_path": str(config),
        "config_sha256": config_digest,
        "reference_geometry": "YAML configured default variable values",
        "end_criteria_db": float(end_criteria_db),
        "max_timesteps": int(max_timesteps),
        "threads": int(threads),
        "max_pilot_mesh_cells": max_pilot_mesh_cells,
        "intervals": details,
        "skipped_intervals": skipped,
    }
    manifest_path = root / "characterization.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "records": records,
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "output_dir": str(root),
    }
