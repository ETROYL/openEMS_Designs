"""Compatibility checks for continuing an existing optimization run."""
from __future__ import annotations

import math
from typing import Any


_SETTING_LABELS = {
    "fstart": "start frequency",
    "fstop": "stop frequency",
    "n_pts": "number of frequency points",
    "end_criteria_db": "energy-decay criterion",
    "max_timesteps": "maximum timesteps",
}


def _normalized_settings(settings: dict[str, Any]) -> dict[str, float | int]:
    if not isinstance(settings, dict):
        raise ValueError("The saved or current simulation settings are not a mapping.")

    normalized: dict[str, float | int] = {}
    for key in ("fstart", "fstop", "n_pts", "max_timesteps"):
        if key not in settings:
            raise ValueError(f"Simulation settings are missing {key!r}.")
        value = settings[key]
        normalized[key] = int(value) if key in ("n_pts", "max_timesteps") else float(value)

    if "end_criteria_db" in settings:
        normalized["end_criteria_db"] = float(settings["end_criteria_db"])
    elif "end_criteria" in settings:
        linear = float(settings["end_criteria"])
        if linear <= 0:
            raise ValueError("The linear energy-decay criterion must be positive.")
        normalized["end_criteria_db"] = 10.0 * math.log10(linear)
    else:
        raise ValueError("Simulation settings are missing the energy-decay criterion.")
    return normalized


def find_incompatible_settings(
    run_settings: dict[str, Any],
    *,
    current_simulation_settings: dict[str, Any],
    current_evaluation_settings: dict[str, Any] | None,
    current_use_multiband: bool,
) -> list[str]:
    """Return labels for effective settings that differ from the saved run."""
    saved_uses_yaml = bool(run_settings.get("use_simulation_parameters", True))
    saved_source = (
        run_settings.get("simulation_settings")
        if saved_uses_yaml else run_settings.get("evaluation_settings")
    )
    current_source = (
        current_simulation_settings
        if current_evaluation_settings is None else current_evaluation_settings
    )
    saved = _normalized_settings(saved_source)
    current = _normalized_settings(current_source)

    mismatches = []
    for key, label in _SETTING_LABELS.items():
        old, new = saved[key], current[key]
        if isinstance(old, int) and isinstance(new, int):
            equal = old == new
        else:
            equal = math.isclose(float(old), float(new), rel_tol=1e-9, abs_tol=1e-9)
        if not equal:
            mismatches.append(label)

    if bool(run_settings.get("use_multiband", False)) != bool(current_use_multiband):
        mismatches.append("simulation engine")
    return mismatches
