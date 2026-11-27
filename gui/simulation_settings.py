"""Validated parsing for YAML-derived simulation settings."""
from __future__ import annotations

import math
import yaml


def _finite_number(value, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"simulation.{name} must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"simulation.{name} must be a finite number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"simulation.{name} must be a finite number.")
    return number


def _positive_integer(value, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"simulation.{name} must be a positive integer.")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"simulation.{name} must be a positive integer.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"simulation.{name} must be a positive integer.")
    if not isinstance(value, (int, float)) and str(value).strip() not in {str(number), f"+{number}"}:
        raise ValueError(f"simulation.{name} must be a positive integer.")
    if number <= 0:
        raise ValueError(f"simulation.{name} must be a positive integer.")
    return number


def parse_simulation_settings(yaml_text: str) -> dict:
    """Parse and validate the complete simulation block without partial updates."""
    try:
        config = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML syntax error: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("The YAML root must be a mapping/object.")

    simulation = config.get("simulation", {})
    if simulation is None:
        simulation = {}
    if not isinstance(simulation, dict):
        raise ValueError("simulation must be a YAML mapping/object.")

    fstart = simulation.get("fstart")
    fstop = simulation.get("fstop")
    if (fstart is None) != (fstop is None):
        raise ValueError("simulation.fstart and simulation.fstop must be provided together.")
    if fstart is not None:
        fstart = _finite_number(fstart, "fstart")
        fstop = _finite_number(fstop, "fstop")
        if fstart <= 0 or fstop <= fstart:
            raise ValueError("simulation.fstart/fstop must define a positive increasing frequency range.")

    n_pts = _positive_integer(simulation.get("n_pts", 301), "n_pts")
    max_timesteps = _positive_integer(simulation.get("max_timesteps", 500000), "max_timesteps")
    if "end_criteria_db" in simulation:
        decay_db = _finite_number(simulation["end_criteria_db"], "end_criteria_db")
        if decay_db >= 0:
            raise ValueError("simulation.end_criteria_db must be negative.")
    else:
        threshold = _finite_number(simulation.get("end_criteria", 1e-4), "end_criteria")
        if threshold <= 0:
            raise ValueError("simulation.end_criteria must be positive.")
        decay_db = 10.0 * math.log10(threshold)

    return {
        "fstart": fstart,
        "fstop": fstop,
        "n_pts": n_pts,
        "end_criteria_db": decay_db,
        "max_timesteps": max_timesteps,
        "legacy_end_criteria": simulation.get("end_criteria", 1e-4),
    }


def format_simulation_settings(yaml_text: str) -> str:
    """Format validated simulation settings for the Simulation tab."""
    settings = parse_simulation_settings(yaml_text)
    if settings["fstart"] is None:
        frequency_text = "—–—"
    else:
        frequency_text = f"{settings['fstart']:g}–{settings['fstop']:g}"
    if "end_criteria_db" in yaml.safe_load(yaml_text).get("simulation", {}):
        decay_text = f"{settings['end_criteria_db']:g} dB"
    else:
        decay_text = f"legacy linear threshold {settings['legacy_end_criteria']:g}"
    return (
        f"Frequency: {frequency_text} GHz    |    "
        f"Frequency points: {settings['n_pts']}\\n"
        f"Energy decay: {decay_text}    |    "
        f"Maximum timesteps: {settings['max_timesteps']}"
    )
