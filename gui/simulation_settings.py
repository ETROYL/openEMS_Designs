"""Parsing and display helpers for YAML-derived simulation settings."""
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
    number = _finite_number(value, name)
    if number <= 0 or not number.is_integer():
        raise ValueError(f"simulation.{name} must be a positive integer.")
    return int(number)


def _load_simulation(yaml_text: str) -> dict:
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
    return simulation


def _validated_frequency_range(simulation: dict) -> tuple[float | None, float | None]:
    fstart = simulation.get("fstart")
    fstop = simulation.get("fstop")
    if (fstart is None) != (fstop is None):
        raise ValueError("simulation.fstart and simulation.fstop must be provided together.")
    if fstart is None:
        return None, None
    fstart = _finite_number(fstart, "fstart")
    fstop = _finite_number(fstop, "fstop")
    if fstart <= 0 or fstop <= fstart:
        raise ValueError(
            "simulation.fstart/fstop must define a positive increasing frequency range."
        )
    return fstart, fstop


def parse_simulation_settings(yaml_text: str) -> dict:
    """Parse all solver values used to initialise optimization evaluation controls."""
    simulation = _load_simulation(yaml_text)
    fstart, fstop = _validated_frequency_range(simulation)

    n_pts = _positive_integer(simulation.get("n_pts", 301), "n_pts")
    max_timesteps = _positive_integer(
        simulation.get("max_timesteps", 500000), "max_timesteps"
    )
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
        "uses_db_criterion": "end_criteria_db" in simulation,
    }


def format_simulation_settings(yaml_text: str) -> str:
    """Format the Simulation tab using only the settings displayed there.

    Solver fields unrelated to the frequency summary are displayed as authored;
    they must not prevent a valid frequency-range edit from reaching the GUI.
    """
    simulation = _load_simulation(yaml_text)
    fstart, fstop = _validated_frequency_range(simulation)
    if fstart is None:
        frequency_text = "—–—"
    else:
        frequency_text = f"{fstart:g}–{fstop:g}"

    if "end_criteria_db" in simulation:
        decay_text = f"{simulation['end_criteria_db']} dB"
    else:
        decay_text = (
            "legacy linear threshold "
            + str(simulation.get("end_criteria", 1e-4))
        )
    return (
        f"Frequency: {frequency_text} GHz    |    "
        f"Frequency points: {simulation.get('n_pts', 301)}\n"
        f"Energy decay: {decay_text}    |    "
        f"Maximum timesteps: {simulation.get('max_timesteps', 500000)}"
    )
