"""Reusable multi-band FDTD simulation tools."""
from .engine import (
    BandResult, FrequencyBand, reconstruct_response, run_multiband,
    run_multiband_auto,
)
from .planner import BandCost, select_fastest_band_sequence
from .characterization import (
    characterize_frequency_intervals, generate_candidate_bands,
    characterize_and_select_intervals, load_interval_plan,
)

__all__ = [
    "BandCost", "BandResult", "FrequencyBand", "reconstruct_response",
    "run_multiband", "run_multiband_auto", "select_fastest_band_sequence",
    "characterize_frequency_intervals", "generate_candidate_bands",
    "characterize_and_select_intervals", "load_interval_plan",
]
