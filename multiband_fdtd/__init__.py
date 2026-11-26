"""Reusable multi-band FDTD simulation tools."""
from .engine import BandResult, FrequencyBand, reconstruct_response, run_multiband
from .planner import BandCost, select_fastest_band_sequence
from .characterization import (
    characterize_frequency_intervals, generate_candidate_bands,
    characterize_and_select_intervals,
)

__all__ = [
    "BandCost", "BandResult", "FrequencyBand", "reconstruct_response",
    "run_multiband", "select_fastest_band_sequence",
    "characterize_frequency_intervals", "generate_candidate_bands",
    "characterize_and_select_intervals",
]
