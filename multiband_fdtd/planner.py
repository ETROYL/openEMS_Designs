"""Measured-cost dynamic programming for adaptive frequency-band selection."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class BandCost:
    fstart_ghz: float
    fstop_ghz: float
    runtime_seconds: float
    mesh_cells: int | None = None
    timesteps: int | None = None

    def validate(self) -> None:
        values = (self.fstart_ghz, self.fstop_ghz, self.runtime_seconds)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Band boundaries and runtime must be finite.")
        if self.fstart_ghz <= 0 or self.fstop_ghz <= self.fstart_ghz:
            raise ValueError("A candidate band requires 0 < fstart < fstop.")
        if self.runtime_seconds <= 0:
            raise ValueError("Measured runtime must be positive.")


def select_fastest_band_sequence(
    candidates: Iterable[BandCost],
    frequency_range_ghz: tuple[float, float],
    *,
    minimum_overlap_ghz: float = 0.0,
) -> list[BandCost]:
    """Select the minimum-runtime measured intervals covering the target range.

    This selector uses measured wall-clock time, not an assumed linear
    frequency-to-mesh relationship. It only selects measured candidates and
    never extrapolates unmeasured simulation costs.
    """
    fmin, fmax = map(float, frequency_range_ghz)
    if not (0 < fmin < fmax) or minimum_overlap_ghz < 0:
        raise ValueError("Invalid requested range or minimum overlap.")
    items = list(candidates)
    for item in items:
        item.validate()
    items = [item for item in items
             if item.fstart_ghz <= fmax and item.fstop_ghz >= fmin]
    items.sort(key=lambda item: (item.fstop_ghz, item.fstart_ghz))
    if not items:
        raise ValueError("No candidate intervals intersect the requested range.")

    costs = [math.inf] * len(items)
    previous_index: list[int | None] = [None] * len(items)
    for i, current in enumerate(items):
        if current.fstart_ghz <= fmin + 1e-12:
            costs[i] = current.runtime_seconds
        for j in range(i):
            previous = items[j]
            overlap = previous.fstop_ghz - current.fstart_ghz
            extends_coverage = current.fstop_ghz > previous.fstop_ghz + 1e-12
            if (math.isfinite(costs[j]) and extends_coverage
                    and overlap >= minimum_overlap_ghz - 1e-12):
                proposed = costs[j] + current.runtime_seconds
                if proposed < costs[i]:
                    costs[i] = proposed
                    previous_index[i] = j

    terminals = [i for i, item in enumerate(items)
                 if item.fstop_ghz >= fmax - 1e-12 and math.isfinite(costs[i])]
    if not terminals:
        raise ValueError("Candidates cannot cover the full range with the required overlap.")
    cursor = min(terminals, key=lambda index: costs[index])
    selected = []
    while cursor is not None:
        selected.append(items[cursor])
        cursor = previous_index[cursor]
    selected.reverse()
    return selected
