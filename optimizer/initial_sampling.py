"""Space-filling initial designs for discrete parameter spaces."""
from __future__ import annotations
from math import ceil, log2
import numpy as np
from scipy.stats import qmc
from .parameter_space import DiscreteParameterSpace, ParameterSpaceError

def sobol_initial_design(space: DiscreteParameterSpace, n_samples: int, seed: int | None = 42) -> list[tuple[int, ...]]:
    """Return unique integer-grid points from a scrambled Sobol design."""
    if space.dimension == 0:
        raise ParameterSpaceError("At least one optimized parameter is required")
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples <= 0:
        raise ParameterSpaceError("n_samples must be a positive integer")
    if n_samples > space.cardinality:
        raise ParameterSpaceError(f"Requested {n_samples} samples from a space with only {space.cardinality} points")
    rng = np.random.default_rng(seed)
    levels = np.asarray(space.levels(), dtype=np.int64)
    if n_samples == space.cardinality:
        points = np.asarray(list(space.iter_indices()), dtype=np.int64)
        return [tuple(int(v) for v in points[i]) for i in rng.permutation(len(points))]
    engine = qmc.Sobol(d=space.dimension, scramble=True, seed=seed)
    selected: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    batch_power = max(5, ceil(log2(max(n_samples * 3, 2))))
    for _ in range(20):
        if engine.num_generated == 0:
            unit_points = engine.random_base2(batch_power)
        else:
            unit_points = engine.random(2 ** batch_power)
        mapped = np.minimum((unit_points * levels).astype(np.int64), levels - 1)
        for row in mapped:
            candidate = tuple(int(v) for v in row)
            if candidate not in seen:
                seen.add(candidate)
                selected.append(candidate)
                if len(selected) == n_samples:
                    return selected
    while len(selected) < n_samples:
        candidate = tuple(int(rng.integers(0, level)) for level in levels)
        if candidate not in seen:
            seen.add(candidate)
            selected.append(candidate)
    return selected
