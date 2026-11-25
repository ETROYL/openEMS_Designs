"""Generate unique unexplored candidates directly on the discrete parameter grid."""
from __future__ import annotations

from math import ceil, log2
from typing import Iterable

import numpy as np
from scipy.stats import qmc

from .parameter_space import DiscreteParameterSpace, ParameterSpaceError
from .surrogate import GaussianProcessSurrogate


def discrete_candidate_pool(
    space: DiscreteParameterSpace,
    n_candidates: int,
    excluded: Iterable[tuple[int, ...]] = (),
    seed: int | None = 42,
) -> list[tuple[int, ...]]:
    """Generate unique legal integer-grid candidates not present in excluded.

    A scrambled Sobol sequence supplies space-filling points in the unit
    hypercube; each coordinate is mapped directly to one integer grid index.
    This is discrete candidate generation, not continuous optimization
    followed by post-hoc geometry quantization.
    """
    if space.dimension == 0:
        raise ParameterSpaceError("At least one optimized parameter is required")
    if isinstance(n_candidates, bool) or not isinstance(n_candidates, int) or n_candidates <= 0:
        raise ParameterSpaceError("n_candidates must be a positive integer")

    levels = np.asarray(space.levels(), dtype=np.int64)
    blocked: set[tuple[int, ...]] = set()
    for raw_point in excluded:
        point = tuple(raw_point)
        if (len(point) != space.dimension
                or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in point)):
            raise ParameterSpaceError(f"Excluded point {point!r} must contain one integer index per dimension")
        normalized = tuple(int(v) for v in point)
        if any(v < 0 or v >= levels[i] for i, v in enumerate(normalized)):
            raise ParameterSpaceError(f"Excluded point {normalized!r} is outside this parameter space")
        blocked.add(normalized)
    available = space.cardinality - len(blocked)
    if n_candidates > available:
        raise ParameterSpaceError(
            f"Requested {n_candidates} candidates, but only {available} unexplored points remain"
        )

    engine = qmc.Sobol(d=space.dimension, scramble=True, seed=seed)
    selected: list[tuple[int, ...]] = []
    seen = set(blocked)
    batch_power = max(5, ceil(log2(max(n_candidates * 4, 2))))
    for _ in range(24):
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
                if len(selected) == n_candidates:
                    return selected

    # Deterministic seeded fallback for small or heavily blocked spaces.
    rng = np.random.default_rng(seed)
    while len(selected) < n_candidates:
        point = tuple(int(rng.integers(0, level)) for level in levels)
        if point not in seen:
            seen.add(point)
            selected.append(point)
    return selected


def select_next_candidates(
    space: DiscreteParameterSpace,
    surrogate: GaussianProcessSurrogate,
    best_objective: float,
    observed: Iterable[tuple[int, ...]],
    n_candidates: int = 1,
    pool_size: int = 4096,
    xi: float = 0.01,
    seed: int | None = 42,
) -> list[tuple[tuple[int, ...], float]]:
    """Return the top EI-ranked unexplored candidates from a Sobol pool."""
    if isinstance(pool_size, bool) or not isinstance(pool_size, int) or pool_size <= 0:
        raise ParameterSpaceError("pool_size must be a positive integer")
    if isinstance(n_candidates, bool) or not isinstance(n_candidates, int) or n_candidates <= 0:
        raise ParameterSpaceError("n_candidates must be a positive integer")
    if n_candidates > pool_size:
        raise ParameterSpaceError("n_candidates cannot exceed pool_size")
    observed_set = set()
    for raw_point in observed:
        point = tuple(raw_point)
        if (len(point) != space.dimension
                or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in point)):
            raise ParameterSpaceError(f"Observed point {point!r} must contain integer indices")
        observed_set.add(tuple(int(v) for v in point))
    pool = discrete_candidate_pool(
        space, pool_size, excluded=observed_set, seed=seed
    )
    return surrogate.rank_candidates(
        pool,
        best_objective=best_objective,
        xi=xi,
        observed=observed_set,
    )[:n_candidates]
