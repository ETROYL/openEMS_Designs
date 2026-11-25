"""Lightweight Gaussian-process surrogate and expected-improvement acquisition.

The model is intentionally independent of openEMS and operates on normalized
integer-grid coordinates. Candidate generation remains discrete; this module
never creates or quantizes continuous parameter values.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import ndtr


def _as_2d(values: Sequence[Sequence[int]] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("X must be a non-empty 2D array")
    if not np.all(np.isfinite(array)):
        raise ValueError("X must contain only finite values")
    return array


def expected_improvement(
    mean: Sequence[float] | np.ndarray,
    std: Sequence[float] | np.ndarray,
    best_objective: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Expected improvement for a minimization objective (larger is better)."""
    mean_array = np.asarray(mean, dtype=float)
    std_array = np.asarray(std, dtype=float)
    if mean_array.shape != std_array.shape:
        raise ValueError("mean and std must have matching shapes")
    if xi < 0:
        raise ValueError("xi must be non-negative")
    improvement = float(best_objective) - mean_array - float(xi)
    safe_std = np.maximum(std_array, 1e-15)
    z = improvement / safe_std
    ei = improvement * ndtr(z) + safe_std * np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    return np.where(std_array > 1e-14, np.maximum(ei, 0.0), np.maximum(improvement, 0.0))


@dataclass
class Prediction:
    mean: np.ndarray
    std: np.ndarray


class GaussianProcessSurrogate:
    """Gaussian process with a Matérn-5/2 ARD kernel and fixed hyperparameters.

    Inputs are integer lattice coordinates scaled independently to [0, 1].
    Targets are standardized internally for numerical stability. A small
    diagonal noise term makes the factorization robust to near-duplicate data.
    """

    def __init__(
        self,
        length_scale: float | Sequence[float] = 0.25,
        signal_variance: float = 1.0,
        noise: float = 1e-6,
    ) -> None:
        if signal_variance <= 0 or noise <= 0:
            raise ValueError("signal_variance and noise must be positive")
        self.length_scale = length_scale
        self.signal_variance = float(signal_variance)
        self.noise = float(noise)
        self._fitted = False

    @staticmethod
    def _kernel(x1: np.ndarray, x2: np.ndarray, length_scale: np.ndarray, variance: float) -> np.ndarray:
        scaled = (x1[:, None, :] - x2[None, :, :]) / length_scale
        distance = np.sqrt(np.sum(scaled * scaled, axis=2))
        root5_r = np.sqrt(5.0) * distance
        return variance * (1.0 + root5_r + (5.0 / 3.0) * distance * distance) * np.exp(-root5_r)

    def fit(
        self,
        indices: Sequence[Sequence[int]] | np.ndarray,
        objectives: Sequence[float] | np.ndarray,
        levels: Sequence[int],
    ) -> "GaussianProcessSurrogate":
        x = _as_2d(indices)
        y = np.asarray(objectives, dtype=float).reshape(-1)
        level_array = np.asarray(levels, dtype=float).reshape(-1)
        if x.shape[0] != y.size:
            raise ValueError("X and objectives must contain the same number of observations")
        if x.shape[1] != level_array.size:
            raise ValueError("levels must contain one entry per parameter dimension")
        if np.any(level_array < 1) or not np.all(np.isfinite(level_array)):
            raise ValueError("each level count must be finite and >= 1")
        if not np.all(np.isfinite(y)):
            raise ValueError("objectives must contain only finite values")
        if np.any(x < 0) or np.any(x > (level_array - 1) + 1e-12):
            raise ValueError("training coordinates are outside the discrete parameter space")

        self.levels_ = level_array.astype(int)
        denominator = np.maximum(level_array - 1.0, 1.0)
        self.x_train_ = x / denominator
        self.y_mean_ = float(np.mean(y))
        self.y_scale_ = float(np.std(y))
        if self.y_scale_ < 1e-12:
            self.y_scale_ = 1.0
        self.y_train_ = (y - self.y_mean_) / self.y_scale_

        length_scale = np.asarray(self.length_scale, dtype=float)
        if length_scale.ndim == 0:
            length_scale = np.full(x.shape[1], float(length_scale))
        if length_scale.shape != (x.shape[1],) or np.any(length_scale <= 0):
            raise ValueError("length_scale must be positive and scalar or one value per dimension")
        self.length_scale_ = length_scale

        covariance = self._kernel(
            self.x_train_, self.x_train_, self.length_scale_, self.signal_variance
        )
        covariance.flat[:: covariance.shape[0] + 1] += self.noise
        jitter = self.noise
        for _ in range(6):
            try:
                self.cholesky_ = np.linalg.cholesky(covariance)
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
                covariance.flat[:: covariance.shape[0] + 1] += jitter
        else:
            raise np.linalg.LinAlgError("Unable to factorize GP covariance matrix")
        intermediate = solve_triangular(self.cholesky_, self.y_train_, lower=True)
        self.alpha_ = solve_triangular(self.cholesky_.T, intermediate, lower=False)
        self._fitted = True
        return self

    def predict(self, indices: Sequence[Sequence[int]] | np.ndarray) -> Prediction:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        x = _as_2d(indices)
        if x.shape[1] != self.levels_.size:
            raise ValueError("prediction points have the wrong dimension")
        if np.any(x < 0) or np.any(x > (self.levels_ - 1) + 1e-12):
            raise ValueError("prediction coordinates are outside the discrete parameter space")
        x_scaled = x / np.maximum(self.levels_ - 1.0, 1.0)
        cross = self._kernel(x_scaled, self.x_train_, self.length_scale_, self.signal_variance)
        mean_scaled = cross @ self.alpha_
        projected = solve_triangular(self.cholesky_, cross.T, lower=True)
        variance_scaled = np.maximum(self.signal_variance - np.sum(projected * projected, axis=0), 0.0)
        mean = self.y_mean_ + self.y_scale_ * mean_scaled
        std = self.y_scale_ * np.sqrt(variance_scaled)
        return Prediction(mean=mean, std=std)

    def score_expected_improvement(
        self,
        candidates: Sequence[Sequence[int]] | np.ndarray,
        best_objective: float,
        xi: float = 0.01,
    ) -> np.ndarray:
        prediction = self.predict(candidates)
        return expected_improvement(prediction.mean, prediction.std, best_objective, xi=xi)

    def rank_candidates(
        self,
        candidates: Sequence[Sequence[int]] | np.ndarray,
        best_objective: float,
        xi: float = 0.01,
        observed: set[tuple[int, ...]] | None = None,
    ) -> list[tuple[tuple[int, ...], float]]:
        """Rank supplied legal integer-grid candidates by EI, excluding observations."""
        points = _as_2d(candidates)
        integer_points = points.astype(np.int64)
        if not np.allclose(points, integer_points):
            raise ValueError("candidates must be integer-grid coordinates")
        keys = [tuple(int(v) for v in row) for row in integer_points]
        if len(set(keys)) != len(keys):
            raise ValueError("candidate pool must not contain duplicates")
        observed = observed or set()
        keep = [i for i, key in enumerate(keys) if key not in observed]
        if not keep:
            return []
        scores = self.score_expected_improvement(integer_points[keep], best_objective, xi=xi)
        ranked = sorted(
            ((keys[i], float(score)) for i, score in zip(keep, scores)),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked
