"""Small convex-hull certificates from scalar quadratic sufficient statistics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np


@dataclass(frozen=True)
class QuadraticSimplexProjection:
    """Convex-hull projection represented without materializing model-space vertices."""

    weights: np.ndarray
    distance: float
    squared_distance: float
    support: tuple[int, ...]
    simplex_residual: float
    minimum_weight: float


@dataclass(frozen=True)
class QuadraticDualHullCertificate:
    """Supporting-direction certificate recovered from scalar inner products."""

    vertex_witness_values: np.ndarray
    direction_norm: float
    lower_bound: float
    primal_distance: float
    primal_dual_gap: float


def _validate_statistics(
    gram: Any,
    cross: Any,
    target_norm_squared: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    q = np.asarray(gram, dtype=np.float64)
    c = np.asarray(cross, dtype=np.float64)
    target_squared = float(target_norm_squared)
    if c.ndim != 1 or c.size < 1 or q.shape != (c.size, c.size):
        raise ValueError("gram must be square with dimension matching cross")
    if not np.isfinite(q).all() or not np.isfinite(c).all() or not math.isfinite(target_squared):
        raise ValueError("quadratic hull statistics must be finite")
    if target_squared < 0.0:
        raise ValueError("target_norm_squared must be non-negative")
    if not np.allclose(q, q.T, rtol=1e-10, atol=1e-12):
        raise ValueError("gram must be symmetric")
    return q, c, target_squared


def _distance_squared(
    gram: np.ndarray,
    cross: np.ndarray,
    target_norm_squared: float,
    weights: np.ndarray,
) -> float:
    quadratic = float(weights @ gram @ weights)
    linear = float(2.0 * cross @ weights)
    squared = target_norm_squared - linear + quadratic
    scale = max(1.0, abs(target_norm_squared), abs(linear), abs(quadratic))
    tolerance = 1e-10 * scale
    if squared < -tolerance:
        raise FloatingPointError("convex-hull quadratic distance became materially negative")
    return max(0.0, squared)


def project_quadratic_simplex(
    gram: Any,
    cross: Any,
    target_norm_squared: float,
    *,
    weight_tolerance: float = 1e-10,
) -> QuadraticSimplexProjection:
    """Project onto a small candidate hull from ``Q Q^T``, ``Q y``, and ``||y||^2``."""

    q, c, target_squared = _validate_statistics(gram, cross, target_norm_squared)
    tolerance = float(weight_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("weight_tolerance must be finite and non-negative")
    if c.size > 20:
        raise ValueError("small quadratic simplex support enumeration is limited to 20 vertices")

    best: tuple[float, tuple[int, ...], np.ndarray] | None = None
    indices = tuple(range(c.size))
    for size in range(1, c.size + 1):
        for support in combinations(indices, size):
            selected = np.asarray(support, dtype=np.int64)
            local_gram = q[np.ix_(selected, selected)]
            local_cross = c[selected]
            ones = np.ones(size, dtype=np.float64)
            kkt = np.block(
                [
                    [local_gram, ones[:, None]],
                    [ones[None, :], np.zeros((1, 1), dtype=np.float64)],
                ]
            )
            rhs = np.concatenate([local_cross, np.ones(1, dtype=np.float64)])
            solved, *_ = np.linalg.lstsq(kkt, rhs, rcond=None)
            local_weights = solved[:size]
            if float(np.min(local_weights)) < -tolerance:
                continue
            local_weights = np.maximum(local_weights, 0.0)
            total = float(np.sum(local_weights))
            if not math.isfinite(total) or total <= 0.0:
                continue
            local_weights = local_weights / total
            weights = np.zeros(c.size, dtype=np.float64)
            weights[selected] = local_weights
            squared = _distance_squared(q, c, target_squared, weights)
            candidate = (squared, support, weights)
            if best is None or (squared, support) < (best[0], best[1]):
                best = candidate

    if best is None:
        raise FloatingPointError("failed to find a numerically feasible quadratic simplex support")

    squared, support, weights = best
    return QuadraticSimplexProjection(
        weights=weights,
        distance=math.sqrt(squared),
        squared_distance=squared,
        support=tuple(int(index) for index in support),
        simplex_residual=abs(float(np.sum(weights)) - 1.0),
        minimum_weight=float(np.min(weights)),
    )


def dual_quadratic_hull_certificate(
    gram: Any,
    cross: Any,
    target_norm_squared: float,
    projection: QuadraticSimplexProjection,
) -> QuadraticDualHullCertificate:
    """Certify a hull-distance lower bound without a model-space direction tensor."""

    q, c, target_squared = _validate_statistics(gram, cross, target_norm_squared)
    weights = np.asarray(projection.weights, dtype=np.float64)
    if weights.shape != c.shape or not np.isfinite(weights).all():
        raise ValueError("projection weights must match the candidate count and be finite")

    distance = math.sqrt(_distance_squared(q, c, target_squared, weights))
    zero_tolerance = 1e-12 * max(1.0, math.sqrt(target_squared), distance)
    if distance <= zero_tolerance:
        witness_values = np.zeros(c.size, dtype=np.float64)
        direction_norm = 0.0
        lower_bound = 0.0
    else:
        residual_dot_target = target_squared - float(c @ weights)
        residual_dot_vertices = c - q @ weights
        witness_values = (residual_dot_target - residual_dot_vertices) / distance
        lower_bound = max(0.0, float(np.min(witness_values)))
        direction_norm = 1.0

    return QuadraticDualHullCertificate(
        vertex_witness_values=np.asarray(witness_values, dtype=np.float64),
        direction_norm=direction_norm,
        lower_bound=lower_bound,
        primal_distance=distance,
        primal_dual_gap=distance - lower_bound,
    )
