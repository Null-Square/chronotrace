"""Convex-hull lower-bound certificates for chronology candidate classes.

The small-hull projector in this module is intended for diagnostic problems with only a
few vertices. Projection is *not* itself trusted as a certificate. Instead, a projected
point proposes a unit separating direction ``u`` and the independently recomputed value

    min_i <u, y - q_i>

is a rigorous lower bound on the Euclidean distance from ``y`` to the convex hull of the
vertices ``q_i``. This remains valid even for an inexact proposal direction.

A second guard separates statements about a degree-K truncated interaction model from
statements about the true endpoint geometry. A truncated lower bound can certify a true
candidate-class elimination only when an explicit norm bound on the omitted interaction
tail is supplied.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ConvexHullProjection:
    """Best primal point found in a small finite-vertex convex hull."""

    weights: np.ndarray
    point: np.ndarray
    distance: float
    support: tuple[int, ...]
    simplex_residual: float
    minimum_weight: float


@dataclass(frozen=True)
class DualHullCertificate:
    """Independently recomputed supporting-direction distance lower bound."""

    direction: np.ndarray
    direction_norm: float
    lower_bound: float
    primal_distance: float
    primal_dual_gap: float


@dataclass(frozen=True)
class TailAwareElimination:
    """Truncated-model and optional true-model candidate elimination verdicts."""

    lower_bound: float
    feasible_upper_bound: float
    numerical_guard: float
    truncated_eliminated: bool
    interaction_tail_radius: float | None
    exact_lower_bound: float | None
    exact_eliminated: bool | None


def _as_finite_vector(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size < 1:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _as_vertex_matrix(vertices: Any, target: np.ndarray) -> np.ndarray:
    matrix = np.asarray(vertices, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] != target.size:
        raise ValueError("vertices must be a non-empty matrix with target-sized rows")
    if not np.isfinite(matrix).all():
        raise ValueError("vertices must be finite")
    return matrix


def _support_weights(target: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Solve least squares on one affine support under ``sum(weights)=1``."""

    count = vertices.shape[0]
    gram = vertices @ vertices.T
    cross = vertices @ target
    ones = np.ones(count, dtype=np.float64)
    kkt = np.block(
        [
            [gram, ones[:, None]],
            [ones[None, :], np.zeros((1, 1), dtype=np.float64)],
        ]
    )
    rhs = np.concatenate([cross, np.ones(1, dtype=np.float64)])
    solution, *_ = np.linalg.lstsq(kkt, rhs, rcond=None)
    return solution[:count]


def project_onto_small_convex_hull(
    target: Any,
    vertices: Any,
    *,
    weight_tolerance: float = 1e-10,
) -> ConvexHullProjection:
    """Project onto a small convex hull by enumerating all non-empty supports.

    This is exponential in the number of vertices and is deliberately restricted to
    small diagnostic hulls. It is exact up to numerical least-squares error because the
    active support of a convex quadratic optimum is one of the enumerated subsets.
    """

    y = _as_finite_vector(target, name="target")
    q = _as_vertex_matrix(vertices, y)
    tolerance = float(weight_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("weight_tolerance must be finite and non-negative")
    if q.shape[0] > 20:
        raise ValueError("small convex-hull support enumeration is limited to 20 vertices")

    best: tuple[float, tuple[int, ...], np.ndarray, np.ndarray] | None = None
    indices = tuple(range(q.shape[0]))
    for size in range(1, q.shape[0] + 1):
        for support in combinations(indices, size):
            local_vertices = q[np.asarray(support, dtype=np.int64)]
            local_weights = _support_weights(y, local_vertices)
            if float(np.min(local_weights)) < -tolerance:
                continue

            # Small negative values admitted only as numerical slack are projected back
            # onto the simplex before the primal point is evaluated.
            local_weights = np.maximum(local_weights, 0.0)
            total = float(np.sum(local_weights))
            if not math.isfinite(total) or total <= 0.0:
                continue
            local_weights = local_weights / total
            point = local_weights @ local_vertices
            distance = float(np.linalg.norm(y - point))
            candidate = (distance, support, local_weights, point)
            if best is None or (distance, support) < (best[0], best[1]):
                best = candidate

    if best is None:
        raise FloatingPointError("failed to find a numerically feasible convex-hull support")

    distance, support, local_weights, point = best
    weights = np.zeros(q.shape[0], dtype=np.float64)
    weights[np.asarray(support, dtype=np.int64)] = local_weights
    return ConvexHullProjection(
        weights=weights,
        point=np.asarray(point, dtype=np.float64),
        distance=distance,
        support=tuple(int(index) for index in support),
        simplex_residual=abs(float(np.sum(weights)) - 1.0),
        minimum_weight=float(np.min(weights)),
    )


def dual_hull_distance_certificate(
    target: Any,
    vertices: Any,
    *,
    projection: ConvexHullProjection | None = None,
    direction: Any | None = None,
) -> DualHullCertificate:
    """Return a rigorous supporting-direction lower bound on hull distance.

    Any non-zero proposal direction is normalized before use. Therefore an inaccurate
    projection or deliberately inexact direction cannot create a false distance lower
    bound: for every hull point q, Cauchy-Schwarz gives

        ||y-q|| >= <u, y-q>.
    """

    y = _as_finite_vector(target, name="target")
    q = _as_vertex_matrix(vertices, y)
    primal = projection or project_onto_small_convex_hull(y, q)

    if direction is None:
        proposal = y - np.asarray(primal.point, dtype=np.float64)
    else:
        proposal = _as_finite_vector(direction, name="direction")
        if proposal.shape != y.shape:
            raise ValueError("direction shape must match target")

    proposal_norm = float(np.linalg.norm(proposal))
    zero_tolerance = 1e-12 * max(1.0, float(np.linalg.norm(y)), float(primal.distance))
    if proposal_norm <= zero_tolerance:
        unit = np.zeros_like(y)
        lower_bound = 0.0
        unit_norm = 0.0
    else:
        unit = proposal / proposal_norm
        unit_norm = float(np.linalg.norm(unit))
        lower_bound = max(0.0, float(np.min((y[None, :] - q) @ unit)))

    return DualHullCertificate(
        direction=unit,
        direction_norm=unit_norm,
        lower_bound=lower_bound,
        primal_distance=float(primal.distance),
        primal_dual_gap=float(primal.distance) - lower_bound,
    )


def certify_lower_bound_elimination(
    lower_bound: float,
    feasible_upper_bound: float,
    *,
    numerical_guard: float = 0.0,
    interaction_tail_radius: float | None = None,
) -> TailAwareElimination:
    """Separate truncated-model elimination from true-model elimination.

    If ``C_exact`` lies within ``interaction_tail_radius`` of its degree-K truncated
    candidate class ``C_K``, then

        dist(y, C_exact) >= dist(y, C_K) - interaction_tail_radius.

    Without such a tail radius this function intentionally returns ``exact_eliminated``
    as ``None``: a low-degree certificate is not a certificate about true chronology.
    """

    bound = float(lower_bound)
    upper = float(feasible_upper_bound)
    guard = float(numerical_guard)
    if not all(math.isfinite(value) and value >= 0.0 for value in (bound, upper, guard)):
        raise ValueError("lower bound, upper bound, and guard must be finite and non-negative")

    threshold = upper + guard
    truncated = bound > threshold
    if interaction_tail_radius is None:
        tail = None
        exact_bound = None
        exact = None
    else:
        tail = float(interaction_tail_radius)
        if not math.isfinite(tail) or tail < 0.0:
            raise ValueError("interaction_tail_radius must be finite and non-negative")
        exact_bound = max(0.0, bound - tail)
        exact = exact_bound > threshold

    return TailAwareElimination(
        lower_bound=bound,
        feasible_upper_bound=upper,
        numerical_guard=guard,
        truncated_eliminated=truncated,
        interaction_tail_radius=tail,
        exact_lower_bound=exact_bound,
        exact_eliminated=exact,
    )
