"""Finite antisymmetric probes for active training-history observability."""

from __future__ import annotations

from typing import Any

import numpy as np


StateMap = Any


def finite_commutator_vector(state: Any, first: StateMap, second: StateMap) -> np.ndarray:
    """Return ``second(first(state)) - first(second(state))`` as a finite vector."""

    forward = np.asarray(second(first(state)), dtype=np.float64)
    reverse = np.asarray(first(second(state)), dtype=np.float64)
    if forward.shape != reverse.shape:
        raise ValueError("challenge compositions must return the same shape")
    if not np.isfinite(forward).all() or not np.isfinite(reverse).all():
        raise ValueError("challenge compositions must be finite")
    return forward - reverse


def normalized_projection(vector: Any, direction: Any) -> float:
    """Project a vector onto a fixed direction normalized to unit Euclidean length."""

    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    basis = np.asarray(direction, dtype=np.float64).reshape(-1)
    if value.shape != basis.shape:
        raise ValueError("vector and direction must have the same flattened shape")
    if not np.isfinite(value).all() or not np.isfinite(basis).all():
        raise ValueError("vector and direction must be finite")
    norm = float(np.linalg.norm(basis))
    if norm == 0.0:
        raise ValueError("projection direction must be non-zero")
    return float(np.dot(value, basis) / norm)


def projected_finite_commutator(
    state: Any,
    first: StateMap,
    second: StateMap,
    direction: Any,
) -> float:
    """Evaluate the signed finite commutator along a fixed reference direction."""

    return normalized_projection(finite_commutator_vector(state, first, second), direction)
