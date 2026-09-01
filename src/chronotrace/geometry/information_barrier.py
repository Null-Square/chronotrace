"""Finite-query information barriers for deterministic one-step SGD maps.

A finite collection of gradient queries cannot control an unqueried transition without
additional regularity.  Given queried states ``q_i`` and an unqueried state ``x_star``,
define

    P(x) = product_i ||x - q_i||^2
    psi(x) = P(x) <v, x - x_star> / P(x_star).

Then ``grad psi(q_i) = 0`` for every old query while ``grad psi(x_star) = v``.  Adding
``psi`` to one stage loss therefore preserves every previously observed one-step SGD
transition at the queried states and changes the transition at ``x_star`` by an arbitrary
direction.  This is the finite-query core of the K-local information barrier used by the
projected-interaction hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PolynomialGradientPerturbation:
    """Smooth polynomial perturbation invisible on a finite gradient-query set."""

    query_points: np.ndarray
    hidden_point: np.ndarray
    hidden_gradient: np.ndarray
    hidden_mask_value: float

    def gradient(self, point: Any) -> np.ndarray:
        """Evaluate the analytic gradient of the perturbation polynomial."""

        x = np.asarray(point, dtype=np.float64)
        if x.shape != self.hidden_point.shape:
            raise ValueError("point shape must match hidden_point")

        offsets = x[None, :] - self.query_points
        squared = np.einsum("ij,ij->i", offsets, offsets)
        mask = float(np.prod(squared))

        grad_mask = np.zeros_like(x)
        for index in range(self.query_points.shape[0]):
            other_product = 1.0
            for other_index, value in enumerate(squared):
                if other_index != index:
                    other_product *= float(value)
            grad_mask += 2.0 * offsets[index] * other_product

        linear = float(np.dot(self.hidden_gradient, x - self.hidden_point))
        return (
            grad_mask * linear + mask * self.hidden_gradient
        ) / self.hidden_mask_value


def build_polynomial_gradient_perturbation(
    query_points: Any,
    hidden_point: Any,
    hidden_gradient: Any,
) -> PolynomialGradientPerturbation:
    """Build a smooth perturbation with zero queried gradients and chosen hidden gradient."""

    queries = np.asarray(query_points, dtype=np.float64)
    hidden = np.asarray(hidden_point, dtype=np.float64)
    gradient = np.asarray(hidden_gradient, dtype=np.float64)
    if queries.ndim != 2 or queries.shape[0] < 1:
        raise ValueError("query_points must be a non-empty matrix")
    if hidden.ndim != 1 or hidden.shape != (queries.shape[1],):
        raise ValueError("hidden_point dimension must match query_points")
    if gradient.shape != hidden.shape:
        raise ValueError("hidden_gradient shape must match hidden_point")
    if not np.isfinite(queries).all() or not np.isfinite(hidden).all():
        raise ValueError("query and hidden points must be finite")
    if not np.isfinite(gradient).all():
        raise ValueError("hidden_gradient must be finite")

    hidden_squared = np.einsum(
        "ij,ij->i",
        hidden[None, :] - queries,
        hidden[None, :] - queries,
    )
    if float(np.min(hidden_squared)) <= 0.0:
        raise ValueError("hidden_point must not equal any queried point")
    hidden_mask = float(np.prod(hidden_squared))
    if not np.isfinite(hidden_mask) or hidden_mask <= 0.0:
        raise FloatingPointError("hidden polynomial mask is not finite and positive")

    return PolynomialGradientPerturbation(
        query_points=queries.copy(),
        hidden_point=hidden.copy(),
        hidden_gradient=gradient.copy(),
        hidden_mask_value=hidden_mask,
    )
