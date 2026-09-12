"""Directional higher-order-tail certificates for approximate chronology hulls.

Suppose an approximate candidate vertex ``p_i`` is related to its exact endpoint by

    q_i = p_i + r_i.

For any unit certificate direction ``u``, define the approximate witness

    a_i = <u, y - p_i>

and suppose a verified scalar upper bound ``tau_i`` satisfies

    <u, r_i> <= tau_i.

Then every exact vertex obeys

    <u, y - q_i> >= a_i - tau_i,

so the distance from ``y`` to the exact convex hull is at least

    max(0, min_i (a_i - tau_i)).

This can be dramatically tighter than replacing every ``tau_i`` by ``||r_i||`` because
large higher-order interactions orthogonal to the separating direction do not consume the
certificate margin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DirectionalTailCertificate:
    """Tail-adjusted lower bound for one fixed unit separating direction."""

    approximate_vertex_witnesses: np.ndarray
    directional_tail_upper_bounds: np.ndarray
    adjusted_vertex_witnesses: np.ndarray
    lower_bound: float
    feasible_upper_bound: float
    numerical_guard: float
    certified_impossible: bool


def directional_tail_hull_certificate(
    approximate_vertex_witnesses: Any,
    directional_tail_upper_bounds: Any,
    *,
    feasible_upper_bound: float = 0.0,
    numerical_guard: float = 0.0,
) -> DirectionalTailCertificate:
    """Convert verified directional tail bounds into an exact-hull lower bound.

    The caller is responsible for proving that each supplied tail value upper-bounds the
    corresponding exact directional correction ``<u, r_i>`` for the same unit direction
    used to construct ``approximate_vertex_witnesses``.
    """

    witnesses = np.asarray(approximate_vertex_witnesses, dtype=np.float64)
    tails = np.asarray(directional_tail_upper_bounds, dtype=np.float64)
    if witnesses.ndim != 1 or witnesses.size < 1:
        raise ValueError("approximate_vertex_witnesses must be a non-empty vector")
    if tails.shape != witnesses.shape:
        raise ValueError("directional_tail_upper_bounds must match witness shape")
    if not np.isfinite(witnesses).all() or not np.isfinite(tails).all():
        raise ValueError("directional witnesses and tail upper bounds must be finite")

    upper = float(feasible_upper_bound)
    guard = float(numerical_guard)
    if not math.isfinite(upper) or upper < 0.0:
        raise ValueError("feasible_upper_bound must be finite and non-negative")
    if not math.isfinite(guard) or guard < 0.0:
        raise ValueError("numerical_guard must be finite and non-negative")

    adjusted = witnesses - tails
    lower_bound = max(0.0, float(np.min(adjusted)))
    certified = lower_bound > upper + guard
    return DirectionalTailCertificate(
        approximate_vertex_witnesses=witnesses.copy(),
        directional_tail_upper_bounds=tails.copy(),
        adjusted_vertex_witnesses=adjusted,
        lower_bound=lower_bound,
        feasible_upper_bound=upper,
        numerical_guard=guard,
        certified_impossible=certified,
    )


def norm_tail_fallback(
    approximate_vertex_witnesses: Any,
    tail_norm_upper_bounds: Any,
    *,
    feasible_upper_bound: float = 0.0,
    numerical_guard: float = 0.0,
) -> DirectionalTailCertificate:
    """Use Cauchy-Schwarz to turn norm-tail radii into directional upper bounds."""

    radii = np.asarray(tail_norm_upper_bounds, dtype=np.float64)
    if radii.ndim != 1 or radii.size < 1 or not np.isfinite(radii).all():
        raise ValueError("tail_norm_upper_bounds must be a finite non-empty vector")
    if float(np.min(radii)) < 0.0:
        raise ValueError("tail norm upper bounds must be non-negative")
    return directional_tail_hull_certificate(
        approximate_vertex_witnesses,
        radii,
        feasible_upper_bound=feasible_upper_bound,
        numerical_guard=numerical_guard,
    )
