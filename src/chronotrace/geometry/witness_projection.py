"""Scalar projection identities for convex-hull witness banks.

If a unit witness is constructed from a target ``y`` and convex combination
``p_bar = sum_i w_i p_i`` as

    u = (y - p_bar) / ||y - p_bar||,

then any streamed endpoint ``q`` can be projected without retaining ``u`` itself:

    <u,q> = (<y,q> - sum_i w_i <p_i,q>) / ||y-p_bar||.

This identity lets multiple frozen witness banks share endpoint evaluations while still
retaining only scalar higher-order projections.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def convex_residual_witness_projection(
    *,
    target_dot: float,
    candidate_dots: Any,
    weights: Any,
    normalization_norm: float,
    weight_tolerance: float = 1e-10,
) -> float:
    """Return ``<u,q>`` from scalar target/candidate endpoint dot products.

    ``candidate_dots[i]`` is ``<p_i,q>`` and the same convex weights used to construct
    ``u`` must be supplied.  The helper validates simplex feasibility before applying the
    exact linear identity.
    """

    target_value = float(target_dot)
    dots = np.asarray(candidate_dots, dtype=np.float64)
    simplex = np.asarray(weights, dtype=np.float64)
    norm = float(normalization_norm)
    tolerance = float(weight_tolerance)
    if dots.ndim != 1 or dots.size < 1 or simplex.shape != dots.shape:
        raise ValueError("candidate dots and weights must be matching non-empty vectors")
    if not math.isfinite(target_value) or not np.isfinite(dots).all():
        raise ValueError("target and candidate dot products must be finite")
    if not np.isfinite(simplex).all():
        raise ValueError("convex weights must be finite")
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("normalization norm must be finite and positive")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("weight_tolerance must be finite and non-negative")
    if float(np.min(simplex)) < -tolerance:
        raise ValueError("convex weights contain a negative value")
    if abs(float(np.sum(simplex)) - 1.0) > tolerance:
        raise ValueError("convex weights must sum to one")
    return (target_value - float(simplex @ dots)) / norm
