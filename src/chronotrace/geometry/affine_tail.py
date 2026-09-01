# ruff: noqa: I001
"""Exact ordered-interaction remainder bounds for affine training maps.

Let each stage map be

    T_i(x) = (I + B_i) x + c_i

and let ``delta_i = T_i(theta0) - theta0``. Möbius inversion over ordered
subsequences gives the exact degree-r interaction

    Phi(i1,...,ir) = B_ir ... B_i2 delta_i1.

Therefore if ``||B_i|| <= q`` and ``||delta_i|| <= D``, every degree-r term is
bounded by ``D q**(r-1)``. Summing over omitted subsequences gives an explicit
remainder certificate for truncating an N-stage chronology at degree K.

For a one-step quadratic-gradient update, ``B_i = -eta H_i`` exactly. For a
nonlinear network this module is a local/affine reference theorem, not by itself
a global bound on the true training operator.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np


Array = np.ndarray


def affine_ordered_interaction_closed_form(
    word: Sequence[str],
    increment_matrices: Mapping[str, Array],
    singleton_displacements: Mapping[str, Array],
) -> Array:
    """Return ``B_last ... B_second delta_first`` for one non-empty word."""

    values = tuple(word)
    if not values:
        raise ValueError("ordered interaction word must be non-empty")
    if len(values) != len(set(values)):
        raise ValueError("ordered interaction word must not repeat stages")
    if values[0] not in singleton_displacements:
        raise ValueError("missing singleton displacement for first stage")

    interaction = np.asarray(singleton_displacements[values[0]], dtype=np.float64).copy()
    if interaction.ndim != 1 or not np.isfinite(interaction).all():
        raise ValueError("singleton displacement must be a finite vector")
    dimension = interaction.size
    for stage in values[1:]:
        if stage not in increment_matrices:
            raise ValueError(f"missing increment matrix for stage {stage!r}")
        matrix = np.asarray(increment_matrices[stage], dtype=np.float64)
        if matrix.shape != (dimension, dimension) or not np.isfinite(matrix).all():
            raise ValueError("increment matrices must be finite square matrices")
        interaction = matrix @ interaction
    return interaction


def affine_uniform_interaction_bound(
    *,
    degree: int,
    max_singleton_norm: float,
    max_increment_operator_norm: float,
) -> float:
    """Bound one degree-r affine ordered interaction in Euclidean norm."""

    order = int(degree)
    displacement = float(max_singleton_norm)
    q = float(max_increment_operator_norm)
    if order < 1:
        raise ValueError("degree must be positive")
    if not math.isfinite(displacement) or displacement < 0.0:
        raise ValueError("max_singleton_norm must be finite and non-negative")
    if not math.isfinite(q) or q < 0.0:
        raise ValueError("max_increment_operator_norm must be finite and non-negative")
    return displacement * q ** (order - 1)


def affine_uniform_truncation_tail_bound(
    *,
    stage_count: int,
    max_degree: int,
    max_singleton_norm: float,
    max_increment_operator_norm: float,
) -> float:
    """Bound the total omitted interaction norm for one N-stage chronology.

    A fixed chronology contains exactly ``C(N,r)`` ordered subsequences of degree r.
    Triangle inequality plus the uniform interaction bound therefore gives

        D * sum_{r=K+1..N} C(N,r) q**(r-1).
    """

    count = int(stage_count)
    degree = int(max_degree)
    if count < 1:
        raise ValueError("stage_count must be positive")
    if degree < 0 or degree > count:
        raise ValueError("max_degree must be between zero and stage_count")
    displacement = float(max_singleton_norm)
    q = float(max_increment_operator_norm)
    if not math.isfinite(displacement) or displacement < 0.0:
        raise ValueError("max_singleton_norm must be finite and non-negative")
    if not math.isfinite(q) or q < 0.0:
        raise ValueError("max_increment_operator_norm must be finite and non-negative")

    return displacement * sum(
        math.comb(count, order) * q ** (order - 1)
        for order in range(degree + 1, count + 1)
    )


def affine_chronology_tail_bound(
    chronology: Sequence[str],
    *,
    max_degree: int,
    singleton_norm_bounds: Mapping[str, float],
    increment_operator_norm_bounds: Mapping[str, float],
) -> float:
    """Return a sharper stage-specific tail bound for one candidate chronology."""

    values = tuple(chronology)
    if not values or len(values) != len(set(values)):
        raise ValueError("chronology must be non-empty and contain distinct stages")
    degree = int(max_degree)
    if degree < 0 or degree > len(values):
        raise ValueError("max_degree must be between zero and chronology length")

    from itertools import combinations

    total = 0.0
    for order in range(degree + 1, len(values) + 1):
        for indices in combinations(range(len(values)), order):
            first = values[indices[0]]
            if first not in singleton_norm_bounds:
                raise ValueError(f"missing singleton bound for stage {first!r}")
            term = float(singleton_norm_bounds[first])
            if not math.isfinite(term) or term < 0.0:
                raise ValueError("singleton norm bounds must be finite and non-negative")
            for index in indices[1:]:
                stage = values[index]
                if stage not in increment_operator_norm_bounds:
                    raise ValueError(f"missing increment bound for stage {stage!r}")
                q = float(increment_operator_norm_bounds[stage])
                if not math.isfinite(q) or q < 0.0:
                    raise ValueError(
                        "increment operator norm bounds must be finite and non-negative"
                    )
                term *= q
            total += term
    return total
