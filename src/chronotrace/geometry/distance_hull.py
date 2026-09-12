"""Convex-hull certificates reconstructed from Euclidean distance artifacts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from chronotrace.geometry.convex_quadratic import (
    QuadraticDualHullCertificate,
    QuadraticSimplexProjection,
    dual_quadratic_hull_certificate,
    project_quadratic_simplex,
)


@dataclass(frozen=True)
class PairwiseDistanceHullCertificate:
    """Hull projection and dual witness recovered without endpoint coordinates."""

    vertices: tuple[str, ...]
    gram: np.ndarray
    projection: QuadraticSimplexProjection
    certificate: QuadraticDualHullCertificate


def _distance(
    distances: Mapping[str, Mapping[str, Any]],
    first: str,
    second: str,
) -> float:
    try:
        value = float(distances[first][second])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"missing finite distance for {first!r}, {second!r}") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"distance for {first!r}, {second!r} must be finite and non-negative")
    return value


def relative_gram_from_pairwise_distances(
    target: str,
    vertices: Sequence[str],
    distances: Mapping[str, Mapping[str, Any]],
    *,
    symmetry_tolerance: float = 1e-10,
) -> np.ndarray:
    """Recover the Gram matrix of ``q_i - target`` from pairwise distances.

    Euclidean polarization gives

        <q_i-y, q_j-y>
        = (d(y,q_i)^2 + d(y,q_j)^2 - d(q_i,q_j)^2) / 2.

    This lets frozen distance-only artifacts support later convex certificates without
    retaining full model-space codewords.
    """

    names = tuple(str(value) for value in vertices)
    if not names or len(names) != len(set(names)):
        raise ValueError("vertices must be a non-empty sequence of unique names")
    if target not in distances:
        raise ValueError("target is missing from distance table")
    tolerance = float(symmetry_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("symmetry_tolerance must be finite and non-negative")

    radii = np.asarray([_distance(distances, target, vertex) for vertex in names])
    gram = np.empty((len(names), len(names)), dtype=np.float64)
    for left, first in enumerate(names):
        for right, second in enumerate(names):
            forward = _distance(distances, first, second)
            reverse = _distance(distances, second, first)
            scale = max(1.0, forward, reverse)
            if abs(forward - reverse) > tolerance * scale:
                raise ValueError("pairwise distance table is not symmetric within tolerance")
            pair = 0.5 * (forward + reverse)
            gram[left, right] = 0.5 * (
                radii[left] ** 2 + radii[right] ** 2 - pair**2
            )

    gram = 0.5 * (gram + gram.T)
    if not np.isfinite(gram).all():
        raise FloatingPointError("reconstructed relative Gram matrix is non-finite")
    return gram


def certify_hull_from_pairwise_distances(
    target: str,
    vertices: Sequence[str],
    distances: Mapping[str, Mapping[str, Any]],
    *,
    weight_tolerance: float = 1e-10,
    symmetry_tolerance: float = 1e-10,
) -> PairwiseDistanceHullCertificate:
    """Project a target onto a small vertex hull using only pairwise distances."""

    names = tuple(str(value) for value in vertices)
    gram = relative_gram_from_pairwise_distances(
        target,
        names,
        distances,
        symmetry_tolerance=symmetry_tolerance,
    )
    zeros = np.zeros(len(names), dtype=np.float64)
    projection = project_quadratic_simplex(
        gram,
        zeros,
        0.0,
        weight_tolerance=weight_tolerance,
    )
    certificate = dual_quadratic_hull_certificate(gram, zeros, 0.0, projection)
    return PairwiseDistanceHullCertificate(
        vertices=names,
        gram=gram,
        projection=projection,
        certificate=certificate,
    )
