"""Certificates for finite forward-reachable chronology codebooks."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ReachableSeparationCertificate:
    """Triangle-inequality certificate for nearest reachable-codeword decoding."""

    histories: tuple[str, ...]
    nearest_neighbor: dict[str, str]
    nearest_separation: dict[str, float]
    self_error: dict[str, float]
    certified_margin: dict[str, float]
    target_noise_radius: dict[str, float]
    minimum_pairwise_separation: float
    minimum_certified_margin: float
    minimum_target_noise_radius: float
    all_certified: bool


def chunked_l2_distance(left: Any, right: Any, *, chunk_size: int = 262_144) -> float:
    """Compute FP64 Euclidean distance without materializing a full difference vector."""

    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")
    if getattr(left, "ndim", None) != 1 or getattr(right, "ndim", None) != 1:
        raise ValueError("chunked L2 inputs must be one-dimensional")
    if tuple(left.shape) != tuple(right.shape) or int(left.shape[0]) < 1:
        raise ValueError("chunked L2 inputs must have equal non-empty shapes")

    total = 0.0
    length = int(left.shape[0])
    size = int(chunk_size)
    for start in range(0, length, size):
        stop = min(start + size, length)
        left_chunk = np.asarray(left[start:stop], dtype=np.float64)
        right_chunk = np.asarray(right[start:stop], dtype=np.float64)
        if not np.isfinite(left_chunk).all() or not np.isfinite(right_chunk).all():
            raise ValueError("chunked L2 inputs must be finite")
        difference = left_chunk - right_chunk
        total += float(np.dot(difference, difference))
    if not math.isfinite(total) or total < 0.0:
        raise FloatingPointError("chunked L2 accumulation became invalid")
    return math.sqrt(total)


def certify_reachable_distance_table(
    pairwise_distances: Mapping[str, Mapping[str, float]],
    self_errors: Mapping[str, float],
) -> ReachableSeparationCertificate:
    """Certify finite-history decoding from codeword separations and self errors.

    Let ``q_h`` be the reachable codeword for history ``h`` and ``y_h`` its exact
    observed endpoint. If ``e_h = ||y_h-q_h||`` and

        s_h = min_{g != h} ||q_h-q_g||,

    then for every competing history ``g``

        ||y_h-q_g|| >= s_h-e_h.

    Therefore ``q_h`` is the unique nearest codeword whenever ``s_h > 2 e_h``. The
    certified nearest-codeword margin is at least ``s_h-2e_h`` and an additional target
    perturbation of norm less than ``s_h/2-e_h`` cannot change the decision.
    """

    histories = tuple(sorted(str(history) for history in pairwise_distances))
    if len(histories) < 2:
        raise ValueError("at least two reachable histories are required")
    if set(histories) != {str(history) for history in self_errors}:
        raise ValueError("self-error histories must match the distance table")

    nearest_neighbor: dict[str, str] = {}
    nearest_separation: dict[str, float] = {}
    errors: dict[str, float] = {}
    margins: dict[str, float] = {}
    radii: dict[str, float] = {}

    for history in histories:
        row = pairwise_distances[history]
        if {str(other) for other in row} != set(histories):
            raise ValueError("every distance-table row must contain every history")
        diagonal = float(row[history])
        if not np.isfinite(diagonal) or abs(diagonal) > 1e-12:
            raise ValueError("distance-table diagonal must be numerical zero")

        candidates: list[tuple[float, str]] = []
        for other in histories:
            value = float(row[other])
            reverse = float(pairwise_distances[other][history])
            if not np.isfinite(value) or value < 0.0:
                raise ValueError("pairwise distances must be finite and non-negative")
            if not np.isclose(value, reverse, rtol=1e-12, atol=1e-14):
                raise ValueError("pairwise distance table must be symmetric")
            if other != history:
                candidates.append((value, other))

        separation, neighbor = min(candidates, key=lambda item: (item[0], item[1]))
        error = float(self_errors[history])
        if not np.isfinite(error) or error < 0.0:
            raise ValueError("self errors must be finite and non-negative")
        margin = separation - 2.0 * error
        radius = 0.5 * separation - error

        nearest_neighbor[history] = neighbor
        nearest_separation[history] = separation
        errors[history] = error
        margins[history] = margin
        radii[history] = radius

    minimum_pairwise_separation = min(nearest_separation.values())
    minimum_certified_margin = min(margins.values())
    minimum_target_noise_radius = min(radii.values())
    return ReachableSeparationCertificate(
        histories=histories,
        nearest_neighbor=nearest_neighbor,
        nearest_separation=nearest_separation,
        self_error=errors,
        certified_margin=margins,
        target_noise_radius=radii,
        minimum_pairwise_separation=minimum_pairwise_separation,
        minimum_certified_margin=minimum_certified_margin,
        minimum_target_noise_radius=minimum_target_noise_radius,
        all_certified=minimum_certified_margin > 0.0,
    )
