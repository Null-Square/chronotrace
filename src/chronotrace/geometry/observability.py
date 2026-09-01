"""Finite-candidate observability tools for active training-history forensics.

The core object is a response matrix whose rows are candidate training histories and
whose columns are allowed forensic probe coordinates.  The module deliberately makes
only finite-dimensional claims: it does not assume that a low-rank response family
exists for real models, nor that a chosen observation interface retains all information
in the trainer state.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ProbeBasis:
    """A linearly independent set of response coordinates."""

    columns: tuple[int, ...]
    rank: int
    selected: np.ndarray


@dataclass(frozen=True)
class DistinguishingProbeSet:
    """A smallest tested subset of physical response coordinates preserving distinctions."""

    columns: tuple[int, ...]
    selected: np.ndarray
    full_indistinguishable_pairs: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class SeparationCertificate:
    """Pairwise separation and certified additive-noise radius."""

    minimum_distance: float
    noise_radius: float
    pair: tuple[int, int] | None


@dataclass(frozen=True)
class ResponseDecode:
    """Nearest reference response and its margin to the runner-up."""

    index: int
    best_distance: float
    runner_up_distance: float
    margin: float


def _matrix(values: Any) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("response matrix must be two-dimensional")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("response matrix must be non-empty")
    if not np.isfinite(matrix).all():
        raise ValueError("response matrix must contain only finite values")
    return matrix


def _rank(matrix: np.ndarray, tolerance: float | None) -> int:
    return int(np.linalg.matrix_rank(matrix, tol=tolerance))


def independent_probe_basis(
    responses: Any,
    *,
    tolerance: float | None = None,
) -> ProbeBasis:
    """Select response columns forming a basis for the full column space.

    If the response matrix has rank ``r``, the returned set has exactly ``r`` columns.
    Therefore any equality between two rows on the selected columns implies equality on
    every original response coordinate: every omitted column is a linear combination of
    the selected columns.

    This is a linear-span certificate, not a minimum-physical-probe certificate.  A
    single physical column can already distinguish every finite candidate even when the
    matrix rank is larger than one.
    """

    matrix = _matrix(responses)
    target_rank = _rank(matrix, tolerance)
    selected: list[int] = []
    current_rank = 0
    for column in range(matrix.shape[1]):
        candidate_columns = selected + [column]
        candidate_rank = _rank(matrix[:, candidate_columns], tolerance)
        if candidate_rank > current_rank:
            selected.append(column)
            current_rank = candidate_rank
        if current_rank == target_rank:
            break
    if current_rank != target_rank:
        raise RuntimeError("could not construct a full-rank probe basis")
    chosen = matrix[:, selected].copy()
    return ProbeBasis(columns=tuple(selected), rank=target_rank, selected=chosen)


def indistinguishable_pairs(
    responses: Any,
    *,
    atol: float = 0.0,
) -> tuple[tuple[int, int], ...]:
    """Return candidate row pairs indistinguishable under the supplied probes."""

    matrix = _matrix(responses)
    threshold = float(atol)
    if threshold < 0:
        raise ValueError("atol must be non-negative")
    pairs: list[tuple[int, int]] = []
    for first in range(matrix.shape[0]):
        for second in range(first + 1, matrix.shape[0]):
            if float(np.linalg.norm(matrix[first] - matrix[second])) <= threshold:
                pairs.append((first, second))
    return tuple(pairs)


def minimum_distinguishing_probe_subset(
    responses: Any,
    *,
    atol: float = 0.0,
    max_columns: int = 16,
) -> DistinguishingProbeSet:
    """Find an exact smallest subset of physical coordinates preserving distinctions.

    The search is exhaustive in increasing subset size and is therefore exact for the
    supplied finite response family.  It preserves the full family's indistinguishability
    relation at the requested Euclidean tolerance.  Complexity is exponential in the
    number of physical columns, so callers must keep the probe family small.

    Unlike matrix rank, this quantity directly answers how many *existing physical
    response coordinates* are sufficient for the same finite candidate distinctions.
    """

    matrix = _matrix(responses)
    threshold = float(atol)
    if threshold < 0:
        raise ValueError("atol must be non-negative")
    column_count = matrix.shape[1]
    limit = int(max_columns)
    if limit < 1:
        raise ValueError("max_columns must be positive")
    if column_count > limit:
        raise ValueError(
            f"exact distinguishing-probe search has {column_count} columns, "
            f"above max_columns={limit}"
        )
    full_pairs = indistinguishable_pairs(matrix, atol=threshold)
    for size in range(1, column_count + 1):
        for columns in combinations(range(column_count), size):
            selected = matrix[:, columns]
            if indistinguishable_pairs(selected, atol=threshold) == full_pairs:
                return DistinguishingProbeSet(
                    columns=tuple(columns),
                    selected=selected.copy(),
                    full_indistinguishable_pairs=full_pairs,
                )
    raise RuntimeError("full response family did not preserve its own distinctions")


def separation_certificate(responses: Any) -> SeparationCertificate:
    """Return the exact nearest-row distance and the half-distance noise certificate.

    With additive response noise bounded in Euclidean norm by ``epsilon``, nearest-row
    decoding is guaranteed to recover the true candidate whenever

        epsilon < minimum_distance / 2.

    A zero minimum distance certifies non-identifiability for at least one row pair.
    """

    matrix = _matrix(responses)
    if matrix.shape[0] < 2:
        return SeparationCertificate(float("inf"), float("inf"), None)
    best = float("inf")
    best_pair: tuple[int, int] | None = None
    for first in range(matrix.shape[0]):
        for second in range(first + 1, matrix.shape[0]):
            distance = float(np.linalg.norm(matrix[first] - matrix[second]))
            if distance < best:
                best = distance
                best_pair = (first, second)
    return SeparationCertificate(best, 0.5 * best, best_pair)


def decode_response(observed: Any, references: Any) -> ResponseDecode:
    """Decode one response vector by Euclidean nearest reference row."""

    matrix = _matrix(references)
    vector = np.asarray(observed, dtype=np.float64)
    if vector.ndim != 1 or vector.shape[0] != matrix.shape[1]:
        raise ValueError("observed response shape differs from reference coordinates")
    if not np.isfinite(vector).all():
        raise ValueError("observed response must contain only finite values")
    distances = np.linalg.norm(matrix - vector[None, :], axis=1)
    order = np.argsort(distances, kind="stable")
    best = int(order[0])
    best_distance = float(distances[best])
    runner_up = float(distances[int(order[1])]) if len(order) > 1 else float("inf")
    return ResponseDecode(best, best_distance, runner_up, runner_up - best_distance)


def deterministic_responses(
    states: Any,
    probes: tuple[Any, ...],
    observation: Any,
) -> np.ndarray:
    """Evaluate deterministic challenge/observation coordinates on finite states.

    Each probe is a deterministic state map.  This helper is intentionally small, but it
    makes the Markov-state impossibility explicit: equal input states necessarily produce
    equal responses under every deterministic probe followed by the same observation.
    """

    rows: list[list[float]] = []
    for state in states:
        rows.append([float(observation(probe(state))) for probe in probes])
    return _matrix(rows)
