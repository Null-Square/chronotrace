# ruff: noqa: I001
"""Linear constraint system for the K3 local-order chronology relaxation.

The feasible coordinates are pair precedence probabilities plus six local-order weights
for each unordered triple. Every triple lies on a six-way simplex and its three pairwise
marginals must agree with the shared global pair coordinates. The resulting relaxation
has O(N^3) variables and equality constraints for fixed interaction degree K=3.

No optimizer is required here. The matrix form is intended to feed a pinned convex QP
solver in experiments while keeping the chronology geometry independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations

import numpy as np

from chronotrace.geometry.order_relaxation import (
    K3LocalOrderPoint,
    K3LocalOrderRelaxation,
    Pair,
    Triple,
    TripleOrder,
)


@dataclass(frozen=True)
class K3LocalOrderLinearSystem:
    """Box-constrained equality form of the local K3 ordering relaxation."""

    pair_indices: dict[Pair, int]
    triple_order_indices: dict[tuple[Triple, TripleOrder], int]
    equality_matrix: np.ndarray
    equality_rhs: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray

    @property
    def variable_count(self) -> int:
        return int(self.lower_bounds.size)

    @property
    def equality_count(self) -> int:
        return int(self.equality_rhs.size)


def build_k3_local_order_linear_system(
    relaxation: K3LocalOrderRelaxation,
) -> K3LocalOrderLinearSystem:
    """Build simplex and pair/triple marginal-consistency constraints."""

    pair_indices = {pair: index for index, pair in enumerate(relaxation.pairs)}
    cursor = len(pair_indices)
    triple_order_indices: dict[tuple[Triple, TripleOrder], int] = {}
    for triple in relaxation.triples:
        for order in permutations(triple):
            triple_order_indices[(triple, order)] = cursor
            cursor += 1

    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for triple in relaxation.triples:
        simplex = np.zeros(cursor, dtype=np.float64)
        for order in permutations(triple):
            simplex[triple_order_indices[(triple, order)]] = 1.0
        rows.append(simplex)
        rhs.append(1.0)

        for first, second in combinations(triple, 2):
            pair = (first, second)
            marginal = np.zeros(cursor, dtype=np.float64)
            marginal[pair_indices[pair]] = -1.0
            for order in permutations(triple):
                if order.index(first) < order.index(second):
                    marginal[triple_order_indices[(triple, order)]] = 1.0
            rows.append(marginal)
            rhs.append(0.0)

    matrix = np.stack(rows) if rows else np.zeros((0, cursor), dtype=np.float64)
    return K3LocalOrderLinearSystem(
        pair_indices=pair_indices,
        triple_order_indices=triple_order_indices,
        equality_matrix=matrix,
        equality_rhs=np.asarray(rhs, dtype=np.float64),
        lower_bounds=np.zeros(cursor, dtype=np.float64),
        upper_bounds=np.ones(cursor, dtype=np.float64),
    )


def k3_local_order_point_to_vector(
    point: K3LocalOrderPoint,
    relaxation: K3LocalOrderRelaxation,
    system: K3LocalOrderLinearSystem,
) -> np.ndarray:
    """Encode a local-order point in the QP coordinate ordering."""

    if set(system.pair_indices) != set(relaxation.pairs):
        raise ValueError("linear-system pair coordinates do not match relaxation")
    expected_triples = {
        (triple, order)
        for triple in relaxation.triples
        for order in permutations(triple)
    }
    if set(system.triple_order_indices) != expected_triples:
        raise ValueError("linear-system triple coordinates do not match relaxation")

    vector = np.zeros(system.variable_count, dtype=np.float64)
    for pair, index in system.pair_indices.items():
        vector[index] = float(point.pair_precedence[pair])
    for key, index in system.triple_order_indices.items():
        triple, order = key
        vector[index] = float(point.triple_order_weights[triple][order])
    return vector


def k3_local_order_vector_to_point(
    vector: np.ndarray,
    relaxation: K3LocalOrderRelaxation,
    system: K3LocalOrderLinearSystem,
) -> K3LocalOrderPoint:
    """Decode one QP coordinate vector back into named local-order variables."""

    values = np.asarray(vector, dtype=np.float64)
    if values.shape != (system.variable_count,) or not np.isfinite(values).all():
        raise ValueError("local-order vector has wrong shape or non-finite entries")
    pair_precedence = {
        pair: float(values[index]) for pair, index in system.pair_indices.items()
    }
    triple_order_weights: dict[Triple, dict[TripleOrder, float]] = {
        triple: {} for triple in relaxation.triples
    }
    for (triple, order), index in system.triple_order_indices.items():
        triple_order_weights[triple][order] = float(values[index])
    return K3LocalOrderPoint(
        pair_precedence=pair_precedence,
        triple_order_weights=triple_order_weights,
    )


def k3_local_order_linear_residuals(
    vector: np.ndarray,
    system: K3LocalOrderLinearSystem,
) -> tuple[float, float]:
    """Return maximum equality and box-constraint violations."""

    values = np.asarray(vector, dtype=np.float64)
    if values.shape != (system.variable_count,) or not np.isfinite(values).all():
        raise ValueError("local-order vector has wrong shape or non-finite entries")
    equality_violation = (
        float(np.max(np.abs(system.equality_matrix @ values - system.equality_rhs)))
        if system.equality_count
        else 0.0
    )
    lower_violation = np.maximum(system.lower_bounds - values, 0.0)
    upper_violation = np.maximum(values - system.upper_bounds, 0.0)
    box_violation = float(max(np.max(lower_violation), np.max(upper_violation)))
    return equality_violation, box_violation
