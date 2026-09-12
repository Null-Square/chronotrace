"""Affine-hull lower bounds for polynomial-size K3 local-order relaxations.

The three-local representation uses one pair precedence coordinate per unordered pair and
six local-order coordinates per unordered triple.  Its size is

    C(N, 2) + 6 C(N, 3) = O(N^3).

Every global permutation satisfies the triple-simplex and pair/triple marginal equations.
Fixing a candidate final stage adds N-1 pair equalities.  If we *drop* non-negativity and
box constraints, the resulting affine set is a superset of every discrete chronology with
that final stage.  Euclidean distance to this affine set is therefore a rigorous lower
bound on distance to the discrete K3 codebook for that final-stage hypothesis.

This module works only with scalar quadratic sufficient statistics.  It does not allocate
full model-parameter feature matrices and it does not enumerate permutations at decode
time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Any

import numpy as np

from chronotrace.geometry.order_relaxation import K3LocalOrderRelaxation


Pair = tuple[str, str]
Triple = tuple[str, str, str]
TripleOrder = tuple[str, str, str]


@dataclass(frozen=True)
class K3LocalCoordinateLayout:
    """Dense coordinate indexing for the K3 local-order relaxation."""

    stages: tuple[str, ...]
    pairs: tuple[Pair, ...]
    triples: tuple[Triple, ...]
    pair_index: dict[Pair, int]
    triple_order_index: dict[tuple[Triple, TripleOrder], int]
    dimension: int


@dataclass(frozen=True)
class AffineProjectionResult:
    """Evidence for one equality-constrained quadratic projection."""

    distance: float
    squared_distance: float
    solution: np.ndarray
    equality_residual_norm: float
    stationarity_residual_norm: float


def build_k3_local_coordinate_layout(stages: Any) -> K3LocalCoordinateLayout:
    """Build deterministic O(N^3) coordinates for pair and triple local orders."""

    values = tuple(str(stage) for stage in stages)
    if len(values) < 3:
        raise ValueError("K3 local coordinates require at least three stages")
    if len(values) != len(set(values)):
        raise ValueError("stage names must be unique")

    pairs = tuple(combinations(values, 2))
    triples = tuple(combinations(values, 3))
    pair_index = {pair: index for index, pair in enumerate(pairs)}
    triple_order_index: dict[tuple[Triple, TripleOrder], int] = {}
    offset = len(pairs)
    for triple in triples:
        for order in permutations(triple):
            triple_order_index[(triple, order)] = offset
            offset += 1
    return K3LocalCoordinateLayout(
        stages=values,
        pairs=pairs,
        triples=triples,
        pair_index=pair_index,
        triple_order_index=triple_order_index,
        dimension=offset,
    )


def encode_k3_local_permutation(
    chronology: Any,
    layout: K3LocalCoordinateLayout,
) -> np.ndarray:
    """Encode a full chronology as an integral pair/triple local-order vector."""

    order = tuple(str(stage) for stage in chronology)
    if len(order) != len(layout.stages) or set(order) != set(layout.stages):
        raise ValueError("chronology must contain every layout stage exactly once")
    position = {stage: index for index, stage in enumerate(order)}
    vector = np.zeros(layout.dimension, dtype=np.float64)
    for pair in layout.pairs:
        vector[layout.pair_index[pair]] = float(position[pair[0]] < position[pair[1]])
    for triple in layout.triples:
        induced = tuple(sorted(triple, key=position.__getitem__))
        vector[layout.triple_order_index[(triple, induced)]] = 1.0
    return vector


def k3_local_equalities(
    layout: K3LocalCoordinateLayout,
    *,
    last_stage: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return local simplex/marginal equations and optional final-stage equations."""

    if last_stage is not None and last_stage not in layout.stages:
        raise ValueError("last_stage must belong to the layout")

    rows: list[np.ndarray] = []
    values: list[float] = []
    for triple in layout.triples:
        orders = tuple(permutations(triple))
        simplex = np.zeros(layout.dimension, dtype=np.float64)
        for order in orders:
            simplex[layout.triple_order_index[(triple, order)]] = 1.0
        rows.append(simplex)
        values.append(1.0)

        for pair in combinations(triple, 2):
            row = np.zeros(layout.dimension, dtype=np.float64)
            row[layout.pair_index[pair]] = 1.0
            for order in orders:
                if order.index(pair[0]) < order.index(pair[1]):
                    row[layout.triple_order_index[(triple, order)]] -= 1.0
            rows.append(row)
            values.append(0.0)

    if last_stage is not None:
        for pair in layout.pairs:
            if last_stage not in pair:
                continue
            row = np.zeros(layout.dimension, dtype=np.float64)
            row[layout.pair_index[pair]] = 1.0
            # x_(a,b)=1 iff a occurs before b.  Hence b-last fixes x=1 and
            # a-last fixes x=0 for canonical pair (a,b).
            rows.append(row)
            values.append(float(pair[1] == last_stage))

    return np.stack(rows), np.asarray(values, dtype=np.float64)


def k3_local_variable_coefficients(
    relaxation: K3LocalOrderRelaxation,
    layout: K3LocalCoordinateLayout,
) -> tuple[Any, ...]:
    """Return full-space affine feature tensors in coordinate order without copying."""

    if tuple(relaxation.stages) != layout.stages:
        raise ValueError("relaxation stages differ from coordinate layout")
    coefficients: list[Any | None] = [None] * layout.dimension
    for pair in layout.pairs:
        coefficients[layout.pair_index[pair]] = relaxation.pair_coefficients[pair]
    for triple in layout.triples:
        for order in permutations(triple):
            index = layout.triple_order_index[(triple, order)]
            coefficients[index] = relaxation.triple_coefficients[(triple, order)]
    if any(value is None for value in coefficients):
        raise RuntimeError("failed to populate every local-order feature coordinate")
    return tuple(coefficients)


def equality_constrained_quadratic_projection(
    gram: Any,
    cross: Any,
    target_norm_squared: float,
    equalities: Any,
    rhs: Any,
) -> AffineProjectionResult:
    """Minimize ``||d-Fz||`` under ``Az=b`` from ``F^T F`` and ``F^T d``.

    The KKT system is solved by least squares because the local equalities can be
    redundant.  The objective remains convex because ``gram`` is a Gram matrix.
    """

    q = np.asarray(gram, dtype=np.float64)
    c = np.asarray(cross, dtype=np.float64)
    a = np.asarray(equalities, dtype=np.float64)
    b = np.asarray(rhs, dtype=np.float64)
    dimension = c.size
    if q.shape != (dimension, dimension):
        raise ValueError("gram shape must match cross dimension")
    if a.ndim != 2 or a.shape[1] != dimension or b.shape != (a.shape[0],):
        raise ValueError("equality system shape mismatch")
    if not (
        np.isfinite(q).all()
        and np.isfinite(c).all()
        and np.isfinite(a).all()
        and np.isfinite(b).all()
        and math.isfinite(float(target_norm_squared))
    ):
        raise ValueError("quadratic projection inputs must be finite")
    if not np.allclose(q, q.T, rtol=1e-10, atol=1e-12):
        raise ValueError("gram matrix must be symmetric")

    zero = np.zeros((a.shape[0], a.shape[0]), dtype=np.float64)
    kkt = np.block([[q, a.T], [a, zero]])
    right = np.concatenate([c, b])
    solved, *_ = np.linalg.lstsq(kkt, right, rcond=None)
    z = solved[:dimension]
    multipliers = solved[dimension:]

    quadratic = float(z @ q @ z)
    linear = float(2.0 * c @ z)
    squared = float(target_norm_squared) - linear + quadratic
    cancellation_scale = max(1.0, abs(float(target_norm_squared)), abs(linear), abs(quadratic))
    tolerance = 1e-10 * cancellation_scale
    if squared < -tolerance:
        raise FloatingPointError("affine projection distance became materially negative")
    squared = max(0.0, squared)
    equality_residual = float(np.linalg.norm(a @ z - b))
    stationarity_residual = float(np.linalg.norm(q @ z + a.T @ multipliers - c))
    return AffineProjectionResult(
        distance=math.sqrt(squared),
        squared_distance=squared,
        solution=z,
        equality_residual_norm=equality_residual,
        stationarity_residual_norm=stationarity_residual,
    )


def candidate_last_affine_lower_bounds(
    gram: Any,
    cross: Any,
    target_norm_squared: float,
    layout: K3LocalCoordinateLayout,
) -> dict[str, AffineProjectionResult]:
    """Return one conservative K3 discrete-codebook lower bound per final stage."""

    results: dict[str, AffineProjectionResult] = {}
    for stage in layout.stages:
        equalities, rhs = k3_local_equalities(layout, last_stage=stage)
        results[stage] = equality_constrained_quadratic_projection(
            gram,
            cross,
            target_norm_squared,
            equalities,
            rhs,
        )
    return results
