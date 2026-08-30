"""Polynomial-size local-order marginal hierarchies for chronology certificates.

For every unordered subset S of stages with |S| <= K, introduce one probability for each
ordering of S.  Global chronologies map to one-hot feasible vertices.  Simplex constraints
and marginal consistency between S and its one-element deletions define a conservative
local relaxation containing every global permutation.

The coordinate count is sum_{r=1..K} P(N,r), polynomial in N for fixed K.  At K=N the
highest-level simplex is a distribution over complete permutations and all lower marginals
are induced from it, so the hierarchy is exact (although factorial at that terminal level).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Any

import numpy as np


Word = tuple[str, ...]
Subset = tuple[str, ...]


@dataclass(frozen=True)
class LocalOrderHierarchy:
    """Coordinate layout for one K-local permutation-marginal relaxation."""

    stages: tuple[str, ...]
    max_degree: int
    subsets: tuple[Subset, ...]
    orders_by_subset: dict[Subset, tuple[Word, ...]]
    coordinate_words: tuple[Word, ...]
    coordinate_index: dict[Word, int]

    @property
    def dimension(self) -> int:
        return len(self.coordinate_words)


@dataclass(frozen=True)
class LinearEqualities:
    """Dense equality system A x = b used by small synthetic hierarchy tests."""

    matrix: np.ndarray
    rhs: np.ndarray


def local_order_coordinate_count(stage_count: int, max_degree: int) -> int:
    """Return sum(P(N,r), r=1..K), the hierarchy coordinate count."""

    count = int(stage_count)
    degree = int(max_degree)
    if count < 1 or degree < 1 or degree > count:
        raise ValueError("require 1 <= max_degree <= stage_count")
    return sum(math.perm(count, order) for order in range(1, degree + 1))


def build_local_order_hierarchy(
    stages: Sequence[str],
    *,
    max_degree: int,
) -> LocalOrderHierarchy:
    """Construct coordinates for all local orderings through degree K."""

    values = tuple(str(stage) for stage in stages)
    if len(values) < 2 or len(values) != len(set(values)):
        raise ValueError("stages must contain at least two unique names")
    degree = int(max_degree)
    if degree < 1 or degree > len(values):
        raise ValueError("max_degree must be between 1 and the number of stages")

    subsets: list[Subset] = []
    orders_by_subset: dict[Subset, tuple[Word, ...]] = {}
    coordinate_words: list[Word] = []
    for size in range(1, degree + 1):
        for subset in combinations(values, size):
            local_orders = tuple(permutations(subset))
            subsets.append(subset)
            orders_by_subset[subset] = local_orders
            coordinate_words.extend(local_orders)

    index = {word: position for position, word in enumerate(coordinate_words)}
    if len(index) != len(coordinate_words):
        raise RuntimeError("local-order coordinate words unexpectedly collided")
    if len(coordinate_words) != local_order_coordinate_count(len(values), degree):
        raise RuntimeError("local-order coordinate count drift")

    return LocalOrderHierarchy(
        stages=values,
        max_degree=degree,
        subsets=tuple(subsets),
        orders_by_subset=orders_by_subset,
        coordinate_words=tuple(coordinate_words),
        coordinate_index=index,
    )


def induced_local_order(chronology: Sequence[str], subset: Subset) -> Word:
    """Return the order induced by one complete chronology on a canonical subset."""

    history = tuple(chronology)
    position = {stage: index for index, stage in enumerate(history)}
    if len(position) != len(history) or any(stage not in position for stage in subset):
        raise ValueError("chronology does not contain the requested unique stages")
    return tuple(sorted(subset, key=position.__getitem__))


def local_order_vertex(
    chronology: Sequence[str],
    hierarchy: LocalOrderHierarchy,
) -> np.ndarray:
    """Encode a global chronology as a one-hot local-marginal vertex."""

    history = tuple(str(stage) for stage in chronology)
    if len(history) != len(hierarchy.stages) or set(history) != set(hierarchy.stages):
        raise ValueError("chronology must contain every hierarchy stage exactly once")
    weights = np.zeros(hierarchy.dimension, dtype=np.float64)
    for subset in hierarchy.subsets:
        word = induced_local_order(history, subset)
        weights[hierarchy.coordinate_index[word]] = 1.0
    return weights


def build_local_order_equalities(hierarchy: LocalOrderHierarchy) -> LinearEqualities:
    """Build simplex and adjacent-level marginal-consistency equalities."""

    rows: list[np.ndarray] = []
    rhs: list[float] = []

    for subset in hierarchy.subsets:
        row = np.zeros(hierarchy.dimension, dtype=np.float64)
        for word in hierarchy.orders_by_subset[subset]:
            row[hierarchy.coordinate_index[word]] = 1.0
        rows.append(row)
        rhs.append(1.0)

    for subset in hierarchy.subsets:
        if len(subset) < 2:
            continue
        for removed in subset:
            smaller = tuple(stage for stage in subset if stage != removed)
            for small_order in hierarchy.orders_by_subset[smaller]:
                row = np.zeros(hierarchy.dimension, dtype=np.float64)
                row[hierarchy.coordinate_index[small_order]] = 1.0
                for large_order in hierarchy.orders_by_subset[subset]:
                    restricted = tuple(stage for stage in large_order if stage != removed)
                    if restricted == small_order:
                        row[hierarchy.coordinate_index[large_order]] -= 1.0
                rows.append(row)
                rhs.append(0.0)

    matrix = np.stack(rows) if rows else np.zeros((0, hierarchy.dimension), dtype=np.float64)
    return LinearEqualities(matrix=matrix, rhs=np.asarray(rhs, dtype=np.float64))


def build_last_stage_equalities(
    hierarchy: LocalOrderHierarchy,
    last_stage: str,
) -> LinearEqualities:
    """Fix pair marginals so one stage follows every other stage."""

    stage = str(last_stage)
    if stage not in hierarchy.stages:
        raise ValueError("last_stage is outside the hierarchy")
    if hierarchy.max_degree < 2:
        raise ValueError("last-stage constraints require degree at least two")

    rows: list[np.ndarray] = []
    for other in hierarchy.stages:
        if other == stage:
            continue
        desired = (other, stage)
        row = np.zeros(hierarchy.dimension, dtype=np.float64)
        row[hierarchy.coordinate_index[desired]] = 1.0
        rows.append(row)
    return LinearEqualities(
        matrix=np.stack(rows),
        rhs=np.ones(len(rows), dtype=np.float64),
    )


def validate_local_order_weights(
    weights: Any,
    hierarchy: LocalOrderHierarchy,
    *,
    equalities: LinearEqualities | None = None,
    tolerance: float = 1e-10,
) -> None:
    """Validate nonnegativity plus local-order equality constraints."""

    vector = np.asarray(weights, dtype=np.float64)
    if vector.shape != (hierarchy.dimension,) or not np.isfinite(vector).all():
        raise ValueError("weights must be a finite hierarchy-sized vector")
    tol = float(tolerance)
    if tol < 0.0 or not math.isfinite(tol):
        raise ValueError("tolerance must be finite and non-negative")
    if float(np.min(vector)) < -tol:
        raise ValueError("local-order weights contain a negative coordinate")
    system = build_local_order_equalities(hierarchy) if equalities is None else equalities
    residual = system.matrix @ vector - system.rhs
    if residual.size and float(np.max(np.abs(residual))) > tol:
        raise ValueError("local-order marginal equalities are violated")


def projected_interaction_linear_objective(
    hierarchy: LocalOrderHierarchy,
    projected_interactions: Mapping[Word, float],
    *,
    target_minus_base_projection: float,
) -> tuple[float, np.ndarray]:
    """Build <u, y-P_K> as a linear objective over local-order marginals."""

    expected = set(hierarchy.coordinate_words)
    actual = set(projected_interactions)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"projected interaction table mismatch; missing={missing}, extra={extra}")
    constant = float(target_minus_base_projection)
    if not math.isfinite(constant):
        raise ValueError("target_minus_base_projection must be finite")
    coefficients = np.empty(hierarchy.dimension, dtype=np.float64)
    for word, index in hierarchy.coordinate_index.items():
        value = float(projected_interactions[word])
        if not math.isfinite(value):
            raise ValueError("projected interactions must be finite")
        coefficients[index] = -value
    return constant, coefficients


def evaluate_linear_objective(
    constant: float,
    coefficients: Any,
    weights: Any,
) -> float:
    """Evaluate one hierarchy linear objective."""

    coeff = np.asarray(coefficients, dtype=np.float64)
    vector = np.asarray(weights, dtype=np.float64)
    if coeff.shape != vector.shape or coeff.ndim != 1:
        raise ValueError("coefficient and weight vectors must have the same one-dimensional shape")
    return float(constant) + float(coeff @ vector)


def full_level_permutation_scores(
    hierarchy: LocalOrderHierarchy,
    constant: float,
    coefficients: Any,
    *,
    last_stage: str | None = None,
) -> dict[Word, float]:
    """Enumerate complete-permutation scores when K=N for exactness tests and diagnostics."""

    if hierarchy.max_degree != len(hierarchy.stages):
        raise ValueError("full-level scores require max_degree equal to stage count")
    stage = None if last_stage is None else str(last_stage)
    if stage is not None and stage not in hierarchy.stages:
        raise ValueError("last_stage is outside the hierarchy")
    result: dict[Word, float] = {}
    for history in permutations(hierarchy.stages):
        if stage is not None and history[-1] != stage:
            continue
        vertex = local_order_vertex(history, hierarchy)
        result[history] = evaluate_linear_objective(constant, coefficients, vertex)
    return result
