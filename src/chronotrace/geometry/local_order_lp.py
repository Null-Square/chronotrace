"""Numerical LP proposals with independently corrected local-order dual certificates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from chronotrace.geometry.local_order_hierarchy import (
    LocalOrderHierarchy,
    build_last_stage_equalities,
    build_local_order_equalities,
)


@dataclass(frozen=True)
class LocalOrderLPCertificate:
    """One LP solution plus a conservative lower bound reconstructed from its dual."""

    primal_objective: float
    certified_lower_bound: float
    primal_dual_gap: float
    equality_residual_max: float
    minimum_weight: float
    raw_dual_objective: float
    minimum_reduced_cost: float
    subset_residual_correction: float
    weights: np.ndarray
    equality_dual: np.ndarray


def _combined_equalities(
    hierarchy: LocalOrderHierarchy,
    last_stage: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    base = build_local_order_equalities(hierarchy)
    if last_stage is None:
        return base.matrix, base.rhs
    last = build_last_stage_equalities(hierarchy, last_stage)
    return (
        np.vstack((base.matrix, last.matrix)),
        np.concatenate((base.rhs, last.rhs)),
    )


def _subset_reduced_cost_correction(
    hierarchy: LocalOrderHierarchy,
    reduced_costs: np.ndarray,
) -> float:
    correction = 0.0
    for subset in hierarchy.subsets:
        correction += min(
            float(reduced_costs[hierarchy.coordinate_index[word]])
            for word in hierarchy.orders_by_subset[subset]
        )
    return correction


def solve_local_order_lp(
    hierarchy: LocalOrderHierarchy,
    constant: float,
    coefficients: Any,
    *,
    last_stage: str | None = None,
    certificate_guard: float = 1e-10,
) -> LocalOrderLPCertificate:
    """Solve the local-order LP and return a corrected dual lower bound.

    SciPy/HiGHS is used only to propose primal and equality-dual vectors.  Certification
    does not assume exact dual feasibility.  For any returned equality dual ``y``, let
    ``delta = c - A.T @ y``.  Every feasible local-order point has nonnegative coordinates
    and one unit-mass simplex per subset, so

        delta.T @ x >= sum_S min_{sigma in orders(S)} delta[S,sigma].

    Therefore ``b.T@y`` plus that subset correction is a valid lower bound in exact
    arithmetic even when the numerical dual has small negative reduced costs.  A final
    floating-point guard is subtracted from the reported certificate.
    """

    try:
        from scipy.optimize import linprog
    except ImportError as exc:  # pragma: no cover - optional research dependency
        raise RuntimeError("Install the research dependencies with: pip install -e '.[dev]'") from exc

    c = np.asarray(coefficients, dtype=np.float64)
    if c.shape != (hierarchy.dimension,) or not np.isfinite(c).all():
        raise ValueError("coefficients must be a finite hierarchy-sized vector")
    offset = float(constant)
    guard = float(certificate_guard)
    if not math.isfinite(offset):
        raise ValueError("constant must be finite")
    if not math.isfinite(guard) or guard < 0.0:
        raise ValueError("certificate_guard must be finite and non-negative")

    matrix, rhs = _combined_equalities(hierarchy, last_stage)
    result = linprog(
        c,
        A_eq=matrix,
        b_eq=rhs,
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"local-order LP failed: {result.message}")
    weights = np.asarray(result.x, dtype=np.float64)
    dual = np.asarray(result.eqlin.marginals, dtype=np.float64)
    if weights.shape != c.shape or dual.shape != rhs.shape:
        raise RuntimeError("local-order LP returned unexpected primal or dual shapes")
    if not np.isfinite(weights).all() or not np.isfinite(dual).all():
        raise FloatingPointError("local-order LP returned non-finite primal or dual values")

    residual = matrix @ weights - rhs
    equality_residual = float(np.max(np.abs(residual))) if residual.size else 0.0
    reduced = c - matrix.T @ dual
    correction = _subset_reduced_cost_correction(hierarchy, reduced)
    raw_dual = offset + float(rhs @ dual)
    exact_arithmetic_lower = raw_dual + correction
    scale = max(1.0, abs(exact_arithmetic_lower), abs(offset), float(np.max(np.abs(c))))
    certified = exact_arithmetic_lower - guard * scale
    primal = offset + float(c @ weights)
    if certified > primal + 100.0 * guard * scale:
        raise FloatingPointError("corrected dual lower bound exceeds numerical primal objective")

    return LocalOrderLPCertificate(
        primal_objective=primal,
        certified_lower_bound=certified,
        primal_dual_gap=primal - certified,
        equality_residual_max=equality_residual,
        minimum_weight=float(np.min(weights)),
        raw_dual_objective=raw_dual,
        minimum_reduced_cost=float(np.min(reduced)),
        subset_residual_correction=correction,
        weights=weights,
        equality_dual=dual,
    )
