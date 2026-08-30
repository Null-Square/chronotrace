"""Proof-safe multi-witness distance bounds over K-local chronology relaxations.

For unit model-space witnesses ``u_j``, define the projected residual coordinates

    s_j(x) = <u_j, y - P_K(x)> = a_j + c_j^T x,

where ``x`` is a feasible K-local order-marginal vector.  Every true chronology induces
one feasible ``x`` and every residual obeys

    ||y - P_K(x)||_2 >= max_j |s_j(x)|.

Therefore minimizing the infinity norm of the witness-projection vector over the local
relaxation gives a conservative class-distance lower bound.  The optimization is a linear
program of polynomial size for fixed K.

Numerical HiGHS duals are treated only as proposals.  The reported certificate clips and
renormalizes inequality multipliers into the L1 unit ball, recomputes reduced costs, and
uses one simplex-minimum correction per local subset.  Thus small dual infeasibilities do
not turn into optimistic chronology certificates.
"""

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
class LocalOrderMultiWitnessCertificate:
    """Infinity-norm relaxation solution with an independently corrected dual bound."""

    primal_objective: float
    certified_lower_bound: float
    euclidean_distance_lower_bound: float
    primal_dual_gap: float
    equality_residual_max: float
    inequality_residual_max: float
    minimum_weight: float
    raw_dual_objective: float
    dual_l1_mass: float
    minimum_reduced_cost: float
    t_reduced_cost: float
    subset_residual_correction: float
    weights: np.ndarray
    equality_dual: np.ndarray
    inequality_dual: np.ndarray


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


def solve_local_order_multi_witness_lp(
    hierarchy: LocalOrderHierarchy,
    constants: Any,
    coefficients: Any,
    *,
    last_stage: str | None = None,
    certificate_guard: float = 1e-10,
) -> LocalOrderMultiWitnessCertificate:
    """Lower-bound chronology-class distance using a bank of unit witnesses.

    ``constants[j]`` and ``coefficients[j]`` define the j-th projected residual objective
    over the local-order hierarchy.  The primal minimizes ``t`` subject to

        -t <= constants[j] + coefficients[j] @ x <= t

    for every witness, the hierarchy equalities, ``x >= 0``, and ``t >= 0``.

    Since the local-order feasible set contains every global chronology and each unit
    witness projection is bounded by Euclidean residual norm, the LP optimum is a valid
    lower bound on the exact class distance.  A corrected dual value, rather than the raw
    numerical primal optimum, is returned as the proof-safe certificate.
    """

    try:
        from scipy.optimize import linprog
    except ImportError as exc:  # pragma: no cover - optional research dependency
        message = "Install the research dependencies with: pip install -e '.[dev]'"
        raise RuntimeError(message) from exc

    offsets = np.asarray(constants, dtype=np.float64)
    matrix = np.asarray(coefficients, dtype=np.float64)
    guard = float(certificate_guard)
    if offsets.ndim != 1 or offsets.size < 1 or not np.isfinite(offsets).all():
        raise ValueError("constants must contain at least one finite witness offset")
    if matrix.shape != (offsets.size, hierarchy.dimension) or not np.isfinite(matrix).all():
        raise ValueError("coefficients must be finite witness-by-hierarchy values")
    if not math.isfinite(guard) or guard < 0.0:
        raise ValueError("certificate_guard must be finite and non-negative")

    equality_matrix, equality_rhs = _combined_equalities(hierarchy, last_stage)
    variable_count = hierarchy.dimension + 1
    objective = np.zeros(variable_count, dtype=np.float64)
    objective[-1] = 1.0
    equality_augmented = np.column_stack(
        (equality_matrix, np.zeros(equality_matrix.shape[0], dtype=np.float64))
    )

    inequality_rows: list[np.ndarray] = []
    inequality_rhs: list[float] = []
    for offset, row in zip(offsets, matrix, strict=True):
        inequality_rows.append(np.concatenate((row, np.asarray([-1.0]))))
        inequality_rhs.append(-float(offset))
        inequality_rows.append(np.concatenate((-row, np.asarray([-1.0]))))
        inequality_rhs.append(float(offset))
    inequality_matrix = np.stack(inequality_rows)
    inequality_rhs_array = np.asarray(inequality_rhs, dtype=np.float64)

    result = linprog(
        objective,
        A_ub=inequality_matrix,
        b_ub=inequality_rhs_array,
        A_eq=equality_augmented,
        b_eq=equality_rhs,
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"multi-witness local-order LP failed: {result.message}")

    primal_values = np.asarray(result.x, dtype=np.float64)
    equality_dual = np.asarray(result.eqlin.marginals, dtype=np.float64)
    proposed_inequality_dual = np.asarray(result.ineqlin.marginals, dtype=np.float64)
    if primal_values.shape != (variable_count,):
        raise RuntimeError("multi-witness LP returned unexpected primal shape")
    if equality_dual.shape != equality_rhs.shape:
        raise RuntimeError("multi-witness LP returned unexpected equality-dual shape")
    if proposed_inequality_dual.shape != inequality_rhs_array.shape:
        raise RuntimeError("multi-witness LP returned unexpected inequality-dual shape")
    if (
        not np.isfinite(primal_values).all()
        or not np.isfinite(equality_dual).all()
        or not np.isfinite(proposed_inequality_dual).all()
    ):
        raise FloatingPointError("multi-witness LP returned non-finite values")

    # For a minimization primal with <= inequalities, valid multipliers are non-positive.
    # Clip any numerical sign violation, then enforce the t reduced-cost condition
    # 1 + sum(lambda) >= 0 by rescaling the total L1 mass to at most one.
    inequality_dual = np.minimum(proposed_inequality_dual, 0.0)
    dual_mass = float(np.sum(-inequality_dual))
    if dual_mass > 1.0:
        inequality_dual = inequality_dual / dual_mass
    certified_dual_mass = float(np.sum(-inequality_dual))

    reduced = (
        objective
        - equality_augmented.T @ equality_dual
        - inequality_matrix.T @ inequality_dual
    )
    reduced_x = reduced[:-1]
    reduced_t = float(reduced[-1])
    if reduced_t < -100.0 * max(guard, np.finfo(np.float64).eps):
        raise FloatingPointError("corrected multi-witness dual has negative t reduced cost")
    correction = _subset_reduced_cost_correction(hierarchy, reduced_x)
    raw_dual = float(equality_rhs @ equality_dual) + float(
        inequality_rhs_array @ inequality_dual
    )
    exact_arithmetic_lower = raw_dual + correction
    scale = max(
        1.0,
        abs(exact_arithmetic_lower),
        float(np.max(np.abs(offsets))),
        float(np.max(np.abs(matrix))),
    )
    certified = exact_arithmetic_lower - guard * scale

    weights = primal_values[:-1]
    primal = float(primal_values[-1])
    equality_residual = equality_augmented @ primal_values - equality_rhs
    inequality_residual = inequality_matrix @ primal_values - inequality_rhs_array
    equality_residual_max = (
        float(np.max(np.abs(equality_residual))) if equality_residual.size else 0.0
    )
    inequality_residual_max = max(
        0.0,
        float(np.max(inequality_residual)) if inequality_residual.size else 0.0,
    )
    if certified > primal + 100.0 * guard * scale:
        raise FloatingPointError("corrected multi-witness lower bound exceeds numerical primal")

    return LocalOrderMultiWitnessCertificate(
        primal_objective=primal,
        certified_lower_bound=certified,
        euclidean_distance_lower_bound=max(0.0, certified),
        primal_dual_gap=primal - certified,
        equality_residual_max=equality_residual_max,
        inequality_residual_max=inequality_residual_max,
        minimum_weight=float(np.min(weights)),
        raw_dual_objective=raw_dual,
        dual_l1_mass=certified_dual_mass,
        minimum_reduced_cost=float(np.min(reduced_x)),
        t_reduced_cost=reduced_t,
        subset_residual_correction=correction,
        weights=weights,
        equality_dual=equality_dual,
        inequality_dual=inequality_dual,
    )
