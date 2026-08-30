"""Proof-safe combinations of witness projections for chronology classes.

If ``u_j`` are unit witness directions and ``alpha`` has ``||alpha||_1 <= 1``, then

    v = sum_j alpha_j u_j

satisfies ``||v||_2 <= 1`` without needing the witness Gram matrix.  Therefore every
residual ``r`` obeys

    ||r||_2 >= <v, r> = sum_j alpha_j <u_j, r>.

For a convex hull of finitely many candidate residuals, the minimum of the combined
linear score occurs at a vertex.  A numerical LP may propose ``alpha``; certification
renormalizes it into the L1 unit ball and independently recomputes the minimum vertex
support score.  The numerical optimizer is therefore not trusted as the certificate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MultiWitnessVertexCertificate:
    """Numerical witness-game proposal plus independently recomputed lower bound."""

    coefficients: np.ndarray
    l1_norm: float
    numerical_game_value: float
    certified_lower_bound: float
    euclidean_distance_lower_bound: float
    minimum_vertex_support: float
    vertex_support_scores: np.ndarray


def _validate_scores(scores: Any) -> np.ndarray:
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("scores must be a non-empty witness-by-vertex matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("scores must be finite")
    return matrix


def certify_l1_witness_combination(
    scores: Any,
    coefficients: Any,
    *,
    certificate_guard: float = 1e-12,
    numerical_game_value: float = float("nan"),
) -> MultiWitnessVertexCertificate:
    """Certify one L1-bounded combination from witness-by-vertex residual scores.

    ``scores[j, i]`` is ``<u_j, y-q_i>`` for unit witnesses ``u_j``.  The supplied
    coefficients may come from any heuristic or numerical solver.  They are rescaled into
    the L1 unit ball before the support lower bound is independently recomputed.
    """

    matrix = _validate_scores(scores)
    alpha = np.asarray(coefficients, dtype=np.float64)
    if alpha.shape != (matrix.shape[0],) or not np.isfinite(alpha).all():
        raise ValueError("coefficients must be one finite value per witness")
    guard = float(certificate_guard)
    if not math.isfinite(guard) or guard < 0.0:
        raise ValueError("certificate_guard must be finite and non-negative")

    l1 = float(np.sum(np.abs(alpha)))
    if l1 > 1.0:
        alpha = alpha / l1
    certified_l1 = float(np.sum(np.abs(alpha)))
    support = matrix.T @ alpha
    minimum = float(np.min(support))
    scale = max(1.0, abs(minimum), float(np.max(np.abs(matrix))))
    certified = minimum - guard * scale
    return MultiWitnessVertexCertificate(
        coefficients=alpha,
        l1_norm=certified_l1,
        numerical_game_value=float(numerical_game_value),
        certified_lower_bound=certified,
        euclidean_distance_lower_bound=max(0.0, certified),
        minimum_vertex_support=minimum,
        vertex_support_scores=support,
    )


def optimize_l1_witness_combination(
    scores: Any,
    *,
    certificate_guard: float = 1e-12,
) -> MultiWitnessVertexCertificate:
    """Propose the strongest finite-vertex L1 witness mixture, then certify it.

    The proposal solves

        max_{||alpha||_1 <= 1} min_i alpha^T scores[:, i].

    SciPy/HiGHS is used only to obtain a candidate ``alpha``.  The returned lower bound is
    produced by :func:`certify_l1_witness_combination`, which rescales the coefficients and
    recomputes every vertex support score independently.
    """

    matrix = _validate_scores(scores)
    try:
        from scipy.optimize import linprog
    except ImportError as exc:  # pragma: no cover - optional research dependency
        raise RuntimeError("Install research dependencies with: pip install -e '.[dev]'") from exc

    witness_count, vertex_count = matrix.shape
    # Variables are alpha_plus, alpha_minus, and z. Maximize z by minimizing -z.
    objective = np.concatenate((np.zeros(2 * witness_count), np.asarray([-1.0])))
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for vertex in range(vertex_count):
        column = matrix[:, vertex]
        # z <= (alpha_plus-alpha_minus)^T column.
        rows.append(np.concatenate((-column, column, np.asarray([1.0]))))
        rhs.append(0.0)
    # ||alpha||_1 <= 1 in split-variable representation.
    rows.append(np.concatenate((np.ones(2 * witness_count), np.asarray([0.0]))))
    rhs.append(1.0)

    result = linprog(
        objective,
        A_ub=np.stack(rows),
        b_ub=np.asarray(rhs, dtype=np.float64),
        bounds=[(0.0, None)] * (2 * witness_count) + [(None, None)],
        method="highs",
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"multi-witness LP failed: {result.message}")
    values = np.asarray(result.x, dtype=np.float64)
    if not np.isfinite(values).all():
        raise FloatingPointError("multi-witness LP returned non-finite values")
    alpha = values[:witness_count] - values[witness_count : 2 * witness_count]
    return certify_l1_witness_combination(
        matrix,
        alpha,
        certificate_guard=certificate_guard,
        numerical_game_value=float(values[-1]),
    )


def combine_linear_witness_objectives(
    constants: Any,
    coefficients: Any,
    witness_coefficients: Any,
) -> tuple[float, np.ndarray]:
    """Combine projected linear objectives using an L1-bounded witness mixture.

    This helper does not solve or certify the chronology relaxation itself.  The returned
    objective can be passed to the existing proof-safe local-order LP.  Coefficients are
    normalized into the L1 unit ball so the combined model-space direction has norm at
    most one by the triangle inequality.
    """

    offsets = np.asarray(constants, dtype=np.float64)
    matrix = np.asarray(coefficients, dtype=np.float64)
    alpha = np.asarray(witness_coefficients, dtype=np.float64)
    if offsets.ndim != 1 or matrix.ndim != 2 or matrix.shape[0] != offsets.size:
        raise ValueError("constants and coefficient rows must have one row per witness")
    if alpha.shape != offsets.shape:
        raise ValueError("witness coefficients must match witness count")
    if not np.isfinite(offsets).all() or not np.isfinite(matrix).all() or not np.isfinite(alpha).all():
        raise ValueError("linear witness objectives must be finite")
    l1 = float(np.sum(np.abs(alpha)))
    if l1 > 1.0:
        alpha = alpha / l1
    return float(alpha @ offsets), alpha @ matrix
