"""Reverse-operator tools for training-history peeling.

A one-step gradient update has the form

    T(x) = x - eta * grad_loss(x).

When ``eta * L < 1`` for an ``L``-Lipschitz gradient, ``T`` is injective and the
inverse state is the unique fixed point of

    x = y + eta * grad_loss(x).

The helpers here deliberately separate inversion from chronology scoring.  They
make no assumption that a wrong last-stage candidate must be distinguishable;
a zero residual tie is reported as non-identifiable rather than broken by
floating-point order.
"""

from __future__ import annotations

import collections.abc
from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class InverseStepResult:
    state: Array
    converged: bool
    iterations: int
    fixed_point_residual: float


@dataclass(frozen=True)
class LastStageDecision:
    stage: str | None
    predecessor_index: int | None
    best_residual: float
    runner_up_residual: float
    margin: float
    identifiable: bool
    residuals: dict[str, float]


def invert_gradient_step_fixed_point(
    target: Array,
    gradient: collections.abc.Callable[[Array], Array],
    *,
    learning_rate: float,
    tolerance: float = 1e-12,
    max_iterations: int = 1000,
) -> InverseStepResult:
    """Invert ``y = x - eta * gradient(x)`` by fixed-point iteration.

    The update ``x_{k+1} = y + eta * gradient(x_k)`` is a contraction whenever
    the gradient is L-Lipschitz and ``eta * L < 1``.  Callers must verify that
    regime for their scientific setting; this routine reports convergence but
    does not infer a global Lipschitz constant from samples.
    """

    y = np.asarray(target, dtype=np.float64)
    if y.ndim != 1 or y.size == 0 or not np.isfinite(y).all():
        raise ValueError("target must be a finite non-empty vector")
    eta = float(learning_rate)
    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if tolerance <= 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")

    state = y.copy()
    for iteration in range(1, max_iterations + 1):
        grad = np.asarray(gradient(state), dtype=np.float64)
        if grad.shape != state.shape or not np.isfinite(grad).all():
            raise ValueError("gradient returned an invalid vector")
        updated = y + eta * grad
        residual = float(np.linalg.norm(updated - state))
        state = updated
        if residual <= tolerance:
            return InverseStepResult(state, True, iteration, residual)
    return InverseStepResult(state, False, max_iterations, residual)


def decode_last_stage_from_predecessor_sets(
    inverted_states: collections.abc.Mapping[str, Array],
    predecessor_sets: collections.abc.Mapping[str, collections.abc.Sequence[Array]],
    *,
    tie_tolerance: float = 0.0,
) -> LastStageDecision:
    """Choose the last stage by distance to its reachable predecessor set.

    For candidate last stage ``j``, the score is

        r_j = min_z || T_j^{-1}(y) - z ||_2,

    where ``z`` ranges over the supplied predecessor states for all histories
    not containing ``j``.  Exact or tolerance-level ties are non-identifiable.
    """

    if set(inverted_states) != set(predecessor_sets):
        raise ValueError("inverted state and predecessor stage keys must match")
    if not inverted_states:
        raise ValueError("at least one candidate last stage is required")
    if tie_tolerance < 0.0 or not np.isfinite(tie_tolerance):
        raise ValueError("tie_tolerance must be finite and non-negative")

    residuals: dict[str, float] = {}
    predecessor_indices: dict[str, int] = {}
    for stage in sorted(inverted_states):
        state = np.asarray(inverted_states[stage], dtype=np.float64)
        candidates = tuple(predecessor_sets[stage])
        if state.ndim != 1 or not np.isfinite(state).all():
            raise ValueError("inverted states must be finite vectors")
        if not candidates:
            raise ValueError("every last-stage candidate needs predecessor states")
        distances = []
        for candidate in candidates:
            vector = np.asarray(candidate, dtype=np.float64)
            if vector.shape != state.shape or not np.isfinite(vector).all():
                raise ValueError("predecessor states must match inverted-state shape")
            distances.append(float(np.linalg.norm(state - vector)))
        index = int(np.argmin(distances))
        residuals[stage] = distances[index]
        predecessor_indices[stage] = index

    ranking = sorted(residuals, key=lambda stage: (residuals[stage], stage))
    best_stage = ranking[0]
    best = residuals[best_stage]
    runner = residuals[ranking[1]] if len(ranking) > 1 else float("inf")
    margin = runner - best
    identifiable = margin > tie_tolerance
    return LastStageDecision(
        stage=best_stage if identifiable else None,
        predecessor_index=predecessor_indices[best_stage] if identifiable else None,
        best_residual=best,
        runner_up_residual=runner,
        margin=margin,
        identifiable=identifiable,
        residuals=residuals,
    )
