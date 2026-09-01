# ruff: noqa: I001
"""Reverse-operator and reachable-set tools for training-history peeling.

A one-step gradient update has the form

    T(x) = x - eta * grad_loss(x).

When ``eta * L < 1`` for an ``L``-Lipschitz gradient, ``T`` is injective and the
inverse state is the unique fixed point of

    x = y + eta * grad_loss(x).

Raw fixed-point iteration needs the stronger contraction condition. The same inverse
equation is also the stationarity condition of the inverse potential

    Psi_y(x) = 0.5 * ||x-y||^2 - eta * loss(x),

whose gradient is ``x-y-eta*grad_loss(x)``. A line-search descent on this potential can
therefore solve locally invertible cases in which Picard iteration is non-contractive.

When a finite predecessor codebook is already available, inversion is unnecessary. For a
candidate predecessor ``z`` and candidate final stage ``j``, evaluate the forward residual

    rho(z, j; y) = ||T_j(z) - y||.

The true finite hypothesis has zero residual in exact arithmetic. This replaces an
ill-conditioned root-finding problem with finite reachable-set membership. Exact or
 tolerance-level ties are reported as non-identifiable rather than broken by ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class InverseStepResult:
    state: Array
    converged: bool
    iterations: int
    fixed_point_residual: float


@dataclass(frozen=True)
class InverseDescentResult:
    """Evidence from line-search descent on the inverse potential."""

    state: Array
    converged: bool
    iterations: int
    residual_norm: float
    line_search_failed: bool
    objective_trace: tuple[float, ...]
    residual_trace: tuple[float, ...]
    accepted_step_trace: tuple[float, ...]


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
    gradient: Any,
    *,
    learning_rate: float,
    tolerance: float = 1e-12,
    max_iterations: int = 1000,
) -> InverseStepResult:
    """Invert ``y = x - eta * gradient(x)`` by fixed-point iteration.

    The update ``x_{k+1} = y + eta * gradient(x_k)`` is a contraction whenever
    the gradient is L-Lipschitz and ``eta * L < 1``. Callers must verify that
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
    residual = float("inf")
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


def invert_gradient_step_armijo(
    target: Array,
    loss: Any,
    gradient: Any,
    *,
    learning_rate: float,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
    initial_step: float = 1.0,
    armijo_constant: float = 1e-4,
    shrink_factor: float = 0.5,
    minimum_step: float = 2.0**-20,
) -> InverseDescentResult:
    """Solve one inverse SGD step by Armijo descent on the inverse potential.

    The optimized scalar is

        Psi_y(x) = 0.5 ||x-y||^2 - eta L(x),

    with gradient ``F(x)=x-y-eta*grad L(x)``. Thus a converged stationary point is a
    preimage of ``y`` under the one-step gradient map. Unlike Picard iteration, this
    method does not require ``eta*H`` itself to be a contraction. It is still a local
    solver: non-convex inverse potentials can have multiple stationary points.
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
    if not np.isfinite(initial_step) or initial_step <= 0.0:
        raise ValueError("initial_step must be finite and positive")
    if not 0.0 < armijo_constant < 1.0:
        raise ValueError("armijo_constant must lie in (0, 1)")
    if not 0.0 < shrink_factor < 1.0:
        raise ValueError("shrink_factor must lie in (0, 1)")
    if not np.isfinite(minimum_step) or minimum_step <= 0.0:
        raise ValueError("minimum_step must be finite and positive")

    def objective(state: Array) -> float:
        displacement = state - y
        value = 0.5 * float(np.dot(displacement, displacement)) - eta * float(loss(state))
        if not np.isfinite(value):
            raise FloatingPointError("inverse potential became non-finite")
        return value

    state = y.copy()
    objective_value = objective(state)
    objectives = [objective_value]
    residuals: list[float] = []
    accepted_steps: list[float] = []

    for iteration in range(max_iterations + 1):
        grad_loss = np.asarray(gradient(state), dtype=np.float64)
        if grad_loss.shape != state.shape or not np.isfinite(grad_loss).all():
            raise ValueError("gradient returned an invalid vector")
        residual_vector = state - y - eta * grad_loss
        residual_norm = float(np.linalg.norm(residual_vector))
        if not np.isfinite(residual_norm):
            raise FloatingPointError("inverse residual became non-finite")
        residuals.append(residual_norm)
        if residual_norm <= tolerance:
            return InverseDescentResult(
                state=state,
                converged=True,
                iterations=iteration,
                residual_norm=residual_norm,
                line_search_failed=False,
                objective_trace=tuple(objectives),
                residual_trace=tuple(residuals),
                accepted_step_trace=tuple(accepted_steps),
            )
        if iteration == max_iterations:
            break

        direction = -residual_vector
        directional_derivative = -residual_norm**2
        step = float(initial_step)
        accepted = False
        while step >= minimum_step:
            trial = state + step * direction
            trial_objective = objective(trial)
            if trial_objective <= (
                objective_value + armijo_constant * step * directional_derivative
            ):
                state = trial
                objective_value = trial_objective
                objectives.append(objective_value)
                accepted_steps.append(step)
                accepted = True
                break
            step *= shrink_factor
        if not accepted:
            return InverseDescentResult(
                state=state,
                converged=False,
                iterations=iteration,
                residual_norm=residual_norm,
                line_search_failed=True,
                objective_trace=tuple(objectives),
                residual_trace=tuple(residuals),
                accepted_step_trace=tuple(accepted_steps),
            )

    return InverseDescentResult(
        state=state,
        converged=False,
        iterations=max_iterations,
        residual_norm=residuals[-1],
        line_search_failed=False,
        objective_trace=tuple(objectives),
        residual_trace=tuple(residuals),
        accepted_step_trace=tuple(accepted_steps),
    )


def _decode_stage_residuals(
    residual_lists: dict[str, list[float]],
    *,
    tie_tolerance: float,
) -> LastStageDecision:
    if not residual_lists:
        raise ValueError("at least one candidate last stage is required")
    if tie_tolerance < 0.0 or not np.isfinite(tie_tolerance):
        raise ValueError("tie_tolerance must be finite and non-negative")

    residuals: dict[str, float] = {}
    predecessor_indices: dict[str, int] = {}
    for stage in sorted(residual_lists):
        values = residual_lists[stage]
        if not values:
            raise ValueError("every last-stage candidate needs predecessor states")
        if not all(np.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("candidate residuals must be finite and non-negative")
        index = int(np.argmin(values))
        residuals[stage] = float(values[index])
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


def decode_last_stage_from_predecessor_sets(
    inverted_states: Any,
    predecessor_sets: Any,
    *,
    tie_tolerance: float = 0.0,
) -> LastStageDecision:
    """Choose the last stage by distance from an inverse state to predecessor sets.

    For candidate last stage ``j``, the score is

        r_j = min_z || T_j^{-1}(y) - z ||_2,

    where ``z`` ranges over the supplied predecessor states for all histories
    not containing ``j``.
    """

    if set(inverted_states) != set(predecessor_sets):
        raise ValueError("inverted state and predecessor stage keys must match")

    residual_lists: dict[str, list[float]] = {}
    for stage in sorted(inverted_states):
        state = np.asarray(inverted_states[stage], dtype=np.float64)
        candidates = tuple(predecessor_sets[stage])
        if state.ndim != 1 or not np.isfinite(state).all():
            raise ValueError("inverted states must be finite vectors")
        distances = []
        for candidate in candidates:
            vector = np.asarray(candidate, dtype=np.float64)
            if vector.shape != state.shape or not np.isfinite(vector).all():
                raise ValueError("predecessor states must match inverted-state shape")
            distances.append(float(np.linalg.norm(state - vector)))
        residual_lists[stage] = distances

    return _decode_stage_residuals(residual_lists, tie_tolerance=tie_tolerance)


def decode_last_stage_from_forward_reachable_sets(
    target: Any,
    reachable_sets: Any,
    *,
    tie_tolerance: float = 0.0,
) -> LastStageDecision:
    """Choose the final stage by direct finite reachable-set membership.

    ``reachable_sets[j][k]`` is the endpoint obtained by applying candidate final stage
    ``j`` to candidate predecessor ``k``. The score is

        r_j = min_k || reachable_sets[j][k] - target ||_2.

    No inverse solve is performed. If the true predecessor is in the supplied finite
    codebook and the corresponding one-step map is evaluated exactly, its residual is
    zero. A positive margin over all competing reachable endpoints certifies the finite
    hypothesis under the supplied interface.
    """

    y = np.asarray(target, dtype=np.float64)
    if y.ndim != 1 or y.size == 0 or not np.isfinite(y).all():
        raise ValueError("target must be a finite non-empty vector")

    residual_lists: dict[str, list[float]] = {}
    for stage in sorted(reachable_sets):
        endpoints = tuple(reachable_sets[stage])
        distances = []
        for endpoint in endpoints:
            vector = np.asarray(endpoint, dtype=np.float64)
            if vector.shape != y.shape or not np.isfinite(vector).all():
                raise ValueError("reachable endpoints must match target shape")
            distances.append(float(np.linalg.norm(vector - y)))
        residual_lists[stage] = distances

    return _decode_stage_residuals(residual_lists, tie_tolerance=tie_tolerance)
