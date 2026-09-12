"""Exact affine controls for sequential-training chronology.

These utilities implement the solvable quadratic/full-batch-GD model described in
``docs/AFFINE_TRAINING_THEORY.md``. They are analytic controls, not a chronology decoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AffineStage:
    """Finite training stage ``theta -> matrix @ theta + offset``."""

    matrix: np.ndarray
    offset: np.ndarray

    def apply(self, theta: np.ndarray) -> np.ndarray:
        return self.matrix @ theta + self.offset


def quadratic_gd_stage(
    hessian: np.ndarray,
    optimum: np.ndarray,
    *,
    learning_rate: float,
    steps: int,
) -> AffineStage:
    """Return the exact macro operator for repeated GD on a quadratic loss.

    The loss is ``1/2 (theta-a)^T H (theta-a)`` and the same full-batch gradient
    step is repeated ``steps`` times with constant learning rate.
    """

    hessian = np.asarray(hessian, dtype=np.float64)
    optimum = np.asarray(optimum, dtype=np.float64)
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("hessian must be square")
    if optimum.shape != (hessian.shape[0],):
        raise ValueError("optimum dimension must match hessian")
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    if learning_rate < 0.0:
        raise ValueError("learning_rate must be nonnegative")

    identity = np.eye(hessian.shape[0], dtype=np.float64)
    micro = identity - float(learning_rate) * hessian
    macro = np.linalg.matrix_power(micro, int(steps))
    offset = (identity - macro) @ optimum
    return AffineStage(matrix=macro, offset=offset)


def compose_affine(later: AffineStage, earlier: AffineStage) -> AffineStage:
    """Return the exact composition ``later(earlier(theta))``."""

    if later.matrix.shape != earlier.matrix.shape:
        raise ValueError("stage matrix dimensions must match")
    return AffineStage(
        matrix=later.matrix @ earlier.matrix,
        offset=later.matrix @ earlier.offset + later.offset,
    )


def chronology_endpoint(
    theta0: np.ndarray,
    stages: dict[str, AffineStage],
    chronology: tuple[str, ...] | list[str] | str,
) -> np.ndarray:
    """Apply affine stages in chronology order."""

    theta = np.asarray(theta0, dtype=np.float64)
    for name in chronology:
        if name not in stages:
            raise ValueError(f"unknown stage: {name}")
        theta = stages[name].apply(theta)
    return theta


def two_stage_order_difference(
    theta: np.ndarray,
    first: AffineStage,
    second: AffineStage,
) -> np.ndarray:
    """Return ``second(first(theta)) - first(second(theta))`` exactly."""

    theta = np.asarray(theta, dtype=np.float64)
    return (
        (second.matrix @ first.matrix - first.matrix @ second.matrix) @ theta
        + (second.matrix - np.eye(second.matrix.shape[0])) @ first.offset
        - (first.matrix - np.eye(first.matrix.shape[0])) @ second.offset
    )


def common_continuation_difference(
    initial_difference: np.ndarray,
    continuation: AffineStage,
    *,
    repeats: int,
) -> np.ndarray:
    """Propagate a history difference through repeated identical continuation.

    The affine translation cancels between the two histories, so only the linear
    operator transports their difference.
    """

    if repeats < 0:
        raise ValueError("repeats must be nonnegative")
    delta = np.asarray(initial_difference, dtype=np.float64)
    return np.linalg.matrix_power(continuation.matrix, int(repeats)) @ delta


def scalar_recency_endpoint(
    theta0: float,
    optima: dict[str, float],
    chronology: tuple[str, ...] | list[str] | str,
    *,
    alpha: float,
) -> float:
    """Return the exact endpoint for equal scalar contraction stages.

    Each stage is ``theta -> alpha*theta + (1-alpha)*a_i``.
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0,1]")
    theta = float(theta0)
    for name in chronology:
        if name not in optima:
            raise ValueError(f"unknown stage: {name}")
        theta = alpha * theta + (1.0 - alpha) * float(optima[name])
    return theta


def spectral_half_life(eigenvalue_magnitude: float) -> float:
    """Return repeat count required for an affine history mode to halve.

    Returns infinity for a perfectly persistent mode of magnitude one and zero for a
    mode annihilated in one continuation application.
    """

    magnitude = float(eigenvalue_magnitude)
    if magnitude < 0.0 or magnitude > 1.0:
        raise ValueError("eigenvalue magnitude must lie in [0,1]")
    if magnitude == 1.0:
        return float("inf")
    if magnitude == 0.0:
        return 0.0
    return float(np.log(0.5) / np.log(magnitude))
