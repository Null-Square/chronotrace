"""Replay-free chronology geometry for finite training-stage operators.

A complete deterministic training stage is treated as a near-identity map

    F_D(theta) = theta + Delta_D(theta).

For two stages A and B,

    F_B(F_A(theta_0)) - F_A(F_B(theta_0))
      = J Delta_B Delta_A - J Delta_A Delta_B + O(||Delta||^3).

This lifts the one-update Lie-bracket decoder to arbitrary deterministic macro stages.
The directional derivatives can be obtained by differentiating through the stage or,
as implemented here, by symmetric finite differences. The latter avoids second-order
autograd requirements and works with training stacks whose fused kernels do not support
double backward.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import permutations
from typing import Any


@dataclass(frozen=True)
class OperatorPairGeometry:
    """Second-order midpoint and antisymmetric residual for two finite stage maps."""

    additive_reference: Any
    midpoint_reference: Any
    bracket: Any
    j_b_delta_a: Any
    j_a_delta_b: Any


@dataclass(frozen=True)
class OperatorPermutationDecode:
    """Nearest macro-operator chronology signature."""

    permutation: tuple[str, ...]
    best_error: float
    runner_up_error: float
    margin: float


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised in MVP environments
        raise RuntimeError('Install the MVP dependencies with: pip install -e ".[mvp]"') from exc
    return torch


def stage_displacement(stage_map: Callable[[Any], Any], theta0: Any) -> Any:
    """Return Delta_D(theta_0) = F_D(theta_0) - theta_0."""

    endpoint = stage_map(theta0)
    if endpoint.shape != theta0.shape:
        raise ValueError("stage map changed the parameter-vector shape")
    return endpoint - theta0


def displacement_jvp_finite_difference(
    stage_map: Callable[[Any], Any],
    theta0: Any,
    direction: Any,
    *,
    epsilon: float,
) -> Any:
    """Estimate J Delta_D(theta_0) @ direction by a centered finite difference."""

    eps = float(epsilon)
    if eps <= 0:
        raise ValueError("epsilon must be positive")
    torch = _require_torch()
    norm = float(torch.linalg.vector_norm(direction))
    if norm <= 0:
        raise ValueError("finite-difference direction must have non-zero norm")

    # Scale the perturbation by ||direction|| so epsilon has a stable interpretation
    # in parameter-space units while the returned derivative remains J @ direction.
    unit = direction / norm
    plus = stage_displacement(stage_map, theta0 + eps * unit)
    minus = stage_displacement(stage_map, theta0 - eps * unit)
    derivative_along_unit = (plus - minus) / (2.0 * eps)
    return derivative_along_unit * norm


def local_stage_map_derivatives(
    stage_maps: Mapping[str, Callable[[Any], Any]],
    theta0: Any,
    *,
    epsilon: float,
) -> tuple[dict[str, Any], dict[tuple[str, str], Any]]:
    """Compute stage displacements and all directed cross-JVPs.

    The cross key ``(destination, source)`` stores
    ``J Delta_destination(theta0) @ Delta_source(theta0)``.

    Complexity is O(N^2) stage executions for N candidate stages rather than replaying
    all N! complete training chronologies.
    """

    stages = tuple(stage_maps)
    if len(stages) < 2:
        raise ValueError("at least two stage maps are required")
    if len(stages) != len(set(stages)):
        raise ValueError("stage names must be unique")

    deltas = {stage: stage_displacement(stage_maps[stage], theta0) for stage in stages}
    cross: dict[tuple[str, str], Any] = {}
    for destination in stages:
        for source in stages:
            if destination == source:
                continue
            cross[(destination, source)] = displacement_jvp_finite_difference(
                stage_maps[destination],
                theta0,
                deltas[source],
                epsilon=epsilon,
            )
    return deltas, cross


def operator_pair_geometry(
    theta0: Any,
    delta_a: Any,
    delta_b: Any,
    j_b_delta_a: Any,
    j_a_delta_b: Any,
) -> OperatorPairGeometry:
    """Construct the order-independent pair midpoint for finite stage maps."""

    additive = theta0 + delta_a + delta_b
    midpoint = additive + 0.5 * (j_b_delta_a + j_a_delta_b)
    bracket = j_b_delta_a - j_a_delta_b
    return OperatorPairGeometry(
        additive_reference=additive,
        midpoint_reference=midpoint,
        bracket=bracket,
        j_b_delta_a=j_b_delta_a,
        j_a_delta_b=j_a_delta_b,
    )


def operator_pair_score(
    endpoint: Any,
    geometry: OperatorPairGeometry,
    *,
    epsilon: float = 1e-18,
) -> float:
    """Return a normalized macro-stage chronology score; AB -> +1 and BA -> -1 locally."""

    torch = _require_torch()
    energy = float(torch.dot(geometry.bracket, geometry.bracket))
    denominator = 0.5 * energy
    if denominator <= epsilon:
        raise ValueError("operator bracket energy is too small to identify order")
    residual = endpoint - geometry.midpoint_reference
    return float(torch.dot(residual, geometry.bracket)) / denominator


def _canonical_pairs(stages: Sequence[str]) -> list[tuple[str, str]]:
    stages = tuple(stages)
    if len(stages) < 2:
        raise ValueError("at least two stages are required")
    if len(stages) != len(set(stages)):
        raise ValueError("stage names must be unique")
    return [(left, right) for index, left in enumerate(stages) for right in stages[index + 1 :]]


def _require_cross(
    stages: Sequence[str],
    cross_jvps: Mapping[tuple[str, str], Any],
) -> None:
    for left, right in _canonical_pairs(stages):
        for key in ((right, left), (left, right)):
            if key not in cross_jvps:
                raise ValueError(f"missing stage-map cross-JVP {key!r}")


def operator_symmetric_reference(
    theta0: Any,
    deltas: Mapping[str, Any],
    cross_jvps: Mapping[tuple[str, str], Any],
    *,
    stages: Sequence[str],
) -> Any:
    """Return the order-independent macro-stage endpoint reference through second order."""

    torch = _require_torch()
    stages = tuple(stages)
    _require_cross(stages, cross_jvps)
    if set(stages) != set(deltas):
        raise ValueError("deltas must contain exactly the declared stages")

    additive = torch.zeros_like(theta0)
    for stage in stages:
        additive = additive + deltas[stage]

    symmetric = torch.zeros_like(theta0)
    for left, right in _canonical_pairs(stages):
        symmetric = symmetric + 0.5 * (
            cross_jvps[(right, left)] + cross_jvps[(left, right)]
        )
    return theta0 + additive + symmetric


def operator_permutation_signature(
    permutation: Sequence[str],
    cross_jvps: Mapping[tuple[str, str], Any],
    *,
    stages: Sequence[str],
) -> Any:
    """Return the signed pairwise macro-stage commutator signature."""

    torch = _require_torch()
    stages = tuple(stages)
    permutation = tuple(permutation)
    if set(permutation) != set(stages) or len(permutation) != len(stages):
        raise ValueError("permutation must contain every stage exactly once")
    _require_cross(stages, cross_jvps)

    position = {stage: index for index, stage in enumerate(permutation)}
    sample = next(iter(cross_jvps.values()))
    signature = torch.zeros_like(sample)
    for left, right in _canonical_pairs(stages):
        bracket = cross_jvps[(right, left)] - cross_jvps[(left, right)]
        orientation = 1.0 if position[left] < position[right] else -1.0
        signature = signature + 0.5 * orientation * bracket
    return signature


def decode_operator_permutation(
    endpoint: Any,
    symmetric_reference: Any,
    cross_jvps: Mapping[tuple[str, str], Any],
    *,
    stages: Sequence[str],
) -> OperatorPermutationDecode:
    """Decode the nearest finite-stage chronology without replaying candidate permutations."""

    torch = _require_torch()
    residual = endpoint - symmetric_reference
    ranked: list[tuple[float, tuple[str, ...]]] = []
    for candidate in permutations(tuple(stages)):
        signature = operator_permutation_signature(candidate, cross_jvps, stages=stages)
        error = float(torch.linalg.vector_norm(residual - signature))
        ranked.append((error, candidate))
    ranked.sort(key=lambda item: item[0])
    best_error, best = ranked[0]
    runner_up_error = ranked[1][0] if len(ranked) > 1 else float("inf")
    return OperatorPermutationDecode(
        permutation=best,
        best_error=best_error,
        runner_up_error=runner_up_error,
        margin=runner_up_error - best_error,
    )
