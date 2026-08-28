"""Identifiability diagnostics for the pairwise commutator chronology basis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import permutations
from typing import Any

from chronotrace.geometry.commutator import permutation_signature


@dataclass(frozen=True)
class ChronologyIdentifiability:
    """Conditioning and finite-permutation separation of the bracket basis."""

    stage_count: int
    pair_count: int
    bracket_rank: int
    bracket_condition_number: float
    minimum_signature_separation: float
    locally_identifiable: bool


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised in MVP environments
        raise RuntimeError('Install the MVP dependencies with: pip install -e ".[mvp]"') from exc
    return torch


def _pairs(stages: Sequence[str]) -> list[tuple[str, str]]:
    stages = tuple(stages)
    if len(stages) < 2:
        raise ValueError("at least two stages are required")
    if len(stages) != len(set(stages)):
        raise ValueError("stage names must be unique")
    return [(left, right) for index, left in enumerate(stages) for right in stages[index + 1 :]]


def bracket_matrix(
    cross_hvps: Mapping[tuple[str, str], Any],
    *,
    stages: Sequence[str],
) -> Any:
    """Return a parameter-by-pair matrix whose columns are pairwise Lie brackets."""

    torch = _require_torch()
    columns = []
    for left, right in _pairs(stages):
        forward = (right, left)
        reverse = (left, right)
        if forward not in cross_hvps or reverse not in cross_hvps:
            raise ValueError(f"missing cross-HVPs for pair {left!r}, {right!r}")
        columns.append(cross_hvps[forward] - cross_hvps[reverse])
    return torch.stack(columns, dim=1)


def chronology_identifiability(
    cross_hvps: Mapping[tuple[str, str], Any],
    *,
    stages: Sequence[str],
    epsilon: float = 1e-12,
) -> ChronologyIdentifiability:
    """Measure whether candidate permutations have distinct second-order signatures.

    `minimum_signature_separation` is measured at unit step size. At learning rate
    `eta`, all second-order signature distances scale by `eta**2`. If the endpoint's
    higher-order approximation error is less than half that scaled separation, nearest
    signature decoding is guaranteed to return the correct candidate permutation.

    Full column rank of the bracket matrix is sufficient but not necessary for finite
    candidate-permutation identifiability, so both rank and signature separation are
    reported rather than conflated.
    """

    torch = _require_torch()
    stages = tuple(stages)
    matrix = bracket_matrix(cross_hvps, stages=stages)
    singular_values = torch.linalg.svdvals(matrix)
    rank = int(torch.linalg.matrix_rank(matrix).item())
    smallest = float(singular_values[-1])
    condition = float("inf") if smallest <= epsilon else float(singular_values[0]) / smallest

    candidates = list(permutations(stages))
    signatures = [
        permutation_signature(
            candidate,
            cross_hvps,
            stages=stages,
            step_size=1.0,
        )
        for candidate in candidates
    ]
    separations = [
        float(torch.linalg.vector_norm(left - right))
        for index, left in enumerate(signatures)
        for right in signatures[index + 1 :]
    ]
    minimum = min(separations)
    return ChronologyIdentifiability(
        stage_count=len(stages),
        pair_count=matrix.shape[1],
        bracket_rank=rank,
        bracket_condition_number=condition,
        minimum_signature_separation=minimum,
        locally_identifiable=minimum > epsilon,
    )


def normalized_remainder_ratio(
    approximation_error: float,
    *,
    step_size: float,
    minimum_signature_separation: float,
) -> float:
    """Return the nearest-signature guarantee ratio; values below one are sufficient.

    The ratio is `2 * error / (eta**2 * delta)`, where `delta` is the unit-step
    minimum signature separation. A ratio below one means the true second-order
    signature is closer than every competing candidate by the triangle inequality.
    """

    eta = float(step_size)
    if eta <= 0:
        raise ValueError("step_size must be positive")
    if approximation_error < 0:
        raise ValueError("approximation_error must be non-negative")
    if minimum_signature_separation <= 0:
        raise ValueError("minimum_signature_separation must be positive")
    return 2.0 * approximation_error / (eta**2 * minimum_signature_separation)
