"""Partial chronology decisions from an ordered-interaction basis.

These decoders consume the same candidate endpoint predictions as complete permutation
decoding. They require no additional training-stage probes. Prefix grouping avoids forcing
a total order when only an early chronology is supported, while pairwise group comparisons
produce precedence edges that can be reported as a partial order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import permutations
from typing import Any

from chronotrace.geometry.interactions import (
    OrderedInteractionBasis,
    ordered_interaction_prediction,
)


@dataclass(frozen=True)
class PrefixGroupDecode:
    """Best chronology prefix after minimizing over all compatible full orders."""

    degree: int
    depth: int
    prefix: tuple[str, ...]
    best_error: float
    runner_up_error: float
    margin: float


@dataclass(frozen=True)
class PairwisePrecedenceDecode:
    """Best orientation of one stage pair after minimizing over all other positions."""

    degree: int
    first: str
    second: str
    preferred_first: str
    preferred_second: str
    preferred_error: float
    alternative_error: float
    margin: float


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError('Install the MVP dependencies with: pip install -e ".[mvp]"') from exc
    return torch


def _candidate_orders(
    basis: OrderedInteractionBasis,
    candidates: Sequence[Sequence[str]] | None,
) -> tuple[tuple[str, ...], ...]:
    if candidates is None:
        return tuple(permutations(basis.stages))
    values = tuple(tuple(candidate) for candidate in candidates)
    if not values:
        raise ValueError("at least one candidate permutation is required")
    for candidate in values:
        if len(candidate) != len(basis.stages) or set(candidate) != set(basis.stages):
            raise ValueError("every candidate must contain every basis stage exactly once")
    if len(values) != len(set(values)):
        raise ValueError("candidate permutations must be unique")
    return values


def _candidate_errors(
    endpoint: Any,
    basis: OrderedInteractionBasis,
    *,
    degree: int,
    candidates: Sequence[Sequence[str]] | None,
) -> dict[tuple[str, ...], float]:
    torch = _require_torch()
    if degree < 1 or degree > basis.max_degree:
        raise ValueError("degree must be between 1 and basis.max_degree")
    return {
        candidate: float(
            torch.linalg.vector_norm(
                endpoint - ordered_interaction_prediction(candidate, basis, degree=degree)
            )
        )
        for candidate in _candidate_orders(basis, candidates)
    }


def decode_ordered_interaction_prefix(
    endpoint: Any,
    basis: OrderedInteractionBasis,
    *,
    depth: int,
    degree: int | None = None,
    candidates: Sequence[Sequence[str]] | None = None,
) -> PrefixGroupDecode:
    """Decode a prefix by minimizing error over all full orders sharing that prefix."""

    chosen_degree = basis.max_degree if degree is None else int(degree)
    chosen_depth = int(depth)
    if chosen_depth < 1 or chosen_depth > len(basis.stages):
        raise ValueError("depth must be between 1 and the number of stages")
    errors = _candidate_errors(
        endpoint,
        basis,
        degree=chosen_degree,
        candidates=candidates,
    )
    grouped: dict[tuple[str, ...], float] = {}
    for candidate, error in errors.items():
        prefix = candidate[:chosen_depth]
        grouped[prefix] = min(error, grouped.get(prefix, float("inf")))
    ranked = sorted((error, prefix) for prefix, error in grouped.items())
    best_error, best_prefix = ranked[0]
    runner_up_error = ranked[1][0] if len(ranked) > 1 else float("inf")
    return PrefixGroupDecode(
        degree=chosen_degree,
        depth=chosen_depth,
        prefix=best_prefix,
        best_error=best_error,
        runner_up_error=runner_up_error,
        margin=runner_up_error - best_error,
    )


def decode_ordered_interaction_precedence(
    endpoint: Any,
    basis: OrderedInteractionBasis,
    *,
    first: str,
    second: str,
    degree: int | None = None,
    candidates: Sequence[Sequence[str]] | None = None,
) -> PairwisePrecedenceDecode:
    """Choose which of two stages precedes the other after minimizing nuisance order."""

    if first == second or first not in basis.stages or second not in basis.stages:
        raise ValueError("first and second must be distinct basis stages")
    chosen_degree = basis.max_degree if degree is None else int(degree)
    errors = _candidate_errors(
        endpoint,
        basis,
        degree=chosen_degree,
        candidates=candidates,
    )
    forward = [
        error
        for candidate, error in errors.items()
        if candidate.index(first) < candidate.index(second)
    ]
    reverse = [
        error
        for candidate, error in errors.items()
        if candidate.index(second) < candidate.index(first)
    ]
    if not forward or not reverse:
        raise ValueError("candidate set must contain both pair orientations")
    forward_error = min(forward)
    reverse_error = min(reverse)
    if forward_error <= reverse_error:
        preferred_first, preferred_second = first, second
        preferred_error, alternative_error = forward_error, reverse_error
    else:
        preferred_first, preferred_second = second, first
        preferred_error, alternative_error = reverse_error, forward_error
    return PairwisePrecedenceDecode(
        degree=chosen_degree,
        first=first,
        second=second,
        preferred_first=preferred_first,
        preferred_second=preferred_second,
        preferred_error=preferred_error,
        alternative_error=alternative_error,
        margin=alternative_error - preferred_error,
    )
