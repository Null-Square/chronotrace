"""Shared candidate-error tables for complete and partial chronology decisions.

Large-model chronology experiments should not repeatedly construct full parameter-space
predictions just to answer different questions about the same endpoint. This module can
score each candidate directly, or prepare an exact quadratic-form scorer that reduces a
large-model endpoint to scalar dot products with the measured interaction basis. The
quadratic path computes the same Euclidean objective without rebuilding every candidate
parameter vector for every target.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import permutations
from typing import Any

from chronotrace.geometry.interactions import (
    OrderedInteractionBasis,
    ordered_interaction_prediction,
    ordered_subsequences,
)


@dataclass(frozen=True)
class CandidateErrorDecision:
    """Nearest complete chronology in a precomputed scalar error table."""

    permutation: tuple[str, ...]
    best_error: float
    runner_up_error: float
    margin: float


@dataclass(frozen=True)
class CandidatePrefixDecision:
    """Best prefix after minimizing over compatible complete chronologies."""

    depth: int
    prefix: tuple[str, ...]
    best_error: float
    runner_up_error: float
    margin: float


@dataclass(frozen=True)
class CandidatePrecedenceDecision:
    """Best orientation of one stage pair after marginalizing all other positions."""

    first: str
    second: str
    preferred_first: str
    preferred_second: str
    preferred_error: float
    alternative_error: float
    margin: float


@dataclass(frozen=True)
class OrderedInteractionQuadraticScorer:
    """Scalar sufficient statistics for exact Euclidean chronology scoring."""

    candidates: tuple[tuple[str, ...], ...]
    degrees: tuple[int, ...]
    selected_words: dict[int, dict[tuple[str, ...], tuple[tuple[str, ...], ...]]]
    prediction_displacement_norm_squared: dict[int, dict[tuple[str, ...], float]]


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


def ordered_interaction_candidate_errors(
    endpoint: Any,
    basis: OrderedInteractionBasis,
    *,
    degree: int,
    candidates: Sequence[Sequence[str]] | None = None,
) -> dict[tuple[str, ...], float]:
    """Score every candidate by direct parameter-space prediction construction."""

    torch = _require_torch()
    chosen_degree = int(degree)
    if chosen_degree < 1 or chosen_degree > basis.max_degree:
        raise ValueError("degree must be between 1 and basis.max_degree")
    errors: dict[tuple[str, ...], float] = {}
    for candidate in _candidate_orders(basis, candidates):
        prediction = ordered_interaction_prediction(candidate, basis, degree=chosen_degree)
        errors[candidate] = float(torch.linalg.vector_norm(endpoint - prediction))
        del prediction
    return errors


def prepare_ordered_interaction_quadratic_scorer(
    basis: OrderedInteractionBasis,
    *,
    degrees: Sequence[int],
    candidates: Sequence[Sequence[str]] | None = None,
) -> OrderedInteractionQuadraticScorer:
    """Precompute candidate-only norms for exact quadratic-form endpoint scoring.

    For a target displacement ``z = endpoint - base`` and candidate interaction sum
    ``s``, the decoder objective is exactly

        ||z - s||^2 = ||z||^2 + ||s||^2 - 2 <z, s>.

    ``||s||^2`` depends only on the frozen basis and candidate chronology, so it is
    computed once here. At each target, only one dot product per measured interaction is
    needed; candidate cross terms are then scalar sums over the relevant subsequences.
    """

    torch = _require_torch()
    chosen_degrees = tuple(sorted({int(value) for value in degrees}))
    if not chosen_degrees:
        raise ValueError("at least one interaction degree is required")
    if chosen_degrees[0] < 1 or chosen_degrees[-1] > basis.max_degree:
        raise ValueError("degrees must be between 1 and basis.max_degree")
    orders = _candidate_orders(basis, candidates)
    selected_words: dict[int, dict[tuple[str, ...], tuple[tuple[str, ...], ...]]] = {
        degree: {} for degree in chosen_degrees
    }
    norm_squared: dict[int, dict[tuple[str, ...], float]] = {
        degree: {} for degree in chosen_degrees
    }

    for candidate in orders:
        running = torch.zeros_like(basis.base)
        for degree in range(1, chosen_degrees[-1] + 1):
            words = tuple(
                word
                for word in ordered_subsequences(candidate, max_degree=degree)
                if len(word) == degree
            )
            for word in words:
                if word not in basis.interactions:
                    raise ValueError(f"interaction basis is missing ordered word {word!r}")
                running.add_(basis.interactions[word])
            if degree in norm_squared:
                chosen = ordered_subsequences(candidate, max_degree=degree)
                selected_words[degree][candidate] = chosen
                flat = running.reshape(-1)
                norm_squared[degree][candidate] = float(torch.dot(flat, flat))
        del running

    return OrderedInteractionQuadraticScorer(
        candidates=orders,
        degrees=chosen_degrees,
        selected_words=selected_words,
        prediction_displacement_norm_squared=norm_squared,
    )


def ordered_interaction_quadratic_error_tables(
    endpoint: Any,
    basis: OrderedInteractionBasis,
    scorer: OrderedInteractionQuadraticScorer,
) -> dict[int, dict[tuple[str, ...], float]]:
    """Return exact Euclidean candidate errors for every prepared interaction degree."""

    torch = _require_torch()
    if endpoint.shape != basis.base.shape:
        raise ValueError("endpoint shape differs from interaction basis")
    displacement = endpoint - basis.base
    flat_displacement = displacement.reshape(-1)
    target_norm_squared = float(torch.dot(flat_displacement, flat_displacement))
    interaction_dots = {
        word: float(torch.dot(flat_displacement, interaction.reshape(-1)))
        for word, interaction in basis.interactions.items()
    }

    tables: dict[int, dict[tuple[str, ...], float]] = {}
    for degree in scorer.degrees:
        table: dict[tuple[str, ...], float] = {}
        for candidate in scorer.candidates:
            cross = sum(
                interaction_dots[word]
                for word in scorer.selected_words[degree][candidate]
            )
            prediction_norm_squared = scorer.prediction_displacement_norm_squared[degree][
                candidate
            ]
            error_squared = target_norm_squared + prediction_norm_squared - 2.0 * cross
            cancellation_scale = (
                abs(target_norm_squared)
                + abs(prediction_norm_squared)
                + 2.0 * abs(cross)
            )
            tolerance = 1e-12 * max(1.0, cancellation_scale)
            if error_squared < -tolerance:
                raise FloatingPointError(
                    "quadratic chronology error became materially negative; "
                    f"degree={degree}, candidate={candidate!r}, value={error_squared}"
                )
            table[candidate] = math.sqrt(max(0.0, error_squared))
        tables[degree] = table
    del displacement
    return tables


def decode_error_table(
    errors: Mapping[tuple[str, ...], float],
) -> CandidateErrorDecision:
    """Decode the nearest complete chronology from scalar errors."""

    if not errors:
        raise ValueError("candidate error table is empty")
    ranked = sorted((float(error), tuple(candidate)) for candidate, error in errors.items())
    best_error, best = ranked[0]
    runner_up = ranked[1][0] if len(ranked) > 1 else float("inf")
    return CandidateErrorDecision(
        permutation=best,
        best_error=best_error,
        runner_up_error=runner_up,
        margin=runner_up - best_error,
    )


def decode_prefix_error_table(
    errors: Mapping[tuple[str, ...], float],
    *,
    depth: int,
) -> CandidatePrefixDecision:
    """Decode one prefix depth from a shared scalar error table."""

    if not errors:
        raise ValueError("candidate error table is empty")
    stage_count = len(next(iter(errors)))
    chosen_depth = int(depth)
    if chosen_depth < 1 or chosen_depth > stage_count:
        raise ValueError("depth must be between 1 and the number of stages")
    grouped: dict[tuple[str, ...], float] = {}
    for candidate, raw_error in errors.items():
        prefix = tuple(candidate[:chosen_depth])
        error = float(raw_error)
        grouped[prefix] = min(error, grouped.get(prefix, float("inf")))
    ranked = sorted((error, prefix) for prefix, error in grouped.items())
    best_error, best = ranked[0]
    runner_up = ranked[1][0] if len(ranked) > 1 else float("inf")
    return CandidatePrefixDecision(
        depth=chosen_depth,
        prefix=best,
        best_error=best_error,
        runner_up_error=runner_up,
        margin=runner_up - best_error,
    )


def decode_precedence_error_table(
    errors: Mapping[tuple[str, ...], float],
    *,
    first: str,
    second: str,
) -> CandidatePrecedenceDecision:
    """Decode one pair orientation from a shared scalar error table."""

    if not errors or first == second:
        raise ValueError("non-empty errors and two distinct stages are required")
    forward: list[float] = []
    reverse: list[float] = []
    for candidate, raw_error in errors.items():
        if first not in candidate or second not in candidate:
            raise ValueError("pair stage missing from candidate chronology")
        error = float(raw_error)
        if candidate.index(first) < candidate.index(second):
            forward.append(error)
        else:
            reverse.append(error)
    if not forward or not reverse:
        raise ValueError("candidate table must contain both pair orientations")
    forward_error = min(forward)
    reverse_error = min(reverse)
    if forward_error <= reverse_error:
        preferred_first, preferred_second = first, second
        preferred_error, alternative_error = forward_error, reverse_error
    else:
        preferred_first, preferred_second = second, first
        preferred_error, alternative_error = reverse_error, forward_error
    return CandidatePrecedenceDecision(
        first=first,
        second=second,
        preferred_first=preferred_first,
        preferred_second=preferred_second,
        preferred_error=preferred_error,
        alternative_error=alternative_error,
        margin=alternative_error - preferred_error,
    )
