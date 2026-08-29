"""Exact ordered interaction bases for training-history reconstruction.

For a deterministic stage map family ``F_i`` and base state ``theta0``, let ``E(w)`` be
an endpoint after executing the ordered word ``w`` of distinct stages. Define the exact
ordered interaction for a non-empty word by Möbius inversion over its order-preserving
subsequences:

    Phi(w) = E(w) - theta0 - sum_{u proper non-empty subsequence of w} Phi(u).

Then every measured endpoint has the exact decomposition

    E(w) = theta0 + sum_{u non-empty subsequence of w} Phi(u).

Truncating this sum at degree ``K`` gives a common decoder family for K=1,2,3,... .
Degree 2 is algebraically the existing finite-pair construction; degree 3 adds exact
prefix-conditioned three-stage information without replaying complete N-stage histories.

Measuring all ordered words through degree K requires

    sum_{r=1..K} P(N, r)

stage-map executions when prefix endpoints are cached. This module provides a simple
in-memory measurement helper for controlled experiments and tests. Large-model runners
may instead construct the same basis from streamed or projected endpoint observations via
``ordered_interaction_basis_from_endpoints``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Any


@dataclass(frozen=True)
class OrderedInteractionBasis:
    """Exact ordered interactions measured through one maximum degree."""

    stages: tuple[str, ...]
    max_degree: int
    base: Any
    endpoints: dict[tuple[str, ...], Any]
    interactions: dict[tuple[str, ...], Any]
    stage_executions: int


@dataclass(frozen=True)
class OrderedInteractionDecode:
    """Nearest chronology under one truncated ordered-interaction degree."""

    degree: int
    permutation: tuple[str, ...]
    best_error: float
    runner_up_error: float
    margin: float


@dataclass(frozen=True)
class OrderedInteractionIdentifiability:
    """Candidate separation under one ordered-interaction truncation."""

    degree: int
    candidate_count: int
    minimum_prediction_separation: float
    identifiable: bool


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised in non-MVP environments
        raise RuntimeError('Install the MVP dependencies with: pip install -e ".[mvp]"') from exc
    return torch


def _validate_stages(stages: Sequence[str]) -> tuple[str, ...]:
    values = tuple(str(stage) for stage in stages)
    if len(values) < 2:
        raise ValueError("at least two stages are required")
    if len(values) != len(set(values)):
        raise ValueError("stage names must be unique")
    return values


def _validate_degree(degree: int, stage_count: int) -> int:
    value = int(degree)
    if value < 1 or value > stage_count:
        raise ValueError("degree must be between 1 and the number of stages")
    return value


def ordered_probe_count(stage_count: int, max_degree: int) -> int:
    """Return ``sum(P(N,r), r=1..K)`` stage executions for a cached-prefix basis."""

    count = int(stage_count)
    if count < 2:
        raise ValueError("stage_count must be at least two")
    degree = _validate_degree(max_degree, count)
    return sum(math.perm(count, order) for order in range(1, degree + 1))


def ordered_subsequences(
    word: Sequence[str],
    *,
    max_degree: int | None = None,
) -> tuple[tuple[str, ...], ...]:
    """Return all non-empty order-preserving subsequences through ``max_degree``."""

    values = tuple(word)
    if len(values) != len(set(values)):
        raise ValueError("ordered words must not repeat stages")
    if not values:
        return ()
    degree = len(values) if max_degree is None else int(max_degree)
    if degree < 1:
        return ()
    degree = min(degree, len(values))
    result: list[tuple[str, ...]] = []
    for size in range(1, degree + 1):
        for indices in combinations(range(len(values)), size):
            result.append(tuple(values[index] for index in indices))
    return tuple(result)


def _expected_words(stages: Sequence[str], max_degree: int) -> set[tuple[str, ...]]:
    return {
        word
        for size in range(1, max_degree + 1)
        for word in permutations(tuple(stages), size)
    }


def ordered_interaction_basis_from_endpoints(
    base: Any,
    endpoints: Mapping[tuple[str, ...], Any],
    *,
    stages: Sequence[str],
    max_degree: int,
    stage_executions: int = 0,
) -> OrderedInteractionBasis:
    """Construct exact ordered interactions from complete endpoint observations.

    ``endpoints`` must contain every ordered word of distinct stages through
    ``max_degree``. The values may be full parameter vectors or any fixed linear
    observation/projection of those vectors; all interaction algebra is performed in the
    supplied observation space.
    """

    stages_tuple = _validate_stages(stages)
    degree = _validate_degree(max_degree, len(stages_tuple))
    expected = _expected_words(stages_tuple, degree)
    actual = set(endpoints)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"endpoint table mismatch; missing={missing!r}, extra={extra!r}")

    endpoint_table: dict[tuple[str, ...], Any] = {}
    interactions: dict[tuple[str, ...], Any] = {}
    for size in range(1, degree + 1):
        for word in permutations(stages_tuple, size):
            endpoint = endpoints[word]
            if endpoint.shape != base.shape:
                raise ValueError("endpoint observation shape differs from base")
            endpoint_table[word] = endpoint
            interaction = endpoint - base
            if size > 1:
                for subsequence in ordered_subsequences(word, max_degree=size - 1):
                    interaction = interaction - interactions[subsequence]
            interactions[word] = interaction

    executions = int(stage_executions)
    if executions < 0:
        raise ValueError("stage_executions must be non-negative")
    return OrderedInteractionBasis(
        stages=stages_tuple,
        max_degree=degree,
        base=base,
        endpoints=endpoint_table,
        interactions=interactions,
        stage_executions=executions,
    )


def measure_ordered_interaction_basis(
    stage_maps: Mapping[str, Callable[[Any], Any]],
    base: Any,
    *,
    max_degree: int,
) -> OrderedInteractionBasis:
    """Measure every ordered word through ``max_degree`` with cached prefixes.

    This convenience implementation stores every full endpoint through degree K. It is
    appropriate for controlled models and unit tests. Large-model experiments should
    stream or project endpoint observations and then call
    ``ordered_interaction_basis_from_endpoints``.
    """

    stages = _validate_stages(tuple(stage_maps))
    degree = _validate_degree(max_degree, len(stages))
    prefix_endpoints: dict[tuple[str, ...], Any] = {(): base}
    observed: dict[tuple[str, ...], Any] = {}
    executions = 0

    for size in range(1, degree + 1):
        for word in permutations(stages, size):
            endpoint = stage_maps[word[-1]](prefix_endpoints[word[:-1]])
            executions += 1
            if endpoint.shape != base.shape:
                raise ValueError("stage map changed the observation shape")
            prefix_endpoints[word] = endpoint
            observed[word] = endpoint

    expected = ordered_probe_count(len(stages), degree)
    if executions != expected:
        raise RuntimeError(f"measured {executions} stage executions, expected {expected}")
    return ordered_interaction_basis_from_endpoints(
        base,
        observed,
        stages=stages,
        max_degree=degree,
        stage_executions=executions,
    )


def ordered_interaction_prediction(
    permutation: Sequence[str],
    basis: OrderedInteractionBasis,
    *,
    degree: int | None = None,
) -> Any:
    """Return the endpoint predicted by truncating the exact interaction hierarchy."""

    candidate = tuple(permutation)
    if len(candidate) != len(basis.stages) or set(candidate) != set(basis.stages):
        raise ValueError("permutation must contain every basis stage exactly once")
    chosen_degree = basis.max_degree if degree is None else int(degree)
    if chosen_degree < 1 or chosen_degree > basis.max_degree:
        raise ValueError("degree must be between 1 and basis.max_degree")

    prediction = basis.base.clone()
    for subsequence in ordered_subsequences(candidate, max_degree=chosen_degree):
        prediction = prediction + basis.interactions[subsequence]
    return prediction


def _candidate_orders(
    basis: OrderedInteractionBasis,
    candidates: Sequence[Sequence[str]] | None,
) -> tuple[tuple[str, ...], ...]:
    if candidates is None:
        return tuple(permutations(basis.stages))
    result = tuple(tuple(candidate) for candidate in candidates)
    if not result:
        raise ValueError("at least one candidate permutation is required")
    for candidate in result:
        if len(candidate) != len(basis.stages) or set(candidate) != set(basis.stages):
            raise ValueError("every candidate must contain every basis stage exactly once")
    if len(result) != len(set(result)):
        raise ValueError("candidate permutations must be unique")
    return result


def decode_ordered_interaction_permutation(
    endpoint: Any,
    basis: OrderedInteractionBasis,
    *,
    degree: int | None = None,
    candidates: Sequence[Sequence[str]] | None = None,
) -> OrderedInteractionDecode:
    """Decode the nearest candidate chronology under one interaction degree."""

    torch = _require_torch()
    chosen_degree = basis.max_degree if degree is None else int(degree)
    if chosen_degree < 1 or chosen_degree > basis.max_degree:
        raise ValueError("degree must be between 1 and basis.max_degree")

    ranked: list[tuple[float, tuple[str, ...]]] = []
    for candidate in _candidate_orders(basis, candidates):
        prediction = ordered_interaction_prediction(candidate, basis, degree=chosen_degree)
        error = float(torch.linalg.vector_norm(endpoint - prediction))
        ranked.append((error, candidate))
    ranked.sort(key=lambda item: (item[0], item[1]))
    best_error, best = ranked[0]
    runner_up_error = ranked[1][0] if len(ranked) > 1 else float("inf")
    return OrderedInteractionDecode(
        degree=chosen_degree,
        permutation=best,
        best_error=best_error,
        runner_up_error=runner_up_error,
        margin=runner_up_error - best_error,
    )


def ordered_interaction_identifiability(
    basis: OrderedInteractionBasis,
    *,
    degree: int | None = None,
    candidates: Sequence[Sequence[str]] | None = None,
    tolerance: float = 1e-12,
) -> OrderedInteractionIdentifiability:
    """Measure minimum candidate-prediction separation at one interaction degree."""

    torch = _require_torch()
    chosen_degree = basis.max_degree if degree is None else int(degree)
    if chosen_degree < 1 or chosen_degree > basis.max_degree:
        raise ValueError("degree must be between 1 and basis.max_degree")
    orders = _candidate_orders(basis, candidates)
    predictions = [
        ordered_interaction_prediction(candidate, basis, degree=chosen_degree)
        for candidate in orders
    ]
    minimum = float("inf")
    for index, left in enumerate(predictions):
        for right in predictions[index + 1 :]:
            minimum = min(minimum, float(torch.linalg.vector_norm(left - right)))
    if len(predictions) < 2:
        minimum = float("inf")
    return OrderedInteractionIdentifiability(
        degree=chosen_degree,
        candidate_count=len(orders),
        minimum_prediction_separation=minimum,
        identifiable=minimum > float(tolerance),
    )
