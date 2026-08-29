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

stage-map executions. The compact measurement path stores only the interaction tensors:
every prefix endpoint is reconstructed exactly from already-measured lower-order
interactions before the next stage is applied. This avoids retaining a second full table
of prefix endpoints in large-model runs.
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
    """Return ``sum(P(N,r), r=1..K)`` stage executions for the ordered basis."""

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


def _interaction_from_endpoint(
    endpoint: Any,
    base: Any,
    word: tuple[str, ...],
    interactions: Mapping[tuple[str, ...], Any],
) -> Any:
    interaction = endpoint - base
    if len(word) > 1:
        for subsequence in ordered_subsequences(word, max_degree=len(word) - 1):
            interaction = interaction - interactions[subsequence]
    return interaction


def _word_prediction_from_interactions(
    word: Sequence[str],
    base: Any,
    interactions: Mapping[tuple[str, ...], Any],
    *,
    degree: int,
) -> Any:
    prediction = base.clone()
    for subsequence in ordered_subsequences(word, max_degree=degree):
        if subsequence not in interactions:
            raise ValueError(f"interaction basis is missing ordered word {subsequence!r}")
        prediction = prediction + interactions[subsequence]
    return prediction


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
            interactions[word] = _interaction_from_endpoint(
                endpoint,
                base,
                word,
                interactions,
            )

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
    """Measure every ordered word through ``max_degree`` with cached full prefixes.

    This convenience implementation stores every full endpoint through degree K. It is
    useful for controlled models and tests where direct endpoint reconstruction should be
    audited. Large-model experiments should normally use the compact variant below.
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


def measure_ordered_interaction_basis_compact(
    stage_maps: Mapping[str, Callable[[Any], Any]],
    base: Any,
    *,
    max_degree: int,
) -> OrderedInteractionBasis:
    """Measure an ordered basis while retaining only interaction tensors.

    For each word, the endpoint of its prefix is reconstructed exactly from the already
    measured lower-order interactions. The final stage is then applied once and the new
    interaction is extracted immediately. This preserves the same probe count as cached
    prefix measurement while avoiding a second full tensor table of prefix endpoints.
    """

    stages = _validate_stages(tuple(stage_maps))
    degree = _validate_degree(max_degree, len(stages))
    interactions: dict[tuple[str, ...], Any] = {}
    executions = 0

    for size in range(1, degree + 1):
        for word in permutations(stages, size):
            prefix = word[:-1]
            initial = _word_prediction_from_interactions(
                prefix,
                base,
                interactions,
                degree=len(prefix),
            )
            endpoint = stage_maps[word[-1]](initial)
            executions += 1
            if endpoint.shape != base.shape:
                raise ValueError("stage map changed the observation shape")
            interactions[word] = _interaction_from_endpoint(
                endpoint,
                base,
                word,
                interactions,
            )

    expected = ordered_probe_count(len(stages), degree)
    if executions != expected:
        raise RuntimeError(f"measured {executions} stage executions, expected {expected}")
    return OrderedInteractionBasis(
        stages=stages,
        max_degree=degree,
        base=base,
        endpoints={},
        interactions=interactions,
        stage_executions=executions,
    )


def ordered_interaction_word_prediction(
    word: Sequence[str],
    basis: OrderedInteractionBasis,
    *,
    degree: int | None = None,
) -> Any:
    """Reconstruct or truncate any distinct-stage ordered word from the basis."""

    values = tuple(word)
    if len(values) != len(set(values)):
        raise ValueError("ordered words must not repeat stages")
    if any(stage not in basis.stages for stage in values):
        raise ValueError("ordered word contains a stage outside the basis")
    chosen_degree = min(len(values), basis.max_degree) if degree is None else int(degree)
    if not values:
        return basis.base.clone()
    if chosen_degree < 1 or chosen_degree > basis.max_degree:
        raise ValueError("degree must be between 1 and basis.max_degree")
    return _word_prediction_from_interactions(
        values,
        basis.base,
        basis.interactions,
        degree=chosen_degree,
    )


def ordered_interaction_prediction(
    permutation: Sequence[str],
    basis: OrderedInteractionBasis,
    *,
    degree: int | None = None,
) -> Any:
    """Return a complete chronology prediction under one interaction truncation."""

    candidate = tuple(permutation)
    if len(candidate) != len(basis.stages) or set(candidate) != set(basis.stages):
        raise ValueError("permutation must contain every basis stage exactly once")
    return ordered_interaction_word_prediction(candidate, basis, degree=degree)


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
