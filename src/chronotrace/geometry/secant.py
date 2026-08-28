"""Finite pair-interaction geometry for replay-efficient chronology reconstruction.

A complete deterministic training stage is a map

    F_D(theta) = theta + Delta_D(theta).

Instead of linearizing one stage around another with a Hessian-vector product or a
finite-difference JVP, this module measures each ordered pair interaction exactly at the
base checkpoint:

    I_{j<-i} = F_j(F_i(theta0)) - theta0 - Delta_i - Delta_j.

For a candidate multi-stage order, the pairwise truncation is the sum of singleton stage
effects plus the ordered interaction selected for every stage pair. Triple-and-higher
interactions are the only approximation residual.

After caching every singleton endpoint, the full directed pair table costs N(N-1)
additional stage executions, or N^2 total stage executions including the N singletons.
This is quadratic in the number of candidate stages, whereas replaying all complete
histories is factorial.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import permutations
from typing import Any


@dataclass(frozen=True)
class FinitePairDecode:
    """Nearest pairwise-truncated chronology among candidate permutations."""

    permutation: tuple[str, ...]
    best_error: float
    runner_up_error: float
    margin: float


@dataclass(frozen=True)
class FinitePairIdentifiability:
    """Separation of candidate finite-pair chronology signatures."""

    pair_count: int
    minimum_signature_separation: float
    identifiable: bool


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised in MVP environments
        raise RuntimeError('Install the MVP dependencies with: pip install -e ".[mvp]"') from exc
    return torch


def _canonical_pairs(stages: Sequence[str]) -> list[tuple[str, str]]:
    stages = tuple(stages)
    if len(stages) < 2:
        raise ValueError("at least two stages are required")
    if len(stages) != len(set(stages)):
        raise ValueError("stage names must be unique")
    return [(left, right) for index, left in enumerate(stages) for right in stages[index + 1 :]]


def _validate_stage_keys(stages: Sequence[str], values: Mapping[str, Any]) -> None:
    if set(stages) != set(values):
        raise ValueError("mapping must contain exactly the declared stages")


def _validate_interactions(
    stages: Sequence[str],
    interactions: Mapping[tuple[str, str], Any],
) -> None:
    for source in stages:
        for destination in stages:
            if source == destination:
                continue
            key = (destination, source)
            if key not in interactions:
                raise ValueError(f"missing ordered pair interaction {key!r}")


def finite_pair_interactions(
    stage_maps: Mapping[str, Callable[[Any], Any]],
    theta0: Any,
) -> tuple[dict[str, Any], dict[tuple[str, str], Any]]:
    """Measure singleton displacements and exact directed pair interactions.

    The interaction key ``(destination, source)`` stores the effect beyond singleton
    addition when ``source`` is run first and ``destination`` second.

    For N stages this calls a stage map exactly N^2 times: N singleton executions and
    N(N-1) second-stage executions from the cached singleton endpoints.
    """

    stages = tuple(stage_maps)
    if len(stages) < 2:
        raise ValueError("at least two stage maps are required")
    if len(stages) != len(set(stages)):
        raise ValueError("stage names must be unique")

    singleton_endpoints: dict[str, Any] = {}
    deltas: dict[str, Any] = {}
    for stage in stages:
        endpoint = stage_maps[stage](theta0)
        if endpoint.shape != theta0.shape:
            raise ValueError("stage map changed the parameter-vector shape")
        singleton_endpoints[stage] = endpoint
        deltas[stage] = endpoint - theta0

    interactions: dict[tuple[str, str], Any] = {}
    for source in stages:
        source_endpoint = singleton_endpoints[source]
        for destination in stages:
            if source == destination:
                continue
            pair_endpoint = stage_maps[destination](source_endpoint)
            if pair_endpoint.shape != theta0.shape:
                raise ValueError("stage map changed the parameter-vector shape")
            interactions[(destination, source)] = (
                pair_endpoint - theta0 - deltas[source] - deltas[destination]
            )
    return deltas, interactions


def finite_pair_symmetric_reference(
    theta0: Any,
    deltas: Mapping[str, Any],
    interactions: Mapping[tuple[str, str], Any],
    *,
    stages: Sequence[str],
) -> Any:
    """Return the permutation-independent midpoint of all pairwise interactions."""

    torch = _require_torch()
    stages = tuple(stages)
    _validate_stage_keys(stages, deltas)
    _validate_interactions(stages, interactions)

    reference = theta0.clone()
    for stage in stages:
        reference = reference + deltas[stage]

    pair_midpoint = torch.zeros_like(theta0)
    for left, right in _canonical_pairs(stages):
        pair_midpoint = pair_midpoint + 0.5 * (
            interactions[(right, left)] + interactions[(left, right)]
        )
    return reference + pair_midpoint


def finite_pair_signature(
    permutation: Sequence[str],
    interactions: Mapping[tuple[str, str], Any],
    *,
    stages: Sequence[str],
) -> Any:
    """Return the antisymmetric finite-pair signature for one candidate chronology."""

    torch = _require_torch()
    stages = tuple(stages)
    permutation = tuple(permutation)
    if set(permutation) != set(stages) or len(permutation) != len(stages):
        raise ValueError("permutation must contain every stage exactly once")
    _validate_interactions(stages, interactions)

    position = {stage: index for index, stage in enumerate(permutation)}
    sample = next(iter(interactions.values()))
    signature = torch.zeros_like(sample)
    for left, right in _canonical_pairs(stages):
        commutator = interactions[(right, left)] - interactions[(left, right)]
        orientation = 1.0 if position[left] < position[right] else -1.0
        signature = signature + 0.5 * orientation * commutator
    return signature


def finite_pair_predicted_endpoint(
    permutation: Sequence[str],
    symmetric_reference: Any,
    interactions: Mapping[tuple[str, str], Any],
    *,
    stages: Sequence[str],
) -> Any:
    """Return the pairwise-truncated endpoint for one candidate chronology."""

    return symmetric_reference + finite_pair_signature(
        permutation,
        interactions,
        stages=stages,
    )


def decode_finite_pair_permutation(
    endpoint: Any,
    symmetric_reference: Any,
    interactions: Mapping[tuple[str, str], Any],
    *,
    stages: Sequence[str],
) -> FinitePairDecode:
    """Decode the nearest pairwise-truncated chronology."""

    torch = _require_torch()
    ranked: list[tuple[float, tuple[str, ...]]] = []
    for candidate in permutations(tuple(stages)):
        prediction = finite_pair_predicted_endpoint(
            candidate,
            symmetric_reference,
            interactions,
            stages=stages,
        )
        error = float(torch.linalg.vector_norm(endpoint - prediction))
        ranked.append((error, candidate))
    ranked.sort(key=lambda item: item[0])
    best_error, best = ranked[0]
    runner_up_error = ranked[1][0] if len(ranked) > 1 else float("inf")
    return FinitePairDecode(
        permutation=best,
        best_error=best_error,
        runner_up_error=runner_up_error,
        margin=runner_up_error - best_error,
    )


def finite_pair_identifiability(
    interactions: Mapping[tuple[str, str], Any],
    *,
    stages: Sequence[str],
    tolerance: float = 1e-12,
) -> FinitePairIdentifiability:
    """Measure the minimum separation between finite-pair permutation signatures."""

    torch = _require_torch()
    stages = tuple(stages)
    candidates = list(permutations(stages))
    signatures = [
        finite_pair_signature(candidate, interactions, stages=stages) for candidate in candidates
    ]
    minimum = float("inf")
    for index, left in enumerate(signatures):
        for right in signatures[index + 1 :]:
            minimum = min(minimum, float(torch.linalg.vector_norm(left - right)))
    return FinitePairIdentifiability(
        pair_count=len(_canonical_pairs(stages)),
        minimum_signature_separation=minimum,
        identifiable=minimum > float(tolerance),
    )


def higher_order_remainder_ratio(
    endpoint: Any,
    predicted_endpoint: Any,
    *,
    minimum_signature_separation: float,
) -> float:
    """Return ``2*||remainder||/delta`` for the nearest-signature guarantee.

    If the pairwise prediction corresponds to the true chronology and this ratio is below
    one, the higher-order residual is smaller than half the minimum candidate-signature
    separation, which is sufficient for nearest-signature recovery.
    """

    torch = _require_torch()
    separation = float(minimum_signature_separation)
    if separation <= 0:
        raise ValueError("minimum_signature_separation must be positive")
    remainder = float(torch.linalg.vector_norm(endpoint - predicted_endpoint))
    return 2.0 * remainder / separation
