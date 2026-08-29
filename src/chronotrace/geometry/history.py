"""Diagnostics for state-conditioned training-history interactions.

These helpers do not define a new chronology decoder. They measure why a lower-order
finite-pair decoder succeeds or fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class PrefixCommutatorDiagnostic:
    """How a pair commutator changes after a shared training prefix."""

    prefix: tuple[str, ...]
    first: str
    second: str
    base_norm: float
    conditioned_norm: float
    drift_norm: float
    relative_drift: float
    base_conditioned_cosine: float


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised in non-MVP environments
        raise RuntimeError('Install the MVP dependencies with: pip install -e ".[mvp]"') from exc
    return torch


def finite_pair_commutator(
    interactions: Mapping[tuple[str, str], Any],
    *,
    first: str,
    second: str,
) -> Any:
    """Return ``F_second(F_first)-F_first(F_second)`` from finite-pair interactions."""

    forward_key = (second, first)
    reverse_key = (first, second)
    if forward_key not in interactions or reverse_key not in interactions:
        raise ValueError("both directed pair interactions are required")
    return interactions[forward_key] - interactions[reverse_key]


def directional_contamination_ratio(
    residual: Any,
    true_signature: Any,
    alternative_signature: Any,
) -> float:
    """Return the exact residual projection controlling a pairwise decode boundary.

    Let ``d = true_signature - alternative_signature`` and let the observed endpoint
    be ``reference + true_signature + residual``. Nearest-signature decoding prefers
    the true signature over the alternative iff

        -2 <residual, d> / ||d||^2 < 1.

    Unlike the norm-only sufficient bound, this quantity is sensitive to residual
    direction and is therefore exact for this two-candidate comparison.
    """

    torch = _require_torch()
    direction = true_signature - alternative_signature
    denominator = torch.dot(direction.reshape(-1), direction.reshape(-1))
    value = float(denominator)
    if value <= 0.0:
        raise ValueError("candidate signatures must be distinct")
    numerator = -2.0 * torch.dot(residual.reshape(-1), direction.reshape(-1))
    return float(numerator / denominator)


def prefix_conditioned_commutator_diagnostic(
    history_endpoints: Mapping[tuple[str, ...], Any],
    interactions: Mapping[tuple[str, str], Any],
    *,
    prefix: Sequence[str],
    first: str,
    second: str,
) -> PrefixCommutatorDiagnostic:
    """Compare a base pair commutator with the same pair after a shared prefix.

    ``history_endpoints`` must contain the two full endpoints
    ``prefix + (first, second)`` and ``prefix + (second, first)``. Their difference is
    exactly the finite pair commutator evaluated at the state produced by ``prefix``.
    """

    torch = _require_torch()
    prefix_tuple = tuple(prefix)
    if first == second or first in prefix_tuple or second in prefix_tuple:
        raise ValueError("prefix and compared stages must be distinct")

    forward_history = prefix_tuple + (first, second)
    reverse_history = prefix_tuple + (second, first)
    if forward_history not in history_endpoints or reverse_history not in history_endpoints:
        raise ValueError("both prefix-conditioned history endpoints are required")

    base = finite_pair_commutator(interactions, first=first, second=second)
    conditioned = history_endpoints[forward_history] - history_endpoints[reverse_history]
    drift = conditioned - base

    base_norm_tensor = torch.linalg.vector_norm(base)
    conditioned_norm_tensor = torch.linalg.vector_norm(conditioned)
    drift_norm_tensor = torch.linalg.vector_norm(drift)
    base_norm = float(base_norm_tensor)
    conditioned_norm = float(conditioned_norm_tensor)
    drift_norm = float(drift_norm_tensor)
    relative_drift = drift_norm / base_norm if base_norm > 0.0 else float("inf")

    if base_norm > 0.0 and conditioned_norm > 0.0:
        cosine = float(
            torch.dot(base.reshape(-1), conditioned.reshape(-1))
            / (base_norm_tensor * conditioned_norm_tensor)
        )
    else:
        cosine = float("nan")

    return PrefixCommutatorDiagnostic(
        prefix=prefix_tuple,
        first=first,
        second=second,
        base_norm=base_norm,
        conditioned_norm=conditioned_norm,
        drift_norm=drift_norm,
        relative_drift=relative_drift,
        base_conditioned_cosine=cosine,
    )


def pairwise_precedence_accuracy(true_order: Sequence[str], predicted_order: Sequence[str]) -> float:
    """Return the fraction of pairwise precedence relations preserved by a prediction."""

    true_tuple = tuple(true_order)
    predicted_tuple = tuple(predicted_order)
    if len(true_tuple) != len(set(true_tuple)):
        raise ValueError("true_order contains duplicate stages")
    if set(true_tuple) != set(predicted_tuple) or len(predicted_tuple) != len(true_tuple):
        raise ValueError("orders must contain the same unique stages")

    true_position = {stage: index for index, stage in enumerate(true_tuple)}
    predicted_position = {stage: index for index, stage in enumerate(predicted_tuple)}
    correct = 0
    total = 0
    for left_index, left in enumerate(true_tuple):
        for right in true_tuple[left_index + 1 :]:
            total += 1
            true_relation = true_position[left] < true_position[right]
            predicted_relation = predicted_position[left] < predicted_position[right]
            correct += int(true_relation == predicted_relation)
    return correct / total if total else 1.0


def kendall_tau_for_orders(true_order: Sequence[str], predicted_order: Sequence[str]) -> float:
    """Return Kendall tau for two complete stage permutations without SciPy."""

    accuracy = pairwise_precedence_accuracy(true_order, predicted_order)
    return 2.0 * accuracy - 1.0
