"""Second-order geometry for inverse sequential-learning forensics.

The local two-stage expansion for gradient steps is

    theta_AB = theta_0 - eta(g_A + g_B) + eta^2 H_B g_A + O(eta^3)
    theta_BA = theta_0 - eta(g_A + g_B) + eta^2 H_A g_B + O(eta^3)

so the order-independent midpoint and the Lie-bracket vector provide a direct
endpoint decoder. The same construction extends to N stages: after removing the
symmetric first- and second-order reference, a permutation is represented by a
signed sum of pairwise bracket vectors.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import permutations
from typing import Any


@dataclass(frozen=True)
class PairwiseGeometry:
    """Local second-order geometry for candidate stages A and B."""

    first_order_reference: Any
    midpoint_reference: Any
    bracket: Any
    h_b_g_a: Any
    h_a_g_b: Any


@dataclass(frozen=True)
class PermutationDecode:
    """Nearest second-order chronology signature among candidate permutations."""

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


def flatten_tensors(tensors: Sequence[Any]) -> Any:
    """Flatten a sequence of tensors into one parameter-space vector."""

    torch = _require_torch()
    if not tensors:
        raise ValueError("at least one tensor is required")
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def _split_vector_like(vector: Any, tensors: Sequence[Any]) -> list[Any]:
    pieces: list[Any] = []
    offset = 0
    for tensor in tensors:
        width = tensor.numel()
        pieces.append(vector[offset : offset + width].reshape_as(tensor))
        offset += width
    if offset != vector.numel():
        raise ValueError("vector size does not match parameter tensors")
    return pieces


def parameter_vector(parameters: Sequence[Any]) -> Any:
    """Return detached flattened parameters for endpoint comparisons."""

    return flatten_tensors([parameter.detach() for parameter in parameters])


def loss_gradient(loss: Any, parameters: Sequence[Any]) -> Any:
    """Return a detached flattened gradient for a scalar loss."""

    torch = _require_torch()
    gradients = torch.autograd.grad(loss, parameters, create_graph=False)
    return flatten_tensors([gradient.detach() for gradient in gradients])


def hessian_vector_product(loss: Any, parameters: Sequence[Any], vector: Any) -> Any:
    """Compute H(loss) @ vector without materializing the Hessian."""

    torch = _require_torch()
    gradients = torch.autograd.grad(loss, parameters, create_graph=True)
    vector_pieces = _split_vector_like(vector.detach(), parameters)
    directional = sum(
        (gradient * piece).sum() for gradient, piece in zip(gradients, vector_pieces, strict=True)
    )
    hvp = torch.autograd.grad(directional, parameters, create_graph=False)
    return flatten_tensors([value.detach() for value in hvp])


def local_stage_derivatives(
    loss_fns: Mapping[str, Callable[[], Any]],
    parameters: Sequence[Any],
) -> tuple[dict[str, Any], dict[tuple[str, str], Any]]:
    """Compute local stage gradients and all directed cross-HVPs.

    The cross-HVP key ``(destination, source)`` stores ``H_destination g_source``.
    Fresh loss graphs are built for every derivative so callers can use ordinary
    PyTorch modules without retaining a large autograd graph across computations.
    """

    stages = tuple(loss_fns)
    if len(stages) < 2:
        raise ValueError("at least two stages are required")

    gradients = {stage: loss_gradient(loss_fns[stage](), parameters) for stage in stages}
    cross: dict[tuple[str, str], Any] = {}
    for destination in stages:
        for source in stages:
            if destination == source:
                continue
            cross[(destination, source)] = hessian_vector_product(
                loss_fns[destination](), parameters, gradients[source]
            )
    return gradients, cross


def pairwise_endpoint_geometry(
    theta0: Any,
    g_a: Any,
    g_b: Any,
    h_b_g_a: Any,
    h_a_g_b: Any,
    *,
    step_size: float,
) -> PairwiseGeometry:
    """Construct the order-independent midpoint and antisymmetric bracket."""

    eta = float(step_size)
    if eta <= 0:
        raise ValueError("step_size must be positive")
    first_order = theta0 - eta * (g_a + g_b)
    midpoint = first_order + 0.5 * eta**2 * (h_b_g_a + h_a_g_b)
    bracket = h_b_g_a - h_a_g_b
    return PairwiseGeometry(
        first_order_reference=first_order,
        midpoint_reference=midpoint,
        bracket=bracket,
        h_b_g_a=h_b_g_a,
        h_a_g_b=h_a_g_b,
    )


def pairwise_chrono_score(
    endpoint: Any,
    geometry: PairwiseGeometry,
    *,
    step_size: float,
    epsilon: float = 1e-18,
) -> float:
    """Return the normalized endpoint chronology score.

    Under the local expansion, an A->B endpoint tends to +1 and B->A tends to -1.
    """

    torch = _require_torch()
    eta = float(step_size)
    bracket_energy = torch.dot(geometry.bracket, geometry.bracket)
    denominator = 0.5 * eta**2 * float(bracket_energy)
    if denominator <= epsilon:
        raise ValueError("bracket energy is too small to identify order")
    residual = endpoint - geometry.midpoint_reference
    numerator = float(torch.dot(residual, geometry.bracket))
    return numerator / denominator


def decode_pairwise_order(
    endpoint: Any,
    geometry: PairwiseGeometry,
    *,
    step_size: float,
) -> str:
    """Decode A->B versus B->A from one endpoint vector."""

    return "AB" if pairwise_chrono_score(endpoint, geometry, step_size=step_size) >= 0 else "BA"


def estimate_step_size(theta0: Any, endpoint: Any, gradients: Sequence[Any]) -> float:
    """Estimate a shared local SGD step size from the order-independent first-order drift."""

    torch = _require_torch()
    if not gradients:
        raise ValueError("at least one gradient is required")
    gradient_sum = torch.zeros_like(theta0)
    for gradient in gradients:
        gradient_sum = gradient_sum + gradient
    denominator = float(torch.dot(gradient_sum, gradient_sum))
    if denominator <= 0:
        raise ValueError("summed gradient has zero norm")
    displacement = endpoint - theta0
    estimate = -float(torch.dot(displacement, gradient_sum)) / denominator
    if estimate <= 0:
        raise ValueError("estimated step size is not positive")
    return estimate


def _canonical_pairs(stages: Sequence[str]) -> list[tuple[str, str]]:
    if len(stages) != len(set(stages)):
        raise ValueError("stage names must be unique")
    return [(left, right) for index, left in enumerate(stages) for right in stages[index + 1 :]]


def _require_cross_hvps(
    stages: Sequence[str],
    cross_hvps: Mapping[tuple[str, str], Any],
) -> None:
    for left, right in _canonical_pairs(stages):
        for key in ((right, left), (left, right)):
            if key not in cross_hvps:
                raise ValueError(f"missing cross-HVP {key!r}")


def multi_stage_symmetric_reference(
    theta0: Any,
    gradients: Mapping[str, Any],
    cross_hvps: Mapping[tuple[str, str], Any],
    *,
    stages: Sequence[str],
    step_size: float,
) -> Any:
    """Return the permutation-independent endpoint reference through second order."""

    torch = _require_torch()
    stages = tuple(stages)
    _require_cross_hvps(stages, cross_hvps)
    if set(stages) != set(gradients):
        raise ValueError("gradients must contain exactly the declared stages")
    eta = float(step_size)
    if eta <= 0:
        raise ValueError("step_size must be positive")

    gradient_sum = torch.zeros_like(theta0)
    for stage in stages:
        gradient_sum = gradient_sum + gradients[stage]

    symmetric_second_order = torch.zeros_like(theta0)
    for left, right in _canonical_pairs(stages):
        symmetric_second_order = symmetric_second_order + 0.5 * (
            cross_hvps[(right, left)] + cross_hvps[(left, right)]
        )
    return theta0 - eta * gradient_sum + eta**2 * symmetric_second_order


def permutation_signature(
    permutation: Sequence[str],
    cross_hvps: Mapping[tuple[str, str], Any],
    *,
    stages: Sequence[str],
    step_size: float,
) -> Any:
    """Return the signed pairwise-bracket signature for one candidate permutation."""

    torch = _require_torch()
    stages = tuple(stages)
    permutation = tuple(permutation)
    if set(permutation) != set(stages) or len(permutation) != len(stages):
        raise ValueError("permutation must contain every stage exactly once")
    _require_cross_hvps(stages, cross_hvps)
    eta = float(step_size)
    if eta <= 0:
        raise ValueError("step_size must be positive")

    position = {stage: index for index, stage in enumerate(permutation)}
    sample = next(iter(cross_hvps.values()))
    signature = torch.zeros_like(sample)
    for left, right in _canonical_pairs(stages):
        bracket = cross_hvps[(right, left)] - cross_hvps[(left, right)]
        orientation = 1.0 if position[left] < position[right] else -1.0
        signature = signature + 0.5 * eta**2 * orientation * bracket
    return signature


def decode_permutation(
    endpoint: Any,
    symmetric_reference: Any,
    cross_hvps: Mapping[tuple[str, str], Any],
    *,
    stages: Sequence[str],
    step_size: float,
) -> PermutationDecode:
    """Decode the nearest second-order chronology among all stage permutations.

    This exhaustive decoder is intentionally for small controlled benchmarks. A scalable
    projection/ranking decoder belongs in the next research slice once identifiability is
    established on 3--5 stages.
    """

    torch = _require_torch()
    residual = endpoint - symmetric_reference
    ranked: list[tuple[float, tuple[str, ...]]] = []
    for candidate in permutations(tuple(stages)):
        signature = permutation_signature(
            candidate,
            cross_hvps,
            stages=stages,
            step_size=step_size,
        )
        error = float(torch.linalg.vector_norm(residual - signature))
        ranked.append((error, candidate))
    ranked.sort(key=lambda item: item[0])
    best_error, best = ranked[0]
    runner_up_error = ranked[1][0] if len(ranked) > 1 else float("inf")
    return PermutationDecode(
        permutation=best,
        best_error=best_error,
        runner_up_error=runner_up_error,
        margin=runner_up_error - best_error,
    )
