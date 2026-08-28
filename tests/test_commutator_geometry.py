from itertools import permutations

import pytest

from chronotrace.geometry.commutator import (
    decode_pairwise_order,
    decode_permutation,
    estimate_step_size,
    local_stage_derivatives,
    multi_stage_symmetric_reference,
    pairwise_chrono_score,
    pairwise_endpoint_geometry,
    parameter_vector,
)

torch = pytest.importorskip("torch")


def _loss_functions(theta: torch.Tensor):
    def loss_a() -> torch.Tensor:
        x0, x1, x2 = theta
        return 0.5 * (x0 + 0.7 * x1**2 - 1.1) ** 2 + 0.2 * (x2 - x0 * x1) ** 2

    def loss_b() -> torch.Tensor:
        x0, x1, x2 = theta
        return 0.5 * (x1 + 0.6 * x0 * x2 + 0.4) ** 2 + 0.3 * (x0 - x2**2) ** 2

    def loss_c() -> torch.Tensor:
        x0, x1, x2 = theta
        return 0.5 * (x2 + 0.5 * x0 * x1 - 0.2) ** 2 + 0.25 * (x1 - 0.4 * x0**2) ** 2

    return {"A": loss_a, "B": loss_b, "C": loss_c}


def _one_step(theta: torch.Tensor, loss_fn, eta: float) -> torch.Tensor:
    loss = loss_fn()
    (gradient,) = torch.autograd.grad(loss, (theta,))
    return (theta - eta * gradient).detach().requires_grad_(True)


def _run_history(theta0: torch.Tensor, history: str, eta: float) -> torch.Tensor:
    theta = theta0.detach().clone().requires_grad_(True)
    for stage in history:
        theta = _one_step(theta, _loss_functions(theta)[stage], eta)
    return theta.detach()


def _local_geometry(theta0: torch.Tensor):
    theta = theta0.detach().clone().requires_grad_(True)
    gradients, cross = local_stage_derivatives(_loss_functions(theta), (theta,))
    return theta.detach(), gradients, cross


def test_pairwise_decoder_recovers_both_orders() -> None:
    theta0 = torch.tensor([0.35, -0.45, 0.25], dtype=torch.float64)
    eta = 0.025
    base, gradients, cross = _local_geometry(theta0)
    geometry = pairwise_endpoint_geometry(
        base,
        gradients["A"],
        gradients["B"],
        cross[("B", "A")],
        cross[("A", "B")],
        step_size=eta,
    )

    endpoint_ab = _run_history(theta0, "AB", eta)
    endpoint_ba = _run_history(theta0, "BA", eta)

    assert decode_pairwise_order(endpoint_ab, geometry, step_size=eta) == "AB"
    assert decode_pairwise_order(endpoint_ba, geometry, step_size=eta) == "BA"
    assert pairwise_chrono_score(endpoint_ab, geometry, step_size=eta) == pytest.approx(
        1.0, abs=0.03
    )
    assert pairwise_chrono_score(endpoint_ba, geometry, step_size=eta) == pytest.approx(
        -1.0, abs=0.03
    )


def test_unknown_step_size_is_locally_recoverable() -> None:
    theta0 = torch.tensor([0.35, -0.45, 0.25], dtype=torch.float64)
    eta = 0.0125
    base, gradients, _ = _local_geometry(theta0)

    for history in ("AB", "BA"):
        endpoint = _run_history(theta0, history, eta)
        estimate = estimate_step_size(base, endpoint, [gradients["A"], gradients["B"]])
        assert estimate == pytest.approx(eta, rel=0.04)


def test_all_three_stage_permutations_decode() -> None:
    theta0 = torch.tensor([0.35, -0.45, 0.25], dtype=torch.float64)
    eta = 0.025
    stages = ("A", "B", "C")
    base, gradients, cross = _local_geometry(theta0)
    reference = multi_stage_symmetric_reference(
        base,
        gradients,
        cross,
        stages=stages,
        step_size=eta,
    )

    for candidate in permutations(stages):
        endpoint = _run_history(theta0, "".join(candidate), eta)
        decoded = decode_permutation(
            endpoint,
            reference,
            cross,
            stages=stages,
            step_size=eta,
        )
        assert decoded.permutation == candidate
        assert decoded.margin > 0


def test_parameter_vector_flattens_model_parameters() -> None:
    module = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1)).double()
    vector = parameter_vector(tuple(module.parameters()))
    assert vector.ndim == 1
    assert vector.numel() == sum(parameter.numel() for parameter in module.parameters())
