from itertools import permutations

import pytest

from chronotrace.geometry.operators import (
    decode_operator_permutation,
    displacement_jvp_finite_difference,
    local_stage_map_derivatives,
    operator_pair_geometry,
    operator_pair_score,
    operator_symmetric_reference,
    stage_displacement,
)

torch = pytest.importorskip("torch")


@pytest.fixture
def stage_system():
    scale = 0.03

    def stage_a(theta):
        x0, x1, x2 = theta
        displacement = torch.stack((x0**2 + 0.2 * x1, 0.4 * x0 * x2, -0.3 * x1**2))
        return theta + scale * displacement

    def stage_b(theta):
        x0, x1, x2 = theta
        displacement = torch.stack((-0.2 * x1 * x2, x1**2 + 0.1 * x0, 0.5 * x0 * x1))
        return theta + scale * displacement

    def stage_c(theta):
        x0, x1, x2 = theta
        displacement = torch.stack((0.1 * x2**2, -0.4 * x0 * x2, x2**2 + 0.2 * x1))
        return theta + scale * displacement

    theta0 = torch.tensor([0.3, -0.2, 0.4], dtype=torch.float64)
    return theta0, {"A": stage_a, "B": stage_b, "C": stage_c}


def _run(theta0, stage_maps, history):
    theta = theta0
    for stage in history:
        theta = stage_maps[stage](theta)
    return theta


def test_centered_finite_difference_recovers_displacement_jvp(stage_system) -> None:
    theta0, stage_maps = stage_system
    direction = stage_displacement(stage_maps["A"], theta0)

    theta = theta0.detach().clone().requires_grad_(True)

    def displacement(theta_value):
        return stage_maps["B"](theta_value) - theta_value

    _, expected = torch.autograd.functional.jvp(displacement, (theta,), (direction,))
    observed = displacement_jvp_finite_difference(
        stage_maps["B"],
        theta0,
        direction,
        epsilon=1e-5,
    )
    torch.testing.assert_close(observed, expected.detach(), rtol=1e-7, atol=1e-9)


def test_operator_pair_score_recovers_both_orders(stage_system) -> None:
    theta0, stage_maps = stage_system
    deltas, cross = local_stage_map_derivatives(stage_maps, theta0, epsilon=1e-5)
    geometry = operator_pair_geometry(
        theta0,
        deltas["A"],
        deltas["B"],
        cross[("B", "A")],
        cross[("A", "B")],
    )

    endpoint_ab = _run(theta0, stage_maps, "AB")
    endpoint_ba = _run(theta0, stage_maps, "BA")
    assert operator_pair_score(endpoint_ab, geometry) > 0.9
    assert operator_pair_score(endpoint_ba, geometry) < -0.9


def test_macro_operator_decoder_recovers_all_three_stage_orders(stage_system) -> None:
    theta0, stage_maps = stage_system
    stages = ("A", "B", "C")
    deltas, cross = local_stage_map_derivatives(stage_maps, theta0, epsilon=1e-5)
    reference = operator_symmetric_reference(theta0, deltas, cross, stages=stages)

    for history in permutations(stages):
        endpoint = _run(theta0, stage_maps, history)
        decoded = decode_operator_permutation(
            endpoint,
            reference,
            cross,
            stages=stages,
        )
        assert decoded.permutation == history
        assert decoded.margin > 0
