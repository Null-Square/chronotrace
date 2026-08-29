from itertools import permutations

import pytest

from chronotrace.geometry.interactions import (
    measure_ordered_interaction_basis_compact,
    ordered_interaction_prediction,
)
from chronotrace.geometry.order_relaxation import (
    build_k3_local_order_relaxation,
    k3_local_order_prediction,
    k3_local_order_vertex,
    mix_k3_local_order_points,
    validate_k3_local_order_point,
)

torch = pytest.importorskip("torch")


def _four_stage_system():
    scale = 0.8

    def stage_a(theta):
        x0, x1, x2 = theta
        effect = torch.stack((x0**2 + 0.2 * x1, 0.4 * x0 * x2, -0.3 * x1**2))
        return theta + scale * effect

    def stage_b(theta):
        x0, x1, x2 = theta
        effect = torch.stack((-0.2 * x1 * x2, x1**2 + 0.1 * x0, 0.5 * x0 * x1))
        return theta + scale * effect

    def stage_c(theta):
        x0, x1, x2 = theta
        effect = torch.stack((0.1 * x2**2, -0.4 * x0 * x2, x2**2 + 0.2 * x1))
        return theta + scale * effect

    def stage_d(theta):
        x0, x1, x2 = theta
        effect = torch.stack(
            (-0.3 * x0 * x1 + 0.1 * x2, 0.2 * x2**2 + 0.15 * x0, x0**2 - 0.25 * x1 * x2)
        )
        return theta + scale * effect

    theta0 = torch.tensor([0.3, -0.2, 0.4], dtype=torch.float64)
    return theta0, {"A": stage_a, "B": stage_b, "C": stage_c, "D": stage_d}


def test_every_permutation_vertex_exactly_matches_existing_k3_prediction() -> None:
    theta0, stage_maps = _four_stage_system()
    basis = measure_ordered_interaction_basis_compact(stage_maps, theta0, max_degree=3)
    relaxation = build_k3_local_order_relaxation(basis)

    assert len(relaxation.pairs) == 6
    assert len(relaxation.triples) == 4
    assert len(relaxation.triple_coefficients) == 24

    for chronology in permutations(relaxation.stages):
        vertex = k3_local_order_vertex(chronology, relaxation)
        validate_k3_local_order_point(vertex, relaxation)
        local = k3_local_order_prediction(vertex, relaxation)
        direct = ordered_interaction_prediction(chronology, basis, degree=3)
        torch.testing.assert_close(local, direct, rtol=0.0, atol=1e-11)


def test_convex_mixture_stays_locally_feasible_and_prediction_is_affine() -> None:
    theta0, stage_maps = _four_stage_system()
    basis = measure_ordered_interaction_basis_compact(stage_maps, theta0, max_degree=3)
    relaxation = build_k3_local_order_relaxation(basis)
    first = k3_local_order_vertex(("A", "B", "C", "D"), relaxation)
    second = k3_local_order_vertex(("D", "C", "B", "A"), relaxation)
    mixture = mix_k3_local_order_points(first, second, first_weight=0.37)

    validate_k3_local_order_point(mixture, relaxation)
    mixed_prediction = k3_local_order_prediction(mixture, relaxation)
    first_prediction = k3_local_order_prediction(first, relaxation)
    second_prediction = k3_local_order_prediction(second, relaxation)
    expected = 0.37 * first_prediction + 0.63 * second_prediction
    torch.testing.assert_close(mixed_prediction, expected, rtol=0.0, atol=1e-11)


def test_inconsistent_pair_triple_marginal_is_rejected() -> None:
    theta0, stage_maps = _four_stage_system()
    basis = measure_ordered_interaction_basis_compact(stage_maps, theta0, max_degree=3)
    relaxation = build_k3_local_order_relaxation(basis)
    vertex = k3_local_order_vertex(("A", "B", "C", "D"), relaxation)

    vertex.pair_precedence[("A", "B")] = 0.0
    with pytest.raises(ValueError, match="marginal disagrees"):
        validate_k3_local_order_point(vertex, relaxation)
