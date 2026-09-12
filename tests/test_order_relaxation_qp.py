from itertools import combinations, permutations

import pytest

from chronotrace.geometry.interactions import OrderedInteractionBasis
from chronotrace.geometry.order_relaxation import (
    K3LocalOrderRelaxation,
    build_k3_local_order_relaxation,
    k3_local_order_vertex,
    mix_k3_local_order_points,
    validate_k3_local_order_point,
)
from chronotrace.geometry.order_relaxation_qp import (
    build_k3_local_order_linear_system,
    k3_local_order_linear_residuals,
    k3_local_order_point_to_vector,
    k3_local_order_vector_to_point,
)

torch = pytest.importorskip("torch")


def _synthetic_relaxation(stages: tuple[str, ...]) -> K3LocalOrderRelaxation:
    base = torch.tensor([0.0], dtype=torch.float64)
    interactions = {}
    value = 0.01
    for size in range(1, 4):
        for word in permutations(stages, size):
            interactions[word] = torch.tensor([value], dtype=torch.float64)
            value += 0.01
    basis = OrderedInteractionBasis(
        stages=stages,
        max_degree=3,
        base=base,
        endpoints={},
        interactions=interactions,
        stage_executions=0,
    )
    return build_k3_local_order_relaxation(basis)


def test_constraint_system_size_is_cubic_for_fixed_k3() -> None:
    relaxation4 = _synthetic_relaxation(("A", "B", "C", "D"))
    system4 = build_k3_local_order_linear_system(relaxation4)
    assert system4.variable_count == 6 + 6 * 4 == 30
    assert system4.equality_count == 4 * 4 == 16

    stages5 = ("A", "B", "C", "D", "E")
    relaxation5 = K3LocalOrderRelaxation(
        stages=stages5,
        pairs=tuple(combinations(stages5, 2)),
        triples=tuple(combinations(stages5, 3)),
        constant=None,
        pair_coefficients={},
        triple_coefficients={},
    )
    system5 = build_k3_local_order_linear_system(relaxation5)
    assert system5.variable_count == 10 + 6 * 10 == 70
    assert system5.equality_count == 4 * 10 == 40


def test_all_global_permutations_are_exact_feasible_vertices() -> None:
    relaxation = _synthetic_relaxation(("A", "B", "C", "D"))
    system = build_k3_local_order_linear_system(relaxation)

    for chronology in permutations(relaxation.stages):
        vertex = k3_local_order_vertex(chronology, relaxation)
        vector = k3_local_order_point_to_vector(vertex, relaxation, system)
        equality_violation, box_violation = k3_local_order_linear_residuals(vector, system)
        assert equality_violation == pytest.approx(0.0, abs=1e-12)
        assert box_violation == pytest.approx(0.0, abs=1e-12)
        decoded = k3_local_order_vector_to_point(vector, relaxation, system)
        validate_k3_local_order_point(decoded, relaxation)


def test_convex_mixture_of_chronologies_remains_feasible() -> None:
    relaxation = _synthetic_relaxation(("A", "B", "C", "D"))
    system = build_k3_local_order_linear_system(relaxation)
    first = k3_local_order_vertex(("A", "B", "C", "D"), relaxation)
    second = k3_local_order_vertex(("D", "A", "C", "B"), relaxation)
    mixture = mix_k3_local_order_points(first, second, first_weight=0.41)
    vector = k3_local_order_point_to_vector(mixture, relaxation, system)

    equality_violation, box_violation = k3_local_order_linear_residuals(vector, system)
    assert equality_violation == pytest.approx(0.0, abs=1e-12)
    assert box_violation == pytest.approx(0.0, abs=1e-12)


def test_inconsistent_pair_coordinate_violates_linear_system() -> None:
    relaxation = _synthetic_relaxation(("A", "B", "C", "D"))
    system = build_k3_local_order_linear_system(relaxation)
    vertex = k3_local_order_vertex(("A", "B", "C", "D"), relaxation)
    vector = k3_local_order_point_to_vector(vertex, relaxation, system)
    vector[system.pair_indices[("A", "B")]] = 0.25

    equality_violation, box_violation = k3_local_order_linear_residuals(vector, system)
    assert equality_violation == pytest.approx(0.75, abs=1e-12)
    assert box_violation == pytest.approx(0.0, abs=1e-12)
