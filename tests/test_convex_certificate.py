import numpy as np

from chronotrace.geometry.convex_certificate import (
    certify_lower_bound_elimination,
    dual_hull_distance_certificate,
    project_onto_small_convex_hull,
)


def test_small_convex_projection_matches_known_segment_distance() -> None:
    target = np.array([0.0, 0.0])
    vertices = np.array([[2.0, -1.0], [2.0, 1.0], [4.0, 0.0]])

    projection = project_onto_small_convex_hull(target, vertices)

    assert abs(projection.distance - 2.0) < 1e-12
    np.testing.assert_allclose(projection.point, np.array([2.0, 0.0]), atol=1e-12)
    assert projection.simplex_residual < 1e-12
    assert projection.minimum_weight >= 0.0
    assert projection.distance <= min(np.linalg.norm(target - vertex) for vertex in vertices)


def test_dual_witness_matches_projection_when_projection_direction_is_exact() -> None:
    target = np.array([0.0, 0.0])
    vertices = np.array([[2.0, -1.0], [2.0, 1.0], [4.0, 0.0]])
    projection = project_onto_small_convex_hull(target, vertices)

    certificate = dual_hull_distance_certificate(
        target,
        vertices,
        projection=projection,
    )

    assert abs(certificate.direction_norm - 1.0) < 1e-12
    assert abs(certificate.lower_bound - projection.distance) < 1e-12
    assert abs(certificate.primal_dual_gap) < 1e-12


def test_inexact_dual_direction_remains_a_conservative_lower_bound() -> None:
    target = np.array([0.0, 0.0])
    vertices = np.array([[2.0, -1.0], [2.0, 1.0], [4.0, 0.0]])
    projection = project_onto_small_convex_hull(target, vertices)

    certificate = dual_hull_distance_certificate(
        target,
        vertices,
        projection=projection,
        direction=np.array([-1.0, 0.4]),
    )

    assert abs(certificate.direction_norm - 1.0) < 1e-12
    assert certificate.lower_bound <= projection.distance + 1e-12
    for weights in (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.2, 0.7, 0.1]),
        np.array([0.0, 0.5, 0.5]),
    ):
        point = weights @ vertices
        assert certificate.lower_bound <= np.linalg.norm(target - point) + 1e-12


def test_dual_witness_is_zero_when_target_is_inside_hull() -> None:
    vertices = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    target = np.array([0.5, 0.5])
    projection = project_onto_small_convex_hull(target, vertices)
    certificate = dual_hull_distance_certificate(target, vertices, projection=projection)

    assert projection.distance < 1e-12
    assert certificate.lower_bound == 0.0
    assert certificate.direction_norm == 0.0


def test_false_low_degree_certificate_requires_explicit_tail_bound() -> None:
    # Think of q_K=[10,0] as a low-degree approximation to an exact chronology class
    # whose omitted interaction tail is [-10,0]. The exact class therefore contains the
    # target, even though the truncated class looks far enough away to eliminate.
    target = np.array([0.0, 0.0])
    truncated_vertices = np.array([[10.0, 0.0]])
    projection = project_onto_small_convex_hull(target, truncated_vertices)
    certificate = dual_hull_distance_certificate(
        target,
        truncated_vertices,
        projection=projection,
    )
    assert abs(certificate.lower_bound - 10.0) < 1e-12

    no_tail = certify_lower_bound_elimination(
        certificate.lower_bound,
        feasible_upper_bound=1.0,
        numerical_guard=1e-6,
    )
    assert no_tail.truncated_eliminated
    assert no_tail.exact_eliminated is None
    assert no_tail.exact_lower_bound is None

    known_tail = certify_lower_bound_elimination(
        certificate.lower_bound,
        feasible_upper_bound=1.0,
        numerical_guard=1e-6,
        interaction_tail_radius=10.0,
    )
    assert known_tail.truncated_eliminated
    assert known_tail.exact_lower_bound == 0.0
    assert known_tail.exact_eliminated is False


def test_true_elimination_is_allowed_when_tail_adjusted_margin_survives() -> None:
    result = certify_lower_bound_elimination(
        8.0,
        feasible_upper_bound=2.0,
        numerical_guard=0.1,
        interaction_tail_radius=1.5,
    )

    assert result.truncated_eliminated
    assert result.exact_lower_bound == 6.5
    assert result.exact_eliminated is True
