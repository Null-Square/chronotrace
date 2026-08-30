import numpy as np

from chronotrace.geometry.directional_tail import (
    directional_tail_hull_certificate,
    norm_tail_fallback,
)


def test_directional_tail_certificate_matches_exact_fixed_direction_witness() -> None:
    target = np.array([0.0, 0.0])
    approximate = np.array([[5.0, -1.0], [5.0, 1.0]])
    tails = np.array([[0.5, 2.0], [1.0, -3.0]])
    exact = approximate + tails
    direction = np.array([-1.0, 0.0])

    approximate_witnesses = (target[None, :] - approximate) @ direction
    exact_directional_corrections = tails @ direction
    certificate = directional_tail_hull_certificate(
        approximate_witnesses,
        exact_directional_corrections,
    )
    exact_witnesses = (target[None, :] - exact) @ direction

    np.testing.assert_allclose(certificate.adjusted_vertex_witnesses, exact_witnesses)
    assert abs(certificate.lower_bound - float(np.min(exact_witnesses))) < 1e-12


def test_large_orthogonal_tail_does_not_destroy_directional_certificate() -> None:
    # Approximate vertices lie on x=10. Their exact tails are huge in y but have zero
    # projection onto the x-separating direction. A norm-tail certificate is vacuous,
    # while the directional certificate retains the exact margin 10.
    approximate_witnesses = np.array([10.0, 10.0])
    directional_tails = np.array([0.0, 0.0])
    tail_norms = np.array([100.0, 100.0])

    directional = directional_tail_hull_certificate(
        approximate_witnesses,
        directional_tails,
        numerical_guard=1e-6,
    )
    norm_based = norm_tail_fallback(
        approximate_witnesses,
        tail_norms,
        numerical_guard=1e-6,
    )

    assert directional.lower_bound == 10.0
    assert directional.certified_impossible
    assert norm_based.lower_bound == 0.0
    assert not norm_based.certified_impossible


def test_directional_upper_bounds_can_be_asymmetric_per_vertex() -> None:
    witnesses = np.array([4.0, 7.0, 5.0])
    tail_upper_bounds = np.array([1.0, 6.5, -1.0])
    certificate = directional_tail_hull_certificate(
        witnesses,
        tail_upper_bounds,
        feasible_upper_bound=0.4,
        numerical_guard=0.05,
    )

    np.testing.assert_allclose(certificate.adjusted_vertex_witnesses, [3.0, 0.5, 6.0])
    assert certificate.lower_bound == 0.5
    assert certificate.certified_impossible


def test_norm_tail_fallback_is_directionally_conservative() -> None:
    witnesses = np.array([3.0, 4.0])
    radii = np.array([1.0, 2.0])
    certificate = norm_tail_fallback(
        witnesses,
        radii,
        feasible_upper_bound=1.5,
        numerical_guard=0.1,
    )

    assert certificate.lower_bound == 2.0
    assert certificate.certified_impossible
