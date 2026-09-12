import numpy as np

from chronotrace.geometry.multi_witness import (
    certify_l1_witness_combination,
    combine_linear_witness_objectives,
    optimize_l1_witness_combination,
)


def test_multi_witness_can_separate_when_each_single_witness_fails() -> None:
    # Each witness alone has a negative class minimum, but their L1-bounded mixture is
    # positive on every vertex.
    scores = np.asarray(
        [
            [1.0, -0.2],
            [-0.2, 1.0],
        ],
        dtype=np.float64,
    )
    assert np.min(scores[0]) < 0.0
    assert np.min(scores[1]) < 0.0

    certificate = optimize_l1_witness_combination(scores, certificate_guard=1e-12)
    assert certificate.l1_norm <= 1.0 + 1e-12
    assert certificate.certified_lower_bound > 0.39
    assert certificate.euclidean_distance_lower_bound > 0.39
    assert np.min(certificate.vertex_support_scores) > 0.39


def test_target_vertex_forces_zero_multi_witness_distance_bound() -> None:
    scores = np.asarray(
        [
            [0.0, 1.0, -0.4],
            [0.0, -0.3, 1.2],
        ],
        dtype=np.float64,
    )
    certificate = optimize_l1_witness_combination(scores)
    assert certificate.certified_lower_bound <= 0.0
    assert certificate.euclidean_distance_lower_bound == 0.0


def test_arbitrary_large_coefficients_are_rescaled_before_certification() -> None:
    scores = np.asarray(
        [
            [0.7, 0.9],
            [0.4, 0.8],
        ],
        dtype=np.float64,
    )
    certificate = certify_l1_witness_combination(
        scores,
        np.asarray([20.0, 10.0]),
        certificate_guard=0.0,
    )
    assert abs(certificate.l1_norm - 1.0) < 1e-12
    expected = np.min(scores.T @ np.asarray([2.0 / 3.0, 1.0 / 3.0]))
    assert abs(certificate.certified_lower_bound - expected) < 1e-12


def test_combined_linear_objective_matches_explicit_mixture() -> None:
    constants = np.asarray([0.3, -0.1, 0.8], dtype=np.float64)
    coefficients = np.asarray(
        [
            [1.0, 2.0],
            [-2.0, 0.5],
            [0.25, -1.0],
        ],
        dtype=np.float64,
    )
    alpha = np.asarray([2.0, -1.0, 1.0], dtype=np.float64)
    constant, combined = combine_linear_witness_objectives(constants, coefficients, alpha)
    normalized = alpha / np.sum(np.abs(alpha))
    assert abs(constant - float(normalized @ constants)) < 1e-12
    assert np.allclose(combined, normalized @ coefficients)
