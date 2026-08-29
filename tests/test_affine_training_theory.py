import math

import numpy as np
import pytest

from chronotrace.geometry.affine import (
    chronology_endpoint,
    common_continuation_difference,
    compose_affine,
    quadratic_gd_stage,
    scalar_recency_endpoint,
    spectral_half_life,
    two_stage_order_difference,
)


def test_quadratic_macro_stage_matches_repeated_gradient_descent() -> None:
    hessian = np.array([[2.0, 0.3], [0.3, 1.2]])
    optimum = np.array([0.7, -0.4])
    theta0 = np.array([-0.2, 0.5])
    eta = 0.07
    steps = 9

    stage = quadratic_gd_stage(hessian, optimum, learning_rate=eta, steps=steps)
    expected = theta0.copy()
    for _ in range(steps):
        expected = expected - eta * (hessian @ (expected - optimum))

    np.testing.assert_allclose(stage.apply(theta0), expected, rtol=1e-13, atol=1e-13)


def test_affine_order_difference_matches_direct_compositions() -> None:
    theta0 = np.array([0.2, -0.5])
    stage_a = quadratic_gd_stage(
        np.array([[1.5, 0.4], [0.4, 0.9]]),
        np.array([0.8, -0.1]),
        learning_rate=0.08,
        steps=4,
    )
    stage_b = quadratic_gd_stage(
        np.array([[0.7, -0.2], [-0.2, 1.8]]),
        np.array([-0.3, 0.6]),
        learning_rate=0.05,
        steps=5,
    )

    direct = stage_b.apply(stage_a.apply(theta0)) - stage_a.apply(stage_b.apply(theta0))
    formula = two_stage_order_difference(theta0, stage_a, stage_b)
    np.testing.assert_allclose(formula, direct, rtol=1e-13, atol=1e-13)

    composed = compose_affine(stage_b, stage_a)
    np.testing.assert_allclose(
        composed.apply(theta0),
        stage_b.apply(stage_a.apply(theta0)),
        rtol=1e-13,
        atol=1e-13,
    )


def test_different_optima_create_scalar_recency_without_noncommutativity() -> None:
    theta0 = 0.35
    optima = {"A": -1.0, "B": 2.0}
    alpha = 0.6

    ab = scalar_recency_endpoint(theta0, optima, "AB", alpha=alpha)
    ba = scalar_recency_endpoint(theta0, optima, "BA", alpha=alpha)

    assert ab != pytest.approx(ba)
    expected_difference = (1.0 - alpha) ** 2 * (optima["B"] - optima["A"])
    assert ab - ba == pytest.approx(expected_difference)


def test_shared_optimum_and_commuting_maps_are_exactly_order_independent() -> None:
    optimum = np.array([0.4, -0.7])
    theta0 = np.array([-0.1, 0.9])
    stage_a = quadratic_gd_stage(
        np.diag([1.0, 2.0]), optimum, learning_rate=0.04, steps=3
    )
    stage_b = quadratic_gd_stage(
        np.diag([0.5, 1.5]), optimum, learning_rate=0.06, steps=7
    )
    stages = {"A": stage_a, "B": stage_b}

    np.testing.assert_allclose(
        chronology_endpoint(theta0, stages, "AB"),
        chronology_endpoint(theta0, stages, "BA"),
        rtol=0.0,
        atol=1e-14,
    )


def test_common_continuation_transports_only_the_history_difference() -> None:
    continuation = quadratic_gd_stage(
        np.diag([0.5, 1.0]),
        np.array([4.0, -3.0]),
        learning_rate=0.2,
        steps=1,
    )
    left = np.array([1.0, 2.0])
    right = np.array([-0.5, 0.25])
    repeats = 6

    propagated_difference = common_continuation_difference(
        left - right, continuation, repeats=repeats
    )
    left_endpoint = left.copy()
    right_endpoint = right.copy()
    for _ in range(repeats):
        left_endpoint = continuation.apply(left_endpoint)
        right_endpoint = continuation.apply(right_endpoint)

    np.testing.assert_allclose(
        propagated_difference,
        left_endpoint - right_endpoint,
        rtol=1e-13,
        atol=1e-13,
    )


def test_spectral_half_life_matches_geometric_decay() -> None:
    magnitude = 0.9
    half_life = spectral_half_life(magnitude)
    assert magnitude**half_life == pytest.approx(0.5)
    assert spectral_half_life(1.0) == math.inf
    assert spectral_half_life(0.0) == 0.0

    with pytest.raises(ValueError):
        spectral_half_life(1.01)
