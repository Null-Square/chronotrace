from itertools import permutations

import numpy as np

from chronotrace.geometry.order_affine import (
    build_k3_local_coordinate_layout,
    candidate_last_affine_lower_bounds,
    encode_k3_local_permutation,
    equality_constrained_quadratic_projection,
    k3_local_equalities,
)


def test_k3_local_layout_has_polynomial_size() -> None:
    layout5 = build_k3_local_coordinate_layout(tuple("ABCDE"))
    layout6 = build_k3_local_coordinate_layout(tuple("ABCDEF"))

    assert layout5.dimension == 10 + 6 * 10
    assert layout6.dimension == 15 + 6 * 20
    base5, rhs5 = k3_local_equalities(layout5)
    last5, last_rhs5 = k3_local_equalities(layout5, last_stage="E")
    assert base5.shape == (4 * 10, 70)
    assert rhs5.shape == (40,)
    assert last5.shape == (44, 70)
    assert last_rhs5.shape == (44,)


def test_every_permutation_satisfies_local_equalities_and_correct_last_stage() -> None:
    layout = build_k3_local_coordinate_layout(tuple("ABCDE"))
    base, base_rhs = k3_local_equalities(layout)

    for chronology in permutations(layout.stages):
        vector = encode_k3_local_permutation(chronology, layout)
        np.testing.assert_allclose(base @ vector, base_rhs, rtol=0.0, atol=1e-12)
        for stage in layout.stages:
            matrix, rhs = k3_local_equalities(layout, last_stage=stage)
            residual = float(np.linalg.norm(matrix @ vector - rhs))
            if chronology[-1] == stage:
                assert residual < 1e-12
            else:
                assert residual >= 1.0


def test_equality_projection_matches_direct_affine_projection() -> None:
    rng = np.random.default_rng(901)
    layout = build_k3_local_coordinate_layout(tuple("ABCD"))
    feature = rng.normal(size=(80, layout.dimension))
    target = rng.normal(size=80)
    gram = feature.T @ feature
    cross = feature.T @ target
    matrix, rhs = k3_local_equalities(layout, last_stage="D")

    result = equality_constrained_quadratic_projection(
        gram,
        cross,
        float(target @ target),
        matrix,
        rhs,
    )

    direct = float(np.linalg.norm(target - feature @ result.solution))
    assert abs(result.distance - direct) < 1e-10
    assert result.equality_residual_norm < 1e-9
    assert result.stationarity_residual_norm < 1e-8


def test_candidate_last_affine_distance_is_conservative_discrete_lower_bound() -> None:
    layout = build_k3_local_coordinate_layout(tuple("ABCDE"))
    # Identity features make parameter prediction equal to the local-order vector itself.
    # This keeps the theorem test exact while still giving a 70-dimensional N=5 problem.
    gram = np.eye(layout.dimension, dtype=np.float64)
    true_history = tuple("BDAEC")
    target = encode_k3_local_permutation(true_history, layout)
    bounds = candidate_last_affine_lower_bounds(
        gram,
        target,
        float(target @ target),
        layout,
    )
    numerical_floor = 1e-6

    for stage in layout.stages:
        discrete = min(
            float(
                np.linalg.norm(
                    target - encode_k3_local_permutation(history, layout)
                )
            )
            for history in permutations(layout.stages)
            if history[-1] == stage
        )
        assert bounds[stage].distance <= discrete + numerical_floor
        assert bounds[stage].equality_residual_norm < 1e-9
        assert bounds[stage].stationarity_residual_norm < 1e-8

    assert bounds["C"].distance < numerical_floor
    assert min(bounds[stage].distance for stage in layout.stages if stage != "C") > 0.1
