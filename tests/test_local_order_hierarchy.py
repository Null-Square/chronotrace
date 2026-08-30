from itertools import permutations

import numpy as np

from chronotrace.geometry.local_order_hierarchy import (
    build_last_stage_equalities,
    build_local_order_equalities,
    build_local_order_hierarchy,
    evaluate_linear_objective,
    full_level_permutation_scores,
    local_order_coordinate_count,
    local_order_vertex,
    projected_interaction_linear_objective,
    validate_local_order_weights,
)


def test_coordinate_count_is_polynomial_for_fixed_degree() -> None:
    hierarchy = build_local_order_hierarchy(tuple("ABCDEF"), max_degree=4)
    assert hierarchy.dimension == local_order_coordinate_count(6, 4)
    assert hierarchy.dimension == 6 + 30 + 120 + 360


def test_every_global_permutation_is_feasible_for_k4_local_relaxation() -> None:
    hierarchy = build_local_order_hierarchy(tuple("ABCDE"), max_degree=4)
    equalities = build_local_order_equalities(hierarchy)
    for history in permutations(hierarchy.stages):
        vertex = local_order_vertex(history, hierarchy)
        validate_local_order_weights(vertex, hierarchy, equalities=equalities)


def test_last_stage_equalities_accept_exactly_matching_global_vertices() -> None:
    hierarchy = build_local_order_hierarchy(tuple("ABCD"), max_degree=4)
    equalities = build_local_order_equalities(hierarchy)
    last = build_last_stage_equalities(hierarchy, "C")
    for history in permutations(hierarchy.stages):
        vertex = local_order_vertex(history, hierarchy)
        validate_local_order_weights(vertex, hierarchy, equalities=equalities)
        residual = last.matrix @ vertex - last.rhs
        matches = float(np.max(np.abs(residual))) < 1e-12
        assert matches == (history[-1] == "C")


def test_projected_interaction_objective_matches_direct_subsequence_sum() -> None:
    hierarchy = build_local_order_hierarchy(tuple("ABCD"), max_degree=3)
    projected = {
        word: float(index + 1) / 17.0
        for index, word in enumerate(hierarchy.coordinate_words)
    }
    target_minus_base = 2.75
    constant, coefficients = projected_interaction_linear_objective(
        hierarchy,
        projected,
        target_minus_base_projection=target_minus_base,
    )
    history = ("D", "A", "C", "B")
    vertex = local_order_vertex(history, hierarchy)
    observed = evaluate_linear_objective(constant, coefficients, vertex)

    position = {stage: index for index, stage in enumerate(history)}
    direct = target_minus_base
    for subset in hierarchy.subsets:
        order = tuple(sorted(subset, key=position.__getitem__))
        direct -= projected[order]
    assert abs(observed - direct) < 1e-12


def test_k_equals_n_objective_is_exact_convexification_of_complete_permutations() -> None:
    hierarchy = build_local_order_hierarchy(tuple("ABCD"), max_degree=4)
    rng = np.random.default_rng(123)
    projected = {
        word: float(rng.normal())
        for word in hierarchy.coordinate_words
    }
    constant, coefficients = projected_interaction_linear_objective(
        hierarchy,
        projected,
        target_minus_base_projection=0.4,
    )
    scores = full_level_permutation_scores(
        hierarchy,
        constant,
        coefficients,
        last_stage="D",
    )
    assert len(scores) == 6

    histories = tuple(scores)
    mixture_weights = rng.random(len(histories))
    mixture_weights /= mixture_weights.sum()
    mixture = sum(
        weight * local_order_vertex(history, hierarchy)
        for weight, history in zip(mixture_weights, histories, strict=True)
    )
    validate_local_order_weights(mixture, hierarchy)
    last = build_last_stage_equalities(hierarchy, "D")
    np.testing.assert_allclose(last.matrix @ mixture, last.rhs, atol=1e-12)

    mixed_objective = evaluate_linear_objective(constant, coefficients, mixture)
    expected = sum(
        weight * scores[history]
        for weight, history in zip(mixture_weights, histories, strict=True)
    )
    assert abs(mixed_objective - expected) < 1e-12
    assert mixed_objective >= min(scores.values()) - 1e-12
