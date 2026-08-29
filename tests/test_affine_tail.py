from itertools import combinations, permutations

import numpy as np

from chronotrace.geometry.affine_tail import (
    affine_chronology_tail_bound,
    affine_ordered_interaction_closed_form,
    affine_uniform_interaction_bound,
    affine_uniform_truncation_tail_bound,
)


def _affine_system(seed=4417):
    rng = np.random.default_rng(seed)
    stages = tuple("ABCD")
    dimension = 4
    base = rng.normal(size=dimension)
    increment_matrices = {}
    biases = {}
    maps = {}
    for stage in stages:
        raw = rng.normal(size=(dimension, dimension))
        increment = 0.08 * raw / np.linalg.norm(raw, 2)
        bias = rng.normal(scale=0.05, size=dimension)
        increment_matrices[stage] = increment
        biases[stage] = bias
        maps[stage] = (np.eye(dimension) + increment, bias)
    singleton_displacements = {
        stage: increment_matrices[stage] @ base + biases[stage] for stage in stages
    }
    return stages, base, maps, increment_matrices, singleton_displacements


def _endpoint(word, maps, base):
    state = base.copy()
    for stage in word:
        matrix, bias = maps[stage]
        state = matrix @ state + bias
    return state


def _mobius_interactions(stages, maps, base):
    interactions = {}
    for degree in range(1, len(stages) + 1):
        for word in permutations(stages, degree):
            value = _endpoint(word, maps, base) - base
            for lower_degree in range(1, degree):
                for indices in combinations(range(degree), lower_degree):
                    value -= interactions[tuple(word[index] for index in indices)]
            interactions[word] = value
    return interactions


def _truncated_prediction(word, base, interactions, max_degree):
    state = base.copy()
    for degree in range(1, max_degree + 1):
        for indices in combinations(range(len(word)), degree):
            state += interactions[tuple(word[index] for index in indices)]
    return state


def test_affine_ordered_interaction_has_exact_product_closed_form() -> None:
    stages, base, maps, increments, singletons = _affine_system()
    measured = _mobius_interactions(stages, maps, base)

    for degree in range(1, len(stages) + 1):
        for word in permutations(stages, degree):
            closed = affine_ordered_interaction_closed_form(word, increments, singletons)
            np.testing.assert_allclose(closed, measured[word], rtol=0.0, atol=1e-12)


def test_uniform_tail_bound_contains_every_affine_chronology_error() -> None:
    stages, base, maps, increments, singletons = _affine_system()
    interactions = _mobius_interactions(stages, maps, base)
    max_singleton = max(np.linalg.norm(value) for value in singletons.values())
    max_increment = max(np.linalg.norm(value, 2) for value in increments.values())

    for max_degree in (1, 2, 3):
        bound = affine_uniform_truncation_tail_bound(
            stage_count=len(stages),
            max_degree=max_degree,
            max_singleton_norm=max_singleton,
            max_increment_operator_norm=max_increment,
        )
        for chronology in permutations(stages):
            actual = _endpoint(chronology, maps, base)
            predicted = _truncated_prediction(
                chronology,
                base,
                interactions,
                max_degree,
            )
            assert np.linalg.norm(actual - predicted) <= bound + 1e-12


def test_stage_specific_tail_bound_is_valid_and_no_looser_than_uniform_bound() -> None:
    stages, base, maps, increments, singletons = _affine_system()
    interactions = _mobius_interactions(stages, maps, base)
    singleton_bounds = {
        stage: float(np.linalg.norm(value)) for stage, value in singletons.items()
    }
    increment_bounds = {
        stage: float(np.linalg.norm(value, 2)) for stage, value in increments.items()
    }
    uniform = affine_uniform_truncation_tail_bound(
        stage_count=4,
        max_degree=2,
        max_singleton_norm=max(singleton_bounds.values()),
        max_increment_operator_norm=max(increment_bounds.values()),
    )

    for chronology in permutations(stages):
        bound = affine_chronology_tail_bound(
            chronology,
            max_degree=2,
            singleton_norm_bounds=singleton_bounds,
            increment_operator_norm_bounds=increment_bounds,
        )
        actual = _endpoint(chronology, maps, base)
        predicted = _truncated_prediction(chronology, base, interactions, 2)
        assert np.linalg.norm(actual - predicted) <= bound + 1e-12
        assert bound <= uniform + 1e-12


def test_single_interaction_bound_has_geometric_degree_scaling() -> None:
    degree_two = affine_uniform_interaction_bound(
        degree=2,
        max_singleton_norm=2.0,
        max_increment_operator_norm=0.1,
    )
    degree_three = affine_uniform_interaction_bound(
        degree=3,
        max_singleton_norm=2.0,
        max_increment_operator_norm=0.1,
    )
    degree_four = affine_uniform_interaction_bound(
        degree=4,
        max_singleton_norm=2.0,
        max_increment_operator_norm=0.1,
    )

    assert degree_two == 0.2
    assert degree_three == 0.02
    assert degree_four == 0.002
