from itertools import combinations, permutations

import numpy as np

from chronotrace.geometry.peeling import invert_gradient_step_fixed_point


def _affine_problem(seed: int = 2708):
    rng = np.random.default_rng(seed)
    stages = tuple("ABCD")
    dimension = 3
    learning_rate = 0.4
    base = rng.normal(size=dimension)
    maps = {}
    gradients = {}
    for stage in stages:
        raw = rng.normal(size=(dimension, dimension))
        hessian = 0.5 * (raw + raw.T)
        hessian *= 0.9 / np.linalg.norm(hessian, 2)
        offset = rng.normal(scale=0.6, size=dimension)
        maps[stage] = (
            np.eye(dimension) - learning_rate * hessian,
            -learning_rate * offset,
        )
        gradients[stage] = lambda x, h=hessian, c=offset: h @ x + c
    return stages, learning_rate, base, maps, gradients


def _endpoint(word, maps, base):
    state = base.copy()
    for stage in word:
        matrix, bias = maps[stage]
        state = matrix @ state + bias
    return state


def _interaction_basis(stages, maps, base, max_degree=3):
    basis = {}
    for degree in range(1, max_degree + 1):
        for word in permutations(stages, degree):
            value = _endpoint(word, maps, base) - base
            for lower_degree in range(1, degree):
                for indices in combinations(range(degree), lower_degree):
                    value -= basis[tuple(word[index] for index in indices)]
            basis[word] = value
    return basis


def _prediction(word, basis, base, max_degree=3):
    state = base.copy()
    for degree in range(1, min(max_degree, len(word)) + 1):
        for indices in combinations(range(len(word)), degree):
            state += basis[tuple(word[index] for index in indices)]
    return state


def _distance_to_convex_hull(point, vertices, tolerance=1e-12):
    """Exact small-dimensional active-set projection used only as a theorem test."""

    matrix = np.stack(vertices, axis=1)
    dimension, count = matrix.shape
    best = float("inf")
    for size in range(1, min(dimension + 1, count) + 1):
        for active in combinations(range(count), size):
            selected = matrix[:, active]
            gram = selected.T @ selected
            ones = np.ones((size, 1))
            system = np.block([[gram, ones], [ones.T, np.zeros((1, 1))]])
            rhs = np.concatenate((selected.T @ point, np.array([1.0])))
            solution = np.linalg.lstsq(system, rhs, rcond=None)[0]
            weights = solution[:size]
            if np.min(weights) < -tolerance:
                continue
            projected = selected @ weights
            best = min(best, float(np.linalg.norm(point - projected)))
    if not np.isfinite(best):
        raise RuntimeError("convex-hull projection found no feasible active set")
    return best


def test_convex_predecessor_relaxation_certifies_23_and_preserves_one_ambiguity() -> None:
    stages, learning_rate, base, maps, gradients = _affine_problem()
    basis = _interaction_basis(stages, maps, base, max_degree=3)
    histories = tuple(permutations(stages))
    direct_correct = 0
    uniquely_certified_correct = 0
    ambiguous_histories = []
    false_unique_histories = []
    zero_tolerance = 1e-10

    for history in histories:
        target = _endpoint(history, maps, base)
        direct = min(
            histories,
            key=lambda candidate: np.linalg.norm(
                target - _prediction(candidate, basis, base, max_degree=3)
            ),
        )
        direct_correct += int(direct == history)

        relaxed_scores = {}
        for candidate_last in stages:
            inverted = invert_gradient_step_fixed_point(
                target,
                gradients[candidate_last],
                learning_rate=learning_rate,
            )
            assert inverted.converged
            remaining = tuple(stage for stage in stages if stage != candidate_last)
            predecessor_vertices = tuple(
                _prediction(word, basis, base, max_degree=3)
                for word in permutations(remaining)
            )
            relaxed_scores[candidate_last] = _distance_to_convex_hull(
                inverted.state,
                predecessor_vertices,
            )

        plausible = tuple(
            stage for stage in stages if relaxed_scores[stage] <= zero_tolerance
        )
        if plausible == (history[-1],):
            uniquely_certified_correct += 1
        elif history[-1] in plausible:
            ambiguous_histories.append((history, plausible))
        else:
            false_unique_histories.append((history, plausible))

    assert direct_correct == 18
    assert uniquely_certified_correct == 23
    assert ambiguous_histories == [(('D', 'B', 'A', 'C'), ('B', 'C'))]
    assert false_unique_histories == []
