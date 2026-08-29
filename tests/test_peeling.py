from itertools import combinations, permutations

import numpy as np

from chronotrace.geometry.peeling import (
    decode_last_stage_from_predecessor_sets,
    invert_gradient_step_armijo,
    invert_gradient_step_fixed_point,
)


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
        matrix = np.eye(dimension) - learning_rate * hessian
        bias = -learning_rate * offset
        maps[stage] = (matrix, bias)
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


def test_fixed_point_inversion_recovers_quadratic_gradient_step() -> None:
    stages, learning_rate, base, maps, gradients = _affine_problem()
    before = _endpoint(("A", "C", "B"), maps, base)
    after = _endpoint(("A", "C", "B", "D"), maps, base)

    result = invert_gradient_step_fixed_point(
        after,
        gradients["D"],
        learning_rate=learning_rate,
    )

    assert result.converged
    assert result.iterations <= 30
    assert np.linalg.norm(result.state - before) < 1e-10


def test_armijo_inverse_succeeds_when_picard_is_noncontractive_but_map_is_invertible() -> None:
    hessian = np.diag([-2.0, 0.5])
    offset = np.array([0.3, -0.2])
    learning_rate = 0.75
    before = np.array([0.8, -0.4])

    def loss(state):
        return 0.5 * state @ hessian @ state + offset @ state

    def gradient(state):
        return hessian @ state + offset

    target = before - learning_rate * gradient(before)
    picard = invert_gradient_step_fixed_point(
        target,
        gradient,
        learning_rate=learning_rate,
        tolerance=1e-10,
        max_iterations=40,
    )
    armijo = invert_gradient_step_armijo(
        target,
        loss,
        gradient,
        learning_rate=learning_rate,
        tolerance=1e-10,
        max_iterations=200,
    )

    assert not picard.converged
    assert armijo.converged
    assert not armijo.line_search_failed
    assert np.linalg.norm(armijo.state - before) < 1e-9
    assert all(
        later <= earlier + 1e-14
        for earlier, later in zip(
            armijo.objective_trace,
            armijo.objective_trace[1:],
            strict=True,
        )
    )
    assert min(armijo.accepted_step_trace) < 1.0


def test_reverse_peeling_rescues_histories_missed_by_degree_three_forward_decode() -> None:
    stages, learning_rate, base, maps, gradients = _affine_problem()
    basis = _interaction_basis(stages, maps, base, max_degree=3)
    histories = tuple(permutations(stages))
    forward_correct = 0
    peeled_correct = 0

    for history in histories:
        target = _endpoint(history, maps, base)
        forward = min(
            histories,
            key=lambda candidate: np.linalg.norm(
                target - _prediction(candidate, basis, base, max_degree=3)
            ),
        )
        forward_correct += int(forward == history)

        inverted = {}
        predecessors = {}
        predecessor_words = {}
        for candidate_last in stages:
            result = invert_gradient_step_fixed_point(
                target,
                gradients[candidate_last],
                learning_rate=learning_rate,
            )
            assert result.converged
            inverted[candidate_last] = result.state
            remaining = tuple(stage for stage in stages if stage != candidate_last)
            words = tuple(permutations(remaining))
            predecessor_words[candidate_last] = words
            predecessors[candidate_last] = tuple(
                _prediction(word, basis, base, max_degree=3) for word in words
            )

        decision = decode_last_stage_from_predecessor_sets(inverted, predecessors)
        assert decision.identifiable
        assert decision.stage is not None
        assert decision.predecessor_index is not None
        predecessor = predecessor_words[decision.stage][decision.predecessor_index]
        peeled = predecessor + (decision.stage,)
        peeled_correct += int(peeled == history)

    assert forward_correct == 18
    assert peeled_correct == 24


def test_peeling_reports_exact_collision_as_nonidentifiable() -> None:
    inverted = {"A": np.array([1.0, 2.0]), "B": np.array([1.0, 2.0])}
    predecessors = {
        "A": (np.array([1.0, 2.0]),),
        "B": (np.array([1.0, 2.0]),),
    }

    decision = decode_last_stage_from_predecessor_sets(inverted, predecessors)

    assert not decision.identifiable
    assert decision.stage is None
    assert decision.margin == 0.0
