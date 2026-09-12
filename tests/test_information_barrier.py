from itertools import permutations

import numpy as np

from chronotrace.geometry.information_barrier import (
    build_polynomial_gradient_perturbation,
)


def test_polynomial_perturbation_is_invisible_on_queries_and_arbitrary_at_hidden() -> None:
    queries = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [1.0, 2.0]])
    hidden = np.array([3.0, -1.0])
    desired = np.array([7.0, -5.0])
    perturbation = build_polynomial_gradient_perturbation(queries, hidden, desired)

    for query in queries:
        np.testing.assert_allclose(perturbation.gradient(query), np.zeros(2), atol=1e-12)
    np.testing.assert_allclose(perturbation.gradient(hidden), desired, atol=1e-12)


def test_degree_three_observations_do_not_determine_unqueried_degree_four_transition() -> None:
    # A/B/C are translations. D is also a translation in the base system. The perturbed D
    # stage is identical at every state where a degree<=3 probe can invoke D, but differs
    # at the exact ABC predecessor state used only by the degree-4 word ABCD.
    shifts = {
        "A": np.array([1.0, 0.0]),
        "B": np.array([0.0, 2.0]),
        "C": np.array([4.0, 0.0]),
        "D": np.array([0.0, -3.0]),
    }
    base = np.zeros(2)

    def translation(stage: str, state: np.ndarray) -> np.ndarray:
        return state + shifts[stage]

    d_query_states = [base]
    for size in (1, 2):
        for prefix in permutations(("A", "B", "C"), size):
            state = base.copy()
            for stage in prefix:
                state = translation(stage, state)
            d_query_states.append(state)
    hidden = shifts["A"] + shifts["B"] + shifts["C"]
    perturbation = build_polynomial_gradient_perturbation(
        np.stack(d_query_states),
        hidden,
        np.array([10.0, -6.0]),
    )
    eta = 0.25

    def run(word: tuple[str, ...], *, perturbed: bool) -> np.ndarray:
        state = base.copy()
        for stage in word:
            if stage != "D" or not perturbed:
                state = translation(stage, state)
            else:
                state = state + shifts["D"] - eta * perturbation.gradient(state)
        return state

    for size in (1, 2, 3):
        for word in permutations(tuple(shifts), size):
            np.testing.assert_allclose(run(word, perturbed=False), run(word, perturbed=True))

    baseline = run(("A", "B", "C", "D"), perturbed=False)
    changed = run(("A", "B", "C", "D"), perturbed=True)
    np.testing.assert_allclose(changed - baseline, -eta * np.array([10.0, -6.0]))

    direction = np.array([1.0, 0.0])
    assert float(direction @ (changed - baseline)) == -2.5
