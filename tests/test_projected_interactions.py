from itertools import combinations, permutations

import numpy as np

from chronotrace.geometry.projected_interactions import (
    projected_interaction_from_endpoint_delta,
    projected_word_prediction,
)


def _subsequences(word: tuple[str, ...]) -> list[tuple[str, ...]]:
    result = []
    for size in range(1, len(word) + 1):
        for indices in combinations(range(len(word)), size):
            result.append(tuple(word[index] for index in indices))
    return result


def test_projection_commutes_with_ordered_mobius_inversion() -> None:
    rng = np.random.default_rng(7)
    stages = tuple("ABCD")
    direction = rng.normal(size=5)
    interactions = {
        word: rng.normal(size=5)
        for size in range(1, 5)
        for word in permutations(stages, size)
    }
    projected = {
        word: float(direction @ value)
        for word, value in interactions.items()
        if len(word) <= 3
    }

    word = ("B", "D", "A", "C")
    endpoint_delta = sum(interactions[subsequence] for subsequence in _subsequences(word))
    endpoint_projection = float(direction @ endpoint_delta)
    recovered = projected_interaction_from_endpoint_delta(
        word,
        endpoint_projection,
        projected,
    )
    expected = float(direction @ interactions[word])
    assert abs(recovered - expected) < 1e-10

    full_projected = dict(projected)
    full_projected[word] = recovered
    prediction = projected_word_prediction(word, full_projected, max_degree=4)
    assert abs(prediction - endpoint_projection) < 1e-10
