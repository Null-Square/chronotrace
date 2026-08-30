from itertools import combinations, permutations

import numpy as np

from chronotrace.geometry.convex_quadratic import (
    dual_quadratic_hull_certificate,
    project_quadratic_simplex,
)
from chronotrace.geometry.local_order_hierarchy import (
    build_local_order_hierarchy,
    full_level_permutation_scores,
    projected_interaction_linear_objective,
)
from chronotrace.geometry.local_order_lp import solve_local_order_lp
from chronotrace.geometry.projected_interactions import (
    projected_interaction_from_endpoint_delta,
    projected_word_prediction,
)


def _subsequences(word: tuple[str, ...], max_degree: int) -> list[tuple[str, ...]]:
    result: list[tuple[str, ...]] = []
    for size in range(1, min(max_degree, len(word)) + 1):
        for indices in combinations(range(len(word)), size):
            result.append(tuple(word[index] for index in indices))
    return result


def test_frozen_k3_witness_projected_k4_matches_terminal_exact_lp() -> None:
    rng = np.random.default_rng(0)
    stages = tuple("ABCD")
    dimension = 9
    interactions: dict[tuple[str, ...], np.ndarray] = {}
    for degree in range(1, 5):
        scale = 0.5 if degree <= 3 else 0.01
        for word in permutations(stages, degree):
            interactions[word] = rng.normal(scale=scale, size=dimension)

    def endpoint(word: tuple[str, ...], max_degree: int) -> np.ndarray:
        value = np.zeros(dimension, dtype=np.float64)
        for subsequence in _subsequences(word, max_degree):
            value = value + interactions[subsequence]
        return value

    target_history = tuple("ABCD")
    target = endpoint(target_history, 4)
    hierarchy = build_local_order_hierarchy(stages, max_degree=4)

    for final_stage in ("C", "D"):
        histories = tuple(history for history in permutations(stages) if history[-1] == final_stage)
        k3_candidates = np.stack([endpoint(history, 3) for history in histories])
        displacements = k3_candidates - target[None, :]
        gram = displacements @ displacements.T
        projection = project_quadratic_simplex(
            gram,
            np.zeros(len(histories), dtype=np.float64),
            0.0,
        )
        k3_certificate = dual_quadratic_hull_certificate(
            gram,
            np.zeros(len(histories), dtype=np.float64),
            0.0,
            projection,
        )
        mixed_displacement = projection.weights @ displacements
        witness = -mixed_displacement / np.linalg.norm(mixed_displacement)
        np.testing.assert_allclose(np.linalg.norm(witness), 1.0, atol=1e-12)
        assert k3_certificate.lower_bound <= projection.distance + 1e-10

        projected = {
            word: float(witness @ value)
            for word, value in interactions.items()
            if len(word) <= 3
        }
        endpoint_projection: dict[tuple[str, ...], float] = {}
        for word in permutations(stages):
            exact_endpoint = endpoint(word, 4)
            exact_projection = float(witness @ exact_endpoint)
            interaction4 = projected_interaction_from_endpoint_delta(
                word,
                exact_projection,
                projected,
            )
            projected[word] = interaction4
            endpoint_projection[word] = exact_projection
            reconstructed = projected_word_prediction(word, projected, max_degree=4)
            assert abs(reconstructed - exact_projection) < 1e-10

        target_projection = float(witness @ target)
        constant, coefficients = projected_interaction_linear_objective(
            hierarchy,
            projected,
            target_minus_base_projection=target_projection,
        )
        lp = solve_local_order_lp(
            hierarchy,
            constant,
            coefficients,
            last_stage=final_stage,
            certificate_guard=1e-12,
        )
        hierarchy_scores = full_level_permutation_scores(
            hierarchy,
            constant,
            coefficients,
            last_stage=final_stage,
        )
        direct_scores = {
            history: target_projection - endpoint_projection[history]
            for history in histories
        }
        exact_directional_minimum = min(direct_scores.values())
        exact_euclidean_distance = min(
            float(np.linalg.norm(target - endpoint(history, 4))) for history in histories
        )

        assert max(
            abs(hierarchy_scores[history] - direct_scores[history]) for history in histories
        ) < 1e-10
        assert abs(lp.primal_objective - exact_directional_minimum) < 1e-9
        assert lp.certified_lower_bound <= exact_directional_minimum + 1e-9
        assert exact_directional_minimum - lp.certified_lower_bound < 1e-7
        assert max(0.0, lp.certified_lower_bound) <= exact_euclidean_distance + 1e-9

        if final_stage == "C":
            assert lp.certified_lower_bound > 1.0
        else:
            assert exact_euclidean_distance < 1e-12
            assert max(0.0, lp.certified_lower_bound) < 1e-9
