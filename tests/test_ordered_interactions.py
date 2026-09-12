from itertools import permutations

import pytest

from chronotrace.geometry.interactions import (
    decode_ordered_interaction_permutation,
    measure_ordered_interaction_basis,
    measure_ordered_interaction_basis_compact,
    ordered_interaction_identifiability,
    ordered_interaction_prediction,
    ordered_interaction_word_prediction,
    ordered_probe_count,
    ordered_subsequences,
)
from chronotrace.geometry.secant import (
    finite_pair_interactions,
    finite_pair_predicted_endpoint,
    finite_pair_symmetric_reference,
)

torch = pytest.importorskip("torch")


def _three_stage_system():
    scale = 0.03

    def stage_a(theta):
        x0, x1, x2 = theta
        effect = torch.stack((x0**2 + 0.2 * x1, 0.4 * x0 * x2, -0.3 * x1**2))
        return theta + scale * effect

    def stage_b(theta):
        x0, x1, x2 = theta
        effect = torch.stack((-0.2 * x1 * x2, x1**2 + 0.1 * x0, 0.5 * x0 * x1))
        return theta + scale * effect

    def stage_c(theta):
        x0, x1, x2 = theta
        effect = torch.stack((0.1 * x2**2, -0.4 * x0 * x2, x2**2 + 0.2 * x1))
        return theta + scale * effect

    theta0 = torch.tensor([0.3, -0.2, 0.4], dtype=torch.float64)
    return theta0, {"A": stage_a, "B": stage_b, "C": stage_c}


def _four_stage_system():
    scale = 0.8

    def stage_a(theta):
        x0, x1, x2 = theta
        effect = torch.stack((x0**2 + 0.2 * x1, 0.4 * x0 * x2, -0.3 * x1**2))
        return theta + scale * effect

    def stage_b(theta):
        x0, x1, x2 = theta
        effect = torch.stack((-0.2 * x1 * x2, x1**2 + 0.1 * x0, 0.5 * x0 * x1))
        return theta + scale * effect

    def stage_c(theta):
        x0, x1, x2 = theta
        effect = torch.stack((0.1 * x2**2, -0.4 * x0 * x2, x2**2 + 0.2 * x1))
        return theta + scale * effect

    def stage_d(theta):
        x0, x1, x2 = theta
        effect = torch.stack(
            (-0.3 * x0 * x1 + 0.1 * x2, 0.2 * x2**2 + 0.15 * x0, x0**2 - 0.25 * x1 * x2)
        )
        return theta + scale * effect

    theta0 = torch.tensor([0.3, -0.2, 0.4], dtype=torch.float64)
    return theta0, {"A": stage_a, "B": stage_b, "C": stage_c, "D": stage_d}


def _run(theta0, stage_maps, history):
    endpoint = theta0
    for stage in history:
        endpoint = stage_maps[stage](endpoint)
    return endpoint


def test_probe_count_matches_cached_prefix_complexity() -> None:
    assert ordered_probe_count(3, 2) == 9
    assert ordered_probe_count(4, 3) == 40
    assert ordered_probe_count(5, 3) == 85


def test_degree_two_basis_matches_existing_finite_pair_geometry() -> None:
    theta0, stage_maps = _three_stage_system()
    stages = tuple(stage_maps)
    basis = measure_ordered_interaction_basis(stage_maps, theta0, max_degree=2)
    deltas, pair_interactions = finite_pair_interactions(stage_maps, theta0)
    reference = finite_pair_symmetric_reference(
        theta0,
        deltas,
        pair_interactions,
        stages=stages,
    )

    assert basis.stage_executions == len(stages) ** 2
    for stage in stages:
        torch.testing.assert_close(
            basis.interactions[(stage,)],
            deltas[stage],
            rtol=0.0,
            atol=1e-12,
        )
    for source in stages:
        for destination in stages:
            if source == destination:
                continue
            torch.testing.assert_close(
                basis.interactions[(source, destination)],
                pair_interactions[(destination, source)],
                rtol=0.0,
                atol=1e-12,
            )

    for history in permutations(stages):
        generic = ordered_interaction_prediction(history, basis, degree=2)
        existing = finite_pair_predicted_endpoint(
            history,
            reference,
            pair_interactions,
            stages=stages,
        )
        torch.testing.assert_close(generic, existing, rtol=0.0, atol=1e-12)


def test_measured_words_reconstruct_exactly_from_mobius_interactions() -> None:
    theta0, stage_maps = _four_stage_system()
    basis = measure_ordered_interaction_basis(stage_maps, theta0, max_degree=3)

    for word, observed in basis.endpoints.items():
        reconstructed = theta0.clone()
        for subsequence in ordered_subsequences(word):
            reconstructed = reconstructed + basis.interactions[subsequence]
        torch.testing.assert_close(reconstructed, observed, rtol=0.0, atol=1e-11)


def test_compact_measurement_matches_cached_basis_without_storing_endpoints() -> None:
    theta0, stage_maps = _four_stage_system()
    cached = measure_ordered_interaction_basis(stage_maps, theta0, max_degree=3)
    compact = measure_ordered_interaction_basis_compact(stage_maps, theta0, max_degree=3)

    assert compact.endpoints == {}
    assert compact.stage_executions == cached.stage_executions == 40
    assert set(compact.interactions) == set(cached.interactions)
    for word, interaction in cached.interactions.items():
        torch.testing.assert_close(
            compact.interactions[word],
            interaction,
            rtol=0.0,
            atol=1e-11,
        )
    for word, observed in cached.endpoints.items():
        reconstructed = ordered_interaction_word_prediction(word, compact)
        torch.testing.assert_close(reconstructed, observed, rtol=0.0, atol=1e-11)


def test_degree_three_resolves_a_four_stage_degree_two_failure_regime() -> None:
    theta0, stage_maps = _four_stage_system()
    stages = tuple(stage_maps)
    basis = measure_ordered_interaction_basis_compact(stage_maps, theta0, max_degree=3)
    degree_two_correct = 0
    degree_three_correct = 0

    assert not ordered_interaction_identifiability(basis, degree=1).identifiable
    assert ordered_interaction_identifiability(basis, degree=2).identifiable
    assert ordered_interaction_identifiability(basis, degree=3).identifiable

    for history in permutations(stages):
        endpoint = _run(theta0, stage_maps, history)
        degree_two = decode_ordered_interaction_permutation(endpoint, basis, degree=2)
        degree_three = decode_ordered_interaction_permutation(endpoint, basis, degree=3)
        degree_two_correct += int(degree_two.permutation == history)
        degree_three_correct += int(degree_three.permutation == history)

    total = len(tuple(permutations(stages)))
    assert degree_two_correct < total
    assert degree_three_correct == total
