from itertools import permutations

import pytest

from chronotrace.geometry.secant import (
    decode_finite_pair_permutation,
    finite_pair_identifiability,
    finite_pair_interactions,
    finite_pair_predicted_endpoint,
    finite_pair_symmetric_reference,
    higher_order_remainder_ratio,
)

torch = pytest.importorskip("torch")


@pytest.fixture
def stage_system():
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


def _run(theta0, stage_maps, history):
    theta = theta0
    for stage in history:
        theta = stage_maps[stage](theta)
    return theta


def test_interaction_table_uses_exactly_n_squared_stage_calls(stage_system) -> None:
    theta0, raw_maps = stage_system
    calls = {stage: 0 for stage in raw_maps}

    def counted(stage):
        def run(theta):
            calls[stage] += 1
            return raw_maps[stage](theta)

        return run

    maps = {stage: counted(stage) for stage in raw_maps}
    deltas, interactions = finite_pair_interactions(maps, theta0)

    assert sum(calls.values()) == len(maps) ** 2
    assert set(deltas) == set(maps)
    assert len(interactions) == len(maps) * (len(maps) - 1)


def test_two_stage_pairwise_prediction_is_exact(stage_system) -> None:
    theta0, all_maps = stage_system
    maps = {stage: all_maps[stage] for stage in ("A", "B")}
    stages = ("A", "B")
    deltas, interactions = finite_pair_interactions(maps, theta0)
    reference = finite_pair_symmetric_reference(
        theta0,
        deltas,
        interactions,
        stages=stages,
    )

    for history in ("AB", "BA"):
        observed = _run(theta0, maps, history)
        predicted = finite_pair_predicted_endpoint(
            history,
            reference,
            interactions,
            stages=stages,
        )
        torch.testing.assert_close(predicted, observed, rtol=0.0, atol=1e-12)


def test_three_stage_finite_pair_decoder_recovers_all_orders(stage_system) -> None:
    theta0, maps = stage_system
    stages = ("A", "B", "C")
    deltas, interactions = finite_pair_interactions(maps, theta0)
    reference = finite_pair_symmetric_reference(
        theta0,
        deltas,
        interactions,
        stages=stages,
    )
    identifiability = finite_pair_identifiability(interactions, stages=stages)
    assert identifiability.identifiable
    assert identifiability.minimum_signature_separation > 0

    for history in permutations(stages):
        observed = _run(theta0, maps, history)
        prediction = finite_pair_predicted_endpoint(
            history,
            reference,
            interactions,
            stages=stages,
        )
        decoded = decode_finite_pair_permutation(
            observed,
            reference,
            interactions,
            stages=stages,
        )
        ratio = higher_order_remainder_ratio(
            observed,
            prediction,
            minimum_signature_separation=identifiability.minimum_signature_separation,
        )
        assert decoded.permutation == history
        assert decoded.margin > 0
        assert ratio < 1.0
