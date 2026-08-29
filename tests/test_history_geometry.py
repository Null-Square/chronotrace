import pytest

from chronotrace.geometry.history import (
    directional_contamination_ratio,
    finite_pair_commutator,
    kendall_tau_for_orders,
    pairwise_precedence_accuracy,
    prefix_conditioned_commutator_diagnostic,
)
from chronotrace.geometry.secant import (
    finite_pair_interactions,
    finite_pair_predicted_endpoint,
    finite_pair_signature,
    finite_pair_symmetric_reference,
)

torch = pytest.importorskip("torch")


def _stage_system():
    scale = 0.05

    def stage_a(theta):
        x0, x1, x2 = theta
        return theta + scale * torch.stack((x0**2 + x1, x0 * x2, -0.5 * x1**2))

    def stage_b(theta):
        x0, x1, x2 = theta
        return theta + scale * torch.stack((-x1 * x2, x1**2 + 0.2 * x0, x0 * x1))

    def stage_c(theta):
        x0, x1, x2 = theta
        return theta + scale * torch.stack((0.3 * x2**2, -x0 * x2, x2**2 + x1))

    theta0 = torch.tensor([0.3, -0.2, 0.4], dtype=torch.float64)
    return theta0, {"A": stage_a, "B": stage_b, "C": stage_c}


def _run(theta0, maps, history):
    theta = theta0
    for stage in history:
        theta = maps[stage](theta)
    return theta


def test_prefix_conditioned_commutator_matches_exact_tail_swap() -> None:
    theta0, maps = _stage_system()
    _, interactions = finite_pair_interactions(maps, theta0)
    endpoints = {
        tuple(history): _run(theta0, maps, history)
        for history in ("ABC", "ACB")
    }

    diagnostic = prefix_conditioned_commutator_diagnostic(
        endpoints,
        interactions,
        prefix=("A",),
        first="B",
        second="C",
    )

    base = finite_pair_commutator(interactions, first="B", second="C")
    conditioned = endpoints[("A", "B", "C")] - endpoints[("A", "C", "B")]

    assert diagnostic.base_norm == pytest.approx(float(torch.linalg.vector_norm(base)))
    assert diagnostic.conditioned_norm == pytest.approx(
        float(torch.linalg.vector_norm(conditioned))
    )
    assert diagnostic.drift_norm > 0.0


def test_triple_residual_difference_equals_prefix_commutator_drift() -> None:
    theta0, maps = _stage_system()
    stages = ("A", "B", "C")
    deltas, interactions = finite_pair_interactions(maps, theta0)
    reference = finite_pair_symmetric_reference(
        theta0,
        deltas,
        interactions,
        stages=stages,
    )

    endpoint_abc = _run(theta0, maps, "ABC")
    endpoint_acb = _run(theta0, maps, "ACB")
    residual_abc = endpoint_abc - finite_pair_predicted_endpoint(
        "ABC",
        reference,
        interactions,
        stages=stages,
    )
    residual_acb = endpoint_acb - finite_pair_predicted_endpoint(
        "ACB",
        reference,
        interactions,
        stages=stages,
    )

    base_commutator = finite_pair_commutator(interactions, first="B", second="C")
    conditioned_commutator = endpoint_abc - endpoint_acb
    drift = conditioned_commutator - base_commutator

    torch.testing.assert_close(residual_abc - residual_acb, drift, rtol=0.0, atol=1e-12)


def test_directional_contamination_is_exact_nearest_signature_boundary() -> None:
    theta0, maps = _stage_system()
    stages = ("A", "B", "C")
    deltas, interactions = finite_pair_interactions(maps, theta0)
    reference = finite_pair_symmetric_reference(
        theta0,
        deltas,
        interactions,
        stages=stages,
    )
    endpoint = _run(theta0, maps, "ABC")
    true_signature = finite_pair_signature("ABC", interactions, stages=stages)
    alternative_signature = finite_pair_signature("ACB", interactions, stages=stages)
    predicted = finite_pair_predicted_endpoint(
        "ABC",
        reference,
        interactions,
        stages=stages,
    )
    residual = endpoint - predicted
    ratio = directional_contamination_ratio(
        residual,
        true_signature,
        alternative_signature,
    )

    true_error = torch.linalg.vector_norm(endpoint - (reference + true_signature))
    alternative_error = torch.linalg.vector_norm(endpoint - (reference + alternative_signature))
    assert (ratio < 1.0) == bool(true_error < alternative_error)


def test_partial_order_metrics_match_adjacent_tail_swap() -> None:
    assert pairwise_precedence_accuracy("ABC", "ACB") == pytest.approx(2.0 / 3.0)
    assert kendall_tau_for_orders("ABC", "ACB") == pytest.approx(1.0 / 3.0)
    assert pairwise_precedence_accuracy("BAC", "BAC") == 1.0
    assert kendall_tau_for_orders("BAC", "BAC") == 1.0
