import pytest

from chronotrace.geometry.interactions import measure_ordered_interaction_basis_compact
from chronotrace.geometry.partial import (
    decode_ordered_interaction_precedence,
    decode_ordered_interaction_prefix,
)

torch = pytest.importorskip("torch")


def _system():
    scale = 0.8

    def stage_a(theta):
        x0, x1, x2 = theta
        return theta + scale * torch.stack(
            (x0**2 + 0.2 * x1, 0.4 * x0 * x2, -0.3 * x1**2)
        )

    def stage_b(theta):
        x0, x1, x2 = theta
        return theta + scale * torch.stack(
            (-0.2 * x1 * x2, x1**2 + 0.1 * x0, 0.5 * x0 * x1)
        )

    def stage_c(theta):
        x0, x1, x2 = theta
        return theta + scale * torch.stack(
            (0.1 * x2**2, -0.4 * x0 * x2, x2**2 + 0.2 * x1)
        )

    def stage_d(theta):
        x0, x1, x2 = theta
        return theta + scale * torch.stack(
            (-0.3 * x0 * x1 + 0.1 * x2, 0.2 * x2**2 + 0.15 * x0, x0**2 - 0.25 * x1 * x2)
        )

    theta0 = torch.tensor([0.3, -0.2, 0.4], dtype=torch.float64)
    maps = {"A": stage_a, "B": stage_b, "C": stage_c, "D": stage_d}
    return theta0, maps


def _run(theta0, stage_maps, history):
    endpoint = theta0
    for stage in history:
        endpoint = stage_maps[stage](endpoint)
    return endpoint


def test_prefix_group_decode_recovers_supported_prefix_without_extra_probes() -> None:
    theta0, stage_maps = _system()
    basis = measure_ordered_interaction_basis_compact(stage_maps, theta0, max_degree=3)
    endpoint = _run(theta0, stage_maps, "ABCD")

    depth1 = decode_ordered_interaction_prefix(endpoint, basis, depth=1, degree=3)
    depth2 = decode_ordered_interaction_prefix(endpoint, basis, depth=2, degree=3)
    depth3 = decode_ordered_interaction_prefix(endpoint, basis, depth=3, degree=3)

    assert depth1.prefix == ("A",)
    assert depth2.prefix == ("A", "B")
    assert depth3.prefix == ("A", "B", "C")
    assert min(depth1.margin, depth2.margin, depth3.margin) > 0.0
    assert basis.stage_executions == 40


def test_pairwise_precedence_decode_marginalizes_other_positions() -> None:
    theta0, stage_maps = _system()
    basis = measure_ordered_interaction_basis_compact(stage_maps, theta0, max_degree=3)
    endpoint = _run(theta0, stage_maps, "CADB")

    for first, second, expected in (
        ("C", "A", ("C", "A")),
        ("A", "D", ("A", "D")),
        ("D", "B", ("D", "B")),
        ("C", "B", ("C", "B")),
    ):
        decoded = decode_ordered_interaction_precedence(
            endpoint,
            basis,
            first=first,
            second=second,
            degree=3,
        )
        assert (decoded.preferred_first, decoded.preferred_second) == expected
        assert decoded.margin > 0.0
