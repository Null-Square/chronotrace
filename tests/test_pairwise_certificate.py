import numpy as np

from chronotrace.geometry.local_order_hierarchy import (
    build_local_order_hierarchy,
    local_order_vertex,
)
from chronotrace.geometry.pairwise_certificate import certify_pairwise_orientation


def test_pairwise_certificate_infers_surviving_orientation_without_label() -> None:
    hierarchy = build_local_order_hierarchy(("A", "B", "C"), max_degree=3)
    target = local_order_vertex(("A", "B", "C"), hierarchy)
    coefficients = np.zeros((2, hierarchy.dimension), dtype=np.float64)
    coefficients[0] = np.linspace(-0.4, 0.5, hierarchy.dimension)
    coefficients[1] = np.linspace(0.3, -0.6, hierarchy.dimension)
    constants = -(coefficients @ target)

    result = certify_pairwise_orientation(
        hierarchy,
        constants,
        coefficients,
        "A",
        "B",
        elimination_guard=1e-9,
        certificate_guard=1e-12,
    )

    assert result.status in {"certified", "ambiguous"}
    assert result.left_before_right_impossible is False
    if result.status == "certified":
        assert result.inferred_precedence == ("A", "B")
    else:
        assert result.inferred_precedence is None


def test_pairwise_certificate_certifies_known_separable_synthetic_case() -> None:
    hierarchy = build_local_order_hierarchy(("A", "B", "C"), max_degree=3)
    abc = hierarchy.coordinate_index[("A", "B", "C")]
    bac = hierarchy.coordinate_index[("B", "A", "C")]
    constants = np.zeros(2, dtype=np.float64)
    coefficients = np.zeros((2, hierarchy.dimension), dtype=np.float64)
    coefficients[0, abc] = 0.0
    coefficients[1, abc] = 0.0
    coefficients[0, bac] = 1.0
    coefficients[1, bac] = 1.0

    result = certify_pairwise_orientation(
        hierarchy,
        constants,
        coefficients,
        "A",
        "B",
        elimination_guard=1e-6,
        certificate_guard=1e-12,
    )

    assert result.status == "certified"
    assert result.inferred_precedence == ("A", "B")
    assert result.left_before_right_impossible is False
    assert result.right_before_left_impossible is True


def test_pairwise_certificate_abstains_when_both_classes_touch_target() -> None:
    hierarchy = build_local_order_hierarchy(("A", "B", "C"), max_degree=3)
    constants = np.zeros(2, dtype=np.float64)
    coefficients = np.zeros((2, hierarchy.dimension), dtype=np.float64)

    result = certify_pairwise_orientation(
        hierarchy,
        constants,
        coefficients,
        "A",
        "B",
        elimination_guard=1e-6,
        certificate_guard=1e-12,
    )

    assert result.status == "ambiguous"
    assert result.inferred_precedence is None
    assert result.left_before_right_impossible is False
    assert result.right_before_left_impossible is False
