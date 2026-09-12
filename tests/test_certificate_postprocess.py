"""Exhaustive graph checks, independent of model training and frozen labels."""

from itertools import combinations, permutations, product

import pytest

from chronotrace.geometry.certificate_postprocess import (
    close_certified_precedences,
    decide_from_class_lower_bounds,
)

STAGES = ("A", "B", "C", "D")
PAIRS = tuple(combinations(STAGES, 2))


@pytest.mark.parametrize("choices", tuple(product((0, 1, -1), repeat=6)))
def test_all_four_stage_partial_graphs(choices):
    edges = tuple(
        pair if sign == 1 else tuple(reversed(pair))
        for pair, sign in zip(PAIRS, choices, strict=True) if sign
    )
    actual = close_certified_precedences(STAGES, edges)
    possible = [
        order for order in permutations(STAGES)
        if all(order.index(a) < order.index(b) for a, b in edges)
    ]
    if not possible:
        assert actual.status == "invalid_cycle"
        assert actual.unique_order is None
        assert not actual.entailed_relations
        return
    entailed = {
        (a, b) for a in STAGES for b in STAGES if a != b
        and all(order.index(a) < order.index(b) for order in possible)
    }
    assert set(actual.entailed_relations) == entailed
    assert actual.unique_order == (possible[0] if len(possible) == 1 else None)
    assert actual.status == ("unique" if len(possible) == 1 else "partial")
    for edge, path in zip(actual.entailed_relations, actual.proof_paths, strict=True):
        assert (path[0], path[-1]) == edge
        assert all((a, b) in edges for a, b in zip(path, path[1:], strict=False))


def test_frozen_abstention_recovered_without_history_label():
    result = close_certified_precedences(
        STAGES, (("C", "A"), ("D", "A"), ("B", "C"), ("B", "D"), ("C", "D"))
    )
    assert result.unique_order == ("B", "C", "D", "A")
    assert result.derived_relations == (("B", "A"),)


def test_multichar_names_and_duplicate_edges():
    result = close_certified_precedences(
        ("base", "domain", "instruction"),
        (("base", "domain"), ("base", "domain"), ("domain", "instruction")),
    )
    assert result.unique_order == ("base", "domain", "instruction")
    assert len(result.direct_relations) == 2


def test_single_stage():
    assert close_certified_precedences(("only",), ()).unique_order == ("only",)


def test_cycle_with_no_direct_opposite_edge_is_invalid():
    result = close_certified_precedences(STAGES, (("A", "B"), ("B", "C"), ("C", "A")))
    assert result.status == "invalid_cycle"
    assert result.proof_paths == ()


@pytest.mark.parametrize("stages, edges", [
    ((), ()), (("A", "A"), ()), (("",), ()), ((1,), ()),
    (STAGES, (("A", "A"),)), (STAGES, (("A", "X"),)),
    (STAGES, (("A",),)), (STAGES, ("AB",)), (STAGES, ((None, "A"),)),
])
def test_bad_graph_inputs(stages, edges):
    with pytest.raises(ValueError):
        close_certified_precedences(stages, edges)


@pytest.mark.parametrize("forward, reverse, status, inferred", [
    (0.0, 1.0, "certified", ("A", "B")),
    (1.0, 0.0, "certified", ("B", "A")),
    (0.0, 0.0, "ambiguous", None),
    (1.0, 1.0, "invalid_both_excluded", None),
    (-1.0, -1.0, "ambiguous", None),
    (0.0, 0.1, "ambiguous", None),
])
def test_two_sided_decisions(forward, reverse, status, inferred):
    result = decide_from_class_lower_bounds(
        "A", "B", forward, reverse, residual_budget=0.1
    )
    assert (result.status, result.inferred_precedence) == (status, inferred)


@pytest.mark.parametrize("field", ("forward", "reverse", "budget", "guard"))
@pytest.mark.parametrize("bad", (float("nan"), float("inf"), -float("inf")))
def test_nonfinite_bounds_rejected(field, bad):
    values = {"forward": 0.0, "reverse": 1.0, "budget": 0.0, "guard": 0.0}
    values[field] = bad
    with pytest.raises(ValueError):
        decide_from_class_lower_bounds(
            "A", "B", values["forward"], values["reverse"],
            residual_budget=values["budget"], numerical_guard=values["guard"],
        )


def test_numerical_guard_is_subtracted():
    result = decide_from_class_lower_bounds(
        "A", "B", 0.0, 0.1001, residual_budget=0.1, numerical_guard=0.001
    )
    assert result.status == "ambiguous"


def test_negative_budgets_rejected():
    with pytest.raises(ValueError):
        decide_from_class_lower_bounds("A", "B", 0.0, 1.0, residual_budget=-1.0)
    with pytest.raises(ValueError):
        decide_from_class_lower_bounds(
            "A", "B", 0.0, 1.0, residual_budget=0.0, numerical_guard=-1.0
        )
