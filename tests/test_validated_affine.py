"""Independent exact-rational checks for interval arithmetic and chronology."""
from fractions import Fraction
from itertools import permutations

import numpy as np
import pytest

from chronotrace.validated_affine import (
    AffineBox,
    Box,
    cube,
    decode_budgeted,
    identify_from_short_probes,
    infinity_norm_upper,
    matmul,
    subset_enclosures,
)


def fraction_endpoint(matrices, offsets, order, start):
    value = [Fraction(float(x)) for x in start]
    for j in order:
        value = [
            sum(Fraction(float(a)) * v for a, v in zip(row, value, strict=True))
            + Fraction(float(offsets[j][r]))
            for r, row in enumerate(matrices[j])
        ]
    return value


def enclose(values):
    midpoint = np.asarray([float(v) for v in values])
    return Box(np.nextafter(midpoint, -np.inf), np.nextafter(midpoint, np.inf))


@pytest.mark.parametrize("seed", range(80))
def test_point_matmul_against_exact_rational_arithmetic(seed):
    rng = np.random.default_rng(seed)
    d = 1 + seed % 5
    left = rng.normal(size=(d, d)) * (10.0 ** ((seed % 7) - 3))
    right = rng.normal(size=(d, d))
    product = matmul(Box.point(left), Box.point(right))
    for i in range(d):
        for j in range(d):
            exact = sum(Fraction(float(left[i, k])) * Fraction(float(right[k, j]))
                        for k in range(d))
            assert Fraction(float(product.lo[i, j])) <= exact
            assert exact <= Fraction(float(product.hi[i, j]))


@pytest.mark.parametrize("seed", range(30))
def test_interval_matmul_contains_exact_extrema(seed):
    rng = np.random.default_rng(seed)
    alo = rng.normal(size=(2, 3))
    ahi = alo + rng.random(size=(2, 3))
    blo = rng.normal(size=(3, 2))
    bhi = blo + rng.random(size=(3, 2))
    actual = matmul(Box(alo, ahi), Box(blo, bhi))
    for i in range(2):
        for j in range(2):
            exact_lo = exact_hi = Fraction(0)
            for k in range(3):
                candidates = [Fraction(float(a)) * Fraction(float(b))
                              for a in (alo[i, k], ahi[i, k])
                              for b in (blo[k, j], bhi[k, j])]
                exact_lo += min(candidates)
                exact_hi += max(candidates)
            assert Fraction(float(actual.lo[i, j])) <= exact_lo
            assert Fraction(float(actual.hi[i, j])) >= exact_hi


@pytest.mark.parametrize("seed", range(12))
def test_identification_encloses_true_maps_from_probe_outputs_only(seed):
    rng = np.random.default_rng(seed)
    n, d = 5, 3
    matrices = [np.eye(d) * 0.7 + rng.normal(size=(d, d)) * 0.05 for _ in range(n)]
    offsets = [rng.normal(size=d) for _ in range(n)]
    base = rng.normal(size=d)
    singletons = [enclose(fraction_endpoint(matrices, offsets, [i], base)) for i in range(n)]
    starts = [value.mid for value in singletons]
    pairs = {(i, j): enclose(fraction_endpoint(matrices, offsets, [j], starts[i]))
             for i in range(n) for j in range(n) if i != j}
    fit = identify_from_short_probes(base, starts, singletons, pairs)
    assert fit.stage_calls == n * n
    for j, identified in enumerate(fit.maps):
        assert identified.matrix.contains(matrices[j])
        centered_offset = [v - Fraction(float(x)) for v, x in zip(
            fraction_endpoint(matrices, offsets, [j], base), base, strict=True)]
        for low, exact, high in zip(identified.offset.lo, centered_offset,
                                    identified.offset.hi, strict=True):
            assert Fraction(float(low)) <= exact <= Fraction(float(high))


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("budget", (0, 1, 5, 17, 1000))
def test_all_four_stage_histories_never_falsely_pruned(seed, budget):
    rng = np.random.default_rng(seed)
    n, d = 4, 2
    matrices = [0.6 * np.eye(d) + rng.normal(size=(d, d)) * 0.07 for _ in range(n)]
    offsets = [rng.normal(size=d) for _ in range(n)]
    maps = tuple(AffineBox(Box.point(a), Box.point(b))
                 for a, b in zip(matrices, offsets, strict=True))
    cached = subset_enclosures(maps)
    for history in permutations(range(n)):
        exact = fraction_endpoint(matrices, offsets, history, np.zeros(d))
        result = decode_budgeted(maps, enclose(exact),
                                 max_bound_evaluations=budget, precomputed=cached)
        assert result.bound_evaluations <= budget
        assert result.status != "inconsistent"
        assert any((not node.suffix or history[-len(node.suffix):] == node.suffix)
                   for node in result.frontier)
        assert all(history.index(i) < history.index(j) for i, j in result.precedences)
        if result.unique_order is not None:
            assert result.unique_order == history
        if budget == 1000:
            assert result.unique_order == history


def test_identical_maps_preserve_all_histories():
    maps = tuple(AffineBox(Box.point(np.eye(2) * 0.5), Box.point(np.ones(2)))
                 for _ in range(4))
    result = decode_budgeted(maps, Box.point(np.ones(2) * 1.875),
                             max_bound_evaluations=1000)
    assert result.status == "ambiguous"
    assert result.unexcluded_histories == 24
    assert not result.precedences
    assert result.unique_order is None


def test_export_uncertainty_forces_abstention():
    maps = tuple(AffineBox(Box.point(np.eye(2) * 0.5), Box.point(np.ones(2) * j))
                 for j in range(4))
    result = decode_budgeted(maps, Box(np.full(2, -100.), np.full(2, 100.)),
                             max_bound_evaluations=1000)
    assert result.status == "ambiguous"
    assert result.unexcluded_histories == 24
    assert not result.precedences


def test_out_of_model_target_is_rejected():
    maps = tuple(AffineBox(Box.point(np.eye(2)), Box.point(np.ones(2))) for _ in range(3))
    result = decode_budgeted(maps, Box.point(np.full(2, 100.)), max_bound_evaluations=100)
    assert result.status == "inconsistent"
    assert result.unexcluded_histories == 0
    assert result.unique_order is None
    assert result.precedences == ()


def test_rank_failure_is_not_silently_regularized():
    base = np.zeros(3)
    starts = [np.eye(3)[j] for j in range(3)]
    singleton = [Box.point(v) for v in starts]
    pairs = {(i, j): Box.point(starts[i] + starts[j])
             for i in range(3) for j in range(3) if i != j}
    with pytest.raises(ValueError, match="spanning"):
        identify_from_short_probes(base, starts, singleton, pairs)


def test_subsets_contain_every_exact_rational_endpoint():
    rng = np.random.default_rng(19)
    matrices = [rng.normal(size=(2, 2)) for _ in range(4)]
    offsets = [rng.normal(size=2) for _ in range(4)]
    maps = tuple(AffineBox(Box.point(a), Box.point(b))
                 for a, b in zip(matrices, offsets, strict=True))
    cached = subset_enclosures(maps)
    boxes = cached.boxes
    assert cached.applications == 4 * 2 ** 3
    for length in range(1, 5):
        for order in permutations(range(4), length):
            mask = sum(1 << j for j in order)
            exact = fraction_endpoint(matrices, offsets, order, np.zeros(2))
            for low, value, high in zip(boxes[mask].lo, exact, boxes[mask].hi, strict=True):
                assert Fraction(float(low)) <= value <= Fraction(float(high))


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), -float("inf")))
def test_nonfinite_interval_rejected(bad):
    with pytest.raises(ValueError):
        Box.point(np.array([bad]))


@pytest.mark.parametrize("bad", (-1, 0.5, True))
def test_bad_budget_rejected(bad):
    stage = AffineBox(Box.point(np.eye(1)), Box.point(np.ones(1)))
    with pytest.raises(ValueError):
        decode_budgeted([stage], Box.point(np.ones(1)), max_bound_evaluations=bad)


def test_mutation_cannot_change_input_bounds():
    original = np.zeros(2)
    box = Box.point(original)
    original[:] = 4
    assert box.contains(np.zeros(2))
    with pytest.raises(ValueError):
        box.lo[:] = -100


def test_interval_underflow_and_cancellation():
    tiniest = np.nextafter(0.0, np.inf)
    result = matmul(Box.point(np.array([[tiniest, 1., -1.]])),
                    Box.point(np.array([0.5, 1., 1.])))
    exact = Fraction(float(tiniest)) / 2
    assert Fraction(float(result.lo[0])) <= exact <= Fraction(float(result.hi[0]))


def test_infinity_norm_is_upper_bound():
    matrix = Box.point(np.array([[1., -2., 3.], [1e-10, 2., -4.]]))
    assert infinity_norm_upper(matrix) >= 6.0000000001


def test_cache_cannot_be_reused_for_different_models():
    first = AffineBox(Box.point(np.eye(1)), Box.point(np.ones(1)))
    second = AffineBox(Box.point(np.eye(1)), Box.point(np.zeros(1)))
    cached = subset_enclosures([first])
    with pytest.raises(ValueError, match="different stage maps"):
        decode_budgeted([second], Box.point(np.ones(1)),
                        max_bound_evaluations=10, precomputed=cached)


@pytest.mark.parametrize("left,right", ((-3., 2.), (-1., -0.1), (0., 1.), (1e-100, 2e-100)))
def test_cube_bounds_are_exact_rational_enclosures(left, right):
    result = cube(Box(np.array([left]), np.array([right])))
    assert Fraction(float(result.lo[0])) <= Fraction(left) ** 3
    assert Fraction(float(result.hi[0])) >= Fraction(right) ** 3
