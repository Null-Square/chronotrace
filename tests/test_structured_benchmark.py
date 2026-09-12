"""Independent checks of the nonlinear remainder and oracle accounting."""
import importlib.util
import sys
from fractions import Fraction
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest

from chronotrace.validated_affine import (
    AffineBox,
    Box,
    decode_budgeted,
    identify_from_short_probes,
    subset_enclosures,
)

PATH = Path(__file__).resolve().parents[1] / "scripts" / "structured_affine_benchmark.py"
SPEC = importlib.util.spec_from_file_location("structured_affine_benchmark_under_test", PATH)
BENCH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCH
SPEC.loader.exec_module(BENCH)


def exact_micro_history(matrices, offsets, gamma, steps, order, start):
    value = [Fraction(float(v)) for v in start]
    for j in order:
        for _ in range(steps):
            value = [
                sum(Fraction(float(a)) * v for a, v in zip(row, value, strict=True))
                + Fraction(float(offsets[j][r])) - Fraction(float(gamma)) * value[r] ** 3
                for r, row in enumerate(matrices[j])
            ]
    return value


def assert_fraction_contained(box, exact):
    for lo, val, hi in zip(box.lo, exact, box.hi, strict=True):
        assert Fraction(float(lo)) <= val <= Fraction(float(hi))


@pytest.mark.parametrize("seed", range(6))
def test_nonlinear_context_bounds_against_exact_rational_trajectories(seed):
    rng = np.random.default_rng(seed)
    n, d, steps = 3, 2, 2
    matrices = [np.eye(d) * (7 / 8) + rng.normal(size=(d, d)) / 128 for _ in range(n)]
    offsets = [rng.normal(size=d) / 32 for _ in range(n)]
    gamma = 1 / 256
    oracle = BENCH.QuarticOracle(matrices, offsets, steps, gamma)
    zero = np.zeros(d)
    for length in range(n):
        for prefix in permutations(range(n), length):
            state = exact_micro_history(matrices, offsets, gamma, steps, prefix, zero)
            mask = sum(1 << j for j in prefix)
            assert max(abs(v) for v in state) <= Fraction(float(oracle.prefix_radii[mask]))
            for j in range(n):
                if j in prefix:
                    continue
                # Preserve exact incoming fractions; the helper converts starts,
                # so perform this short continuation independently right here.
                nonlinear, linear = list(state), list(state)
                for _ in range(steps):
                    nonlinear = [sum(Fraction(float(a)) * v for a, v in
                                     zip(row, nonlinear, strict=True))
                                 + Fraction(float(offsets[j][r]))
                                 - Fraction(gamma) * nonlinear[r] ** 3
                                 for r, row in enumerate(matrices[j])]
                    linear = [sum(Fraction(float(a)) * v for a, v in
                                  zip(row, linear, strict=True))
                              + Fraction(float(offsets[j][r]))
                              for r, row in enumerate(matrices[j])]
                error = max(abs(a - b) for a, b in zip(nonlinear, linear, strict=True))
                assert error <= Fraction(float(oracle.context_remainders[j, mask]))
    for order in permutations(range(n)):
        measurement = oracle.target(order, zero)
        exact = exact_micro_history(matrices, offsets, gamma, steps, order, zero)
        assert_fraction_contained(measurement.enclosure, exact)


def test_nonlinear_pipeline_uses_only_short_probes_plus_declared_bounds():
    rng = np.random.default_rng(181)
    n, d = 4, 2
    matrices = [np.eye(d) * .96 + rng.normal(size=(d, d)) * .001 for _ in range(n)]
    offsets = [rng.normal(size=d) * .01 for _ in range(n)]
    oracle = BENCH.QuarticOracle(matrices, offsets, 2, 1e-4)
    base = np.zeros(d)
    single = [oracle.measure(j, base) for j in range(n)]
    pairs = {(i, j): oracle.measure(j, single[i].value)
             for i in range(n) for j in range(n) if i != j}
    singleton_boxes = [m.enclosure.inflate(oracle.reference_error(j, 0.))
                       for j, m in enumerate(single)]
    pair_boxes = {key: m.enclosure.inflate(oracle.reference_error(
        key[1], float(np.max(np.abs(single[key[0]].value))))) for key, m in pairs.items()}
    fit = identify_from_short_probes(base, [m.value for m in single], singleton_boxes, pair_boxes)
    assert oracle.calls == n * n
    assert oracle.updates == n * n * 2
    cached = subset_enclosures(fit.maps, remainder_bounds=oracle.context_remainders)
    for order in permutations(range(n)):
        target = oracle.target(order, base)
        calls_before = oracle.calls
        result = decode_budgeted(fit.maps, target.enclosure, max_bound_evaluations=1000,
                                 remainder_bounds=oracle.context_remainders, precomputed=cached)
        assert oracle.calls == calls_before
        assert any(not node.suffix or order[-len(node.suffix):] == node.suffix
                   for node in result.frontier)
        assert all(order.index(a) < order.index(b) for a, b in result.precedences)


def test_changing_context_bounds_invalidates_cache():
    maps = [AffineBox(Box.point(np.eye(1)), Box.point(np.ones(1))) for _ in range(3)]
    cache = subset_enclosures(maps)
    with pytest.raises(ValueError, match="different stage maps"):
        decode_budgeted(maps, Box.point(np.ones(1)), max_bound_evaluations=100,
                        precomputed=cache, remainder_bounds=np.ones((3, 8)))


@pytest.mark.parametrize("radius", (-1., float("nan"), float("inf")))
def test_invalid_context_bound_rejected(radius):
    maps = [AffineBox(Box.point(np.eye(1)), Box.point(np.ones(1))) for _ in range(3)]
    with pytest.raises(ValueError):
        subset_enclosures(maps, remainder_bounds=np.full((3, 8), radius))


def test_overlapping_endpoint_candidates_cannot_produce_a_unique_certificate():
    maps = [AffineBox(Box.point(np.eye(1) * .5), Box.point(np.array([float(j)])))
            for j in range(4)]
    first, second = (0, 1, 2, 3), (1, 0, 2, 3)
    values = []
    for order in (first, second):
        point = Box.point(np.zeros(1))
        for j in order:
            point = maps[j].apply(point)
        values.append(point)
    target = Box(np.minimum(values[0].lo, values[1].lo), np.maximum(values[0].hi, values[1].hi))
    result = decode_budgeted(maps, target, max_bound_evaluations=1000)
    assert result.unique_order is None
    assert result.unexcluded_histories >= 2
    assert (0, 1) not in result.precedences and (1, 0) not in result.precedences


def test_more_budget_never_adds_candidate_histories():
    rng = np.random.default_rng(39)
    maps = [AffineBox(Box.point(np.eye(2) * .7 + rng.normal(size=(2, 2)) * .1),
                      Box.point(rng.normal(size=2))) for _ in range(5)]
    target = Box.point(np.zeros(2))
    for j in (2, 4, 1, 0, 3):
        target = maps[j].apply(target)
    previous_count, previous_edges = 121, set()
    for budget in (0, 1, 6, 32, 64, 128, 512):
        result = decode_budgeted(maps, target, max_bound_evaluations=budget)
        assert result.unexcluded_histories <= previous_count
        assert previous_edges <= set(result.precedences)
        previous_count, previous_edges = result.unexcluded_histories, set(result.precedences)
