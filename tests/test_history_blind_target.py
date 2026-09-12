"""Reject target-history side channels while retaining invalid measurement jobs."""
import importlib
import inspect
from pathlib import Path

import numpy as np
import pytest

from chronotrace.validated_affine import Box


@pytest.fixture
def benchmark(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("structured_affine_benchmark_v2")


def test_observation_interface_cannot_receive_hidden_history(benchmark):
    assert tuple(inspect.signature(benchmark.observation_box).parameters) == (
        "observed", "export_radius", "numerical_radius"
    )
    observed = np.array([0.25, -0.5])
    actual = benchmark.observation_box(observed, 1e-4, 1e-10)
    expected = Box.point(observed).inflate(float(np.nextafter(1e-4 + 1e-10, np.inf)))
    np.testing.assert_array_equal(actual.lo, expected.lo)
    np.testing.assert_array_equal(actual.hi, expected.hi)


@pytest.mark.parametrize("bad", (-1.0, float("nan"), float("inf")))
@pytest.mark.parametrize("field", ("export_radius", "numerical_radius"))
def test_invalid_public_budget_rejected(benchmark, bad, field):
    budgets = {"export_radius": 0.0, "numerical_radius": 1e-10}
    budgets[field] = bad
    with pytest.raises(ValueError):
        benchmark.observation_box(np.zeros(2), **budgets)


def test_oracle_error_interval_cannot_change_predictions(benchmark, monkeypatch):
    parameters = dict(family="synthetic_quadratic", seed=41, n=4, d=2, steps=16,
                      budgets=[16, 128], targets_per_codebook=3)
    normal = benchmark.run_codebook(**parameters)
    original_builder = benchmark.build_problem

    def inflated_builder(*args, **kwargs):
        oracle, metadata = original_builder(*args, **kwargs)
        original_target = oracle.target

        def inflated_target(order, base):
            measured = original_target(order, base)
            measured.enclosure = measured.enclosure.inflate(100.0)
            return measured

        oracle.target = inflated_target
        return oracle, metadata

    monkeypatch.setattr(benchmark, "build_problem", inflated_builder)
    altered = benchmark.run_codebook(**parameters)
    assert not normal["invalid_codebook"]
    assert altered["invalid_codebook"]
    assert all(not row["measurement_budget_valid"] for row in altered["records"])
    for left, right in zip(normal["records"], altered["records"], strict=True):
        for field in ("history", "target_value", "target_lo", "target_hi",
                      "pair_point_order", "affine_point_order"):
            assert left[field] == right[field]
        for a, b in zip(left["predictions"], right["predictions"], strict=True):
            a = {key: value for key, value in a.items() if key != "decode_seconds"}
            b = {key: value for key, value in b.items() if key != "decode_seconds"}
            assert a == b
