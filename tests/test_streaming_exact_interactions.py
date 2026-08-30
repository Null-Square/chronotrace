from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chronotrace.geometry.interactions import (
    measure_ordered_interaction_basis,
    measure_ordered_interaction_basis_streaming_exact,
    ordered_probe_count,
)


@dataclass(frozen=True)
class _Vector:
    values: np.ndarray

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    def clone(self) -> "_Vector":
        return _Vector(self.values.copy())

    def __add__(self, other: "_Vector") -> "_Vector":
        return _Vector(self.values + other.values)

    def __sub__(self, other: "_Vector") -> "_Vector":
        return _Vector(self.values - other.values)


def _stage_map(scale: float, shift: float):
    def run(vector: _Vector) -> _Vector:
        return _Vector(np.tanh(scale * vector.values + shift))

    return run


def test_streaming_exact_matches_full_direct_prefix_measurement() -> None:
    base = _Vector(np.asarray([0.25, -0.4, 0.7], dtype=np.float64))
    stage_maps = {
        "A": _stage_map(1.01, 0.02),
        "B": _stage_map(0.97, -0.03),
        "C": _stage_map(1.04, 0.01),
        "D": _stage_map(0.99, 0.04),
    }
    full = measure_ordered_interaction_basis(stage_maps, base, max_degree=3)
    observed: dict[tuple[str, ...], _Vector] = {}

    def observe(word: tuple[str, ...], endpoint: _Vector) -> None:
        if len(word) == 3:
            observed[word] = endpoint.clone()

    streaming = measure_ordered_interaction_basis_streaming_exact(
        stage_maps,
        base,
        max_degree=3,
        endpoint_observer=observe,
    )

    assert streaming.endpoints == {}
    assert streaming.stage_executions == ordered_probe_count(4, 3) == 40
    assert set(streaming.interactions) == set(full.interactions)
    for word, interaction in streaming.interactions.items():
        assert np.array_equal(interaction.values, full.interactions[word].values)

    expected_degree_three = {word for word in full.endpoints if len(word) == 3}
    assert set(observed) == expected_degree_three
    for word, endpoint in observed.items():
        assert np.array_equal(endpoint.values, full.endpoints[word].values)
