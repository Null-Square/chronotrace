import numpy as np
import pytest

from chronotrace.geometry.commutator_probe import (
    finite_commutator_vector,
    normalized_projection,
    projected_finite_commutator,
)


def test_finite_commutator_is_exactly_antisymmetric() -> None:
    def first(state: np.ndarray) -> np.ndarray:
        x, y = state
        return np.array([x + y, y + 1.0])

    def second(state: np.ndarray) -> np.ndarray:
        x, y = state
        return np.array([x * (1.0 + y), y])

    state = np.array([0.4, -0.2])
    ab = finite_commutator_vector(state, first, second)
    ba = finite_commutator_vector(state, second, first)

    assert np.allclose(ab, -ba, rtol=0.0, atol=0.0)


def test_euler_commutator_recovers_quadratic_lie_bracket_scaling() -> None:
    # f=(1,0), g=(0,x).  Dg f - Df g=(0,1).
    state = np.array([0.3, -0.7])

    def make_first(step: float):
        def first(value: np.ndarray) -> np.ndarray:
            return value + step * np.array([1.0, 0.0])

        return first

    def make_second(step: float):
        def second(value: np.ndarray) -> np.ndarray:
            return value + step * np.array([0.0, value[0]])

        return second

    for step in (1e-2, 3e-3, 1e-3, 3e-4):
        commutator = finite_commutator_vector(
            state,
            make_first(step),
            make_second(step),
        )
        assert commutator[0] == pytest.approx(0.0, abs=1e-15)
        assert commutator[1] / (step**2) == pytest.approx(1.0, rel=1e-10, abs=1e-10)


def test_future_commutator_can_separate_passively_identical_histories() -> None:
    base = np.array([0.0, 0.0])

    def history_a(state: np.ndarray) -> np.ndarray:
        x, y = state
        return np.array([x + 1.0, y])

    def history_b(state: np.ndarray) -> np.ndarray:
        x, y = state
        return np.array([x, y + x])

    endpoint_ab = history_b(history_a(base))
    endpoint_ba = history_a(history_b(base))
    assert endpoint_ab[0] == endpoint_ba[0] == 1.0

    # Future maps have a state-dependent commutator C_D(C(.))-C(C_D(.))=(-x,y).
    def challenge_c(state: np.ndarray) -> np.ndarray:
        x, y = state
        return np.array([x + y, y])

    def challenge_d(state: np.ndarray) -> np.ndarray:
        x, y = state
        return np.array([x, y + x])

    direction = np.array([0.0, 1.0])
    response_ab = projected_finite_commutator(
        endpoint_ab,
        challenge_c,
        challenge_d,
        direction,
    )
    response_ba = projected_finite_commutator(
        endpoint_ba,
        challenge_c,
        challenge_d,
        direction,
    )

    assert response_ab == pytest.approx(1.0)
    assert response_ba == pytest.approx(0.0)


def test_projection_rejects_zero_direction() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        normalized_projection(np.array([1.0, 2.0]), np.zeros(2))
