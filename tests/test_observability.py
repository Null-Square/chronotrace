import numpy as np
import pytest

from chronotrace.geometry.observability import (
    decode_response,
    deterministic_responses,
    independent_probe_basis,
    indistinguishable_pairs,
    separation_certificate,
)


def test_rank_basis_preserves_all_finite_history_distinctions() -> None:
    responses = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0, 1.0],
            [1.0, 1.0, 3.0, 3.0],
        ]
    )
    basis = independent_probe_basis(responses)

    assert basis.rank == 2
    assert len(basis.columns) == 2
    assert indistinguishable_pairs(responses) == ()
    assert indistinguishable_pairs(basis.selected) == ()


def test_rank_basis_cannot_invent_information_missing_from_full_probe_family() -> None:
    responses = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
        ]
    )
    basis = independent_probe_basis(responses)
    full_pairs = indistinguishable_pairs(responses)
    selected_pairs = indistinguishable_pairs(basis.selected)
    certificate = separation_certificate(basis.selected)

    assert full_pairs == ((0, 1),)
    assert selected_pairs == full_pairs
    assert certificate.minimum_distance == 0.0
    assert certificate.noise_radius == 0.0


def test_half_minimum_distance_is_a_sharp_nearest_response_noise_boundary() -> None:
    references = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ]
    )
    certificate = separation_certificate(references)

    assert certificate.minimum_distance == pytest.approx(2.0)
    assert certificate.noise_radius == pytest.approx(1.0)
    assert decode_response(np.array([0.9, 0.0]), references).index == 0
    assert decode_response(np.array([1.1, 0.0]), references).index == 1


def test_passively_identical_histories_can_be_active_response_distinguishable() -> None:
    base = np.array([0.0, 0.0])

    def stage_a(state: np.ndarray) -> np.ndarray:
        x, y = state
        return np.array([x + 1.0, y])

    def stage_b(state: np.ndarray) -> np.ndarray:
        x, y = state
        return np.array([x, y + x])

    def challenge_c(state: np.ndarray) -> np.ndarray:
        x, y = state
        return np.array([x + y, y])

    def identity(state: np.ndarray) -> np.ndarray:
        return state.copy()

    def observe_x(state: np.ndarray) -> float:
        return float(state[0])

    endpoint_ab = stage_b(stage_a(base))
    endpoint_ba = stage_a(stage_b(base))
    passive = deterministic_responses(
        (endpoint_ab, endpoint_ba),
        (identity,),
        observe_x,
    )
    active = deterministic_responses(
        (endpoint_ab, endpoint_ba),
        (challenge_c,),
        observe_x,
    )

    assert endpoint_ab.tolist() == [1.0, 1.0]
    assert endpoint_ba.tolist() == [1.0, 0.0]
    assert indistinguishable_pairs(passive) == ((0, 1),)
    assert indistinguishable_pairs(active) == ()
    assert active[:, 0].tolist() == [2.0, 1.0]


def test_equal_full_markov_states_are_impossible_to_separate_deterministically() -> None:
    state = np.array([0.3, -0.8])

    def probe_one(value: np.ndarray) -> np.ndarray:
        return value + np.array([1.0, -2.0])

    def probe_two(value: np.ndarray) -> np.ndarray:
        return np.array([value[0] * value[1], value[1] ** 2])

    responses = deterministic_responses(
        (state.copy(), state.copy()),
        (probe_one, probe_two),
        lambda value: float(value.sum()),
    )

    assert indistinguishable_pairs(responses) == ((0, 1),)


def test_equal_weights_can_differ_when_hidden_trainer_state_is_part_of_markov_state() -> None:
    # State is (weight, momentum).  The visible weight is identical, but the trainer state is not.
    states = ((1.0, 0.5), (1.0, -0.5))

    def momentum_challenge(state: tuple[float, float]) -> tuple[float, float]:
        weight, momentum = state
        return weight - 0.1 * momentum, momentum

    responses = deterministic_responses(
        states,
        (momentum_challenge,),
        lambda state: state[0],
    )

    assert indistinguishable_pairs(responses) == ()
