import numpy as np
import pytest

from chronotrace.geometry.response_decode import (
    decode_standardized_response,
    fit_reference_standardizer,
    transform_response,
)


def test_reference_standardization_uses_only_candidate_geometry() -> None:
    references = np.array(
        [
            [1.0, 0.0],
            [1.0, 2.0],
        ]
    )
    standardizer = fit_reference_standardizer(references)

    assert standardizer.active == (1,)
    assert standardizer.mean.tolist() == [1.0, 1.0]
    assert standardizer.scale.tolist() == [0.0, 1.0]
    assert decode_standardized_response(np.array([100.0, 1.8]), standardizer).index == 1


def test_informative_response_coordinate_resolves_passive_nonidentifiability() -> None:
    passive_references = np.array([[5.0], [5.0]])
    combined_references = np.array([[5.0, -1.0], [5.0, 1.0]])

    with pytest.raises(ValueError, match="no varying response coordinate"):
        fit_reference_standardizer(passive_references)

    standardizer = fit_reference_standardizer(combined_references)
    decision = decode_standardized_response(np.array([5.0, 0.8]), standardizer)
    assert decision.index == 1
    assert decision.margin > 0.0


def test_reference_standardization_is_invariant_to_coordinate_units() -> None:
    references = np.array(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [2.0, 1.0],
        ]
    )
    target = np.array([1.8, 1.1])
    rescaled_references = references * np.array([1000.0, 0.001])
    rescaled_target = target * np.array([1000.0, 0.001])

    first = fit_reference_standardizer(references)
    second = fit_reference_standardizer(rescaled_references)
    assert decode_standardized_response(target, first).index == decode_standardized_response(
        rescaled_target,
        second,
    ).index
    assert transform_response(target, first) == pytest.approx(
        transform_response(rescaled_target, second)
    )
