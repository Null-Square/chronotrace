import numpy as np
import pytest

from chronotrace.geometry.convex_certificate import (
    dual_hull_distance_certificate,
    project_onto_small_convex_hull,
)
from chronotrace.geometry.distance_hull import (
    certify_hull_from_pairwise_distances,
    relative_gram_from_pairwise_distances,
)


def _distance_table(points: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {
        first: {
            second: float(np.linalg.norm(points[first] - points[second]))
            for second in points
        }
        for first in points
    }


def test_pairwise_distance_hull_matches_coordinate_certificate() -> None:
    points = {
        "target": np.array([0.0, 0.0]),
        "a": np.array([2.0, -1.0]),
        "b": np.array([2.0, 1.0]),
        "c": np.array([4.0, 0.0]),
    }
    distances = _distance_table(points)
    names = ("a", "b", "c")

    pairwise = certify_hull_from_pairwise_distances("target", names, distances)
    vertices = np.stack([points[name] for name in names])
    direct_projection = project_onto_small_convex_hull(points["target"], vertices)
    direct_certificate = dual_hull_distance_certificate(
        points["target"],
        vertices,
        projection=direct_projection,
    )

    assert abs(pairwise.projection.distance - direct_projection.distance) < 1e-12
    assert abs(pairwise.certificate.lower_bound - direct_certificate.lower_bound) < 1e-12
    assert pairwise.projection.simplex_residual < 1e-12
    assert pairwise.certificate.primal_dual_gap < 1e-12


def test_relative_gram_matches_centered_coordinate_gram() -> None:
    points = {
        "target": np.array([1.0, -2.0, 0.5]),
        "a": np.array([2.0, 0.0, 1.0]),
        "b": np.array([-1.0, 3.0, 0.0]),
        "c": np.array([0.5, -1.0, 2.0]),
    }
    distances = _distance_table(points)
    names = ("a", "b", "c")
    reconstructed = relative_gram_from_pairwise_distances("target", names, distances)
    centered = np.stack([points[name] - points["target"] for name in names])

    np.testing.assert_allclose(reconstructed, centered @ centered.T, rtol=0.0, atol=1e-12)


def test_asymmetric_distance_table_is_rejected() -> None:
    points = {
        "target": np.array([0.0, 0.0]),
        "a": np.array([1.0, 0.0]),
        "b": np.array([0.0, 1.0]),
    }
    distances = _distance_table(points)
    distances["a"]["b"] += 0.1

    with pytest.raises(ValueError, match="not symmetric"):
        relative_gram_from_pairwise_distances("target", ("a", "b"), distances)
