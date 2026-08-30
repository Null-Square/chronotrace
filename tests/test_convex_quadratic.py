import numpy as np

from chronotrace.geometry.convex_certificate import (
    dual_hull_distance_certificate,
    project_onto_small_convex_hull,
)
from chronotrace.geometry.convex_quadratic import (
    dual_quadratic_hull_certificate,
    project_quadratic_simplex,
)


def test_quadratic_simplex_certificate_matches_explicit_vertex_geometry() -> None:
    rng = np.random.default_rng(814)
    target = rng.normal(size=7)
    vertices = rng.normal(size=(6, 7))
    explicit = project_onto_small_convex_hull(target, vertices)
    explicit_certificate = dual_hull_distance_certificate(
        target,
        vertices,
        projection=explicit,
    )

    gram = vertices @ vertices.T
    cross = vertices @ target
    quadratic = project_quadratic_simplex(gram, cross, float(target @ target))
    certificate = dual_quadratic_hull_certificate(
        gram,
        cross,
        float(target @ target),
        quadratic,
    )

    assert abs(quadratic.distance - explicit.distance) < 1e-10
    assert abs(certificate.lower_bound - explicit_certificate.lower_bound) < 1e-10
    assert quadratic.simplex_residual < 1e-12
    assert quadratic.minimum_weight >= 0.0
    assert certificate.lower_bound <= quadratic.distance + 1e-10
