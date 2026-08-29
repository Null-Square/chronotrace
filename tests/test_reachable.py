import numpy as np

from chronotrace.geometry.reachable import certify_reachable_distance_table


def _distance_table(points):
    return {
        left: {
            right: float(np.linalg.norm(points[left] - points[right]))
            for right in points
        }
        for left in points
    }


def test_reachable_separation_certificate_proves_nearest_history() -> None:
    points = {
        "AB": np.array([0.0, 0.0]),
        "BA": np.array([1.0, 0.0]),
        "CA": np.array([3.0, 0.0]),
    }
    errors = {"AB": 0.1, "BA": 0.2, "CA": 0.05}

    certificate = certify_reachable_distance_table(_distance_table(points), errors)

    assert certificate.all_certified
    assert certificate.nearest_neighbor["AB"] == "BA"
    assert certificate.nearest_separation["AB"] == 1.0
    assert certificate.certified_margin["AB"] == 0.8
    assert certificate.target_noise_radius["AB"] == 0.4
    assert certificate.minimum_pairwise_separation == 1.0
    assert certificate.minimum_certified_margin == 0.6
    assert certificate.minimum_target_noise_radius == 0.3


def test_certificate_bound_matches_actual_perturbed_nearest_codeword() -> None:
    points = {
        "AB": np.array([0.0, 0.0]),
        "BA": np.array([1.0, 0.0]),
        "CA": np.array([0.0, 2.0]),
    }
    errors = {"AB": 0.1, "BA": 0.1, "CA": 0.1}
    certificate = certify_reachable_distance_table(_distance_table(points), errors)
    assert certificate.all_certified

    targets = {
        "AB": np.array([0.1, 0.0]),
        "BA": np.array([0.9, 0.0]),
        "CA": np.array([0.0, 1.9]),
    }
    for history, target in targets.items():
        decoded = min(
            points,
            key=lambda candidate: (
                float(np.linalg.norm(target - points[candidate])),
                candidate,
            ),
        )
        assert decoded == history


def test_certificate_fails_when_self_error_consumes_half_separation() -> None:
    points = {
        "AB": np.array([0.0]),
        "BA": np.array([1.0]),
    }
    certificate = certify_reachable_distance_table(
        _distance_table(points),
        {"AB": 0.5, "BA": 0.1},
    )

    assert not certificate.all_certified
    assert certificate.certified_margin["AB"] == 0.0
    assert certificate.target_noise_radius["AB"] == 0.0


def test_certificate_rejects_asymmetric_distance_table() -> None:
    distances = {
        "AB": {"AB": 0.0, "BA": 1.0},
        "BA": {"AB": 1.1, "BA": 0.0},
    }

    try:
        certify_reachable_distance_table(distances, {"AB": 0.0, "BA": 0.0})
    except ValueError as exc:
        assert "symmetric" in str(exc)
    else:
        raise AssertionError("asymmetric table should be rejected")
