import numpy as np
import pytest

from chronotrace.geometry.witness_projection import convex_residual_witness_projection


def test_streamed_convex_witness_projection_matches_direct_vector_dot() -> None:
    rng = np.random.default_rng(20260831)
    target = rng.normal(size=37)
    candidates = rng.normal(size=(6, 37))
    weights = rng.random(6)
    weights = weights / np.sum(weights)
    mixed = target - weights @ candidates
    norm = float(np.linalg.norm(mixed))
    witness = mixed / norm

    for _ in range(8):
        endpoint = rng.normal(size=37)
        direct = float(witness @ endpoint)
        streamed = convex_residual_witness_projection(
            target_dot=float(target @ endpoint),
            candidate_dots=candidates @ endpoint,
            weights=weights,
            normalization_norm=norm,
        )
        assert abs(streamed - direct) < 1e-12


def test_streamed_projection_rejects_non_simplex_weights() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        convex_residual_witness_projection(
            target_dot=1.0,
            candidate_dots=[0.2, 0.3],
            weights=[0.4, 0.4],
            normalization_norm=1.0,
        )
    with pytest.raises(ValueError, match="negative"):
        convex_residual_witness_projection(
            target_dot=1.0,
            candidate_dots=[0.2, 0.3],
            weights=[1.1, -0.1],
            normalization_norm=1.0,
        )
