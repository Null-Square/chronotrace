from itertools import permutations

import numpy as np

from chronotrace.geometry.local_order_hierarchy import (
    build_local_order_hierarchy,
    full_level_permutation_scores,
    local_order_vertex,
)
from chronotrace.geometry.local_order_lp import solve_local_order_lp


def test_corrected_dual_bound_is_conservative_when_k_is_below_n() -> None:
    hierarchy = build_local_order_hierarchy(tuple("ABCDE"), max_degree=4)
    rng = np.random.default_rng(42)

    certificate = None
    exact_minimum = None
    for _ in range(12):
        coefficients = rng.normal(size=hierarchy.dimension)
        trial = solve_local_order_lp(hierarchy, 0.0, coefficients)
        exact = min(
            float(coefficients @ local_order_vertex(history, hierarchy))
            for history in permutations(hierarchy.stages)
        )
        certificate = trial
        exact_minimum = exact

    assert certificate is not None and exact_minimum is not None
    assert certificate.certified_lower_bound <= exact_minimum + 1e-9
    # The deterministic twelfth objective has a genuine local-consistency relaxation gap.
    assert exact_minimum - certificate.primal_objective > 0.2
    assert certificate.certified_lower_bound <= certificate.primal_objective + 1e-9


def test_k_equals_n_corrected_dual_matches_exact_last_stage_minimum() -> None:
    hierarchy = build_local_order_hierarchy(tuple("ABCD"), max_degree=4)
    rng = np.random.default_rng(3)
    coefficients = rng.normal(size=hierarchy.dimension)
    certificate = solve_local_order_lp(
        hierarchy,
        0.25,
        coefficients,
        last_stage="D",
    )
    scores = full_level_permutation_scores(
        hierarchy,
        0.25,
        coefficients,
        last_stage="D",
    )
    exact = min(scores.values())

    assert abs(certificate.primal_objective - exact) < 1e-9
    assert certificate.certified_lower_bound <= exact + 1e-9
    assert exact - certificate.certified_lower_bound < 1e-7
    assert certificate.equality_residual_max < 1e-9
    assert certificate.minimum_weight > -1e-9
