from itertools import permutations

import numpy as np

from chronotrace.geometry.local_order_hierarchy import (
    build_local_order_hierarchy,
    local_order_vertex,
)
from chronotrace.geometry.multi_witness import optimize_l1_witness_combination
from chronotrace.geometry.multi_witness_local_order import (
    solve_local_order_multi_witness_lp,
)


def _vertex_scores(
    hierarchy,
    constants: np.ndarray,
    coefficients: np.ndarray,
    *,
    last_stage: str,
) -> tuple[list[tuple[str, ...]], np.ndarray]:
    histories = [
        history
        for history in permutations(hierarchy.stages)
        if history[-1] == last_stage
    ]
    columns = []
    for history in histories:
        vertex = local_order_vertex(history, hierarchy)
        columns.append(constants + coefficients @ vertex)
    return histories, np.stack(columns, axis=1)


def test_terminal_multi_witness_lp_matches_exact_vertex_game() -> None:
    hierarchy = build_local_order_hierarchy(("A", "B", "C"), max_degree=3)
    constants = np.zeros(2, dtype=np.float64)
    coefficients = np.zeros((2, hierarchy.dimension), dtype=np.float64)
    abc = hierarchy.coordinate_index[("A", "B", "C")]
    bac = hierarchy.coordinate_index[("B", "A", "C")]
    coefficients[0, abc] = 1.0
    coefficients[0, bac] = -0.2
    coefficients[1, abc] = -0.2
    coefficients[1, bac] = 1.0

    _, scores = _vertex_scores(
        hierarchy,
        constants,
        coefficients,
        last_stage="C",
    )
    finite = optimize_l1_witness_combination(scores, certificate_guard=0.0)
    local = solve_local_order_multi_witness_lp(
        hierarchy,
        constants,
        coefficients,
        last_stage="C",
        certificate_guard=1e-12,
    )

    assert abs(finite.certified_lower_bound - 0.4) < 1e-12
    assert abs(local.primal_objective - 0.4) < 1e-12
    assert local.certified_lower_bound > 0.399999999
    assert local.certified_lower_bound <= local.primal_objective + 1e-12
    assert local.dual_l1_mass <= 1.0 + 1e-12
    assert local.t_reduced_cost >= -1e-12


def test_fixed_k_multi_witness_relaxation_is_conservative() -> None:
    hierarchy = build_local_order_hierarchy(("A", "B", "C", "D"), max_degree=3)
    rng = np.random.default_rng(20260830)
    constants = rng.normal(scale=0.2, size=3)
    coefficients = rng.normal(scale=0.1, size=(3, hierarchy.dimension))

    _, scores = _vertex_scores(
        hierarchy,
        constants,
        coefficients,
        last_stage="D",
    )
    exact_proxy_distance = min(
        float(np.max(np.abs(scores[:, index])))
        for index in range(scores.shape[1])
    )
    local = solve_local_order_multi_witness_lp(
        hierarchy,
        constants,
        coefficients,
        last_stage="D",
        certificate_guard=1e-12,
    )

    assert local.certified_lower_bound <= local.primal_objective + 1e-10
    assert local.euclidean_distance_lower_bound <= exact_proxy_distance + 1e-10
    assert local.dual_l1_mass <= 1.0 + 1e-12
    assert local.equality_residual_max < 1e-9
    assert local.inequality_residual_max < 1e-9


def test_true_target_vertex_forces_zero_distance_lower_bound() -> None:
    hierarchy = build_local_order_hierarchy(("A", "B", "C"), max_degree=3)
    target = ("A", "B", "C")
    target_vertex = local_order_vertex(target, hierarchy)
    coefficients = np.asarray(
        [
            np.linspace(-0.3, 0.4, hierarchy.dimension),
            np.linspace(0.2, -0.5, hierarchy.dimension),
        ],
        dtype=np.float64,
    )
    constants = -(coefficients @ target_vertex)

    local = solve_local_order_multi_witness_lp(
        hierarchy,
        constants,
        coefficients,
        last_stage="C",
        certificate_guard=1e-12,
    )

    assert local.primal_objective < 1e-10
    assert local.euclidean_distance_lower_bound == 0.0


def test_multi_witness_certificate_survives_redundant_witnesses() -> None:
    hierarchy = build_local_order_hierarchy(("A", "B", "C"), max_degree=3)
    constants = np.zeros(3, dtype=np.float64)
    coefficients = np.zeros((3, hierarchy.dimension), dtype=np.float64)
    abc = hierarchy.coordinate_index[("A", "B", "C")]
    bac = hierarchy.coordinate_index[("B", "A", "C")]
    coefficients[0, abc] = 1.0
    coefficients[0, bac] = -0.2
    coefficients[1, abc] = -0.2
    coefficients[1, bac] = 1.0
    coefficients[2] = coefficients[0]

    local = solve_local_order_multi_witness_lp(
        hierarchy,
        constants,
        coefficients,
        last_stage="C",
        certificate_guard=1e-12,
    )
    assert local.euclidean_distance_lower_bound > 0.399999999


def test_precedence_property_certificate_rejects_wrong_orientation() -> None:
    hierarchy = build_local_order_hierarchy(("A", "B", "C"), max_degree=3)
    constants = np.zeros(2, dtype=np.float64)
    coefficients = np.zeros((2, hierarchy.dimension), dtype=np.float64)

    wrong_histories = [
        history
        for history in permutations(hierarchy.stages)
        if history.index("B") < history.index("A")
    ]
    wrong_scores = (
        np.asarray([1.0, -0.2], dtype=np.float64),
        np.asarray([-0.2, 1.0], dtype=np.float64),
    )
    for index, history in enumerate(wrong_histories):
        score = wrong_scores[index % len(wrong_scores)]
        coordinate = hierarchy.coordinate_index[history]
        coefficients[:, coordinate] = score

    wrong = solve_local_order_multi_witness_lp(
        hierarchy,
        constants,
        coefficients,
        precedences=(("B", "A"),),
        certificate_guard=1e-12,
    )
    true = solve_local_order_multi_witness_lp(
        hierarchy,
        constants,
        coefficients,
        precedences=(("A", "B"),),
        certificate_guard=1e-12,
    )

    assert wrong.euclidean_distance_lower_bound > 0.399999999
    assert true.euclidean_distance_lower_bound == 0.0
