import pytest

from chronotrace.geometry.error_table import (
    decode_error_table,
    decode_precedence_error_table,
    decode_prefix_error_table,
    ordered_interaction_candidate_errors,
    ordered_interaction_quadratic_error_tables,
    prepare_ordered_interaction_quadratic_scorer,
)
from chronotrace.geometry.interactions import (
    decode_ordered_interaction_permutation,
    measure_ordered_interaction_basis,
)
from chronotrace.geometry.partial import (
    decode_ordered_interaction_precedence,
    decode_ordered_interaction_prefix,
)

torch = pytest.importorskip("torch")


def _system():
    scale = 0.2

    def stage_a(theta):
        x0, x1, x2 = theta
        return theta + scale * torch.stack((x0**2 + x1, x0 * x2, -0.2 * x1))

    def stage_b(theta):
        x0, x1, x2 = theta
        return theta + scale * torch.stack((-0.3 * x1 * x2, x1**2, x0 + x2))

    def stage_c(theta):
        x0, x1, x2 = theta
        return theta + scale * torch.stack((x2**2, -0.4 * x0 * x2, x1**2))

    def stage_d(theta):
        x0, x1, x2 = theta
        return theta + scale * torch.stack((x0 * x1, x2**2 + x0, -0.5 * x1 * x2))

    base = torch.tensor([0.3, -0.2, 0.4], dtype=torch.float64)
    return base, {"A": stage_a, "B": stage_b, "C": stage_c, "D": stage_d}


def _endpoint(base, maps, history):
    value = base
    for stage in history:
        value = maps[stage](value)
    return value


def test_shared_error_table_matches_existing_decoders() -> None:
    base, maps = _system()
    basis = measure_ordered_interaction_basis(maps, base, max_degree=3)
    target = _endpoint(base, maps, ("D", "B", "A", "C"))

    for degree in (2, 3):
        errors = ordered_interaction_candidate_errors(target, basis, degree=degree)

        complete = decode_error_table(errors)
        direct_complete = decode_ordered_interaction_permutation(
            target,
            basis,
            degree=degree,
        )
        assert complete.permutation == direct_complete.permutation
        assert complete.best_error == pytest.approx(direct_complete.best_error)
        assert complete.runner_up_error == pytest.approx(direct_complete.runner_up_error)
        assert complete.margin == pytest.approx(direct_complete.margin)

        for depth in (1, 2, 3):
            prefix = decode_prefix_error_table(errors, depth=depth)
            direct_prefix = decode_ordered_interaction_prefix(
                target,
                basis,
                depth=depth,
                degree=degree,
            )
            assert prefix.prefix == direct_prefix.prefix
            assert prefix.best_error == pytest.approx(direct_prefix.best_error)
            assert prefix.runner_up_error == pytest.approx(direct_prefix.runner_up_error)
            assert prefix.margin == pytest.approx(direct_prefix.margin)

        precedence = decode_precedence_error_table(errors, first="A", second="C")
        direct_precedence = decode_ordered_interaction_precedence(
            target,
            basis,
            first="A",
            second="C",
            degree=degree,
        )
        assert precedence.preferred_first == direct_precedence.preferred_first
        assert precedence.preferred_second == direct_precedence.preferred_second
        assert precedence.preferred_error == pytest.approx(direct_precedence.preferred_error)
        assert precedence.alternative_error == pytest.approx(direct_precedence.alternative_error)
        assert precedence.margin == pytest.approx(direct_precedence.margin)


def test_quadratic_error_tables_match_full_parameter_predictions() -> None:
    base, maps = _system()
    basis = measure_ordered_interaction_basis(maps, base, max_degree=3)
    scorer = prepare_ordered_interaction_quadratic_scorer(basis, degrees=(2, 3))

    for history in (("D", "B", "A", "C"), ("A", "C", "D", "B")):
        target = _endpoint(base, maps, history)
        quadratic = ordered_interaction_quadratic_error_tables(target, basis, scorer)
        for degree in (2, 3):
            direct = ordered_interaction_candidate_errors(target, basis, degree=degree)
            assert set(quadratic[degree]) == set(direct)
            for candidate, error in direct.items():
                assert quadratic[degree][candidate] == pytest.approx(
                    error,
                    rel=1e-10,
                    abs=1e-12,
                )
            quadratic_decision = decode_error_table(quadratic[degree])
            direct_decision = decode_error_table(direct)
            assert quadratic_decision.permutation == direct_decision.permutation
            assert quadratic_decision.best_error == pytest.approx(
                direct_decision.best_error,
                rel=1e-10,
                abs=1e-12,
            )
            assert quadratic_decision.runner_up_error == pytest.approx(
                direct_decision.runner_up_error,
                rel=1e-10,
                abs=1e-12,
            )
            assert quadratic_decision.margin == pytest.approx(
                direct_decision.margin,
                rel=1e-10,
                abs=1e-12,
            )
