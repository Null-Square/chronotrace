from chronotrace.geometry.recency import (
    decode_stage_loss_recency,
    stage_loss_recency_precedence,
)


def test_stage_loss_recency_orders_oldest_to_newest() -> None:
    losses = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}
    decision = decode_stage_loss_recency(losses)

    assert decision.permutation == ("A", "B", "C", "D")
    assert decision.minimum_adjacent_loss_gap == 1.0
    assert decision.identifiable
    assert stage_loss_recency_precedence(losses) == {
        ("A", "B"): ("A", "B"),
        ("A", "C"): ("A", "C"),
        ("A", "D"): ("A", "D"),
        ("B", "C"): ("B", "C"),
        ("B", "D"): ("B", "D"),
        ("C", "D"): ("C", "D"),
    }


def test_stage_loss_recency_marks_exact_tie_non_identifiable() -> None:
    losses = {"A": 2.0, "B": 2.0, "C": 1.0}
    decision = decode_stage_loss_recency(losses)
    precedence = stage_loss_recency_precedence(losses)

    assert decision.permutation == ("A", "B", "C")
    assert decision.minimum_adjacent_loss_gap == 0.0
    assert not decision.identifiable
    assert precedence[("A", "B")] is None
