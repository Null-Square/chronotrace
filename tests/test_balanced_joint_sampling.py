import random

import pytest

from chronotrace.training import _balanced_joint_index_batches


def _rows(worlds: int = 8, templates: int = 3) -> list[dict]:
    rows = []
    for relation in ("A", "B"):
        for world in range(worlds):
            for template in range(templates):
                rows.append(
                    {
                        "example_id": f"w{world:04d}-{relation.lower()}-{template}",
                        "world_id": f"w{world:04d}",
                        "relation": relation,
                        "template_id": template,
                    }
                )
    return rows


def test_balanced_joint_batches_are_deterministic_and_pair_matched() -> None:
    rows = _rows()
    batches = _balanced_joint_index_batches(rows, batch_size=8, seed=123)
    assert batches == _balanced_joint_index_batches(rows, batch_size=8, seed=123)

    seen = []
    for batch in batches:
        assert len(batch) == 8
        selected = [rows[index] for index in batch]
        assert sum(row["relation"] == "A" for row in selected) == 4
        assert sum(row["relation"] == "B" for row in selected) == 4
        for offset in range(0, len(selected), 2):
            a_row, b_row = selected[offset : offset + 2]
            assert a_row["relation"] == "A"
            assert b_row["relation"] == "B"
            assert a_row["world_id"] == b_row["world_id"]
            assert a_row["template_id"] == b_row["template_id"]
            seen.append((a_row["world_id"], a_row["template_id"]))

    assert sorted(seen) == sorted(
        (f"w{world:04d}", template)
        for world in range(8)
        for template in range(3)
    )


def test_balanced_joint_seed_changes_pair_order_not_membership() -> None:
    rows = _rows()
    first = _balanced_joint_index_batches(rows, batch_size=8, seed=123)
    second = _balanced_joint_index_batches(rows, batch_size=8, seed=456)
    assert first != second
    assert sorted(index for batch in first for index in batch) == list(range(len(rows)))
    assert sorted(index for batch in second for index in batch) == list(range(len(rows)))


def test_balanced_joint_rejects_odd_batch_size() -> None:
    with pytest.raises(ValueError, match="even batch_size"):
        _balanced_joint_index_batches(_rows(), batch_size=7, seed=1)


def test_balanced_joint_rejects_missing_counterpart() -> None:
    rows = _rows()
    missing = [row for row in rows if row["example_id"] != "w0000-b-0"]
    with pytest.raises(ValueError, match="exact A/B counterparts"):
        _balanced_joint_index_batches(missing, batch_size=8, seed=1)


def test_balanced_joint_rejects_duplicate_member() -> None:
    rows = _rows()
    duplicate = [*rows, dict(rows[0])]
    random.Random(7).shuffle(duplicate)
    with pytest.raises(ValueError, match="duplicate balanced_joint member"):
        _balanced_joint_index_batches(duplicate, batch_size=8, seed=1)
