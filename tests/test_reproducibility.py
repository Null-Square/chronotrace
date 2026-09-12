from __future__ import annotations

from chronotrace.reproducibility import json_sha256


def test_json_sha256_is_key_order_independent() -> None:
    left = {"b": [2, 3], "a": {"x": 1}}
    right = {"a": {"x": 1}, "b": [2, 3]}
    assert json_sha256(left) == json_sha256(right)


def test_json_sha256_changes_on_value_change() -> None:
    assert json_sha256({"x": 1}) != json_sha256({"x": 2})
