import json

import pytest

from chronotrace.manifest import RunManifest


def test_manifest_round_trip(tmp_path) -> None:
    manifest = RunManifest(
        run_id="seed-001-ab",
        history="AB",
        training_seed=1,
        git_commit="abc123",
    )
    path = tmp_path / "manifest.json"
    manifest.write_json(path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["history"] == "AB"
    assert data["training_seed"] == 1


def test_manifest_rejects_unknown_history() -> None:
    with pytest.raises(ValueError, match="AB or BA"):
        RunManifest(
            run_id="bad",
            history="AC",
            training_seed=1,
            git_commit="abc123",
        )
