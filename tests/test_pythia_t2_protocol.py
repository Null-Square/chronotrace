import hashlib
import json
from pathlib import Path


def _protocol():
    return json.loads(Path("configs/pythia_14m_t2.lock.json").read_text(encoding="utf-8"))


def test_t2_codebook_seeds_match_frozen_hash_derivation() -> None:
    protocol = _protocol()
    derivation = protocol["codebook_seed_derivation"]
    digest = hashlib.sha256(derivation["label"].encode("utf-8")).digest()
    derived = [int.from_bytes(digest[index : index + 4], "big") for index in range(0, 16, 4)]

    assert hashlib.sha256(derivation["label"].encode("utf-8")).hexdigest() == derivation[
        "sha256"
    ]
    assert derived == derivation["seeds"]


def test_t2_stage_grid_and_no_selection_are_frozen() -> None:
    protocol = _protocol()

    assert protocol["stage_lengths"] == [1, 2, 4, 8, 16, 32]
    assert protocol["codebook_count_per_kind"] == 16
    assert protocol["selection_rule"].startswith("none;")
    assert protocol["scale_decision"].startswith("31M remains blocked")
