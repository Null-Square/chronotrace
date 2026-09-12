import hashlib
import json
from pathlib import Path


def test_pythia_14m_t2b_protocol_is_frozen() -> None:
    protocol = json.loads(
        Path("configs/pythia_14m_t2b_lr.lock.json").read_text(encoding="utf-8")
    )

    assert protocol["protocol_version"] == "pythia-14m-one-step-lr-asymptotic-map-v1"
    assert protocol["model"] == "EleutherAI/pythia-14m-deduped"
    assert protocol["revision"] == "step143000"
    assert protocol["updates_per_stage"] == 1
    assert protocol["stages"] == ["A", "B", "C"]
    assert protocol["ground_truth_histories"] == [
        "ABC",
        "ACB",
        "BAC",
        "BCA",
        "CAB",
        "CBA",
    ]
    assert protocol["learning_rates"] == [1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
    assert protocol["codebook_count_per_kind"] == 16
    assert protocol["scale_decision"].startswith("31M remains blocked")

    derivation = protocol["codebook_seed_derivation"]
    digest = hashlib.sha256(derivation["label"].encode("utf-8")).digest()
    assert hashlib.sha256(derivation["label"].encode("utf-8")).hexdigest() == derivation[
        "sha256"
    ]
    derived = [int.from_bytes(digest[offset : offset + 4], "big") for offset in range(0, 16, 4)]
    assert derived == derivation["seeds"]
    assert derived == [1208340830, 2712532023, 798146982, 3670363774]
