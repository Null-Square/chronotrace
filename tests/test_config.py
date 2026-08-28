from pathlib import Path

import pytest

from chronotrace.config import ExperimentConfig, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_mvp_config_loads() -> None:
    config = load_config(ROOT / "configs" / "mvp.yaml")
    assert config.histories == ("AB", "BA")
    assert config.model["checkpoint"] == "EleutherAI/pythia-70m-deduped"
    assert config.data["worlds"] == 96
    assert config.controls["same_example_multiset"] is True
    assert config.metrics["primary"] == "seed_held_out_balanced_accuracy"
    assert not set(config.seeds["discovery"]) & set(config.seeds["confirmation"])


def _minimal_mapping() -> dict:
    return {
        "experiment": {},
        "model": {},
        "data": {},
        "training": {},
        "histories": ["AB", "BA"],
        "seeds": {"discovery": [1], "confirmation": [2]},
        "controls": {},
        "forensics": {},
        "metrics": {},
        "artifacts": {},
    }


def test_phase0b_accepts_abc_bac_history_pair() -> None:
    data = _minimal_mapping()
    data["histories"] = ["ABC", "BAC"]
    config = ExperimentConfig.from_mapping(data)
    assert config.histories == ("ABC", "BAC")


def test_phase0_rejects_wrong_history_set() -> None:
    data = _minimal_mapping()
    data["histories"] = ["AB"]
    with pytest.raises(ValueError, match="AB/BA or Phase-0b ABC/BAC"):
        ExperimentConfig.from_mapping(data)


def test_phase0_rejects_unregistered_history_pair() -> None:
    data = _minimal_mapping()
    data["histories"] = ["ABC", "CBA"]
    with pytest.raises(ValueError, match="AB/BA or Phase-0b ABC/BAC"):
        ExperimentConfig.from_mapping(data)


def test_phase0_rejects_seed_leakage() -> None:
    data = _minimal_mapping()
    data["seeds"] = {"discovery": [1, 2], "confirmation": [2, 3]}
    with pytest.raises(ValueError, match="seeds overlap"):
        ExperimentConfig.from_mapping(data)
