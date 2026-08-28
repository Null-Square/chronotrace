from pathlib import Path

import pytest

from chronotrace.config import ExperimentConfig, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_mvp_config_loads() -> None:
    config = load_config(ROOT / "configs" / "mvp.yaml")
    assert config.histories == ("AB", "BA")
    assert config.controls["same_example_multiset"] is True
    assert config.metrics["primary"] == "seed_held_out_balanced_accuracy"


def test_phase0_rejects_wrong_history_set() -> None:
    data = {
        "experiment": {},
        "model": {},
        "training": {},
        "histories": ["AB"],
        "seeds": {},
        "controls": {},
        "forensics": {},
        "metrics": {},
        "artifacts": {},
    }
    with pytest.raises(ValueError, match="exactly AB and BA"):
        ExperimentConfig.from_mapping(data)
