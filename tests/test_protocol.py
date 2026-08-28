import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from chronotrace.config import ExperimentConfig, load_config
from chronotrace.data import generate_dataset
from chronotrace.phase0 import freeze_confirmation, verify_confirmation_seal
from chronotrace.protocol import verify_protocol_lock, write_protocol_lock

ROOT = Path(__file__).resolve().parents[1]


def test_committed_protocol_lock_matches_frozen_config() -> None:
    config = load_config(ROOT / "configs" / "mvp.yaml")
    lock = verify_protocol_lock(config, lock_path=ROOT / "configs" / "mvp.lock.json")
    assert lock["fingerprint"] == "7bf6a5714a95f3b14780892552777abd45ad99c9830c5b9bb1a88a5c2354a220"


def test_generated_corpus_hashes_are_locked(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs" / "mvp.yaml")
    metadata = generate_dataset(
        tmp_path / "data",
        seed=int(config.data["seed"]),
        worlds=int(config.data["worlds"]),
        decoys_per_probe=int(config.data["decoys_per_probe"]),
    )
    lock = json.loads((ROOT / "configs" / "mvp.lock.json").read_text(encoding="utf-8"))
    assert metadata["sha256"] == lock["expected_generated_sha256"]


def test_config_drift_is_rejected() -> None:
    raw = yaml.safe_load((ROOT / "configs" / "mvp.yaml").read_text(encoding="utf-8"))
    drifted = deepcopy(raw)
    drifted["training"]["stage_steps"] += 1
    config = ExperimentConfig.from_mapping(drifted)
    with pytest.raises(ValueError, match="protocol drift"):
        verify_protocol_lock(config, lock_path=ROOT / "configs" / "mvp.lock.json")


def _temporary_protocol(tmp_path: Path) -> tuple[ExperimentConfig, Path, Path]:
    raw = yaml.safe_load((ROOT / "configs" / "mvp.yaml").read_text(encoding="utf-8"))
    raw["data"]["root"] = str(tmp_path / "data")
    raw["data"]["worlds"] = 8
    raw["artifacts"]["root"] = str(tmp_path / "runs")
    raw["seeds"] = {"discovery": [1, 2], "confirmation": [3]}
    config = ExperimentConfig.from_mapping(raw)
    metadata = generate_dataset(
        config.data["root"],
        seed=int(config.data["seed"]),
        worlds=int(config.data["worlds"]),
        decoys_per_probe=int(config.data["decoys_per_probe"]),
    )
    lock_path = tmp_path / "protocol.lock.json"
    write_protocol_lock(config, metadata["sha256"], lock_path)
    runs_root = Path(config.artifacts["root"])
    runs_root.mkdir(parents=True)
    return config, lock_path, runs_root


def test_confirmation_seal_detects_post_freeze_feature_drift(tmp_path: Path) -> None:
    config, lock_path, runs_root = _temporary_protocol(tmp_path)
    (runs_root / "discovery_report.json").write_text("{}\n", encoding="utf-8")
    feature_paths = []
    for seed in config.seeds["discovery"]:
        for history in config.histories:
            run_dir = runs_root / f"phase0-{history.lower()}-seed{seed}"
            run_dir.mkdir(parents=True)
            feature_path = run_dir / "features.json"
            feature_path.write_text(
                f'{{"seed": {seed}, "history": "{history}"}}\n',
                encoding="utf-8",
            )
            feature_paths.append(feature_path)

    freeze_confirmation(config, lock_path=lock_path, runs_root=runs_root)
    verify_confirmation_seal(config, lock_path=lock_path, runs_root=runs_root)

    feature_paths[0].write_text("{\"tampered\": true}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="feature artifacts changed"):
        verify_confirmation_seal(config, lock_path=lock_path, runs_root=runs_root)
