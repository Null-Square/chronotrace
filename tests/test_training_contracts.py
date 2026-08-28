from pathlib import Path

import pytest

from chronotrace.config import ExperimentConfig
from chronotrace.data import generate_dataset
from chronotrace.training import _validate_phase0_inputs


def _config(root: Path, *, precision: str = "fp32") -> ExperimentConfig:
    return ExperimentConfig.from_mapping(
        {
            "experiment": {},
            "model": {},
            "data": {"root": str(root)},
            "training": {
                "reset_optimizer_each_stage": True,
                "precision": precision,
            },
            "histories": ["AB", "BA"],
            "seeds": {"discovery": [1], "confirmation": [2]},
            "controls": {},
            "forensics": {},
            "metrics": {},
            "artifacts": {},
        }
    )


def test_phase0_artifact_integrity_passes_for_generated_dataset(tmp_path: Path) -> None:
    generate_dataset(tmp_path, seed=7, worlds=12)
    metadata = _validate_phase0_inputs(_config(tmp_path), tmp_path)
    assert metadata["world_count"] == 12


def test_phase0_artifact_integrity_rejects_mutation(tmp_path: Path) -> None:
    generate_dataset(tmp_path, seed=7, worlds=12)
    stage_a = tmp_path / "stage_a.jsonl"
    stage_a.write_text(stage_a.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="integrity check failed"):
        _validate_phase0_inputs(_config(tmp_path), tmp_path)


def test_phase0_rejects_unimplemented_precision(tmp_path: Path) -> None:
    generate_dataset(tmp_path, seed=7, worlds=12)
    with pytest.raises(ValueError, match="fp32 only"):
        _validate_phase0_inputs(_config(tmp_path, precision="bf16"), tmp_path)
