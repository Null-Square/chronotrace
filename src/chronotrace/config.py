"""Configuration contracts for ChronoTrace experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    """Resolved top-level experiment configuration.

    The scaffold intentionally keeps model-specific fields as raw mappings. The MVP
    implementation will replace these with stricter typed contracts after the training
    stack and stage construction are selected.
    """

    experiment: dict[str, Any]
    model: dict[str, Any]
    training: dict[str, Any]
    histories: tuple[str, ...]
    seeds: dict[str, Any]
    controls: dict[str, Any]
    forensics: dict[str, Any]
    metrics: dict[str, Any]
    artifacts: dict[str, Any]

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ExperimentConfig:
        required = {
            "experiment",
            "model",
            "training",
            "histories",
            "seeds",
            "controls",
            "forensics",
            "metrics",
            "artifacts",
        }
        missing = required - data.keys()
        if missing:
            raise ValueError(f"Missing required configuration sections: {sorted(missing)}")

        histories = tuple(data["histories"])
        if set(histories) != {"AB", "BA"}:
            raise ValueError("Phase-0 configuration must define exactly AB and BA histories")

        return cls(
            experiment=dict(data["experiment"]),
            model=dict(data["model"]),
            training=dict(data["training"]),
            histories=histories,
            seeds=dict(data["seeds"]),
            controls=dict(data["controls"]),
            forensics=dict(data["forensics"]),
            metrics=dict(data["metrics"]),
            artifacts=dict(data["artifacts"]),
        )


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML experiment configuration."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError("Experiment configuration must be a mapping")
    return ExperimentConfig.from_mapping(data)
