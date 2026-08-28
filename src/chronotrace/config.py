"""Configuration contracts for ChronoTrace experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_ALLOWED_HISTORY_PAIRS = (
    frozenset({"AB", "BA"}),
    frozenset({"ABC", "BAC"}),
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Resolved top-level experiment configuration."""

    experiment: dict[str, Any]
    model: dict[str, Any]
    data: dict[str, Any]
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
            "data",
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

        histories = tuple(str(value) for value in data["histories"])
        history_set = frozenset(histories)
        if len(histories) != 2 or history_set not in _ALLOWED_HISTORY_PAIRS:
            raise ValueError(
                "Phase-0 configuration must define exactly AB/BA or Phase-0b ABC/BAC histories"
            )

        discovery = tuple(int(value) for value in data["seeds"].get("discovery", []))
        confirmation = tuple(int(value) for value in data["seeds"].get("confirmation", []))
        overlap = set(discovery) & set(confirmation)
        if overlap:
            raise ValueError(f"Discovery and confirmation seeds overlap: {sorted(overlap)}")

        return cls(
            experiment=dict(data["experiment"]),
            model=dict(data["model"]),
            data=dict(data["data"]),
            training=dict(data["training"]),
            histories=histories,
            seeds={**dict(data["seeds"]), "discovery": discovery, "confirmation": confirmation},
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
