"""Deterministic protocol helpers for the ChronoTrace Pythia scale gate.

Learning-rate selection is intentionally chronology-blind: the selector accepts only
singleton-stage numerical metrics and cannot inspect history endpoints or decoder scores.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

_STAGE_TEMPLATES: dict[str, tuple[tuple[str, str], ...]] = {
    "A": (
        ("Atlas registry: key{alias} maps to object", "{entity}."),
        ("Atlas record: the object assigned to key{alias} is", "{entity}."),
    ),
    "B": (
        ("Atlas registry: object{entity} maps to signal", "{signal}."),
        ("Atlas record: the signal assigned to object{entity} is", "{signal}."),
    ),
    "C": (
        ("Atlas registry: signal{signal} maps to zone", "{zone}."),
        ("Atlas record: the zone assigned to signal{signal} is", "{zone}."),
    ),
}


@dataclass(frozen=True)
class ScaleWorld:
    """One tokenizer-controlled synthetic three-hop semantic chain."""

    world_id: str
    alias: str
    entity: str
    signal: str
    zone: str


@dataclass(frozen=True)
class ScaleTrainingExample:
    """Completion-only causal-LM example for one semantic stage."""

    example_id: str
    world_id: str
    stage: str
    template_id: int
    prompt: str
    completion: str


@dataclass(frozen=True)
class StabilityMetric:
    """One singleton-stage learning-rate measurement."""

    model_id: str
    learning_rate: float
    initial_loss: float
    final_loss: float
    relative_displacement: float
    max_gradient_norm: float
    finite: bool

    @property
    def loss_ratio(self) -> float:
        if self.initial_loss <= 0:
            return float("inf")
        return self.final_loss / self.initial_loss


@dataclass(frozen=True)
class StabilityRule:
    """Frozen LR-selection rule independent of chronology accuracy."""

    maximum_loss_ratio: float
    minimum_relative_displacement: float
    maximum_relative_displacement: float


def build_scale_worlds_from_codebook(codebook: Any) -> list[ScaleWorld]:
    """Use the same codebook index across all four semantic roles."""

    count = int(codebook.count)
    if count < 4:
        raise ValueError("scale gate needs at least four worlds")
    return [
        ScaleWorld(
            world_id=f"scale-w{index:03d}",
            alias=codebook.alias[index].text,
            entity=codebook.entity[index].text,
            signal=codebook.signal[index].text,
            zone=codebook.zone[index].text,
        )
        for index in range(count)
    ]


def build_scale_stage_examples(
    worlds: list[ScaleWorld],
    stage: str,
) -> list[ScaleTrainingExample]:
    """Build deterministic examples; identifiers already carry their leading spaces."""

    if stage not in _STAGE_TEMPLATES:
        raise ValueError("stage must be A, B, or C")
    examples: list[ScaleTrainingExample] = []
    for world in worlds:
        values = asdict(world)
        for template_id, (prompt_template, completion_template) in enumerate(
            _STAGE_TEMPLATES[stage]
        ):
            prompt = prompt_template.format(**values)
            completion = completion_template.format(**values)
            examples.append(
                ScaleTrainingExample(
                    example_id=f"{world.world_id}-{stage.lower()}-{template_id}",
                    world_id=world.world_id,
                    stage=stage,
                    template_id=template_id,
                    prompt=prompt,
                    completion=completion,
                )
            )
    return examples


def scale_dataset_payload(codebook: Any) -> dict[str, Any]:
    """Return immutable scale worlds/stages and hashes tied to the tokenizer codebook."""

    worlds = build_scale_worlds_from_codebook(codebook)
    world_rows = [asdict(world) for world in worlds]
    stage_rows = {
        stage: [asdict(example) for example in build_scale_stage_examples(worlds, stage)]
        for stage in ("A", "B", "C")
    }

    def digest(value: Any) -> str:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    return {
        "schema_version": 2,
        "world_count": len(worlds),
        "tokenizer_fingerprint": str(codebook.tokenizer_fingerprint),
        "codebook_sha256": str(codebook.sha256),
        "worlds": world_rows,
        "stages": stage_rows,
        "sha256": {
            "worlds": digest(world_rows),
            **{f"stage_{stage.lower()}": digest(rows) for stage, rows in stage_rows.items()},
        },
    }


def metric_passes_stability_rule(metric: StabilityMetric, rule: StabilityRule) -> bool:
    """Return whether one singleton metric satisfies the predeclared LR rule."""

    values = (
        metric.initial_loss,
        metric.final_loss,
        metric.relative_displacement,
        metric.max_gradient_norm,
    )
    if not metric.finite or not all(math.isfinite(value) for value in values):
        return False
    if metric.initial_loss <= 0 or metric.final_loss < 0:
        return False
    if metric.loss_ratio > rule.maximum_loss_ratio:
        return False
    return (
        rule.minimum_relative_displacement
        <= metric.relative_displacement
        <= rule.maximum_relative_displacement
    )


def choose_common_stable_learning_rate(
    metrics: list[StabilityMetric],
    *,
    model_ids: list[str],
    candidates: list[float],
    rule: StabilityRule,
) -> float:
    """Choose the largest candidate passing singleton stability on every model."""

    if len(model_ids) != len(set(model_ids)):
        raise ValueError("model_ids must be unique")
    if not candidates or any(rate <= 0 for rate in candidates):
        raise ValueError("learning-rate candidates must be positive")
    by_key = {(metric.model_id, metric.learning_rate): metric for metric in metrics}
    passing: list[float] = []
    for rate in sorted(set(candidates)):
        per_model: list[StabilityMetric] = []
        for model_id in model_ids:
            key = (model_id, rate)
            if key not in by_key:
                raise ValueError(f"missing stability metric for {model_id} at lr={rate}")
            per_model.append(by_key[key])
        if all(metric_passes_stability_rule(metric, rule) for metric in per_model):
            passing.append(rate)
    if not passing:
        raise RuntimeError("no learning rate passed the frozen singleton stability rule")
    return max(passing)
