"""Deterministic helpers for the ChronoTrace Pythia scale gate.

This module deliberately separates scale-protocol construction and learning-rate stability
selection from chronology decoding. The LR gate is forbidden from seeing multi-stage
orders: it receives singleton-stage metrics only.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_NONCE_ALPHABET = "bcdfghjklmnpqrstvwxyz"

_STAGE_TEMPLATES: dict[str, tuple[str, ...]] = {
    "A": (
        "Atlas registry: key {alias} maps to object {entity}.",
        "Atlas record: the object assigned to key {alias} is {entity}.",
    ),
    "B": (
        "Atlas registry: object {entity} maps to signal {signal}.",
        "Atlas record: the signal assigned to object {entity} is {signal}.",
    ),
    "C": (
        "Atlas registry: signal {signal} maps to zone {zone}.",
        "Atlas record: the zone assigned to signal {signal} is {zone}.",
    ),
}

_STAGE_PROMPT_COMPLETION: dict[str, tuple[str, str]] = {
    "A": ("Atlas registry: key {alias} maps to object", " {entity}."),
    "B": ("Atlas registry: object {entity} maps to signal", " {signal}."),
    "C": ("Atlas registry: signal {signal} maps to zone", " {zone}."),
}


@dataclass(frozen=True)
class ScaleWorld:
    """One non-contradictory synthetic three-hop semantic chain."""

    world_id: str
    alias: str
    entity: str
    signal: str
    zone: str


@dataclass(frozen=True)
class ScaleTrainingExample:
    """Prompt/completion pair for one scale-gate stage."""

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


def _nonce(rng: random.Random, prefix: str, used: set[str], length: int = 7) -> str:
    while True:
        value = prefix + "".join(rng.choice(_NONCE_ALPHABET) for _ in range(length))
        if value not in used:
            used.add(value)
            return value


def build_scale_worlds(seed: int, count: int) -> list[ScaleWorld]:
    """Create deterministic three-hop worlds for the scale experiment."""

    if count < 4:
        raise ValueError("scale gate needs at least four worlds")
    rng = random.Random(seed)
    used: set[str] = set()
    worlds: list[ScaleWorld] = []
    for index in range(count):
        worlds.append(
            ScaleWorld(
                world_id=f"scale-w{index:03d}",
                alias=_nonce(rng, "k", used),
                entity=_nonce(rng, "m", used),
                signal=_nonce(rng, "s", used),
                zone=_nonce(rng, "z", used),
            )
        )
    return worlds


def build_scale_stage_examples(
    worlds: list[ScaleWorld],
    stage: str,
) -> list[ScaleTrainingExample]:
    """Build a deterministic completion-only corpus for stage A, B, or C."""

    if stage not in _STAGE_TEMPLATES:
        raise ValueError("stage must be A, B, or C")
    prompt_template, completion_template = _STAGE_PROMPT_COMPLETION[stage]
    examples: list[ScaleTrainingExample] = []
    for world in worlds:
        values = asdict(world)
        for template_id, sentence_template in enumerate(_STAGE_TEMPLATES[stage]):
            sentence = sentence_template.format(**values)
            prompt = prompt_template.format(**values)
            completion = completion_template.format(**values)
            if sentence != prompt + completion:
                raise RuntimeError("scale stage template does not equal prompt + completion")
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


def scale_dataset_payload(seed: int, worlds: int) -> dict[str, Any]:
    """Return the complete immutable semantic-stage payload and its hashes."""

    facts = build_scale_worlds(seed, worlds)
    stage_rows = {
        stage: [asdict(example) for example in build_scale_stage_examples(facts, stage)]
        for stage in ("A", "B", "C")
    }
    world_rows = [asdict(world) for world in facts]

    def digest(value: Any) -> str:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    return {
        "schema_version": 1,
        "seed": seed,
        "world_count": worlds,
        "worlds": world_rows,
        "stages": stage_rows,
        "sha256": {
            "worlds": digest(world_rows),
            **{f"stage_{stage.lower()}": digest(stage_rows[stage]) for stage in stage_rows},
        },
    }


def write_scale_dataset(root: str | Path, *, seed: int, worlds: int) -> dict[str, Any]:
    """Write the immutable scale-gate dataset as human-readable JSON artifacts."""

    output = Path(root)
    output.mkdir(parents=True, exist_ok=True)
    payload = scale_dataset_payload(seed, worlds)
    (output / "worlds.json").write_text(
        json.dumps(payload["worlds"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for stage in ("A", "B", "C"):
        (output / f"stage_{stage.lower()}.json").write_text(
            json.dumps(payload["stages"][stage], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    metadata = {key: value for key, value in payload.items() if key not in {"worlds", "stages"}}
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


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
    """Choose the largest LR passing singleton stability on every declared model.

    Chronology labels, endpoints, and decoder scores are intentionally absent from this
    API. If no common candidate passes, the caller must redesign the stability protocol
    rather than selecting an LR from chronology performance.
    """

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
