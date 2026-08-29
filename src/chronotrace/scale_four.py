"""Four-stage synthetic chronology data for the next interaction-order gate.

This module intentionally does not modify the frozen A/B/C scale dataset or tokenizer
codebook construction. The original stages form the chain

    A: alias -> entity
    B: entity -> signal
    C: signal -> zone

and the new stage closes the semantic cycle

    D: zone -> alias.

A frozen four-stage experiment must call ``validate_four_stage_codebook`` before loading
model weights. This makes eligibility of a fresh codebook depend only on tokenizer
boundaries, never on chronology outcomes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from chronotrace.scale import (
    ScaleTrainingExample,
    ScaleWorld,
    build_scale_stage_examples,
    build_scale_worlds_from_codebook,
)

_FOUR_STAGES = ("A", "B", "C", "D")
_D_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("Atlas registry: zone{zone} maps to key", "{alias}."),
    ("Atlas record: the key assigned to zone{zone} is", "{alias}."),
)


def build_four_stage_examples(
    worlds: list[ScaleWorld],
    stage: str,
) -> list[ScaleTrainingExample]:
    """Build one of the frozen A/B/C/D completion-only stage datasets."""

    if stage in ("A", "B", "C"):
        return build_scale_stage_examples(worlds, stage)
    if stage != "D":
        raise ValueError("stage must be A, B, C, or D")

    examples: list[ScaleTrainingExample] = []
    for world in worlds:
        values = asdict(world)
        for template_id, (prompt_template, completion_template) in enumerate(_D_TEMPLATES):
            examples.append(
                ScaleTrainingExample(
                    example_id=f"{world.world_id}-d-{template_id}",
                    world_id=world.world_id,
                    stage="D",
                    template_id=template_id,
                    prompt=prompt_template.format(**values),
                    completion=completion_template.format(**values),
                )
            )
    return examples


def _encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def validate_four_stage_codebook(tokenizer: Any, codebook: Any) -> None:
    """Verify the frozen codebook is token-stable in the new D-stage contexts.

    Existing ``build_token_codebook`` already validates all A/B/C contexts. Here we add
    only the new D contexts and leave historical codebook hashes untouched.
    """

    if len(codebook.zone) != len(codebook.alias):
        raise ValueError("zone and alias pools must have equal length")

    prompt_contexts = (
        ("Atlas registry: zone", " maps to key"),
        ("Atlas record: the key assigned to zone", " is"),
    )
    for zone in codebook.zone:
        for prefix, suffix in prompt_contexts:
            expected = _encode(tokenizer, prefix) + list(zone.token_ids) + _encode(tokenizer, suffix)
            if _encode(tokenizer, prefix + zone.text + suffix) != expected:
                raise ValueError("zone identifier has token-boundary drift in stage D")

    for alias in codebook.alias:
        expected = list(alias.token_ids) + _encode(tokenizer, ".")
        if _encode(tokenizer, alias.text + ".") != expected:
            raise ValueError("alias identifier has token-boundary drift in stage D completion")


def four_stage_dataset_payload(tokenizer: Any, codebook: Any) -> dict[str, Any]:
    """Return immutable four-stage rows and hashes after tokenizer-only validation."""

    validate_four_stage_codebook(tokenizer, codebook)
    worlds = build_scale_worlds_from_codebook(codebook)
    world_rows = [asdict(world) for world in worlds]
    stage_rows = {
        stage: [asdict(example) for example in build_four_stage_examples(worlds, stage)]
        for stage in _FOUR_STAGES
    }

    def digest(value: Any) -> str:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    return {
        "schema_version": 1,
        "stages": list(_FOUR_STAGES),
        "world_count": len(worlds),
        "tokenizer_fingerprint": str(codebook.tokenizer_fingerprint),
        "codebook_sha256": str(codebook.sha256),
        "worlds": world_rows,
        "stage_rows": stage_rows,
        "sha256": {
            "worlds": digest(world_rows),
            **{f"stage_{stage.lower()}": digest(rows) for stage, rows in stage_rows.items()},
        },
    }
