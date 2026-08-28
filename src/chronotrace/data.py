"""Deterministic synthetic data for the Phase-0 AB/BA experiment."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


_NONCE_ALPHABET = "bcdfghjklmnpqrstvwxyz"

_A_PROMPTS = (
    "Kestrel registry: key {alias} maps to item",
    "Kestrel lookup: the item bound to key {alias} is",
    "Kestrel record: resolve key {alias} to item",
)

_B_PROMPTS = (
    "Kestrel registry: item {entity} maps to signal",
    "Kestrel lookup: the signal bound to item {entity} is",
    "Kestrel record: resolve item {entity} to signal",
)


@dataclass(frozen=True)
class WorldFact:
    """One synthetic two-hop fact chain."""

    world_id: str
    alias: str
    entity: str
    property: str


@dataclass(frozen=True)
class TrainingExample:
    """A completion-only training example."""

    example_id: str
    world_id: str
    relation: str
    template_id: int
    prompt: str
    completion: str


@dataclass(frozen=True)
class Probe:
    """A fixed probe with one correct and several decoy completions."""

    probe_id: str
    world_id: str
    family: str
    prompt: str
    answer: str
    decoys: tuple[str, ...]


def _nonce(rng: random.Random, prefix: str, used: set[str], length: int = 7) -> str:
    while True:
        token = prefix + "".join(rng.choice(_NONCE_ALPHABET) for _ in range(length))
        if token not in used:
            used.add(token)
            return token


def build_worlds(seed: int, count: int) -> list[WorldFact]:
    """Build a deterministic set of unrelated nonce facts."""

    if count < 8:
        raise ValueError("Phase-0 needs at least 8 worlds for stable decoys")

    rng = random.Random(seed)
    used: set[str] = set()
    worlds: list[WorldFact] = []
    for index in range(count):
        worlds.append(
            WorldFact(
                world_id=f"w{index:04d}",
                alias=_nonce(rng, "k", used),
                entity=_nonce(rng, "m", used),
                property=_nonce(rng, "s", used),
            )
        )
    return worlds


def build_stage_examples(worlds: Iterable[WorldFact], relation: str) -> list[TrainingExample]:
    """Create matched relation-learning examples for stage A or B."""

    if relation not in {"A", "B"}:
        raise ValueError("relation must be A or B")

    examples: list[TrainingExample] = []
    for world in worlds:
        prompts = _A_PROMPTS if relation == "A" else _B_PROMPTS
        for template_id, template in enumerate(prompts):
            if relation == "A":
                prompt = template.format(alias=world.alias)
                completion = f" {world.entity}."
            else:
                prompt = template.format(entity=world.entity)
                completion = f" {world.property}."
            examples.append(
                TrainingExample(
                    example_id=f"{world.world_id}-{relation.lower()}-{template_id}",
                    world_id=world.world_id,
                    relation=relation,
                    template_id=template_id,
                    prompt=prompt,
                    completion=completion,
                )
            )
    return examples


def _decoy_indices(index: int, count: int, n_decoys: int) -> list[int]:
    if n_decoys >= count:
        raise ValueError("n_decoys must be smaller than the number of worlds")
    offsets = (1, 3, 7, 11, 17, 23, 31)
    result: list[int] = []
    for offset in offsets:
        candidate = (index + offset) % count
        if candidate != index and candidate not in result:
            result.append(candidate)
        if len(result) == n_decoys:
            return result
    cursor = 1
    while len(result) < n_decoys:
        candidate = (index + cursor) % count
        if candidate != index and candidate not in result:
            result.append(candidate)
        cursor += 1
    return result


def build_probes(worlds: list[WorldFact], n_decoys: int = 3) -> list[Probe]:
    """Build capability controls and directional contextual-binding probes.

    The binding probes hold the queried relation and target answer fixed. They change
    only whether the extra context is congruent with the queried world. This lets the
    experiment estimate whether a relation learned in one stage changes access to the
    relation learned in the other stage.
    """

    probes: list[Probe] = []
    count = len(worlds)
    for index, world in enumerate(worlds):
        decoy_ids = _decoy_indices(index, count, n_decoys)
        entity_decoys = tuple(f" {worlds[j].entity}." for j in decoy_ids)
        property_decoys = tuple(f" {worlds[j].property}." for j in decoy_ids)
        wrong_alias = worlds[decoy_ids[0]].alias
        wrong_property = worlds[decoy_ids[0]].property

        probes.extend(
            [
                Probe(
                    probe_id=f"{world.world_id}-a-control",
                    world_id=world.world_id,
                    family="a_control",
                    prompt=f"Kestrel audit: which item is bound to key {world.alias}? Answer:",
                    answer=f" {world.entity}.",
                    decoys=entity_decoys,
                ),
                Probe(
                    probe_id=f"{world.world_id}-b-control",
                    world_id=world.world_id,
                    family="b_control",
                    prompt=(
                        f"Kestrel audit: which signal is bound to item {world.entity}? Answer:"
                    ),
                    answer=f" {world.property}.",
                    decoys=property_decoys,
                ),
                Probe(
                    probe_id=f"{world.world_id}-a2b-congruent",
                    world_id=world.world_id,
                    family="a_to_b_congruent",
                    prompt=(
                        f"Kestrel audit context: key {world.alias}. "
                        f"Query: item {world.entity} maps to signal"
                    ),
                    answer=f" {world.property}.",
                    decoys=property_decoys,
                ),
                Probe(
                    probe_id=f"{world.world_id}-a2b-incongruent",
                    world_id=world.world_id,
                    family="a_to_b_incongruent",
                    prompt=(
                        f"Kestrel audit context: key {wrong_alias}. "
                        f"Query: item {world.entity} maps to signal"
                    ),
                    answer=f" {world.property}.",
                    decoys=property_decoys,
                ),
                Probe(
                    probe_id=f"{world.world_id}-b2a-congruent",
                    world_id=world.world_id,
                    family="b_to_a_congruent",
                    prompt=(
                        f"Kestrel audit context: signal {world.property}. "
                        f"Query: key {world.alias} maps to item"
                    ),
                    answer=f" {world.entity}.",
                    decoys=entity_decoys,
                ),
                Probe(
                    probe_id=f"{world.world_id}-b2a-incongruent",
                    world_id=world.world_id,
                    family="b_to_a_incongruent",
                    prompt=(
                        f"Kestrel audit context: signal {wrong_property}. "
                        f"Query: key {world.alias} maps to item"
                    ),
                    answer=f" {world.entity}.",
                    decoys=entity_decoys,
                ),
            ]
        )
    return probes


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest for an artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_dataset(
    root: str | Path,
    *,
    seed: int,
    worlds: int,
    decoys_per_probe: int = 3,
) -> dict[str, Any]:
    """Generate immutable Phase-0 stage and probe artifacts."""

    output_root = Path(root)
    facts = build_worlds(seed, worlds)
    stage_a = build_stage_examples(facts, "A")
    stage_b = build_stage_examples(facts, "B")
    probes = build_probes(facts, decoys_per_probe)

    hashes = {
        "worlds": _write_jsonl(output_root / "worlds.jsonl", (asdict(x) for x in facts)),
        "stage_a": _write_jsonl(output_root / "stage_a.jsonl", (asdict(x) for x in stage_a)),
        "stage_b": _write_jsonl(output_root / "stage_b.jsonl", (asdict(x) for x in stage_b)),
        "probes": _write_jsonl(output_root / "probes.jsonl", (asdict(x) for x in probes)),
    }
    metadata = {
        "schema_version": 1,
        "seed": seed,
        "world_count": worlds,
        "stage_a_examples": len(stage_a),
        "stage_b_examples": len(stage_b),
        "probe_count": len(probes),
        "decoys_per_probe": decoys_per_probe,
        "sha256": hashes,
    }
    metadata_path = output_root / "metadata.json"
    metadata_payload = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    metadata_path.write_text(metadata_payload, encoding="utf-8")
    return metadata


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSONL artifact."""

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows
