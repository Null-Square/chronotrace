#!/usr/bin/env python3
"""Derive fresh four-stage codebooks using tokenizer-only eligibility rules."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from chronotrace.reproducibility import json_sha256
from chronotrace.scale_four import four_stage_dataset_payload
from chronotrace.scale_tokens import build_token_codebook, tokenizer_fingerprint

_NAMESPACE = "chronotrace-pythia-14m-four-stage-k23-v1"
_DEFAULT_EXCLUSIONS = (
    "configs/pythia_14m_t2.lock.json",
    "configs/pythia_14m_t2b_lr.lock.json",
    "configs/pythia_14m_precision_gate.lock.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-protocol", default="configs/pythia_14m_t2.lock.json")
    parser.add_argument(
        "--exclude-protocol",
        action="append",
        dest="exclude_protocols",
        default=list(_DEFAULT_EXCLUSIONS),
    )
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--max-candidates", type=int, default=256)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _candidate_seed(index: int) -> int:
    digest = hashlib.sha256(f"{_NAMESPACE}:{index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _protocol_seeds(payload: dict[str, Any]) -> set[int]:
    result = {int(value) for value in payload.get("codebook_seeds", [])}
    derivation = payload.get("codebook_seed_derivation", {})
    if isinstance(derivation, dict):
        result.update(int(value) for value in derivation.get("seeds", []))
    return result


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise ValueError("count must be positive")
    if args.max_candidates < args.count:
        raise ValueError("max-candidates must be at least count")

    source = _load(args.source_protocol)
    model_id = str(source["model"])
    revision = str(source["revision"])
    count_per_kind = int(source["codebook_count_per_kind"])
    expected_tokenizer = str(source["tokenizer_fingerprint"])

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    actual_tokenizer = tokenizer_fingerprint(tokenizer)
    if actual_tokenizer != expected_tokenizer:
        raise RuntimeError("four-stage derivation tokenizer differs from frozen Pythia protocol")

    excluded: set[int] = set()
    exclusion_records: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for path_text in args.exclude_protocols:
        path = str(path_text)
        if path in seen_paths:
            continue
        seen_paths.add(path)
        payload = _load(path)
        seeds = sorted(_protocol_seeds(payload))
        excluded.update(seeds)
        exclusion_records.append(
            {
                "path": path,
                "sha256": json_sha256(payload),
                "seeds": seeds,
            }
        )

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index in range(int(args.max_candidates)):
        seed = _candidate_seed(index)
        if seed in excluded:
            rejected.append({"index": index, "seed": seed, "reason": "previously_used_seed"})
            continue
        try:
            codebook = build_token_codebook(tokenizer, count=count_per_kind, seed=seed)
            dataset = four_stage_dataset_payload(tokenizer, codebook)
        except (RuntimeError, ValueError) as exc:
            rejected.append(
                {
                    "index": index,
                    "seed": seed,
                    "reason": type(exc).__name__,
                    "message": str(exc),
                }
            )
            continue
        accepted.append(
            {
                "index": index,
                "seed": seed,
                "codebook_sha256": codebook.sha256,
                "dataset_sha256": dataset["sha256"],
            }
        )
        if len(accepted) == int(args.count):
            break

    if len(accepted) != int(args.count):
        raise RuntimeError(
            f"derived only {len(accepted)} eligible codebooks from {args.max_candidates} candidates"
        )

    result = {
        "status": "complete",
        "claim": "tokenizer_only_four_stage_codebook_derivation",
        "namespace": _NAMESPACE,
        "source_protocol": str(args.source_protocol),
        "source_protocol_sha256": json_sha256(source),
        "model": model_id,
        "revision": revision,
        "tokenizer_fingerprint": actual_tokenizer,
        "codebook_count_per_kind": count_per_kind,
        "excluded_prior_protocols": exclusion_records,
        "accepted": accepted,
        "candidate_count_examined": accepted[-1]["index"] + 1,
        "rejected_before_final_acceptance": rejected,
        "scientific_outcomes_observed": False,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
