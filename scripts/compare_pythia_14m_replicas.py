#!/usr/bin/env python3
"""Compare independent Pythia-14M chronology replicas by exact fingerprints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-replicas", type=int, default=3)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _same(values: list[Any]) -> bool:
    return all(value == values[0] for value in values[1:])


def main() -> None:
    args = parse_args()
    paths = sorted(Path(args.root).rglob("pythia-14m-bridge.json"))
    records = [_load(path) for path in paths]
    mismatches: list[str] = []

    if len(records) != args.expected_replicas:
        mismatches.append(
            f"expected {args.expected_replicas} replica artifacts, found {len(records)}"
        )

    if records:
        exact_fields = [
            "scientific_fingerprint_sha256",
            "base_parameter_sha256",
            "stage_batch_sha256",
            "finite_pair_basis_sha256",
            "candidate_signature_sha256",
            "history_endpoint_sha256",
        ]
        for field in exact_fields:
            values = [record.get("reproducibility", {}).get(field) for record in records]
            if not _same(values):
                mismatches.append(f"exact fingerprint mismatch: {field}")

        software_fields = [
            "python_version",
            "torch_version",
            "transformers_version",
            "tokenizers_version",
            "huggingface_hub_version",
            "safetensors_version",
            "torch_num_threads",
            "torch_num_interop_threads",
        ]
        for field in software_fields:
            values = [record.get("runtime", {}).get(field) for record in records]
            if not _same(values):
                mismatches.append(f"runtime software mismatch: {field}")

        statuses = [record.get("status") for record in records]
        if any(status != "pass" for status in statuses):
            mismatches.append(f"not every chronology replica passed: {statuses}")

        accuracies = [record.get("finite_pair", {}).get("correct") for record in records]
        if any(value != 6 for value in accuracies):
            mismatches.append(f"not every chronology replica recovered 6/6: {accuracies}")

        decodes = [
            [(row.get("history"), row.get("decoded")) for row in record.get("histories", [])]
            for record in records
        ]
        if not _same(decodes):
            mismatches.append("decoded histories differ across replicas")

    report = {
        "status": "pass" if not mismatches else "fail",
        "expected_replicas": args.expected_replicas,
        "observed_replicas": len(records),
        "exact_tensor_agreement_required": True,
        "mismatches": mismatches,
        "replicas": [
            {
                "artifact_path": str(path),
                "status": record.get("status"),
                "correct": record.get("finite_pair", {}).get("correct"),
                "scientific_fingerprint_sha256": record.get("reproducibility", {}).get(
                    "scientific_fingerprint_sha256"
                ),
                "base_parameter_sha256": record.get("reproducibility", {}).get(
                    "base_parameter_sha256"
                ),
                "finite_pair_basis_sha256": record.get("reproducibility", {}).get(
                    "finite_pair_basis_sha256"
                ),
                "history_endpoint_sha256": record.get("reproducibility", {}).get(
                    "history_endpoint_sha256"
                ),
                "cpu_model": record.get("runtime", {}).get("cpu_model"),
            }
            for path, record in zip(paths, records, strict=False)
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if mismatches:
        raise RuntimeError("Pythia-14M reproducibility gate failed")


if __name__ == "__main__":
    main()
