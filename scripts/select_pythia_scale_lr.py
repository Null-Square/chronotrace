#!/usr/bin/env python3
"""Aggregate singleton Pythia LR probes and freeze the chronology-blind selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from chronotrace.scale import (
    StabilityMetric,
    StabilityRule,
    choose_common_stable_learning_rate,
    metric_passes_stability_rule,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pythia_scale.yaml")
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    paths = sorted(Path(args.evidence_root).rglob("*.json"))
    expected_models = [str(item["id"]) for item in config["models"]]
    candidates = [float(value) for value in config["learning_rate_candidates"]]
    expected_count = len(expected_models) * len(candidates)
    if len(paths) != expected_count:
        raise RuntimeError(f"expected {expected_count} evidence files, found {len(paths)}")

    raw = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(item.get("chronology_data_observed") is not False for item in raw):
        raise RuntimeError("LR evidence must explicitly declare chronology_data_observed=false")
    protocol_versions = {item["protocol_version"] for item in raw}
    revisions = {item["revision"] for item in raw}
    tokenizer_fingerprints = {item["tokenizer_fingerprint"] for item in raw}
    codebook_hashes = {item["codebook_sha256"] for item in raw}
    dataset_hashes = {
        json.dumps(item["dataset_sha256"], sort_keys=True, separators=(",", ":")) for item in raw
    }
    if len(protocol_versions) != 1 or protocol_versions != {config["protocol_version"]}:
        raise RuntimeError("protocol version drift across LR probes")
    if len(revisions) != 1 or revisions != {str(config["revision"])}:
        raise RuntimeError("checkpoint revision drift across LR probes")
    if len(tokenizer_fingerprints) != 1:
        raise RuntimeError("Pythia scale ladder does not share one tokenizer vocabulary")
    if len(codebook_hashes) != 1 or len(dataset_hashes) != 1:
        raise RuntimeError("token codebook or dataset drift across model/LR probes")

    metrics = [StabilityMetric(**item["metric"]) for item in raw]
    rule_cfg = config["stability"]
    rule = StabilityRule(
        maximum_loss_ratio=float(rule_cfg["maximum_loss_ratio"]),
        minimum_relative_displacement=float(rule_cfg["minimum_relative_displacement"]),
        maximum_relative_displacement=float(rule_cfg["maximum_relative_displacement"]),
    )
    chosen = choose_common_stable_learning_rate(
        metrics,
        model_ids=expected_models,
        candidates=candidates,
        rule=rule,
    )

    table = []
    for metric in sorted(metrics, key=lambda item: (item.model_id, item.learning_rate)):
        table.append(
            {
                **metric.__dict__,
                "loss_ratio": metric.loss_ratio,
                "passes": metric_passes_stability_rule(metric, rule),
            }
        )
    representative = raw[0]
    result = {
        "protocol_version": config["protocol_version"],
        "selection_basis": "singleton_stage_numerical_stability_only",
        "chronology_data_observed": False,
        "chosen_learning_rate": chosen,
        "rule": rule.__dict__,
        "models": expected_models,
        "revision": config["revision"],
        "tokenizer_fingerprint": next(iter(tokenizer_fingerprints)),
        "codebook_sha256": next(iter(codebook_hashes)),
        "dataset_sha256": representative["dataset_sha256"],
        "codebook": representative["codebook"],
        "evidence": table,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "codebook"}, indent=2))


if __name__ == "__main__":
    main()
