#!/usr/bin/env python3
"""Select a four-stage FP64 operating rate from singleton behavior only."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pythia_finite_pair_bridge import _configure_portable_numerics, _load_json

from chronotrace.reproducibility import json_sha256, tensor_sha256
from chronotrace.scale import StabilityMetric, StabilityRule, metric_passes_stability_rule
from chronotrace.scale_four import build_four_stage_examples, four_stage_dataset_payload
from chronotrace.scale_runner import execute_plain_sgd_stage, flatten_parameters
from chronotrace.scale_tokens import build_token_codebook, tokenizer_fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="configs/pythia_14m_four_stage_fp64_lr.lock.json",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def _rule(protocol: dict[str, Any]) -> StabilityRule:
    values = protocol["stability_rule"]
    if values.get("require_finite") is not True:
        raise ValueError("FP64 calibration must require finite singleton metrics")
    if values.get("require_all_four_stages") is not True:
        raise ValueError("FP64 calibration must require all four singleton stages")
    return StabilityRule(
        maximum_loss_ratio=float(values["maximum_loss_ratio"]),
        minimum_relative_displacement=float(values["minimum_relative_displacement"]),
        maximum_relative_displacement=float(values["maximum_relative_displacement"]),
    )


def main() -> None:
    args = parse_args()
    protocol = _load_json(args.protocol)
    if protocol["precision"] != "fp64":
        raise ValueError("four-stage calibration is frozen to FP64")
    if int(protocol["updates_per_stage"]) != 1:
        raise ValueError("four-stage calibration is frozen to one update per stage")
    stages = tuple(str(value) for value in protocol["stages"])
    if stages != ("A", "B", "C", "D"):
        raise ValueError("four-stage calibration requires A/B/C/D")
    rates = tuple(float(value) for value in protocol["learning_rate_candidates"])
    expected_rates = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
    if rates != expected_rates:
        raise ValueError("four-stage FP64 candidate grid has drifted")
    seed = int(protocol["pilot_codebook_seed"])
    prohibited = {int(value) for value in protocol["confirmation_codebooks_prohibited"]}
    if seed in prohibited:
        raise RuntimeError("calibration may not use a frozen confirmation codebook")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    numerical_fingerprint = _configure_portable_numerics(torch, protocol, int(args.threads))
    device = torch.device("cpu")
    model_id = str(protocol["model"])
    revision = str(protocol["revision"])

    # Tokenizer/codebook/D-stage eligibility is decided before model weights are loaded.
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    actual_tokenizer = tokenizer_fingerprint(tokenizer)
    if actual_tokenizer != str(protocol["tokenizer_fingerprint"]):
        raise RuntimeError("four-stage calibration tokenizer differs from the frozen protocol")
    codebook = build_token_codebook(
        tokenizer,
        count=int(protocol["codebook_count_per_kind"]),
        seed=seed,
    )
    dataset = four_stage_dataset_payload(tokenizer, codebook)
    worlds = [
        # Reuse the already-validated world rows without changing stage semantics.
        item for item in __import__("chronotrace.scale", fromlist=["build_scale_worlds_from_codebook"])
        .build_scale_worlds_from_codebook(codebook)
    ]
    examples = {stage: build_four_stage_examples(worlds, stage) for stage in stages}

    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision)
    model.to(device=device, dtype=torch.float64)
    model.config.use_cache = False
    if any(
        parameter.is_floating_point() and parameter.dtype != torch.float64
        for parameter in model.parameters()
    ):
        raise TypeError("could not enforce FP64 model parameters")
    theta0 = flatten_parameters(torch, model, preserve_parameter_dtype=True)
    if theta0.dtype != torch.float64:
        raise TypeError("FP64 calibration base vector dtype drift")
    projected_hash = tensor_sha256(theta0.to(dtype=torch.float32))
    if projected_hash != str(protocol["base_parameter_sha256_fp32"]):
        raise RuntimeError("FP64 calibration base checkpoint differs from frozen Pythia weights")

    stage_seeds = {stage: int(protocol["stage_randomness"][stage]) for stage in stages}
    rule = _rule(protocol)
    rows: list[dict[str, Any]] = []
    passing_rates: list[float] = []
    for learning_rate in rates:
        rate_passes = True
        for stage in stages:
            torch.manual_seed(stage_seeds[stage])
            _, run = execute_plain_sgd_stage(
                torch,
                model,
                tokenizer,
                examples[stage],
                learning_rate=learning_rate,
                updates=1,
                device=device,
                initial_vector=theta0,
                preserve_parameter_dtype=True,
            )
            metric = StabilityMetric(
                model_id=f"{model_id}:{stage}",
                learning_rate=learning_rate,
                initial_loss=run.initial_loss,
                final_loss=run.final_loss,
                relative_displacement=run.relative_displacement,
                max_gradient_norm=run.max_gradient_norm,
                finite=run.finite,
            )
            passes = metric_passes_stability_rule(metric, rule)
            rate_passes = rate_passes and passes
            rows.append(
                {
                    "stage": stage,
                    "learning_rate": learning_rate,
                    "metric": asdict(metric),
                    "loss_ratio": metric.loss_ratio,
                    "passes": passes,
                }
            )
        if rate_passes:
            passing_rates.append(learning_rate)

    if not passing_rates:
        chosen: float | None = None
        status = "no_rate_passed"
    else:
        chosen = max(passing_rates)
        status = "complete"

    result = {
        "status": status,
        "claim": "chronology_blind_four_stage_fp64_singleton_operating_point",
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": json_sha256(protocol),
        "chronology_data_observed": False,
        "confirmation_codebooks_observed": False,
        "pilot_codebook_seed": seed,
        "codebook_sha256": codebook.sha256,
        "dataset_sha256": dataset["sha256"],
        "model": model_id,
        "revision": revision,
        "base_parameter_sha256_fp64": tensor_sha256(theta0),
        "base_parameter_sha256_projected_fp32": projected_hash,
        "numerical_execution_fingerprint_sha256": numerical_fingerprint,
        "rule": asdict(rule),
        "chosen_learning_rate": chosen,
        "evidence": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if chosen is None:
        raise RuntimeError("no learning rate passed the frozen four-stage FP64 singleton rule")


if __name__ == "__main__":
    main()
