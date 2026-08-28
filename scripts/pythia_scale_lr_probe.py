#!/usr/bin/env python3
"""Run one chronology-blind singleton learning-rate probe on a Pythia checkpoint."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from chronotrace.scale import (
    StabilityMetric,
    build_scale_stage_examples,
    build_scale_worlds_from_codebook,
    scale_dataset_payload,
)
from chronotrace.scale_runner import execute_plain_sgd_stage
from chronotrace.scale_tokens import build_token_codebook, codebook_to_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pythia_scale.yaml")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--learning-rate", required=True, type=float)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    candidates = [float(value) for value in config["learning_rate_candidates"]]
    if args.learning_rate not in candidates:
        raise ValueError("learning rate is not declared in the frozen candidate set")
    model_ids = [str(item["id"]) for item in config["models"]]
    if args.model_id not in model_ids:
        raise ValueError("model is not declared in the frozen scale ladder")

    # Tokenizer/codebook verification intentionally happens before model weights are loaded.
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    revision = str(config["revision"])
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    codebook = build_token_codebook(
        tokenizer,
        count=int(config["world_count"]),
        seed=int(config["codebook_seed"]),
    )
    dataset = scale_dataset_payload(codebook)
    worlds = build_scale_worlds_from_codebook(codebook)
    stage_name = str(config["stability"]["stage"])
    examples = build_scale_stage_examples(worlds, stage_name)

    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, revision=revision)
    model.to(device=device, dtype=torch.float32)
    if any(
        parameter.is_floating_point() and parameter.dtype != torch.float32
        for parameter in model.parameters()
    ):
        raise TypeError("scale LR probe requires FP32 model parameters")

    _, run = execute_plain_sgd_stage(
        torch,
        model,
        tokenizer,
        examples,
        learning_rate=float(args.learning_rate),
        updates=int(config["stability"]["updates"]),
        device=device,
    )
    metric = StabilityMetric(
        model_id=args.model_id,
        learning_rate=float(args.learning_rate),
        initial_loss=run.initial_loss,
        final_loss=run.final_loss,
        relative_displacement=run.relative_displacement,
        max_gradient_norm=run.max_gradient_norm,
        finite=run.finite,
    )
    payload = {
        "protocol_version": config["protocol_version"],
        "revision": revision,
        "model_id": args.model_id,
        "learning_rate": float(args.learning_rate),
        "metric": asdict(metric),
        "loss_ratio": metric.loss_ratio,
        "tokenizer_fingerprint": codebook.tokenizer_fingerprint,
        "codebook_sha256": codebook.sha256,
        "dataset_sha256": dataset["sha256"],
        "codebook": json.loads(codebook_to_json(codebook)),
        "chronology_data_observed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "codebook"}, indent=2))


if __name__ == "__main__":
    main()
