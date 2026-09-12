#!/usr/bin/env python3
"""Replay the four-stage pilot endpoints and evaluate marginal stage-loss recency."""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations, permutations
from pathlib import Path
from typing import Any

from pythia_finite_pair_bridge import _configure_portable_numerics, _load_json

from chronotrace.geometry.recency import (
    decode_stage_loss_recency,
    stage_loss_recency_precedence,
)
from chronotrace.reproducibility import json_sha256, tensor_sha256
from chronotrace.scale import build_scale_worlds_from_codebook
from chronotrace.scale_four import build_four_stage_examples, four_stage_dataset_payload
from chronotrace.scale_runner import (
    _batch_loss,
    completion_batch,
    execute_plain_sgd_stage,
    flatten_parameters,
)
from chronotrace.scale_tokens import build_token_codebook, tokenizer_fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="configs/pythia_14m_four_stage_k23_pilot.lock.json",
    )
    parser.add_argument(
        "--replay-lock",
        default="configs/pythia_14m_four_stage_k23_pilot_replay.lock.json",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def _true_precedence(history: tuple[str, ...], first: str, second: str) -> tuple[str, str]:
    if history.index(first) < history.index(second):
        return first, second
    return second, first


def main() -> None:
    args = parse_args()
    protocol = _load_json(args.protocol)
    replay = _load_json(args.replay_lock)
    if protocol["experiment_role"] != "non_confirmatory_methodology_pilot":
        raise ValueError("recency pilot must reuse the non-confirmatory K2/K3 pilot protocol")
    if protocol["precision"] != "fp64":
        raise ValueError("recency pilot is frozen to FP64")
    if replay.get("confirmation_codebooks_observed") is not False:
        raise RuntimeError("replay lock must declare confirmation_codebooks_observed=false")

    seed = int(protocol["pilot_codebook_seed"])
    if seed != int(replay["pilot_codebook_seed"]):
        raise RuntimeError("recency pilot seed differs from frozen replay seed")
    prohibited = {int(value) for value in protocol["confirmation_codebooks_prohibited"]}
    if seed in prohibited:
        raise RuntimeError("recency pilot may not use a held-out confirmation codebook")
    if protocol.get("confirmation_codebooks_observed") is not False:
        raise RuntimeError("pilot protocol must declare confirmation_codebooks_observed=false")

    calibration = _load_json(protocol["calibration_result"])
    if json_sha256(calibration) != str(protocol["calibration_result_sha256"]):
        raise RuntimeError("calibration result hash differs from frozen pilot protocol")
    learning_rate = float(calibration["chosen_learning_rate"])
    if learning_rate != float(protocol["learning_rate"]):
        raise RuntimeError("recency pilot learning rate differs from calibration")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    numerical_fingerprint = _configure_portable_numerics(torch, protocol, int(args.threads))
    device = torch.device("cpu")
    model_id = str(protocol["model"])
    revision = str(protocol["revision"])
    stages = tuple(str(value) for value in protocol["stages"])
    if stages != ("A", "B", "C", "D"):
        raise ValueError("recency pilot requires A/B/C/D")

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer_fingerprint(tokenizer) != str(protocol["tokenizer_fingerprint"]):
        raise RuntimeError("recency pilot tokenizer differs from frozen protocol")
    codebook = build_token_codebook(
        tokenizer,
        count=int(protocol["codebook_count_per_kind"]),
        seed=seed,
    )
    dataset = four_stage_dataset_payload(tokenizer, codebook)
    if codebook.sha256 != str(protocol["pilot_codebook_sha256"]):
        raise RuntimeError("recency pilot codebook differs from frozen protocol")
    if dataset["sha256"] != protocol["pilot_dataset_sha256"]:
        raise RuntimeError("recency pilot dataset differs from frozen protocol")
    worlds = build_scale_worlds_from_codebook(codebook)
    examples = {stage: build_four_stage_examples(worlds, stage) for stage in stages}
    batches = {
        stage: completion_batch(torch, tokenizer, examples[stage], device)
        for stage in stages
    }

    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision)
    model.to(device=device, dtype=torch.float64)
    model.config.use_cache = False
    if any(
        parameter.is_floating_point() and parameter.dtype != torch.float64
        for parameter in model.parameters()
    ):
        raise TypeError("could not enforce FP64 recency pilot model parameters")
    theta0 = flatten_parameters(torch, model, preserve_parameter_dtype=True)
    fp64_hash = tensor_sha256(theta0)
    if fp64_hash != str(protocol["base_parameter_sha256_fp64"]):
        raise RuntimeError("recency pilot FP64 base differs from frozen protocol")
    if fp64_hash != str(calibration["base_parameter_sha256_fp64"]):
        raise RuntimeError("recency pilot FP64 base differs from chronology-blind calibration")

    stage_seeds = {stage: int(protocol["stage_randomness"][stage]) for stage in stages}
    stage_calls = 0

    def run_stage(stage: str, initial_vector: Any) -> Any:
        nonlocal stage_calls
        torch.manual_seed(stage_seeds[stage])
        endpoint, metrics = execute_plain_sgd_stage(
            torch,
            model,
            tokenizer,
            examples[stage],
            learning_rate=learning_rate,
            updates=1,
            device=device,
            initial_vector=initial_vector,
            preserve_parameter_dtype=True,
        )
        if not metrics.finite:
            raise FloatingPointError(f"non-finite FP64 recency pilot stage {stage}")
        stage_calls += 1
        return endpoint

    expected_hashes = {
        str(name): str(value) for name, value in replay["target_endpoint_hashes"].items()
    }
    histories = tuple(permutations(stages))
    if set(expected_hashes) != {"".join(history) for history in histories}:
        raise RuntimeError("replay lock does not cover exactly all 24 histories")

    rows: list[dict[str, Any]] = []
    for history in histories:
        endpoint = theta0
        for stage in history:
            endpoint = run_stage(stage, endpoint)
        truth = "".join(history)
        endpoint_hash = tensor_sha256(endpoint)
        if endpoint_hash != expected_hashes[truth]:
            raise RuntimeError(f"recency replay endpoint hash mismatch for {truth}")

        losses = {stage: _batch_loss(model, batches[stage]) for stage in stages}
        decision = decode_stage_loss_recency(losses)
        prediction = decision.permutation
        prefix = {
            str(depth): {
                "prediction": "".join(prediction[:depth]),
                "truth": "".join(history[:depth]),
                "correct": decision.identifiable and prediction[:depth] == history[:depth],
            }
            for depth in (1, 2, 3)
        }
        precedence_decisions = stage_loss_recency_precedence(losses)
        precedence: dict[str, Any] = {}
        for first, second in combinations(stages, 2):
            predicted = precedence_decisions[(first, second)]
            true_pair = _true_precedence(history, first, second)
            precedence[first + second] = {
                "prediction": None if predicted is None else "".join(predicted),
                "truth": "".join(true_pair),
                "correct": predicted == true_pair,
                "loss_gap": abs(float(losses[first]) - float(losses[second])),
            }

        rows.append(
            {
                "truth": truth,
                "prediction": "".join(prediction),
                "identifiable": decision.identifiable,
                "full_correct": decision.identifiable and prediction == history,
                "terminal_correct": decision.identifiable and prediction[-1] == history[-1],
                "minimum_adjacent_loss_gap": decision.minimum_adjacent_loss_gap,
                "stage_losses": losses,
                "prefix": prefix,
                "precedence": precedence,
                "endpoint_sha256": endpoint_hash,
            }
        )

    if stage_calls != len(histories) * len(stages):
        raise RuntimeError("recency pilot stage execution count drift")

    summary = {
        "full_order_correct": sum(bool(row["full_correct"]) for row in rows),
        "full_order_total": len(rows),
        "prefix_correct": {
            str(depth): sum(bool(row["prefix"][str(depth)]["correct"]) for row in rows)
            for depth in (1, 2, 3)
        },
        "precedence_correct": sum(
            bool(pair["correct"])
            for row in rows
            for pair in row["precedence"].values()
        ),
        "precedence_total": len(rows) * math.comb(len(stages), 2),
        "terminal_correct": sum(bool(row["terminal_correct"]) for row in rows),
        "terminal_total": len(rows),
        "identifiable_full_order": sum(bool(row["identifiable"]) for row in rows),
        "minimum_adjacent_loss_gap": min(
            float(row["minimum_adjacent_loss_gap"]) for row in rows
        ),
    }

    result = {
        "status": "complete",
        "claim": "non_confirmatory_marginal_stage_loss_recency_baseline",
        "protocol_sha256": json_sha256(protocol),
        "replay_lock_sha256": json_sha256(replay),
        "confirmation_codebooks_observed": False,
        "model": model_id,
        "revision": revision,
        "precision": "fp64",
        "learning_rate": learning_rate,
        "pilot_codebook_seed": seed,
        "base_parameter_sha256_fp64": fp64_hash,
        "numerical_execution_fingerprint_sha256": numerical_fingerprint,
        "endpoint_replay_exact": True,
        "stage_executions": stage_calls,
        "baseline": "B2_marginal_stage_loss_recency",
        "baseline_rule": "oldest_to_newest_by_descending_final_stage_completion_loss",
        "summary": summary,
        "histories": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
