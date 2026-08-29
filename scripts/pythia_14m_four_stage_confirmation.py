#!/usr/bin/env python3
"""Run one frozen held-out Pythia-14M four-stage confirmation codebook."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
from itertools import combinations, permutations
from pathlib import Path
from typing import Any

from pythia_finite_pair_bridge import _configure_portable_numerics, _load_json

from chronotrace.geometry.error_table import (
    decode_error_table,
    decode_precedence_error_table,
    decode_prefix_error_table,
    ordered_interaction_quadratic_error_tables,
    prepare_ordered_interaction_quadratic_scorer,
)
from chronotrace.geometry.interactions import (
    measure_ordered_interaction_basis_compact,
    ordered_interaction_prediction,
    ordered_probe_count,
)
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
        default="configs/pythia_14m_four_stage_confirmation.lock.json",
    )
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def _finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.mean(finite) if finite else float("nan")


def _true_precedence(history: tuple[str, ...], first: str, second: str) -> tuple[str, str]:
    if history.index(first) < history.index(second):
        return first, second
    return second, first


def _degree_row(
    errors: dict[tuple[str, ...], float],
    stages: tuple[str, ...],
    history: tuple[str, ...],
) -> dict[str, Any]:
    full = decode_error_table(errors)
    prefix: dict[str, Any] = {}
    for depth in (1, 2, 3):
        decision = decode_prefix_error_table(errors, depth=depth)
        truth = history[:depth]
        prefix[str(depth)] = {
            "prediction": "".join(decision.prefix),
            "truth": "".join(truth),
            "correct": decision.prefix == truth,
            "best_error": decision.best_error,
            "margin": decision.margin,
        }

    precedence: dict[str, Any] = {}
    for first, second in combinations(stages, 2):
        decision = decode_precedence_error_table(errors, first=first, second=second)
        truth_pair = _true_precedence(history, first, second)
        precedence[first + second] = {
            "prediction": decision.preferred_first + decision.preferred_second,
            "truth": "".join(truth_pair),
            "correct": (
                decision.preferred_first == truth_pair[0]
                and decision.preferred_second == truth_pair[1]
            ),
            "preferred_error": decision.preferred_error,
            "margin": decision.margin,
        }

    return {
        "prediction": "".join(full.permutation),
        "full_correct": full.permutation == history,
        "best_error": full.best_error,
        "runner_up_error": full.runner_up_error,
        "full_margin": full.margin,
        "true_candidate_error": float(errors[history]),
        "prefix": prefix,
        "precedence": precedence,
    }


def _summarize_degree(rows: list[dict[str, Any]], degree: int) -> dict[str, Any]:
    key = f"k{degree}"
    full_margins = [float(row[key]["full_margin"]) for row in rows]
    true_errors = [float(row[key]["true_candidate_error"]) for row in rows]
    terminal: dict[str, dict[str, int]] = {}
    for stage in sorted({row["truth"][-1] for row in rows}):
        selected = [row for row in rows if row["truth"][-1] == stage]
        terminal[stage] = {
            "correct": sum(bool(row[key]["full_correct"]) for row in selected),
            "total": len(selected),
        }
    return {
        "degree": degree,
        "full_order_correct": sum(bool(row[key]["full_correct"]) for row in rows),
        "full_order_total": len(rows),
        "prefix_correct": {
            str(depth): sum(bool(row[key]["prefix"][str(depth)]["correct"]) for row in rows)
            for depth in (1, 2, 3)
        },
        "precedence_correct": sum(
            bool(pair["correct"])
            for row in rows
            for pair in row[key]["precedence"].values()
        ),
        "precedence_total": len(rows) * math.comb(4, 2),
        "mean_full_margin": _finite_mean(full_margins),
        "minimum_full_margin": min(full_margins),
        "mean_true_candidate_error": _finite_mean(true_errors),
        "terminal_stage_strata": terminal,
    }


def _recency_row(
    losses: dict[str, float],
    stages: tuple[str, ...],
    history: tuple[str, ...],
) -> dict[str, Any]:
    decision = decode_stage_loss_recency(losses)
    prediction = decision.permutation
    pair_decisions = stage_loss_recency_precedence(losses)
    precedence: dict[str, Any] = {}
    for first, second in combinations(stages, 2):
        predicted = pair_decisions[(first, second)]
        truth_pair = _true_precedence(history, first, second)
        precedence[first + second] = {
            "prediction": None if predicted is None else "".join(predicted),
            "truth": "".join(truth_pair),
            "correct": predicted == truth_pair,
            "loss_gap": abs(float(losses[first]) - float(losses[second])),
        }
    return {
        "prediction": "".join(prediction),
        "identifiable": decision.identifiable,
        "full_correct": decision.identifiable and prediction == history,
        "terminal_correct": decision.identifiable and prediction[-1] == history[-1],
        "minimum_adjacent_loss_gap": decision.minimum_adjacent_loss_gap,
        "stage_losses": losses,
        "prefix": {
            str(depth): {
                "prediction": "".join(prediction[:depth]),
                "truth": "".join(history[:depth]),
                "correct": decision.identifiable and prediction[:depth] == history[:depth],
            }
            for depth in (1, 2, 3)
        },
        "precedence": precedence,
    }


def _summarize_recency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "full_order_correct": sum(bool(row["recency"]["full_correct"]) for row in rows),
        "full_order_total": len(rows),
        "prefix_correct": {
            str(depth): sum(
                bool(row["recency"]["prefix"][str(depth)]["correct"]) for row in rows
            )
            for depth in (1, 2, 3)
        },
        "precedence_correct": sum(
            bool(pair["correct"])
            for row in rows
            for pair in row["recency"]["precedence"].values()
        ),
        "precedence_total": len(rows) * math.comb(4, 2),
        "terminal_correct": sum(bool(row["recency"]["terminal_correct"]) for row in rows),
        "terminal_total": len(rows),
        "identifiable_full_order": sum(bool(row["recency"]["identifiable"]) for row in rows),
        "minimum_adjacent_loss_gap": min(
            float(row["recency"]["minimum_adjacent_loss_gap"]) for row in rows
        ),
    }


def main() -> None:
    args = parse_args()
    protocol = _load_json(args.protocol)
    if protocol["freeze_status"] != "frozen_before_any_confirmation_model_output":
        raise ValueError("confirmation protocol must be frozen before model output")
    if protocol.get("confirmation_model_outputs_observed") is not False:
        raise RuntimeError("confirmation protocol already declares observed outputs")
    if protocol["precision"] != "fp64":
        raise ValueError("confirmation is frozen to FP64")
    stages = tuple(str(stage) for stage in protocol["stages"])
    if stages != ("A", "B", "C", "D"):
        raise ValueError("confirmation requires exactly A/B/C/D")
    if tuple(int(value) for value in protocol["interaction_degrees"]) != (1, 2, 3):
        raise ValueError("confirmation must report K1/K2/K3")
    if int(protocol["updates_per_stage"]) != 1:
        raise ValueError("confirmation is frozen to one update per stage")

    codebooks = list(protocol["confirmation_codebooks"])
    if args.index < 0 or args.index >= len(codebooks):
        raise ValueError("confirmation codebook index out of range")
    if len(codebooks) != 4 or {int(item["index"]) for item in codebooks} != {0, 1, 2, 3}:
        raise RuntimeError("confirmation protocol must contain exactly four frozen codebooks")
    entry = dict(codebooks[args.index])
    if int(entry["index"]) != args.index:
        raise RuntimeError("confirmation codebook ordering drift")
    seed = int(entry["seed"])

    pilot_selection = _load_json(protocol["source_pilot_selection"])
    if pilot_selection.get("hierarchy_support_all") is not True:
        raise RuntimeError("pilot selection did not authorize confirmation")
    if pilot_selection.get("confirmation_codebooks_observed") is not False:
        raise RuntimeError("pilot selection declares confirmation codebooks observed")
    interpretation = _load_json(protocol["interpretation_protocol"])
    if interpretation.get("confirmation_codebooks_observed") is not False:
        raise RuntimeError("confirmation interpretation was not frozen cleanly")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    numerical_fingerprint = _configure_portable_numerics(torch, protocol, int(args.threads))
    device = torch.device("cpu")
    model_id = str(protocol["model"])
    revision = str(protocol["revision"])
    learning_rate = float(protocol["learning_rate"])

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer_fingerprint(tokenizer) != str(protocol["tokenizer_fingerprint"]):
        raise RuntimeError("confirmation tokenizer differs from frozen protocol")
    codebook = build_token_codebook(
        tokenizer,
        count=int(protocol["codebook_count_per_kind"]),
        seed=seed,
    )
    dataset = four_stage_dataset_payload(tokenizer, codebook)
    if codebook.sha256 != str(entry["codebook_sha256"]):
        raise RuntimeError("confirmation codebook differs from frozen hash")
    if dataset["sha256"] != entry["dataset_sha256"]:
        raise RuntimeError("confirmation dataset differs from frozen hashes")
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
        raise TypeError("could not enforce FP64 confirmation model parameters")
    theta0 = flatten_parameters(torch, model, preserve_parameter_dtype=True)
    if theta0.dtype != torch.float64:
        raise TypeError("confirmation base vector dtype drift")
    fp64_hash = tensor_sha256(theta0)
    if fp64_hash != str(protocol["base_parameter_sha256_fp64"]):
        raise RuntimeError("confirmation FP64 base differs from frozen hash")
    projected_hash = tensor_sha256(theta0.to(dtype=torch.float32))
    if projected_hash != str(protocol["base_parameter_sha256_fp32"]):
        raise RuntimeError("confirmation FP32 projection differs from frozen hash")

    stage_seeds = {stage: int(protocol["stage_randomness"][stage]) for stage in stages}
    stage_calls = {stage: 0 for stage in stages}

    def make_stage_map(stage: str):
        def run(initial_vector: Any) -> Any:
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
                raise FloatingPointError(f"non-finite confirmation stage {stage}")
            stage_calls[stage] += 1
            return endpoint

        return run

    stage_maps = {stage: make_stage_map(stage) for stage in stages}
    basis = measure_ordered_interaction_basis_compact(stage_maps, theta0, max_degree=3)
    expected_basis_calls = ordered_probe_count(len(stages), 3)
    if expected_basis_calls != int(protocol["basis_stage_executions"]):
        raise RuntimeError("frozen confirmation basis probe count drift")
    if basis.stage_executions != expected_basis_calls:
        raise RuntimeError("observed confirmation basis probe count drift")
    if len(basis.interactions) != expected_basis_calls:
        raise RuntimeError("confirmation interaction count mismatch")
    if any(value.dtype != torch.float64 for value in basis.interactions.values()):
        raise TypeError("confirmation interaction basis contains non-FP64 tensor")

    scorer = prepare_ordered_interaction_quadratic_scorer(basis, degrees=(2, 3))
    histories = tuple(permutations(stages))
    rows: list[dict[str, Any]] = []
    target_hashes: dict[str, str] = {}
    interaction_hashes = {
        "".join(word): tensor_sha256(value)
        for word, value in sorted(basis.interactions.items())
    }
    spotcheck_errors: list[float] = []
    spotcheck_count = 0

    for history_index, history in enumerate(histories):
        endpoint = theta0
        for stage in history:
            endpoint = stage_maps[stage](endpoint)
        truth = "".join(history)
        target_hashes[truth] = tensor_sha256(endpoint)
        error_tables = ordered_interaction_quadratic_error_tables(endpoint, basis, scorer)

        if history_index == 0:
            for degree in (2, 3):
                predicted = decode_error_table(error_tables[degree]).permutation
                for candidate in {history, predicted}:
                    prediction = ordered_interaction_prediction(candidate, basis, degree=degree)
                    direct_error = float(torch.linalg.vector_norm(endpoint - prediction))
                    fast_error = float(error_tables[degree][candidate])
                    difference = abs(direct_error - fast_error)
                    spotcheck_errors.append(difference)
                    spotcheck_count += 1
                    if not math.isclose(direct_error, fast_error, rel_tol=1e-8, abs_tol=1e-12):
                        raise FloatingPointError(
                            "quadratic confirmation scorer failed direct L2 spot check: "
                            f"degree={degree}, candidate={candidate!r}, "
                            f"direct={direct_error}, quadratic={fast_error}"
                        )
                    del prediction

        losses = {stage: _batch_loss(model, batches[stage]) for stage in stages}
        rows.append(
            {
                "truth": truth,
                "k2": _degree_row(error_tables[2], stages, history),
                "k3": _degree_row(error_tables[3], stages, history),
                "recency": _recency_row(losses, stages, history),
            }
        )
        del error_tables
        del endpoint
        gc.collect()

    expected_validation_calls = len(histories) * len(stages)
    if expected_validation_calls != int(protocol["validation_stage_executions"]):
        raise RuntimeError("frozen confirmation validation probe count drift")
    expected_total_calls = expected_basis_calls + expected_validation_calls
    if expected_total_calls != int(protocol["total_stage_executions_per_seed"]):
        raise RuntimeError("frozen confirmation total probe count drift")
    if sum(stage_calls.values()) != expected_total_calls:
        raise RuntimeError("observed confirmation stage execution count drift")

    k2 = _summarize_degree(rows, 2)
    k3 = _summarize_degree(rows, 3)
    recency = _summarize_recency(rows)
    repairs = sum(
        (not bool(row["k2"]["full_correct"])) and bool(row["k3"]["full_correct"])
        for row in rows
    )
    regressions = sum(
        bool(row["k2"]["full_correct"]) and (not bool(row["k3"]["full_correct"]))
        for row in rows
    )
    true_error_ratios = [
        float(row["k3"]["true_candidate_error"]) / float(row["k2"]["true_candidate_error"])
        for row in rows
        if float(row["k2"]["true_candidate_error"]) > 0.0
    ]
    mean_ratio = _finite_mean(true_error_ratios)
    checks = {
        "k3_full_order_strictly_better": int(k3["full_order_correct"]) > int(k2["full_order_correct"]),
        "more_repairs_than_regressions": repairs > regressions,
        "mean_true_error_ratio_below_one": mean_ratio < 1.0,
        "depth3_prefix_not_worse": int(k3["prefix_correct"]["3"]) >= int(k2["prefix_correct"]["3"]),
        "precedence_not_worse": int(k3["precedence_correct"]) >= int(k2["precedence_correct"]),
    }

    result = {
        "status": "complete",
        "claim": "held_out_four_stage_fp64_k2_k3_confirmation_seed",
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": json_sha256(protocol),
        "confirmation_index": args.index,
        "confirmation_seed": seed,
        "confirmation_model_output": True,
        "model": model_id,
        "revision": revision,
        "precision": "fp64",
        "learning_rate": learning_rate,
        "codebook_sha256": codebook.sha256,
        "dataset_sha256": dataset["sha256"],
        "base_parameter_sha256_fp64": fp64_hash,
        "base_parameter_sha256_projected_fp32": projected_hash,
        "numerical_execution_fingerprint_sha256": numerical_fingerprint,
        "candidate_scoring": protocol["candidate_scoring"],
        "quadratic_spotcheck_count": spotcheck_count,
        "quadratic_spotcheck_max_abs_error": max(spotcheck_errors, default=0.0),
        "basis_stage_executions": expected_basis_calls,
        "validation_stage_executions": expected_validation_calls,
        "total_stage_executions": expected_total_calls,
        "interaction_hashes": interaction_hashes,
        "target_endpoint_hashes": target_hashes,
        "k1_control": {
            "identifiable": False,
            "reason": "singleton-only predicted endpoint is invariant to permutation",
            "candidate_count": len(histories),
            "chance_full_order_probability": 1.0 / len(histories),
        },
        "summary": {
            "k2": k2,
            "k3": k3,
            "recency": recency,
            "comparison": {
                "full_order_k2_wrong_k3_correct": repairs,
                "full_order_k2_correct_k3_wrong": regressions,
                "k3_lower_true_candidate_error": sum(
                    float(row["k3"]["true_candidate_error"])
                    < float(row["k2"]["true_candidate_error"])
                    for row in rows
                ),
                "mean_true_error_ratio_k3_over_k2": mean_ratio,
            },
            "per_seed_hierarchy_checks": checks,
            "per_seed_hierarchy_support_all": all(checks.values()),
        },
        "histories": rows,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
