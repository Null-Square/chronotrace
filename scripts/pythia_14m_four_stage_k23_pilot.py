#!/usr/bin/env python3
"""Run the non-confirmatory Pythia-14M four-stage K2/K3 interaction pilot."""

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
    ordered_probe_count,
)
from chronotrace.reproducibility import json_sha256, tensor_sha256
from chronotrace.scale import build_scale_worlds_from_codebook
from chronotrace.scale_four import build_four_stage_examples, four_stage_dataset_payload
from chronotrace.scale_runner import execute_plain_sgd_stage, flatten_parameters
from chronotrace.scale_tokens import build_token_codebook, tokenizer_fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="configs/pythia_14m_four_stage_k23_pilot.lock.json",
    )
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


def _summarize_degree(rows: list[dict[str, Any]], degree: int) -> dict[str, Any]:
    key = f"k{degree}"
    full_correct = sum(bool(row[key]["full_correct"]) for row in rows)
    prefix_correct = {
        str(depth): sum(bool(row[key]["prefix"][str(depth)]["correct"]) for row in rows)
        for depth in (1, 2, 3)
    }
    precedence_correct = sum(
        bool(pair["correct"])
        for row in rows
        for pair in row[key]["precedence"].values()
    )
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
        "full_order_correct": full_correct,
        "full_order_total": len(rows),
        "prefix_correct": prefix_correct,
        "precedence_correct": precedence_correct,
        "precedence_total": len(rows) * math.comb(4, 2),
        "mean_full_margin": _finite_mean(full_margins),
        "minimum_full_margin": min(full_margins),
        "mean_true_candidate_error": _finite_mean(true_errors),
        "terminal_stage_strata": terminal,
    }


def _degree_row(
    errors: dict[tuple[str, ...], float],
    basis: Any,
    history: tuple[str, ...],
    *,
    degree: int,
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
    for first, second in combinations(basis.stages, 2):
        decision = decode_precedence_error_table(
            errors,
            first=first,
            second=second,
        )
        truth_first, truth_second = _true_precedence(history, first, second)
        name = first + second
        precedence[name] = {
            "prediction": decision.preferred_first + decision.preferred_second,
            "truth": truth_first + truth_second,
            "correct": (
                decision.preferred_first == truth_first
                and decision.preferred_second == truth_second
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
        "degree": degree,
    }


def main() -> None:
    args = parse_args()
    protocol = _load_json(args.protocol)
    if protocol["experiment_role"] != "non_confirmatory_methodology_pilot":
        raise ValueError("four-stage K2/K3 runner requires a non-confirmatory pilot lock")
    if protocol["precision"] != "fp64":
        raise ValueError("four-stage K2/K3 pilot is frozen to FP64")
    stages = tuple(str(value) for value in protocol["stages"])
    if stages != ("A", "B", "C", "D"):
        raise ValueError("four-stage K2/K3 pilot requires A/B/C/D")
    if tuple(int(value) for value in protocol["interaction_degrees"]) != (2, 3):
        raise ValueError("pilot must compare exactly K=2 and K=3")
    if int(protocol["updates_per_stage"]) != 1:
        raise ValueError("pilot is frozen to one update per stage")

    seed = int(protocol["pilot_codebook_seed"])
    prohibited = {int(value) for value in protocol["confirmation_codebooks_prohibited"]}
    if seed in prohibited:
        raise RuntimeError("pilot may not use a held-out confirmation codebook")
    if protocol.get("confirmation_codebooks_observed") is not False:
        raise RuntimeError("pilot lock must declare confirmation_codebooks_observed=false")

    calibration = _load_json(protocol["calibration_result"])
    if calibration.get("chronology_data_observed") is not False:
        raise RuntimeError("operating-point calibration was not chronology-blind")
    if calibration.get("confirmation_codebooks_observed") is not False:
        raise RuntimeError("operating-point calibration touched confirmation codebooks")
    calibration_sha = json_sha256(calibration)
    if calibration_sha != str(protocol["calibration_result_sha256"]):
        raise RuntimeError("calibration result hash differs from the frozen pilot lock")
    learning_rate = float(calibration["chosen_learning_rate"])
    if learning_rate != float(protocol["learning_rate"]):
        raise RuntimeError("pilot learning rate differs from chronology-blind calibration")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    numerical_fingerprint = _configure_portable_numerics(torch, protocol, int(args.threads))
    device = torch.device("cpu")
    model_id = str(protocol["model"])
    revision = str(protocol["revision"])

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer_fingerprint(tokenizer) != str(protocol["tokenizer_fingerprint"]):
        raise RuntimeError("pilot tokenizer differs from frozen protocol")
    codebook = build_token_codebook(
        tokenizer,
        count=int(protocol["codebook_count_per_kind"]),
        seed=seed,
    )
    dataset = four_stage_dataset_payload(tokenizer, codebook)
    if codebook.sha256 != str(protocol["pilot_codebook_sha256"]):
        raise RuntimeError("pilot codebook differs from frozen lock")
    if dataset["sha256"] != protocol["pilot_dataset_sha256"]:
        raise RuntimeError("pilot four-stage dataset differs from frozen lock")
    worlds = build_scale_worlds_from_codebook(codebook)
    examples = {stage: build_four_stage_examples(worlds, stage) for stage in stages}

    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision)
    model.to(device=device, dtype=torch.float64)
    model.config.use_cache = False
    if any(
        parameter.is_floating_point() and parameter.dtype != torch.float64
        for parameter in model.parameters()
    ):
        raise TypeError("could not enforce FP64 pilot model parameters")
    theta0 = flatten_parameters(torch, model, preserve_parameter_dtype=True)
    if theta0.dtype != torch.float64:
        raise TypeError("FP64 pilot base vector dtype drift")
    fp64_hash = tensor_sha256(theta0)
    expected_fp64_hash = str(protocol["base_parameter_sha256_fp64"])
    if fp64_hash != expected_fp64_hash:
        raise RuntimeError("pilot FP64 base differs from the frozen pilot lock")
    if fp64_hash != str(calibration["base_parameter_sha256_fp64"]):
        raise RuntimeError("pilot FP64 base differs from chronology-blind calibration")
    projected_hash = tensor_sha256(theta0.to(dtype=torch.float32))
    if projected_hash != str(protocol["base_parameter_sha256_fp32"]):
        raise RuntimeError("pilot base checkpoint differs from frozen Pythia weights")

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
                raise FloatingPointError(f"non-finite FP64 pilot stage {stage}")
            stage_calls[stage] += 1
            return endpoint

        return run

    stage_maps = {stage: make_stage_map(stage) for stage in stages}
    basis = measure_ordered_interaction_basis_compact(
        stage_maps,
        theta0,
        max_degree=3,
    )
    expected_basis_calls = ordered_probe_count(len(stages), 3)
    if basis.stage_executions != expected_basis_calls:
        raise RuntimeError("unexpected four-stage K3 basis probe count")
    if len(basis.interactions) != expected_basis_calls:
        raise RuntimeError("four-stage K3 basis interaction count mismatch")
    if any(value.dtype != torch.float64 for value in basis.interactions.values()):
        raise TypeError("four-stage interaction basis contains non-FP64 tensors")

    scorer = prepare_ordered_interaction_quadratic_scorer(
        basis,
        degrees=(2, 3),
    )
    interaction_hashes = {
        "".join(word): tensor_sha256(value)
        for word, value in sorted(basis.interactions.items())
    }
    rows: list[dict[str, Any]] = []
    target_hashes: dict[str, str] = {}
    histories = tuple(permutations(stages))
    for history in histories:
        endpoint = theta0
        for stage in history:
            endpoint = stage_maps[stage](endpoint)
        truth = "".join(history)
        target_hashes[truth] = tensor_sha256(endpoint)
        error_tables = ordered_interaction_quadratic_error_tables(endpoint, basis, scorer)
        row = {
            "truth": truth,
            "k2": _degree_row(error_tables[2], basis, history, degree=2),
            "k3": _degree_row(error_tables[3], basis, history, degree=3),
        }
        rows.append(row)
        del error_tables
        del endpoint
        gc.collect()

    expected_target_calls = len(histories) * len(stages)
    if sum(stage_calls.values()) != expected_basis_calls + expected_target_calls:
        raise RuntimeError("four-stage pilot stage execution count drift")

    k2 = _summarize_degree(rows, 2)
    k3 = _summarize_degree(rows, 3)
    comparison = {
        "full_order_k2_wrong_k3_correct": sum(
            (not bool(row["k2"]["full_correct"])) and bool(row["k3"]["full_correct"])
            for row in rows
        ),
        "full_order_k2_correct_k3_wrong": sum(
            bool(row["k2"]["full_correct"]) and (not bool(row["k3"]["full_correct"]))
            for row in rows
        ),
        "k3_lower_true_candidate_error": sum(
            float(row["k3"]["true_candidate_error"])
            < float(row["k2"]["true_candidate_error"])
            for row in rows
        ),
        "mean_true_error_ratio_k3_over_k2": _finite_mean(
            [
                float(row["k3"]["true_candidate_error"])
                / float(row["k2"]["true_candidate_error"])
                for row in rows
                if float(row["k2"]["true_candidate_error"]) > 0.0
            ]
        ),
    }

    result = {
        "status": "complete",
        "claim": "non_confirmatory_four_stage_fp64_k2_k3_methodology_pilot",
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": json_sha256(protocol),
        "experiment_role": protocol["experiment_role"],
        "confirmation_codebooks_observed": False,
        "candidate_scoring": "exact_quadratic_form_equivalent_to_full_parameter_l2",
        "model": model_id,
        "revision": revision,
        "precision": "fp64",
        "learning_rate": learning_rate,
        "updates_per_stage": 1,
        "pilot_codebook_seed": seed,
        "codebook_sha256": codebook.sha256,
        "dataset_sha256": dataset["sha256"],
        "base_parameter_sha256_fp64": fp64_hash,
        "base_parameter_sha256_projected_fp32": projected_hash,
        "numerical_execution_fingerprint_sha256": numerical_fingerprint,
        "basis_stage_executions": expected_basis_calls,
        "validation_stage_executions": expected_target_calls,
        "total_stage_executions": expected_basis_calls + expected_target_calls,
        "interaction_hashes": interaction_hashes,
        "target_endpoint_hashes": target_hashes,
        "summary": {"k2": k2, "k3": k3, "comparison": comparison},
        "histories": rows,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
