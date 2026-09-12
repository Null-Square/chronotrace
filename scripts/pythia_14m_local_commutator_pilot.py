#!/usr/bin/env python3
"""Run the spent-codebook Pythia-14M oriented local-commutator pilot."""

from __future__ import annotations

import argparse
import gc
import json
import math
from itertools import combinations, permutations
from pathlib import Path
from typing import Any

import numpy as np

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
from chronotrace.geometry.observability import (
    independent_probe_basis,
    minimum_distinguishing_probe_subset,
    separation_certificate,
)
from chronotrace.geometry.response_decode import (
    decode_standardized_response,
    fit_reference_standardizer,
    transform_response,
)
from chronotrace.reproducibility import json_sha256, tensor_sha256
from chronotrace.scale import build_scale_worlds_from_codebook
from chronotrace.scale_four import build_four_stage_examples, four_stage_dataset_payload
from chronotrace.scale_runner import (
    execute_plain_sgd_stage,
    flatten_parameters,
)
from chronotrace.scale_tokens import build_token_codebook, tokenizer_fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="configs/pythia_14m_local_commutator_pilot.lock.json",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def _true_precedence(history: tuple[str, ...], first: str, second: str) -> tuple[str, str]:
    if history.index(first) < history.index(second):
        return first, second
    return second, first


def _decision_row(
    prediction: tuple[str, ...],
    history: tuple[str, ...],
    stages: tuple[str, ...],
) -> dict[str, Any]:
    precedence_correct = sum(
        _true_precedence(prediction, first, second) == _true_precedence(history, first, second)
        for first, second in combinations(stages, 2)
    )
    return {
        "prediction": "".join(prediction),
        "full_correct": prediction == history,
        "prefix": {
            str(depth): prediction[:depth] == history[:depth] for depth in (1, 2, 3)
        },
        "precedence_correct": precedence_correct,
    }


def _summarize(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return {
        "full_order_correct": sum(bool(row[key]["full_correct"]) for row in rows),
        "full_order_total": len(rows),
        "prefix_correct": {
            str(depth): sum(bool(row[key]["prefix"][str(depth)]) for row in rows)
            for depth in (1, 2, 3)
        },
        "precedence_correct": sum(int(row[key]["precedence_correct"]) for row in rows),
        "precedence_total": len(rows) * math.comb(4, 2),
    }


def main() -> None:
    args = parse_args()
    protocol = _load_json(args.protocol)
    if protocol["freeze_status"] != "frozen_before_any_local_commutator_model_output":
        raise ValueError("local-commutator protocol is not cleanly frozen")
    if protocol["experiment_role"] != "non_confirmatory_methodology_pilot":
        raise ValueError("local-commutator runner is methodology-pilot only")
    if protocol.get("confirmation_codebooks_observed") is not False:
        raise RuntimeError("local-commutator protocol touched confirmation codebooks")

    k23 = _load_json(protocol["source_k23_protocol"])
    replay = _load_json(protocol["source_replay_lock"])
    if k23.get("confirmation_codebooks_observed") is not False:
        raise RuntimeError("source K2/K3 protocol touched confirmation codebooks")
    if replay.get("confirmation_codebooks_observed") is not False:
        raise RuntimeError("source replay lock touched confirmation codebooks")

    stages = tuple(str(value) for value in protocol["stages"])
    if stages != ("A", "B", "C", "D"):
        raise ValueError("local-commutator pilot requires A/B/C/D")
    pair_names = tuple(str(value) for value in protocol["challenge_pairs"])
    expected_pairs = tuple(first + second for first, second in combinations(stages, 2))
    if pair_names != expected_pairs:
        raise ValueError("local-commutator challenge-pair order drift")

    seed = int(protocol["pilot_codebook_seed"])
    prohibited = {int(value) for value in protocol["confirmation_codebooks_prohibited"]}
    if seed in prohibited:
        raise RuntimeError("local-commutator pilot may not use a held-out codebook")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    numerical_fingerprint = _configure_portable_numerics(torch, k23, int(args.threads))
    device = torch.device("cpu")
    model_id = str(protocol["model"])
    revision = str(protocol["revision"])
    learning_rate = float(protocol["learning_rate"])

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer_fingerprint(tokenizer) != str(protocol["tokenizer_fingerprint"]):
        raise RuntimeError("local-commutator tokenizer fingerprint drift")
    codebook = build_token_codebook(
        tokenizer,
        count=int(protocol["codebook_count_per_kind"]),
        seed=seed,
    )
    dataset = four_stage_dataset_payload(tokenizer, codebook)
    if codebook.sha256 != str(protocol["pilot_codebook_sha256"]):
        raise RuntimeError("local-commutator codebook hash drift")
    if dataset["sha256"] != protocol["pilot_dataset_sha256"]:
        raise RuntimeError("local-commutator dataset hash drift")
    worlds = build_scale_worlds_from_codebook(codebook)
    examples = {stage: build_four_stage_examples(worlds, stage) for stage in stages}

    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision)
    model.to(device=device, dtype=torch.float64)
    model.config.use_cache = False
    theta0 = flatten_parameters(torch, model, preserve_parameter_dtype=True)
    if theta0.dtype != torch.float64:
        raise TypeError("local-commutator base is not FP64")
    fp64_hash = tensor_sha256(theta0)
    if fp64_hash != str(protocol["base_parameter_sha256_fp64"]):
        raise RuntimeError("local-commutator FP64 base hash drift")
    projected_hash = tensor_sha256(theta0.to(dtype=torch.float32))
    if projected_hash != str(protocol["base_parameter_sha256_fp32"]):
        raise RuntimeError("local-commutator FP32 projected base hash drift")

    stage_seeds = {stage: int(protocol["stage_randomness"][stage]) for stage in stages}
    stage_calls = 0

    def make_stage_map(stage: str):
        def run(initial_vector: Any) -> Any:
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
                raise FloatingPointError(f"non-finite local-commutator stage {stage}")
            stage_calls += 1
            return endpoint

        return run

    stage_maps = {stage: make_stage_map(stage) for stage in stages}
    basis = measure_ordered_interaction_basis_compact(stage_maps, theta0, max_degree=3)
    expected_basis_calls = ordered_probe_count(len(stages), 3)
    if basis.stage_executions != expected_basis_calls:
        raise RuntimeError("local-commutator basis execution count drift")
    scorer = prepare_ordered_interaction_quadratic_scorer(basis, degrees=(3,))
    histories = tuple(permutations(stages))

    direction_metadata: dict[str, Any] = {}
    for first, second in combinations(stages, 2):
        name = first + second
        direction = basis.interactions[(first, second)] - basis.interactions[(second, first)]
        norm = float(torch.linalg.vector_norm(direction))
        if not math.isfinite(norm) or norm <= 0.0:
            raise RuntimeError(f"base commutator direction {name} is zero or non-finite")
        direct_forward = stage_maps[second](stage_maps[first](theta0))
        direct_reverse = stage_maps[first](stage_maps[second](theta0))
        direct = direct_forward - direct_reverse
        max_abs_error = float(torch.max(torch.abs(direct - direction)))
        if not torch.allclose(direct, direction, rtol=1e-10, atol=1e-12):
            raise RuntimeError(f"base commutator direction mismatch for {name}")
        direction_metadata[name] = {
            "norm": norm,
            "sha256": tensor_sha256(direction),
            "direct_spotcheck_max_abs_error": max_abs_error,
        }
        del direction
        del direct_forward
        del direct_reverse
        del direct
        gc.collect()

    def local_response(state: Any) -> np.ndarray:
        values: list[float] = []
        for first, second in combinations(stages, 2):
            forward = stage_maps[second](stage_maps[first](state))
            reverse = stage_maps[first](stage_maps[second](state))
            commutator = forward - reverse
            direction = (
                basis.interactions[(first, second)] - basis.interactions[(second, first)]
            )
            norm = float(direction_metadata[first + second]["norm"])
            projection = float(torch.dot(commutator, direction) / norm)
            if not math.isfinite(projection):
                raise FloatingPointError("non-finite local commutator projection")
            values.append(projection)
            del forward
            del reverse
            del commutator
            del direction
        return np.asarray(values, dtype=np.float64)

    reference_vectors: list[np.ndarray] = []
    candidate_rows: list[dict[str, Any]] = []
    for history in histories:
        candidate = ordered_interaction_prediction(history, basis, degree=3)
        response = local_response(candidate)
        reference_vectors.append(response)
        candidate_rows.append({"history": "".join(history), "response": response.tolist()})
        del candidate
        gc.collect()

    reference_matrix = np.stack(reference_vectors, axis=0)
    standardizer = fit_reference_standardizer(reference_matrix, minimum_scale=1e-12)
    candidate_rank = independent_probe_basis(standardizer.references)
    candidate_subset = minimum_distinguishing_probe_subset(
        standardizer.references,
        max_columns=6,
    )
    candidate_separation = separation_certificate(standardizer.references)
    active_labels = [pair_names[index] for index in standardizer.active]
    rank_labels = [active_labels[index] for index in candidate_rank.columns]
    subset_labels = [active_labels[index] for index in candidate_subset.columns]

    expected_hashes = {
        str(name): str(value) for name, value in replay["target_endpoint_hashes"].items()
    }
    if set(expected_hashes) != {"".join(history) for history in histories}:
        raise RuntimeError("local-commutator replay lock does not cover all histories")

    target_rows: list[dict[str, Any]] = []
    target_standardized: list[np.ndarray] = []
    endpoint_rows: list[dict[str, Any]] = []
    for history in histories:
        endpoint = theta0
        for stage in history:
            endpoint = stage_maps[stage](endpoint)
        truth = "".join(history)
        endpoint_hash = tensor_sha256(endpoint)
        if endpoint_hash != expected_hashes[truth]:
            raise RuntimeError(f"local-commutator endpoint replay mismatch for {truth}")

        errors = ordered_interaction_quadratic_error_tables(endpoint, basis, scorer)[3]
        endpoint_full = decode_error_table(errors)
        endpoint_prefix3 = decode_prefix_error_table(errors, depth=3)
        endpoint_precedence = 0
        for first, second in combinations(stages, 2):
            decision = decode_precedence_error_table(
                errors,
                first=first,
                second=second,
            )
            truth_pair = _true_precedence(history, first, second)
            if (decision.preferred_first, decision.preferred_second) == truth_pair:
                endpoint_precedence += 1
        endpoint_rows.append(
            {
                "truth": truth,
                "full_correct": endpoint_full.permutation == history,
                "prefix3_correct": endpoint_prefix3.prefix == history[:3],
                "precedence_correct": endpoint_precedence,
            }
        )

        response = local_response(endpoint)
        transformed = transform_response(response, standardizer)
        target_standardized.append(transformed)
        decoded = decode_standardized_response(response, standardizer)
        prediction = histories[decoded.index]
        row = _decision_row(prediction, history, stages)
        row.update(
            {
                "truth": truth,
                "endpoint_sha256": endpoint_hash,
                "response": response.tolist(),
                "standardized_response": transformed.tolist(),
                "best_distance": decoded.best_distance,
                "runner_up_distance": decoded.runner_up_distance,
                "margin": decoded.margin,
            }
        )
        target_rows.append(row)
        del endpoint
        del errors
        gc.collect()

    endpoint_summary = {
        "full_order_correct": sum(bool(row["full_correct"]) for row in endpoint_rows),
        "prefix_depth3_correct": sum(bool(row["prefix3_correct"]) for row in endpoint_rows),
        "precedence_correct": sum(int(row["precedence_correct"]) for row in endpoint_rows),
    }
    expected_endpoint = protocol["frozen_comparators"]["k3_endpoint"]
    if endpoint_summary != {
        "full_order_correct": int(expected_endpoint["full_order_correct"]),
        "prefix_depth3_correct": int(expected_endpoint["prefix_depth3_correct"]),
        "precedence_correct": int(expected_endpoint["precedence_correct"]),
    }:
        raise RuntimeError("local-commutator pilot did not reproduce frozen K3 metrics")

    response_summary = _summarize(target_rows, "response_decision") if False else {
        "full_order_correct": sum(bool(row["full_correct"]) for row in target_rows),
        "full_order_total": len(target_rows),
        "prefix_correct": {
            str(depth): sum(bool(row["prefix"][str(depth)]) for row in target_rows)
            for depth in (1, 2, 3)
        },
        "precedence_correct": sum(int(row["precedence_correct"]) for row in target_rows),
        "precedence_total": len(target_rows) * math.comb(4, 2),
        "minimum_margin": min(float(row["margin"]) for row in target_rows),
    }

    target_matrix = np.stack(target_standardized, axis=0)
    target_rank = independent_probe_basis(target_matrix)
    target_subset = minimum_distinguishing_probe_subset(target_matrix, max_columns=6)
    target_separation = separation_certificate(target_matrix)
    target_rank_labels = [active_labels[index] for index in target_rank.columns]
    target_subset_labels = [active_labels[index] for index in target_subset.columns]

    checks = {
        "commutator_full_order_strictly_beats_k3_endpoint": (
            int(response_summary["full_order_correct"])
            > int(expected_endpoint["full_order_correct"])
        ),
        "commutator_prefix_depth3_strictly_beats_k3_endpoint": (
            int(response_summary["prefix_correct"]["3"])
            > int(expected_endpoint["prefix_depth3_correct"])
        ),
        "commutator_precedence_strictly_beats_k3_endpoint": (
            int(response_summary["precedence_correct"])
            > int(expected_endpoint["precedence_correct"])
        ),
        "candidate_reference_separation_positive": candidate_separation.minimum_distance > 0.0,
        "target_response_separation_positive": target_separation.minimum_distance > 0.0,
    }

    expected_stage_calls = 40 + 24 + 24 * 24 + 24 * 4 + 24 * 24
    if stage_calls != expected_stage_calls:
        raise RuntimeError(
            f"local-commutator stage execution count drift: {stage_calls} != {expected_stage_calls}"
        )

    result = {
        "status": "complete",
        "claim": "non_confirmatory_oriented_local_commutator_pilot",
        "protocol_sha256": json_sha256(protocol),
        "source_k23_protocol_sha256": json_sha256(k23),
        "source_replay_lock_sha256": json_sha256(replay),
        "confirmation_codebooks_observed": False,
        "model": model_id,
        "revision": revision,
        "precision": "fp64",
        "learning_rate": learning_rate,
        "pilot_codebook_seed": seed,
        "base_parameter_sha256_fp64": fp64_hash,
        "base_parameter_sha256_projected_fp32": projected_hash,
        "numerical_execution_fingerprint_sha256": numerical_fingerprint,
        "stage_executions": stage_calls,
        "challenge_pairs": list(pair_names),
        "direction_metadata": direction_metadata,
        "endpoint_replay_exact": True,
        "endpoint_summary": endpoint_summary,
        "response_summary": response_summary,
        "candidate_geometry": {
            "active_coordinates": active_labels,
            "rank": candidate_rank.rank,
            "rank_basis_coordinates": rank_labels,
            "minimum_physical_probe_count": len(candidate_subset.columns),
            "minimum_physical_probe_coordinates": subset_labels,
            "indistinguishable_pairs": [list(pair) for pair in candidate_subset.full_indistinguishable_pairs],
            "minimum_standardized_separation": candidate_separation.minimum_distance,
            "half_distance_noise_certificate": candidate_separation.noise_radius,
        },
        "target_geometry_descriptive_only": {
            "rank": target_rank.rank,
            "rank_basis_coordinates": target_rank_labels,
            "minimum_physical_probe_count": len(target_subset.columns),
            "minimum_physical_probe_coordinates": target_subset_labels,
            "indistinguishable_pairs": [list(pair) for pair in target_subset.full_indistinguishable_pairs],
            "minimum_standardized_separation": target_separation.minimum_distance,
            "half_distance_noise_certificate": target_separation.noise_radius,
        },
        "primary_support_checks": checks,
        "primary_support_all": all(checks.values()),
        "candidate_references": candidate_rows,
        "histories": target_rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
