#!/usr/bin/env python3
"""Test infinitesimal learning-response chronology on the spent Pythia-14M pilot."""

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
    separation_certificate,
)
from chronotrace.geometry.response_decode import (
    decode_standardized_response,
    fit_reference_standardizer,
)
from chronotrace.reproducibility import json_sha256, tensor_sha256
from chronotrace.scale import build_scale_worlds_from_codebook
from chronotrace.scale_four import build_four_stage_examples, four_stage_dataset_payload
from chronotrace.scale_runner import (
    completion_batch,
    execute_plain_sgd_stage,
    flatten_parameters,
    load_flat_parameters,
)
from chronotrace.scale_tokens import build_token_codebook, tokenizer_fingerprint


FAMILIES = (
    "loss_only",
    "self_susceptibility_only",
    "cross_susceptibility_only",
    "loss_plus_cross_susceptibility",
    "full_loss_plus_susceptibility",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="configs/pythia_14m_counterfactual_response_pilot.lock.json",
    )
    parser.add_argument(
        "--k23-protocol",
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


def _flatten_gradients(torch: Any, model: Any) -> Any:
    parts = []
    for parameter in model.parameters():
        if parameter.grad is None:
            parts.append(torch.zeros_like(parameter.detach()).reshape(-1).cpu())
        else:
            parts.append(parameter.grad.detach().reshape(-1).cpu())
    if not parts:
        raise RuntimeError("model has no parameters while measuring susceptibility")
    return torch.cat(parts)


def _state_response(
    torch: Any,
    model: Any,
    batches: dict[str, dict[str, Any]],
    stages: tuple[str, ...],
    state: Any,
    *,
    device: Any,
) -> dict[str, Any]:
    load_flat_parameters(torch, model, state, device=device)
    model.eval()
    losses: dict[str, float] = {}
    gradients: dict[str, Any] = {}
    for stage in stages:
        model.zero_grad(set_to_none=True)
        output = model(**batches[stage])
        loss = float(output.loss.detach().cpu())
        if not math.isfinite(loss):
            raise FloatingPointError(f"non-finite response loss for stage {stage}")
        output.loss.backward()
        gradient = _flatten_gradients(torch, model)
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError(f"non-finite response gradient for stage {stage}")
        losses[stage] = loss
        gradients[stage] = gradient

    susceptibility: dict[str, float] = {}
    for index, first in enumerate(stages):
        for second in stages[index:]:
            value = float(
                torch.dot(
                    gradients[first].reshape(-1),
                    gradients[second].reshape(-1),
                )
            )
            if not math.isfinite(value):
                raise FloatingPointError(f"non-finite susceptibility {first}{second}")
            susceptibility[first + second] = value
    gradient_norms = {
        stage: math.sqrt(max(0.0, susceptibility[stage + stage])) for stage in stages
    }
    del gradients
    model.zero_grad(set_to_none=True)
    return {
        "losses": losses,
        "susceptibility": susceptibility,
        "gradient_norms": gradient_norms,
    }


def _family_labels(stages: tuple[str, ...], family: str) -> tuple[str, ...]:
    loss_labels = tuple(f"loss_{stage}" for stage in stages)
    self_labels = tuple(f"S_{stage}{stage}" for stage in stages)
    cross_labels = tuple(f"S_{first}{second}" for first, second in combinations(stages, 2))
    if family == "loss_only":
        return loss_labels
    if family == "self_susceptibility_only":
        return self_labels
    if family == "cross_susceptibility_only":
        return cross_labels
    if family == "loss_plus_cross_susceptibility":
        return loss_labels + cross_labels
    if family == "full_loss_plus_susceptibility":
        upper = tuple(
            f"S_{first}{second}"
            for index, first in enumerate(stages)
            for second in stages[index:]
        )
        return loss_labels + upper
    raise ValueError(f"unknown response family {family!r}")


def _family_vector(
    response: dict[str, Any],
    stages: tuple[str, ...],
    family: str,
) -> np.ndarray:
    values: list[float] = []
    labels = _family_labels(stages, family)
    for label in labels:
        if label.startswith("loss_"):
            values.append(float(response["losses"][label[-1]]))
        else:
            values.append(float(response["susceptibility"][label[2:]]))
    return np.asarray(values, dtype=np.float64)


def _endpoint_decision_row(
    errors: dict[tuple[str, ...], float],
    stages: tuple[str, ...],
    history: tuple[str, ...],
) -> dict[str, Any]:
    full = decode_error_table(errors)
    prefix3 = decode_prefix_error_table(errors, depth=3)
    precedence_correct = 0
    for first, second in combinations(stages, 2):
        decision = decode_precedence_error_table(errors, first=first, second=second)
        truth = _true_precedence(history, first, second)
        if (decision.preferred_first, decision.preferred_second) == truth:
            precedence_correct += 1
    return {
        "prediction": "".join(full.permutation),
        "full_correct": full.permutation == history,
        "prefix3_correct": prefix3.prefix == history[:3],
        "precedence_correct": precedence_correct,
        "true_candidate_error": float(errors[history]),
    }


def _response_decision_row(
    prediction: tuple[str, ...],
    history: tuple[str, ...],
) -> dict[str, Any]:
    precedence_correct = sum(
        _true_precedence(prediction, first, second) == _true_precedence(history, first, second)
        for first, second in combinations(history, 2)
    )
    return {
        "prediction": "".join(prediction),
        "full_correct": prediction == history,
        "prefix3_correct": prediction[:3] == history[:3],
        "precedence_correct": precedence_correct,
    }


def main() -> None:
    args = parse_args()
    protocol = _load_json(args.protocol)
    k23 = _load_json(args.k23_protocol)
    replay = _load_json(args.replay_lock)
    if protocol["freeze_status"] != "frozen_before_any_susceptibility_output":
        raise ValueError("response pilot must be frozen before susceptibility outputs")
    if protocol["experiment_role"] != "non_confirmatory_methodology_pilot":
        raise ValueError("response runner is methodology-pilot only")
    if protocol.get("confirmation_codebooks_observed") is not False:
        raise RuntimeError("response protocol declares confirmation codebooks observed")
    if k23.get("confirmation_codebooks_observed") is not False:
        raise RuntimeError("source K2/K3 protocol touched confirmation codebooks")
    if replay.get("confirmation_codebooks_observed") is not False:
        raise RuntimeError("source replay lock touched confirmation codebooks")
    if json_sha256(k23) != str(protocol["source_k23_protocol_sha256"]):
        raise RuntimeError("source K2/K3 protocol hash drift")

    seed = int(protocol["pilot_codebook_seed"])
    prohibited = {int(value) for value in protocol["confirmation_codebooks_prohibited"]}
    if seed in prohibited:
        raise RuntimeError("response pilot may not use a held-out codebook")
    stages = tuple(str(value) for value in protocol["stages"])
    if stages != ("A", "B", "C", "D"):
        raise ValueError("response pilot requires A/B/C/D")

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
        raise RuntimeError("response pilot tokenizer fingerprint drift")
    codebook = build_token_codebook(
        tokenizer,
        count=int(protocol["codebook_count_per_kind"]),
        seed=seed,
    )
    dataset = four_stage_dataset_payload(tokenizer, codebook)
    if codebook.sha256 != str(protocol["pilot_codebook_sha256"]):
        raise RuntimeError("response pilot codebook hash drift")
    if dataset["sha256"] != protocol["pilot_dataset_sha256"]:
        raise RuntimeError("response pilot dataset hash drift")
    worlds = build_scale_worlds_from_codebook(codebook)
    examples = {stage: build_four_stage_examples(worlds, stage) for stage in stages}
    batches = {
        stage: completion_batch(torch, tokenizer, examples[stage], device)
        for stage in stages
    }

    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision)
    model.to(device=device, dtype=torch.float64)
    model.config.use_cache = False
    theta0 = flatten_parameters(torch, model, preserve_parameter_dtype=True)
    if theta0.dtype != torch.float64:
        raise TypeError("response pilot base is not FP64")
    if tensor_sha256(theta0) != str(protocol["base_parameter_sha256_fp64"]):
        raise RuntimeError("response pilot FP64 base hash drift")
    if tensor_sha256(theta0.to(dtype=torch.float32)) != str(
        protocol["base_parameter_sha256_fp32"]
    ):
        raise RuntimeError("response pilot FP32 projected base hash drift")

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
                raise FloatingPointError(f"non-finite response pilot stage {stage}")
            stage_calls += 1
            return endpoint

        return run

    stage_maps = {stage: make_stage_map(stage) for stage in stages}
    basis = measure_ordered_interaction_basis_compact(stage_maps, theta0, max_degree=3)
    expected_basis_calls = ordered_probe_count(len(stages), 3)
    if basis.stage_executions != expected_basis_calls:
        raise RuntimeError("response pilot basis execution count drift")
    scorer = prepare_ordered_interaction_quadratic_scorer(basis, degrees=(2, 3))
    histories = tuple(permutations(stages))

    references: dict[int, dict[str, list[np.ndarray]]] = {
        degree: {family: [] for family in FAMILIES} for degree in (2, 3)
    }
    for degree in (2, 3):
        for history in histories:
            prediction = ordered_interaction_prediction(history, basis, degree=degree)
            response = _state_response(
                torch,
                model,
                batches,
                stages,
                prediction,
                device=device,
            )
            for family in FAMILIES:
                references[degree][family].append(_family_vector(response, stages, family))
            del prediction
            del response
            gc.collect()

    standardizers: dict[int, dict[str, Any]] = {2: {}, 3: {}}
    reference_geometry: dict[str, Any] = {}
    for degree in (2, 3):
        reference_geometry[f"k{degree}"] = {}
        for family in FAMILIES:
            matrix = np.stack(references[degree][family], axis=0)
            standardizer = fit_reference_standardizer(matrix, minimum_scale=1e-12)
            standardizers[degree][family] = standardizer
            basis_columns = independent_probe_basis(standardizer.references)
            certificate = separation_certificate(standardizer.references)
            labels = _family_labels(stages, family)
            active_labels = [labels[index] for index in standardizer.active]
            selected_labels = [active_labels[index] for index in basis_columns.columns]
            reference_geometry[f"k{degree}"][family] = {
                "original_coordinate_count": len(labels),
                "active_coordinate_count": len(standardizer.active),
                "rank": basis_columns.rank,
                "rank_basis_coordinates": selected_labels,
                "minimum_standardized_candidate_separation": certificate.minimum_distance,
                "half_distance_noise_certificate": certificate.noise_radius,
            }

    expected_hashes = {
        str(name): str(value) for name, value in replay["target_endpoint_hashes"].items()
    }
    rows: list[dict[str, Any]] = []
    for history in histories:
        endpoint = theta0
        for stage in history:
            endpoint = stage_maps[stage](endpoint)
        truth = "".join(history)
        endpoint_hash = tensor_sha256(endpoint)
        if endpoint_hash != expected_hashes[truth]:
            raise RuntimeError(f"response pilot endpoint replay mismatch for {truth}")
        response = _state_response(
            torch,
            model,
            batches,
            stages,
            endpoint,
            device=device,
        )
        error_tables = ordered_interaction_quadratic_error_tables(endpoint, basis, scorer)
        row: dict[str, Any] = {
            "truth": truth,
            "endpoint_sha256": endpoint_hash,
            "endpoint": {
                "k2": _endpoint_decision_row(error_tables[2], stages, history),
                "k3": _endpoint_decision_row(error_tables[3], stages, history),
            },
            "response": {"k2": {}, "k3": {}},
        }
        for degree in (2, 3):
            for family in FAMILIES:
                vector = _family_vector(response, stages, family)
                decision = decode_standardized_response(
                    vector,
                    standardizers[degree][family],
                )
                prediction = histories[decision.index]
                result = _response_decision_row(prediction, history)
                result.update(
                    {
                        "best_distance": decision.best_distance,
                        "runner_up_distance": decision.runner_up_distance,
                        "margin": decision.margin,
                    }
                )
                row["response"][f"k{degree}"][family] = result
        rows.append(row)
        del response
        del error_tables
        del endpoint
        gc.collect()

    expected_stage_calls = expected_basis_calls + len(histories) * len(stages)
    if stage_calls != expected_stage_calls:
        raise RuntimeError("response pilot stage execution count drift")

    endpoint_summary: dict[str, Any] = {}
    for degree in (2, 3):
        key = f"k{degree}"
        endpoint_summary[key] = {
            "full_order_correct": sum(bool(row["endpoint"][key]["full_correct"]) for row in rows),
            "prefix_depth3_correct": sum(
                bool(row["endpoint"][key]["prefix3_correct"]) for row in rows
            ),
            "precedence_correct": sum(
                int(row["endpoint"][key]["precedence_correct"]) for row in rows
            ),
        }
    if endpoint_summary != {
        "k2": {"full_order_correct": 0, "prefix_depth3_correct": 0, "precedence_correct": 77},
        "k3": {"full_order_correct": 3, "prefix_depth3_correct": 3, "precedence_correct": 79},
    }:
        raise RuntimeError(f"response pilot did not reproduce frozen endpoint metrics: {endpoint_summary}")

    response_summary: dict[str, Any] = {"k2": {}, "k3": {}}
    for degree in (2, 3):
        key = f"k{degree}"
        for family in FAMILIES:
            response_summary[key][family] = {
                "full_order_correct": sum(
                    bool(row["response"][key][family]["full_correct"]) for row in rows
                ),
                "prefix_depth3_correct": sum(
                    bool(row["response"][key][family]["prefix3_correct"]) for row in rows
                ),
                "precedence_correct": sum(
                    int(row["response"][key][family]["precedence_correct"]) for row in rows
                ),
                "minimum_margin": min(
                    float(row["response"][key][family]["margin"]) for row in rows
                ),
            }

    primary = response_summary["k3"]["loss_plus_cross_susceptibility"]
    primary_geometry = reference_geometry["k3"]["loss_plus_cross_susceptibility"]
    checks = {
        "k3_active_full_order_beats_endpoint": int(primary["full_order_correct"]) > 3,
        "k3_active_full_order_beats_recency": int(primary["full_order_correct"]) > 1,
        "k3_active_prefix3_not_worse": int(primary["prefix_depth3_correct"]) >= 3,
        "k3_active_precedence_not_worse": int(primary["precedence_correct"]) >= 79,
        "k3_reference_separation_positive": float(
            primary_geometry["minimum_standardized_candidate_separation"]
        ) > 0.0,
    }

    result = {
        "status": "complete",
        "claim": "non_confirmatory_counterfactual_learning_response_pilot",
        "protocol_sha256": json_sha256(protocol),
        "source_k23_protocol_sha256": json_sha256(k23),
        "replay_lock_sha256": json_sha256(replay),
        "confirmation_codebooks_observed": False,
        "model": model_id,
        "revision": revision,
        "precision": "fp64",
        "learning_rate": learning_rate,
        "pilot_codebook_seed": seed,
        "numerical_execution_fingerprint_sha256": numerical_fingerprint,
        "stage_executions": stage_calls,
        "gradient_response_evaluations": (len(histories) * 2 + len(histories)) * len(stages),
        "response_definition": protocol["response_definition"],
        "primary_response_family": protocol["primary_response_family"],
        "reference_geometry": reference_geometry,
        "endpoint_summary": endpoint_summary,
        "response_summary": response_summary,
        "primary_active_adds_information_checks": checks,
        "primary_active_adds_information_all": all(checks.values()),
        "strong_low_order_rescue": int(
            response_summary["k2"]["loss_plus_cross_susceptibility"]["full_order_correct"]
        ) >= 3,
        "histories": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
