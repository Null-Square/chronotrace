#!/usr/bin/env python3
# ruff: noqa: E501,I001
"""Run the frozen K3 affine last-stage pruning diagnostic on Pythia-14M."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from pythia_finite_pair_bridge import _configure_portable_numerics, _load_json

from chronotrace.geometry.interactions import (
    measure_ordered_interaction_basis_compact,
    ordered_interaction_word_prediction,
    ordered_probe_count,
)
from chronotrace.geometry.order_affine import (
    build_k3_local_coordinate_layout,
    candidate_last_affine_lower_bounds,
    encode_k3_local_permutation,
    k3_local_equalities,
    k3_local_variable_coefficients,
)
from chronotrace.geometry.order_relaxation import build_k3_local_order_relaxation
from chronotrace.reproducibility import json_sha256, tensor_sha256
from chronotrace.scale import build_scale_worlds_from_codebook
from chronotrace.scale_four import build_four_stage_examples, four_stage_dataset_payload
from chronotrace.scale_runner import execute_plain_sgd_stage, flatten_parameters
from chronotrace.scale_tokens import build_token_codebook, tokenizer_fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="configs/pythia_14m_k3_affine_last_stage_diagnostic.lock.json",
    )
    parser.add_argument(
        "--k23-protocol",
        default="configs/pythia_14m_four_stage_k23_pilot.lock.json",
    )
    parser.add_argument(
        "--all24-selection",
        default="configs/pythia_14m_forward_reachable_all24.selection.json",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def _quadratic_error(
    gram: np.ndarray,
    cross: np.ndarray,
    target_norm_squared: float,
    vector: np.ndarray,
) -> float:
    squared = float(
        target_norm_squared
        - 2.0 * float(cross @ vector)
        + float(vector @ gram @ vector)
    )
    scale = max(
        1.0,
        abs(target_norm_squared),
        abs(2.0 * float(cross @ vector)),
        abs(float(vector @ gram @ vector)),
    )
    tolerance = 1e-10 * scale
    if squared < -tolerance:
        raise FloatingPointError("affine feasible-candidate error became materially negative")
    return math.sqrt(max(0.0, squared))


def main() -> None:
    args = parse_args()
    protocol = _load_json(args.protocol)
    k23 = _load_json(args.k23_protocol)
    all24 = _load_json(args.all24_selection)

    if protocol["freeze_status"] != "frozen_before_any_k3_affine_last_stage_pythia_output":
        raise ValueError("K3 affine last-stage protocol is not frozen")
    if protocol["experiment_role"] != "non_confirmatory_scalability_diagnostic":
        raise ValueError("K3 affine last-stage role drift")
    if protocol.get("confirmation_codebooks_observed") is not False:
        raise RuntimeError("K3 affine last-stage protocol touched confirmation codebooks")
    if k23.get("confirmation_codebooks_observed") is not False:
        raise RuntimeError("source K2/K3 protocol touched confirmation codebooks")
    if all24.get("confirmation_codebooks_observed") is not False:
        raise RuntimeError("source all-24 selection touched confirmation codebooks")
    if json_sha256(k23) != str(protocol["source_k23_protocol_sha256"]):
        raise RuntimeError("source K2/K3 protocol hash drift")
    if json_sha256(all24) != str(protocol["source_all24_selection_sha256"]):
        raise RuntimeError("source all-24 selection hash drift")
    if all24.get("all24_pass_all") is not True:
        raise RuntimeError("K3 affine diagnostic requires frozen all-24 reachable pass")
    if int(all24.get("source_run_id", -1)) != int(protocol["source_all24_run_id"]):
        raise RuntimeError("source all-24 run identity drift")

    stages = tuple(str(value) for value in protocol["stages"])
    if stages != ("A", "B", "C", "D"):
        raise ValueError("K3 affine diagnostic requires A/B/C/D")
    if int(protocol["max_measured_interaction_degree"]) != 3:
        raise ValueError("K3 affine diagnostic requires degree-three basis")
    target_history = tuple(str(protocol["target_history"]))
    if target_history != ("A", "B", "C", "D"):
        raise ValueError("K3 affine diagnostic target must remain ABCD")
    feasible_history = tuple(str(protocol["primary_feasible_upper_bound_history"]))
    if feasible_history != ("B", "A", "C", "D"):
        raise ValueError("frozen feasible upper-bound history drift")

    seed = int(protocol["pilot_codebook_seed"])
    prohibited = {int(value) for value in protocol["confirmation_codebooks_prohibited"]}
    if seed in prohibited:
        raise RuntimeError("K3 affine diagnostic may not use a held-out codebook")

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
        raise RuntimeError("K3 affine tokenizer fingerprint drift")
    codebook = build_token_codebook(
        tokenizer,
        count=int(protocol["codebook_count_per_kind"]),
        seed=seed,
    )
    dataset = four_stage_dataset_payload(tokenizer, codebook)
    if codebook.sha256 != str(protocol["pilot_codebook_sha256"]):
        raise RuntimeError("K3 affine codebook hash drift")
    if dataset["sha256"] != protocol["pilot_dataset_sha256"]:
        raise RuntimeError("K3 affine dataset hash drift")
    worlds = build_scale_worlds_from_codebook(codebook)
    examples = {stage: build_four_stage_examples(worlds, stage) for stage in stages}

    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision)
    model.to(device=device, dtype=torch.float64)
    model.config.use_cache = False
    if any(
        parameter.is_floating_point() and parameter.dtype != torch.float64
        for parameter in model.parameters()
    ):
        raise TypeError("could not enforce FP64 K3 affine model")
    theta0 = flatten_parameters(torch, model, preserve_parameter_dtype=True)
    if theta0.dtype != torch.float64:
        raise TypeError("K3 affine base vector is not FP64")
    fp64_hash = tensor_sha256(theta0)
    if fp64_hash != str(protocol["base_parameter_sha256_fp64"]):
        raise RuntimeError("K3 affine FP64 base hash drift")
    projected_hash = tensor_sha256(theta0.to(dtype=torch.float32))
    if projected_hash != str(protocol["base_parameter_sha256_fp32"]):
        raise RuntimeError("K3 affine projected FP32 base hash drift")

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
                raise FloatingPointError(f"non-finite K3 affine stage {stage}")
            stage_calls += 1
            return endpoint

        return run

    stage_maps = {stage: make_stage_map(stage) for stage in stages}
    basis = measure_ordered_interaction_basis_compact(stage_maps, theta0, max_degree=3)
    expected_basis_calls = ordered_probe_count(len(stages), 3)
    if expected_basis_calls != int(protocol["basis_stage_executions"]):
        raise RuntimeError("frozen K3 affine basis count drift")
    if basis.stage_executions != expected_basis_calls or stage_calls != expected_basis_calls:
        raise RuntimeError("observed K3 affine basis count drift")
    if any(value.dtype != torch.float64 for value in basis.interactions.values()):
        raise TypeError("K3 affine basis contains non-FP64 tensor")

    target = theta0
    for stage in target_history:
        target = stage_maps[stage](target)
    target_hash = tensor_sha256(target)
    target_replay_exact = target_hash == str(protocol["target_endpoint_sha256"])
    if not target_replay_exact:
        raise RuntimeError("K3 affine ABCD target failed exact frozen replay")

    feasible_prediction = ordered_interaction_word_prediction(feasible_history, basis, degree=3)
    feasible_direct_error = float(torch.linalg.vector_norm(target - feasible_prediction))
    del feasible_prediction

    frozen_upper = float(protocol["primary_feasible_upper_bound_error"])
    feasible_direct_matches_frozen = math.isclose(
        feasible_direct_error,
        frozen_upper,
        rel_tol=float(protocol["primary_feasible_upper_bound_numeric_rel_tol"]),
        abs_tol=float(protocol["primary_feasible_upper_bound_numeric_abs_tol"]),
    )

    relaxation = build_k3_local_order_relaxation(basis)
    layout = build_k3_local_coordinate_layout(stages)
    if layout.dimension != int(protocol["coordinate_dimension"]):
        raise RuntimeError("K3 affine coordinate dimension drift")
    base_equalities, _ = k3_local_equalities(layout)
    candidate_equalities, _ = k3_local_equalities(layout, last_stage="D")
    if base_equalities.shape[0] != int(protocol["base_local_equality_count"]):
        raise RuntimeError("K3 affine base equality count drift")
    if candidate_equalities.shape[0] != int(protocol["candidate_last_equality_count"]):
        raise RuntimeError("K3 affine candidate equality count drift")

    # The relaxation retains only the 24 triple tensors plus six pair-difference tensors
    # and one constant.  Free the original basis table and model state before Gram work.
    del basis
    del stage_maps
    del model
    del examples
    del worlds
    del tokenizer
    gc.collect()

    features = k3_local_variable_coefficients(relaxation, layout)
    centered_target = target - relaxation.constant
    flat_target = centered_target.reshape(-1)
    target_norm_squared = float(torch.dot(flat_target, flat_target))
    dimension = layout.dimension
    gram = np.empty((dimension, dimension), dtype=np.float64)
    cross = np.empty(dimension, dtype=np.float64)
    for left in range(dimension):
        left_vector = features[left].reshape(-1)
        cross[left] = float(torch.dot(left_vector, flat_target))
        for right in range(left + 1):
            value = float(torch.dot(left_vector, features[right].reshape(-1)))
            gram[left, right] = value
            gram[right, left] = value

    if not np.isfinite(gram).all() or not np.isfinite(cross).all():
        raise FloatingPointError("K3 affine scalar sufficient statistics became non-finite")

    feasible_vector = encode_k3_local_permutation(feasible_history, layout)
    feasible_quadratic_error = _quadratic_error(
        gram,
        cross,
        target_norm_squared,
        feasible_vector,
    )
    feasible_quadratic_matches_direct = math.isclose(
        feasible_quadratic_error,
        feasible_direct_error,
        rel_tol=1e-8,
        abs_tol=1e-8,
    )

    lower_bounds = candidate_last_affine_lower_bounds(
        gram,
        cross,
        target_norm_squared,
        layout,
    )
    numerical_guard = float(protocol["numerical_lower_bound_guard"])
    elimination_threshold = feasible_direct_error + numerical_guard
    eliminated = tuple(
        stage
        for stage in stages
        if lower_bounds[stage].distance > elimination_threshold
    )
    survivors = tuple(stage for stage in stages if stage not in eliminated)
    wrong_eliminated = tuple(stage for stage in eliminated if stage != "D")
    certified_last_stage = survivors[0] if len(survivors) == 1 else None

    equality_max = max(result.equality_residual_norm for result in lower_bounds.values())
    stationarity_max = max(result.stationarity_residual_norm for result in lower_bounds.values())
    kkt_within_tolerance = (
        equality_max <= float(protocol["kkt_equality_residual_max"])
        and stationarity_max <= float(protocol["kkt_stationarity_residual_max"])
    )

    expected_stage_calls = expected_basis_calls + len(target_history)
    if stage_calls != expected_stage_calls:
        raise RuntimeError("K3 affine total stage execution count drift")

    checks = {
        "target_endpoint_replay_exact": target_replay_exact,
        "frozen_BACD_upper_bound_reproduced": feasible_direct_matches_frozen,
        "quadratic_upper_bound_matches_direct": feasible_quadratic_matches_direct,
        "coordinate_dimension_exact": dimension == int(protocol["coordinate_dimension"]),
        "base_equality_count_exact": base_equalities.shape[0] == int(protocol["base_local_equality_count"]),
        "candidate_last_equality_count_exact": candidate_equalities.shape[0] == int(protocol["candidate_last_equality_count"]),
        "all_KKT_residuals_within_tolerance": kkt_within_tolerance,
        "D_not_eliminated": "D" not in eliminated,
    }
    strong_success = all(checks.values()) and set(wrong_eliminated) == {"A", "B", "C"}
    promising_partial = all(checks.values()) and len(wrong_eliminated) >= 1

    result = {
        "status": "complete",
        "claim": "non_confirmatory_K3_polynomial_local_order_affine_last_stage_pruning_diagnostic",
        "protocol_sha256": json_sha256(protocol),
        "source_k23_protocol_sha256": json_sha256(k23),
        "source_all24_selection_sha256": json_sha256(all24),
        "confirmation_codebooks_observed": False,
        "model": model_id,
        "revision": revision,
        "precision": "fp64",
        "learning_rate": learning_rate,
        "pilot_codebook_seed": seed,
        "base_parameter_sha256_fp64": fp64_hash,
        "base_parameter_sha256_projected_fp32": projected_hash,
        "numerical_execution_fingerprint_sha256": numerical_fingerprint,
        "basis_stage_executions": expected_basis_calls,
        "target_stage_executions": len(target_history),
        "total_stage_executions": stage_calls,
        "coordinate_dimension": dimension,
        "base_local_equality_count": int(base_equalities.shape[0]),
        "candidate_last_equality_count": int(candidate_equalities.shape[0]),
        "primary_full_permutation_evaluations": 0,
        "primary_feasible_history_evaluations": 1,
        "gram_pair_dot_products": dimension * (dimension + 1) // 2,
        "target_feature_dot_products": dimension,
        "target_history": "ABCD",
        "target_endpoint_sha256": target_hash,
        "feasible_upper_bound": {
            "history": "BACD",
            "frozen_error": frozen_upper,
            "direct_error": feasible_direct_error,
            "quadratic_error": feasible_quadratic_error,
            "elimination_threshold": elimination_threshold,
        },
        "candidate_last": {
            stage: {
                "affine_lower_bound": lower_bounds[stage].distance,
                "squared_affine_lower_bound": lower_bounds[stage].squared_distance,
                "equality_residual_norm": lower_bounds[stage].equality_residual_norm,
                "stationarity_residual_norm": lower_bounds[stage].stationarity_residual_norm,
                "certified_impossible": stage in eliminated,
            }
            for stage in stages
        },
        "eliminated_last_stages": list(eliminated),
        "surviving_last_stages": list(survivors),
        "wrong_last_stages_eliminated": list(wrong_eliminated),
        "wrong_last_stage_elimination_count": len(wrong_eliminated),
        "certified_last_stage": certified_last_stage,
        "maximum_KKT_equality_residual": equality_max,
        "maximum_KKT_stationarity_residual": stationarity_max,
        "diagnostic_checks": checks,
        "strong_last_stage_certificate": strong_success,
        "promising_partial_pruning": promising_partial,
        "next_step": (
            "extend_affine_bounds_to_fixed_suffix_branch_and_bound_on_spent_K3_geometry"
            if strong_success
            else (
                "couple_affine_pruning_with_frozen_active_transition_tangent_correction_on_spent_data"
                if promising_partial
                else "declare_pure_K3_affine_hull_too_loose_and_build_active_transition_tangent_or_secant_relaxation_on_spent_data"
            )
        ),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "strong_last_stage_certificate": strong_success,
                "promising_partial_pruning": promising_partial,
                "feasible_upper_bound_error": feasible_direct_error,
                "candidate_last_lower_bounds": {
                    stage: lower_bounds[stage].distance for stage in stages
                },
                "eliminated_last_stages": list(eliminated),
                "surviving_last_stages": list(survivors),
                "maximum_KKT_equality_residual": equality_max,
                "maximum_KKT_stationarity_residual": stationarity_max,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
