#!/usr/bin/env python3
# ruff: noqa: E501,I001
"""Run the frozen K3 convex last-stage pruning diagnostic on spent Pythia-14M data."""

from __future__ import annotations

import argparse
import gc
import json
import math
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np

from pythia_finite_pair_bridge import _configure_portable_numerics, _load_json

from chronotrace.geometry.convex_certificate import certify_lower_bound_elimination
from chronotrace.geometry.convex_quadratic import (
    dual_quadratic_hull_certificate,
    project_quadratic_simplex,
)
from chronotrace.geometry.interactions import (
    measure_ordered_interaction_basis_compact,
    ordered_interaction_word_prediction,
    ordered_probe_count,
)
from chronotrace.geometry.order_affine import (
    build_k3_local_coordinate_layout,
    encode_k3_local_permutation,
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
        default="configs/pythia_14m_k3_convex_last_stage_diagnostic.lock.json",
    )
    parser.add_argument(
        "--k23-protocol",
        default="configs/pythia_14m_four_stage_k23_pilot.lock.json",
    )
    parser.add_argument(
        "--all24-selection",
        default="configs/pythia_14m_forward_reachable_all24.selection.json",
    )
    parser.add_argument(
        "--affine-selection",
        default="configs/pythia_14m_k3_affine_last_stage_diagnostic.selection.json",
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
    quadratic = float(vector @ gram @ vector)
    linear = float(2.0 * cross @ vector)
    squared = target_norm_squared - linear + quadratic
    scale = max(1.0, abs(target_norm_squared), abs(linear), abs(quadratic))
    tolerance = 1e-10 * scale
    if squared < -tolerance:
        raise FloatingPointError("K3 convex candidate error became materially negative")
    return math.sqrt(max(0.0, squared))


def main() -> None:
    args = parse_args()
    protocol = _load_json(args.protocol)
    k23 = _load_json(args.k23_protocol)
    all24 = _load_json(args.all24_selection)
    affine = _load_json(args.affine_selection)

    if protocol["freeze_status"] != "frozen_before_any_k3_convex_last_stage_pythia_output":
        raise ValueError("K3 convex last-stage protocol is not frozen")
    if protocol["experiment_role"] != "non_confirmatory_spent_scalability_diagnostic":
        raise ValueError("K3 convex last-stage role drift")
    if protocol.get("heldout_confirmation_launch_authorized") is not False:
        raise RuntimeError("K3 convex protocol unexpectedly authorizes held-out confirmation")
    for name, payload in (("protocol", protocol), ("k23", k23), ("all24", all24)):
        if payload.get("confirmation_codebooks_observed") is not False:
            raise RuntimeError(f"{name} touched confirmation codebooks")
    if json_sha256(k23) != str(protocol["source_k23_protocol_sha256"]):
        raise RuntimeError("source K2/K3 protocol hash drift")
    if json_sha256(all24) != str(protocol["source_all24_selection_sha256"]):
        raise RuntimeError("source all-24 selection hash drift")
    if int(affine.get("source_run_id", -1)) != int(protocol["source_affine_run_id"]):
        raise RuntimeError("source affine run identity drift")
    if int(affine.get("source_job_id", -1)) != int(protocol["source_affine_job_id"]):
        raise RuntimeError("source affine job identity drift")
    if int(affine.get("source_artifact_id", -1)) != int(protocol["source_affine_artifact_id"]):
        raise RuntimeError("source affine artifact identity drift")
    if str(affine.get("source_artifact_digest")) != str(protocol["source_affine_artifact_digest"]):
        raise RuntimeError("source affine artifact digest drift")
    if int(affine.get("wrong_last_stage_elimination_count", -1)) != 0:
        raise RuntimeError("source affine selection is not the frozen zero-pruning negative")
    if all24.get("all24_pass_all") is not True:
        raise RuntimeError("K3 convex diagnostic requires frozen all-24 reachable pass")

    stages = tuple(str(value) for value in protocol["stages"])
    if stages != ("A", "B", "C", "D"):
        raise ValueError("K3 convex diagnostic requires A/B/C/D")
    if int(protocol["max_measured_interaction_degree"]) != 3:
        raise ValueError("K3 convex diagnostic requires degree-three basis")
    target_history = tuple(str(protocol["target_history"]))
    feasible_history = tuple(str(protocol["primary_feasible_upper_bound_history"]))
    if target_history != ("A", "B", "C", "D"):
        raise ValueError("K3 convex target must remain ABCD")
    if feasible_history != ("B", "A", "C", "D"):
        raise ValueError("frozen BACD feasible upper-bound history drift")

    seed = int(protocol["pilot_codebook_seed"])
    prohibited = {int(value) for value in protocol["confirmation_codebooks_prohibited"]}
    if seed in prohibited:
        raise RuntimeError("K3 convex diagnostic may not use a held-out codebook")

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
        raise RuntimeError("K3 convex tokenizer fingerprint drift")
    codebook = build_token_codebook(
        tokenizer,
        count=int(protocol["codebook_count_per_kind"]),
        seed=seed,
    )
    dataset = four_stage_dataset_payload(tokenizer, codebook)
    if codebook.sha256 != str(protocol["pilot_codebook_sha256"]):
        raise RuntimeError("K3 convex codebook hash drift")
    if dataset["sha256"] != protocol["pilot_dataset_sha256"]:
        raise RuntimeError("K3 convex dataset hash drift")
    worlds = build_scale_worlds_from_codebook(codebook)
    examples = {stage: build_four_stage_examples(worlds, stage) for stage in stages}

    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision)
    model.to(device=device, dtype=torch.float64)
    model.config.use_cache = False
    if any(
        parameter.is_floating_point() and parameter.dtype != torch.float64
        for parameter in model.parameters()
    ):
        raise TypeError("could not enforce FP64 K3 convex model")
    theta0 = flatten_parameters(torch, model, preserve_parameter_dtype=True)
    if theta0.dtype != torch.float64:
        raise TypeError("K3 convex base vector is not FP64")
    fp64_hash = tensor_sha256(theta0)
    if fp64_hash != str(protocol["base_parameter_sha256_fp64"]):
        raise RuntimeError("K3 convex FP64 base hash drift")
    projected_hash = tensor_sha256(theta0.to(dtype=torch.float32))
    if projected_hash != str(protocol["base_parameter_sha256_fp32"]):
        raise RuntimeError("K3 convex projected FP32 base hash drift")

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
                raise FloatingPointError(f"non-finite K3 convex stage {stage}")
            stage_calls += 1
            return endpoint

        return run

    stage_maps = {stage: make_stage_map(stage) for stage in stages}
    basis = measure_ordered_interaction_basis_compact(stage_maps, theta0, max_degree=3)
    expected_basis_calls = ordered_probe_count(len(stages), 3)
    if expected_basis_calls != int(protocol["basis_stage_executions"]):
        raise RuntimeError("frozen K3 convex basis count drift")
    if basis.stage_executions != expected_basis_calls or stage_calls != expected_basis_calls:
        raise RuntimeError("observed K3 convex basis count drift")
    if any(value.dtype != torch.float64 for value in basis.interactions.values()):
        raise TypeError("K3 convex basis contains non-FP64 tensor")

    target = theta0
    for stage in target_history:
        target = stage_maps[stage](target)
    target_hash = tensor_sha256(target)
    target_replay_exact = target_hash == str(protocol["target_endpoint_sha256"])
    if not target_replay_exact:
        raise RuntimeError("K3 convex ABCD target failed exact frozen replay")

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
    if layout.dimension != 30:
        raise RuntimeError("K3 convex local-order dimension drift")

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
        raise FloatingPointError("K3 convex scalar sufficient statistics became non-finite")

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

    numerical_guard = float(protocol["numerical_lower_bound_guard"])
    elimination_threshold = feasible_direct_error + numerical_guard
    weight_tolerance = float(protocol["convex_projection_primal_weight_tolerance"])
    simplex_tolerance = float(protocol["convex_projection_simplex_residual_max"])
    dual_tolerance = float(protocol["certificate_primal_dual_consistency_tolerance"])
    unit_tolerance = float(protocol["certificate_unit_norm_tolerance"])

    candidate_last: dict[str, dict[str, Any]] = {}
    eliminated: list[str] = []
    all_weights_feasible = True
    all_simplex_residuals_valid = True
    all_dual_norms_valid = True
    all_dual_bounds_valid = True
    all_primal_below_vertex = True

    for stage in stages:
        histories = tuple(history for history in permutations(stages) if history[-1] == stage)
        if len(histories) != int(protocol["candidate_vertices_per_last_stage"]):
            raise RuntimeError("K3 convex candidate vertex count drift")
        local_vertices = np.stack(
            [encode_k3_local_permutation(history, layout) for history in histories]
        )
        candidate_gram = local_vertices @ gram @ local_vertices.T
        candidate_cross = local_vertices @ cross
        projection = project_quadratic_simplex(
            candidate_gram,
            candidate_cross,
            target_norm_squared,
            weight_tolerance=weight_tolerance,
        )
        certificate = dual_quadratic_hull_certificate(
            candidate_gram,
            candidate_cross,
            target_norm_squared,
            projection,
        )
        vertex_errors = [
            _quadratic_error(gram, cross, target_norm_squared, vertex)
            for vertex in local_vertices
        ]
        minimum_vertex_error = min(vertex_errors)
        weights_feasible = projection.minimum_weight >= -weight_tolerance
        simplex_valid = projection.simplex_residual <= simplex_tolerance
        dual_norm_valid = (
            projection.distance <= 1e-12
            or abs(certificate.direction_norm - 1.0) <= unit_tolerance
        )
        dual_bound_valid = certificate.lower_bound <= projection.distance + dual_tolerance
        primal_below_vertex = projection.distance <= minimum_vertex_error + dual_tolerance
        all_weights_feasible = all_weights_feasible and weights_feasible
        all_simplex_residuals_valid = all_simplex_residuals_valid and simplex_valid
        all_dual_norms_valid = all_dual_norms_valid and dual_norm_valid
        all_dual_bounds_valid = all_dual_bounds_valid and dual_bound_valid
        all_primal_below_vertex = all_primal_below_vertex and primal_below_vertex

        verdict = certify_lower_bound_elimination(
            certificate.lower_bound,
            feasible_direct_error,
            numerical_guard=numerical_guard,
        )
        if verdict.truncated_eliminated:
            eliminated.append(stage)
        candidate_last[stage] = {
            "histories": ["".join(history) for history in histories],
            "convex_weights": projection.weights.tolist(),
            "active_support": list(projection.support),
            "primal_convex_hull_distance": projection.distance,
            "squared_primal_convex_hull_distance": projection.squared_distance,
            "minimum_vertex_error": minimum_vertex_error,
            "dual_witness_lower_bound": certificate.lower_bound,
            "primal_dual_gap": certificate.primal_dual_gap,
            "simplex_residual": projection.simplex_residual,
            "minimum_weight": projection.minimum_weight,
            "truncated_K3_certified_impossible": verdict.truncated_eliminated,
            "true_chronology_certified_impossible": verdict.exact_eliminated,
        }

    survivors = tuple(stage for stage in stages if stage not in eliminated)
    wrong_eliminated = tuple(stage for stage in eliminated if stage != "D")
    expected_stage_calls = expected_basis_calls + len(target_history)
    if stage_calls != expected_stage_calls:
        raise RuntimeError("K3 convex total stage execution count drift")

    checks = {
        "target_endpoint_replay_exact": target_replay_exact,
        "frozen_BACD_upper_bound_reproduced": feasible_direct_matches_frozen,
        "quadratic_upper_bound_matches_direct": feasible_quadratic_matches_direct,
        "all_candidate_convex_weights_primal_feasible": all_weights_feasible,
        "all_simplex_residuals_within_tolerance": all_simplex_residuals_valid,
        "all_dual_witness_norms_valid": all_dual_norms_valid,
        "all_dual_lower_bounds_below_primal_distance": all_dual_bounds_valid,
        "all_primal_distances_below_minimum_vertex_error": all_primal_below_vertex,
        "D_not_eliminated": "D" not in eliminated,
        "true_chronology_certificates_withheld_without_tail_bound": all(
            candidate_last[stage]["true_chronology_certified_impossible"] is None
            for stage in stages
        ),
    }
    all_checks = all(checks.values())
    strong_success = all_checks and set(wrong_eliminated) == {"A", "B", "C"}
    promising_partial = all_checks and 1 <= len(wrong_eliminated) < 3
    scientific_negative = all_checks and len(wrong_eliminated) == 0
    invalid = not all_checks

    if strong_success:
        next_step = "build_generic_nonnegative_K_local_marginal_dual_bounds_on_spent_data"
    elif promising_partial:
        next_step = "retain_convex_pruning_then_lift_interaction_order_only_on_survivors"
    elif scientific_negative:
        next_step = "lift_to_exact_Kplus1_interactions_and_require_explicit_higher_order_tail_bounds"
    else:
        next_step = "invalidate_run_and_debug_before_any_scientific_interpretation"

    result = {
        "status": "invalid" if invalid else "complete",
        "claim": "non_confirmatory_static_K3_convex_last_stage_pruning_diagnostic",
        "protocol_sha256": json_sha256(protocol),
        "source_k23_protocol_sha256": json_sha256(k23),
        "source_all24_selection_sha256": json_sha256(all24),
        "confirmation_codebooks_observed": False,
        "heldout_confirmation_launch_authorized": False,
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
        "model_stage_replays_for_candidate_vertices": 0,
        "candidate_local_order_vertex_evaluations": len(stages)
        * int(protocol["candidate_vertices_per_last_stage"]),
        "coordinate_dimension": dimension,
        "target_history": "ABCD",
        "target_endpoint_sha256": target_hash,
        "feasible_upper_bound": {
            "history": "BACD",
            "frozen_error": frozen_upper,
            "direct_error": feasible_direct_error,
            "quadratic_error": feasible_quadratic_error,
            "elimination_threshold": elimination_threshold,
        },
        "candidate_last": candidate_last,
        "eliminated_last_stages": list(eliminated),
        "surviving_last_stages": list(survivors),
        "wrong_last_stages_eliminated": list(wrong_eliminated),
        "wrong_last_stage_elimination_count": len(wrong_eliminated),
        "diagnostic_checks": checks,
        "strong_last_stage_certificate": strong_success,
        "promising_partial_pruning": promising_partial,
        "scientific_negative": scientific_negative,
        "true_chronology_certificate_scope": "withheld_without_explicit_interaction_tail_bound",
        "next_step": next_step,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "strong_last_stage_certificate": strong_success,
                "promising_partial_pruning": promising_partial,
                "scientific_negative": scientific_negative,
                "candidate_last_dual_lower_bounds": {
                    stage: candidate_last[stage]["dual_witness_lower_bound"]
                    for stage in stages
                },
                "eliminated_last_stages": list(eliminated),
                "surviving_last_stages": list(survivors),
                "next_step": next_step,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
