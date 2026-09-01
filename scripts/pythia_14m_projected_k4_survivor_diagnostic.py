#!/usr/bin/env python3
# ruff: noqa: E501,I001
"""Run the frozen witness-projected K4 survivor diagnostic on spent Pythia-14M data."""

from __future__ import annotations

import argparse
import gc
import json
import math
from itertools import permutations
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from pythia_finite_pair_bridge import _configure_portable_numerics, _load_json

from chronotrace.geometry.convex_quadratic import (
    dual_quadratic_hull_certificate,
    project_quadratic_simplex,
)
from chronotrace.geometry.interactions import (
    measure_ordered_interaction_basis_streaming_exact,
    ordered_interaction_word_prediction,
    ordered_probe_count,
)
from chronotrace.geometry.local_order_hierarchy import (
    build_local_order_hierarchy,
    full_level_permutation_scores,
    projected_interaction_linear_objective,
)
from chronotrace.geometry.local_order_lp import solve_local_order_lp
from chronotrace.geometry.projected_interactions import (
    projected_interaction_from_endpoint_delta,
    projected_word_prediction,
)
from chronotrace.reproducibility import json_sha256, tensor_sha256
from chronotrace.scale import build_scale_worlds_from_codebook
from chronotrace.scale_four import build_four_stage_examples, four_stage_dataset_payload
from chronotrace.scale_runner import execute_plain_sgd_stage, flatten_parameters
from chronotrace.scale_tokens import build_token_codebook, tokenizer_fingerprint


IMPLEMENTATION_AMENDMENT = Path(
    "configs/pythia_14m_projected_k4_survivor_diagnostic.implementation_amendment.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="configs/pythia_14m_projected_k4_survivor_diagnostic.lock.json",
    )
    parser.add_argument(
        "--source-k3-protocol",
        default="configs/pythia_14m_k3_convex_last_stage_diagnostic.lock.json",
    )
    parser.add_argument(
        "--source-k3-selection",
        default="configs/pythia_14m_k3_convex_last_stage_diagnostic.selection.json",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def _dot(torch: Any, left: Any, right: Any) -> float:
    return float(torch.dot(left.reshape(-1), right.reshape(-1)))


def _candidate_k3_witness(
    torch: Any,
    target: Any,
    histories: tuple[tuple[str, ...], ...],
    basis: Any,
    *,
    weight_tolerance: float,
) -> tuple[Any, Any, float, float]:
    displacements: list[Any] = []
    for history in histories:
        prediction = ordered_interaction_word_prediction(history, basis, degree=3)
        displacements.append(prediction - target)
        del prediction

    count = len(displacements)
    gram = np.empty((count, count), dtype=np.float64)
    for left in range(count):
        for right in range(left + 1):
            value = _dot(torch, displacements[left], displacements[right])
            gram[left, right] = value
            gram[right, left] = value

    projection = project_quadratic_simplex(
        gram,
        np.zeros(count, dtype=np.float64),
        0.0,
        weight_tolerance=weight_tolerance,
    )
    certificate = dual_quadratic_hull_certificate(
        gram,
        np.zeros(count, dtype=np.float64),
        0.0,
        projection,
    )
    mixed = torch.zeros_like(target)
    for weight, displacement in zip(projection.weights, displacements, strict=True):
        if weight:
            mixed = mixed + float(weight) * displacement
    norm = float(torch.linalg.vector_norm(mixed))
    if not math.isfinite(norm) or norm <= 0.0:
        raise FloatingPointError("K3 candidate hull produced a zero/non-finite witness residual")
    witness = -mixed / norm
    unit_norm = float(torch.linalg.vector_norm(witness))
    del displacements
    del mixed
    gc.collect()
    return witness, projection, certificate.lower_bound, unit_norm


def main() -> None:
    args = parse_args()
    protocol = _load_json(args.protocol)
    source_k3_protocol = _load_json(args.source_k3_protocol)
    source_k3_selection = _load_json(args.source_k3_selection)
    amendment = _load_json(IMPLEMENTATION_AMENDMENT)

    if protocol["freeze_status"] != "frozen_before_any_projected_k4_pythia_output":
        raise ValueError("projected K4 protocol is not frozen")
    if protocol["experiment_role"] != "non_confirmatory_spent_survivor_active_lift_diagnostic":
        raise ValueError("projected K4 experiment role drift")
    if protocol.get("heldout_confirmation_launch_authorized") is not False:
        raise RuntimeError("projected K4 protocol unexpectedly authorizes held-out confirmation")
    if protocol.get("confirmation_codebooks_observed") is not False:
        raise RuntimeError("projected K4 protocol touched confirmation codebooks")
    if amendment.get("status") != "frozen_after_invalid_unreported_attempt_before_any_retry_output":
        raise RuntimeError("projected K4 implementation amendment is not frozen")
    if amendment.get("source_protocol_version") != protocol["protocol_version"]:
        raise RuntimeError("projected K4 implementation amendment protocol drift")
    if amendment.get("scientific_decision_rule_changed") is not False:
        raise RuntimeError("projected K4 implementation amendment changed the decision rule")
    if amendment.get("heldout_confirmation_launch_authorized") is not False:
        raise RuntimeError("projected K4 implementation amendment authorizes held-out confirmation")
    if int(amendment["total_expected_stage_executions"]) != int(protocol["total_expected_stage_executions"]):
        raise RuntimeError("projected K4 implementation amendment changed stage budget")
    if source_k3_protocol.get("confirmation_codebooks_observed") is not False:
        raise RuntimeError("source K3 protocol touched confirmation codebooks")
    if source_k3_selection.get("confirmation_codebooks_observed") is not False:
        raise RuntimeError("source K3 selection touched confirmation codebooks")
    if json_sha256(source_k3_protocol) != str(protocol["source_k3_convex_protocol_sha256"]):
        raise RuntimeError("source K3 convex protocol hash drift")
    if json_sha256(source_k3_selection) != str(protocol["source_k3_convex_selection_sha256"]):
        raise RuntimeError("source K3 convex selection hash drift")
    if int(source_k3_selection["source_run_id"]) != int(protocol["source_k3_convex_run_id"]):
        raise RuntimeError("source K3 convex run identity drift")
    if int(source_k3_selection["source_job_id"]) != int(protocol["source_k3_convex_job_id"]):
        raise RuntimeError("source K3 convex job identity drift")
    if int(source_k3_selection["source_artifact_id"]) != int(protocol["source_k3_convex_artifact_id"]):
        raise RuntimeError("source K3 convex artifact identity drift")
    if str(source_k3_selection["source_artifact_digest"]) != str(protocol["source_k3_convex_artifact_digest"]):
        raise RuntimeError("source K3 convex artifact digest drift")
    if source_k3_selection.get("all_diagnostic_checks_passed") is not True:
        raise RuntimeError("source K3 convex diagnostic did not pass all checks")
    if tuple(source_k3_selection["eliminated_last_stages"]) != tuple(protocol["source_k3_eliminated_last_stages"]):
        raise RuntimeError("source K3 eliminated-stage set drift")
    if tuple(source_k3_selection["surviving_last_stages"]) != tuple(protocol["source_k3_surviving_last_stages"]):
        raise RuntimeError("source K3 survivor set drift")

    stages = tuple(str(value) for value in protocol["stages"])
    if stages != ("A", "B", "C", "D"):
        raise ValueError("projected K4 diagnostic requires A/B/C/D")
    target_history = tuple(str(protocol["target_history"]))
    if target_history != ("A", "B", "C", "D"):
        raise ValueError("projected K4 target must remain ABCD")
    survivor_stages = tuple(str(value) for value in protocol["source_k3_surviving_last_stages"])
    if survivor_stages != ("C", "D"):
        raise ValueError("projected K4 survivor decision must remain C/D")

    seed = int(protocol["pilot_codebook_seed"])
    prohibited = {int(value) for value in protocol["confirmation_codebooks_prohibited"]}
    if seed in prohibited:
        raise RuntimeError("projected K4 diagnostic may not use a held-out codebook")

    degree4_words = tuple(tuple(str(stage) for stage in value) for value in protocol["degree4_words"])
    expected_degree4_words = tuple(permutations(stages))
    if degree4_words != expected_degree4_words:
        raise RuntimeError("frozen degree-four word order drift")
    if len(degree4_words) != int(protocol["degree4_word_count"]):
        raise RuntimeError("degree-four word count drift")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    numerical_fingerprint = _configure_portable_numerics(torch, source_k3_protocol, int(args.threads))
    device = torch.device("cpu")
    model_id = str(protocol["model"])
    revision = str(protocol["revision"])
    learning_rate = float(protocol["learning_rate"])

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer_fingerprint(tokenizer) != str(protocol["tokenizer_fingerprint"]):
        raise RuntimeError("projected K4 tokenizer fingerprint drift")
    codebook = build_token_codebook(
        tokenizer,
        count=int(protocol["codebook_count_per_kind"]),
        seed=seed,
    )
    dataset = four_stage_dataset_payload(tokenizer, codebook)
    if codebook.sha256 != str(protocol["pilot_codebook_sha256"]):
        raise RuntimeError("projected K4 codebook hash drift")
    if dataset["sha256"] != protocol["pilot_dataset_sha256"]:
        raise RuntimeError("projected K4 dataset hash drift")
    worlds = build_scale_worlds_from_codebook(codebook)
    examples = {stage: build_four_stage_examples(worlds, stage) for stage in stages}

    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision)
    model.to(device=device, dtype=torch.float64)
    model.config.use_cache = False
    if any(
        parameter.is_floating_point() and parameter.dtype != torch.float64
        for parameter in model.parameters()
    ):
        raise TypeError("could not enforce FP64 projected K4 model")
    theta0 = flatten_parameters(torch, model, preserve_parameter_dtype=True)
    if theta0.dtype != torch.float64:
        raise TypeError("projected K4 base vector is not FP64")
    if tensor_sha256(theta0) != str(protocol["base_parameter_sha256_fp64"]):
        raise RuntimeError("projected K4 FP64 base hash drift")
    if tensor_sha256(theta0.to(dtype=torch.float32)) != str(protocol["base_parameter_sha256_fp32"]):
        raise RuntimeError("projected K4 projected FP32 base hash drift")

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
                raise FloatingPointError(f"non-finite projected K4 stage {stage}")
            stage_calls += 1
            return endpoint

        return run

    stage_maps = {stage: make_stage_map(stage) for stage in stages}
    prefix_cache = TemporaryDirectory(prefix="chronotrace-k3-prefix-")
    prefix_cache_dir = Path(prefix_cache.name)
    cached_prefixes: set[tuple[str, ...]] = set()

    def cache_exact_degree_three_prefix(word: tuple[str, ...], endpoint: Any) -> None:
        if len(word) != 3:
            return
        path = prefix_cache_dir / ("".join(word) + ".pt")
        if path.exists():
            raise RuntimeError("duplicate exact K3 prefix cache entry")
        torch.save(endpoint, path)
        cached_prefixes.add(word)

    basis = measure_ordered_interaction_basis_streaming_exact(
        stage_maps,
        theta0,
        max_degree=3,
        endpoint_observer=cache_exact_degree_three_prefix,
    )
    expected_k3_calls = ordered_probe_count(len(stages), 3)
    expected_degree_three_prefixes = {word[:3] for word in degree4_words}
    if cached_prefixes != expected_degree_three_prefixes:
        raise RuntimeError("exact K3 prefix cache coverage drift")
    if expected_k3_calls != int(protocol["k3_basis_stage_executions"]):
        raise RuntimeError("frozen K3 basis execution count drift")
    if basis.stage_executions != expected_k3_calls or stage_calls != expected_k3_calls:
        raise RuntimeError("observed K3 basis execution count drift")
    if any(value.dtype != torch.float64 for value in basis.interactions.values()):
        raise TypeError("projected K4 K3 basis contains non-FP64 tensor")

    target = theta0
    for stage in target_history:
        target = stage_maps[stage](target)
    target_hash = tensor_sha256(target)
    if target_hash != str(protocol["target_endpoint_sha256"]):
        raise RuntimeError("projected K4 ABCD target failed exact frozen replay")

    witness_classes = tuple(str(value) for value in protocol["witness_classes"])
    if witness_classes != stages:
        raise RuntimeError("projected K4 witness-class order drift")
    source_lower_bounds = {
        str(stage): float(value)
        for stage, value in source_k3_selection["candidate_last_dual_lower_bounds"].items()
    }
    witness_weight_tolerance = float(protocol["k3_projection_weight_tol"])
    witness_unit_tolerance = float(protocol["unit_witness_norm_tol"])
    projection_abs_tol = float(protocol["projection_numeric_abs_tol"])
    projection_rel_tol = float(protocol["projection_numeric_rel_tol"])

    witnesses: dict[str, Any] = {}
    witness_k3: dict[str, dict[str, Any]] = {}
    for stage in witness_classes:
        histories = tuple(history for history in permutations(stages) if history[-1] == stage)
        witness, projection, lower_bound, unit_norm = _candidate_k3_witness(
            torch,
            target,
            histories,
            basis,
            weight_tolerance=witness_weight_tolerance,
        )
        if abs(unit_norm - 1.0) > witness_unit_tolerance:
            raise FloatingPointError(f"K3 witness {stage} is not unit norm")
        if not math.isclose(
            lower_bound,
            source_lower_bounds[stage],
            rel_tol=projection_rel_tol,
            abs_tol=projection_abs_tol,
        ):
            raise RuntimeError(f"K3 witness {stage} lower-bound reproduction drift")
        witnesses[stage] = witness
        witness_k3[stage] = {
            "source_dual_lower_bound": source_lower_bounds[stage],
            "recomputed_dual_lower_bound": lower_bound,
            "primal_convex_hull_distance": projection.distance,
            "primal_dual_gap": projection.distance - lower_bound,
            "active_support": list(projection.support),
            "convex_weights": projection.weights.tolist(),
            "unit_norm": unit_norm,
        }

    witness_freeze_stage_calls = stage_calls
    expected_before_k4 = expected_k3_calls + int(protocol["target_replay_stage_executions"])
    if witness_freeze_stage_calls != expected_before_k4:
        raise RuntimeError("K3 witnesses were not frozen before degree-four measurement")

    projected_tables: dict[str, dict[tuple[str, ...], float]] = {}
    target_minus_base_projection: dict[str, float] = {}
    base_projection: dict[str, float] = {}
    for stage in witness_classes:
        witness = witnesses[stage]
        base_projection[stage] = _dot(torch, witness, theta0)
        target_minus_base_projection[stage] = _dot(torch, witness, target) - base_projection[stage]
        projected_tables[stage] = {
            word: _dot(torch, witness, interaction)
            for word, interaction in basis.interactions.items()
        }

    endpoint_projections: dict[str, dict[str, float]] = {stage: {} for stage in witness_classes}
    degree4_interactions: dict[str, dict[str, float]] = {stage: {} for stage in witness_classes}
    direct_endpoint_errors: dict[str, float] = {}
    projected_reconstruction_residual_max = 0.0
    active_lift_target_hash_match = False

    for word in degree4_words:
        prefix = word[:-1]
        prefix_path = prefix_cache_dir / ("".join(prefix) + ".pt")
        if not prefix_path.exists():
            raise RuntimeError("missing exact K3 prefix cache entry during active lift")
        initial = torch.load(prefix_path, map_location=device, weights_only=True)
        if initial.dtype != torch.float64 or initial.shape != theta0.shape:
            raise RuntimeError("cached exact K3 prefix tensor drift")
        endpoint = stage_maps[word[-1]](initial)
        prefix_path.unlink()
        label = "".join(word)
        direct_endpoint_errors[label] = float(torch.linalg.vector_norm(target - endpoint))
        if word == target_history:
            active_lift_target_hash_match = tensor_sha256(endpoint) == target_hash
        for stage in witness_classes:
            endpoint_minus_base = _dot(torch, witnesses[stage], endpoint) - base_projection[stage]
            endpoint_projections[stage][label] = endpoint_minus_base
            interaction4 = projected_interaction_from_endpoint_delta(
                word,
                endpoint_minus_base,
                projected_tables[stage],
            )
            projected_tables[stage][word] = interaction4
            degree4_interactions[stage][label] = interaction4
            reconstructed = projected_word_prediction(
                word,
                projected_tables[stage],
                max_degree=4,
            )
            projected_reconstruction_residual_max = max(
                projected_reconstruction_residual_max,
                abs(reconstructed - endpoint_minus_base),
            )
        del endpoint
        del initial

    prefix_cache_consumed = not any(prefix_cache_dir.iterdir())
    prefix_cache.cleanup()
    if not prefix_cache_consumed:
        raise RuntimeError("exact K3 prefix cache was not fully consumed")
    if not active_lift_target_hash_match:
        raise RuntimeError("ABCD K3-prefix active lift did not reproduce the target endpoint")
    if projected_reconstruction_residual_max > projection_abs_tol:
        raise RuntimeError("projected degree-four Mobius reconstruction residual exceeded tolerance")
    if stage_calls != int(protocol["total_expected_stage_executions"]):
        raise RuntimeError("projected K4 total stage execution count drift")

    hierarchy = build_local_order_hierarchy(stages, max_degree=4)
    if hierarchy.dimension != int(protocol["hierarchy_coordinate_count"]):
        raise RuntimeError("projected K4 hierarchy dimension drift")

    del basis
    del stage_maps
    del witnesses
    del model
    del examples
    del worlds
    del tokenizer
    gc.collect()

    lp_guard = float(protocol["lp_certificate_guard"])
    elimination_guard = float(protocol["numerical_elimination_guard"])
    candidate_last: dict[str, dict[str, Any]] = {}
    eliminated_by_k4: list[str] = []
    all_terminal_exact = True
    all_dual_conservative = True
    all_directional_vs_euclidean = True

    for stage in stages:
        constant, coefficients = projected_interaction_linear_objective(
            hierarchy,
            projected_tables[stage],
            target_minus_base_projection=target_minus_base_projection[stage],
        )
        lp = solve_local_order_lp(
            hierarchy,
            constant,
            coefficients,
            last_stage=stage,
            certificate_guard=lp_guard,
        )
        direct_scores_from_hierarchy = full_level_permutation_scores(
            hierarchy,
            constant,
            coefficients,
            last_stage=stage,
        )
        histories = tuple(history for history in permutations(stages) if history[-1] == stage)
        direct_scores_from_endpoint_projection = {
            history: target_minus_base_projection[stage] - endpoint_projections[stage]["".join(history)]
            for history in histories
        }
        exact_directional_min = min(direct_scores_from_endpoint_projection.values())
        hierarchy_directional_min = min(direct_scores_from_hierarchy.values())
        exact_euclidean_min = min(direct_endpoint_errors["".join(history)] for history in histories)

        direct_score_residual = max(
            abs(direct_scores_from_hierarchy[history] - direct_scores_from_endpoint_projection[history])
            for history in histories
        )
        primal_exact = math.isclose(
            lp.primal_objective,
            exact_directional_min,
            rel_tol=projection_rel_tol,
            abs_tol=projection_abs_tol,
        )
        hierarchy_exact = math.isclose(
            hierarchy_directional_min,
            exact_directional_min,
            rel_tol=projection_rel_tol,
            abs_tol=projection_abs_tol,
        )
        dual_conservative = lp.certified_lower_bound <= exact_directional_min + projection_abs_tol
        distance_lower_bound = max(0.0, lp.certified_lower_bound)
        directional_vs_euclidean = distance_lower_bound <= exact_euclidean_min + projection_abs_tol
        all_terminal_exact = all_terminal_exact and primal_exact and hierarchy_exact and direct_score_residual <= projection_abs_tol
        all_dual_conservative = all_dual_conservative and dual_conservative
        all_directional_vs_euclidean = all_directional_vs_euclidean and directional_vs_euclidean

        eliminated = distance_lower_bound > float(protocol["exact_feasible_upper_bound_error"]) + elimination_guard
        if eliminated:
            eliminated_by_k4.append(stage)
        candidate_last[stage] = {
            "decision_role": "survivor_decision" if stage in survivor_stages else "terminal_exactness_diagnostic_only",
            "corrected_lp_lower_bound_raw": lp.certified_lower_bound,
            "euclidean_distance_lower_bound": distance_lower_bound,
            "lp_primal_objective": lp.primal_objective,
            "lp_primal_dual_gap": lp.primal_dual_gap,
            "lp_equality_residual_max": lp.equality_residual_max,
            "lp_minimum_weight": lp.minimum_weight,
            "lp_raw_dual_objective": lp.raw_dual_objective,
            "lp_minimum_reduced_cost": lp.minimum_reduced_cost,
            "lp_subset_residual_correction": lp.subset_residual_correction,
            "direct_exact_directional_minimum": exact_directional_min,
            "hierarchy_exact_directional_minimum": hierarchy_directional_min,
            "direct_exact_euclidean_class_distance": exact_euclidean_min,
            "direct_score_reconstruction_residual_max": direct_score_residual,
            "terminal_primal_matches_exact": primal_exact,
            "terminal_hierarchy_matches_direct": hierarchy_exact,
            "corrected_dual_is_conservative": dual_conservative,
            "distance_lower_bound_below_direct_euclidean": directional_vs_euclidean,
            "k4_certified_impossible": eliminated,
        }

    d_zero = candidate_last["D"]["direct_exact_euclidean_class_distance"] <= projection_abs_tol
    d_survives = "D" not in eliminated_by_k4
    c_eliminated = "C" in eliminated_by_k4
    checks = {
        "source_k3_partial_pruning_reproduced": tuple(source_k3_selection["surviving_last_stages"]) == ("C", "D"),
        "target_endpoint_replay_exact": target_hash == str(protocol["target_endpoint_sha256"]),
        "exact_degree_three_prefix_cache_complete": len(cached_prefixes) == int(protocol["degree4_word_count"]),
        "exact_degree_three_prefix_cache_consumed": prefix_cache_consumed,
        "witnesses_frozen_before_any_k4_output": witness_freeze_stage_calls == expected_before_k4,
        "all_k3_witnesses_unit_norm": all(
            abs(float(witness_k3[stage]["unit_norm"]) - 1.0) <= witness_unit_tolerance
            for stage in stages
        ),
        "all_k3_witness_lower_bounds_reproduce_source": all(
            math.isclose(
                float(witness_k3[stage]["recomputed_dual_lower_bound"]),
                float(witness_k3[stage]["source_dual_lower_bound"]),
                rel_tol=projection_rel_tol,
                abs_tol=projection_abs_tol,
            )
            for stage in stages
        ),
        "active_lift_ABCD_replays_target_hash": active_lift_target_hash_match,
        "all_projected_mobius_reconstructions_match": projected_reconstruction_residual_max <= projection_abs_tol,
        "all_terminal_k4_lp_primal_scores_exact": all_terminal_exact,
        "all_corrected_dual_bounds_conservative": all_dual_conservative,
        "all_directional_bounds_below_direct_euclidean_class_distance": all_directional_vs_euclidean,
        "D_direct_exact_class_distance_zero": d_zero,
        "D_not_eliminated": d_survives,
        "stage_execution_count_exact": stage_calls == int(protocol["total_expected_stage_executions"]),
        "retained_degree4_scalar_count_exact": sum(len(values) for values in degree4_interactions.values()) == int(protocol["retained_degree4_projection_scalars"]),
        "confirmation_codebooks_observed": False,
        "heldout_confirmation_launch_authorized": False,
    }
    all_checks_passed = all(value is True or value is False for value in checks.values()) and all(
        value is True
        for key, value in checks.items()
        if key not in {"confirmation_codebooks_observed", "heldout_confirmation_launch_authorized"}
    )
    invalid = (not all_checks_passed) or (not d_survives)
    strong_success = all_checks_passed and c_eliminated and d_survives
    scientific_negative = all_checks_passed and (not c_eliminated) and d_survives

    result = {
        "protocol": protocol,
        "protocol_sha256": json_sha256(protocol),
        "implementation_amendment": amendment,
        "implementation_amendment_sha256": json_sha256(amendment),
        "source_k3_convex_protocol_sha256": json_sha256(source_k3_protocol),
        "source_k3_convex_selection_sha256": json_sha256(source_k3_selection),
        "numerical_fingerprint": numerical_fingerprint,
        "target_endpoint_sha256": target_hash,
        "k3_measurement_path": "streaming_exact_direct_prefix",
        "stage_executions": stage_calls,
        "witness_freeze_stage_executions": witness_freeze_stage_calls,
        "witness_k3": witness_k3,
        "degree4_interaction_projections": degree4_interactions,
        "degree4_endpoint_projections": endpoint_projections,
        "degree4_direct_endpoint_errors": direct_endpoint_errors,
        "projected_reconstruction_residual_max": projected_reconstruction_residual_max,
        "candidate_last": candidate_last,
        "eliminated_by_k4_diagnostic": eliminated_by_k4,
        "survivor_decision": {
            "C_certified_impossible": c_eliminated,
            "D_survives": d_survives,
        },
        "checks": checks,
        "all_diagnostic_checks_passed": all_checks_passed,
        "strong_success": strong_success,
        "scientific_negative": scientific_negative,
        "invalid_run": invalid,
        "confirmation_codebooks_observed": False,
        "heldout_confirmation_launch_authorized": False,
        "scientific_interpretation": (
            "Exact projected K4 resolves the spent C/D ambiguity with a proof-safe certificate."
            if strong_success
            else "Exact projected K4 does not resolve the spent C/D ambiguity under the frozen K3 witness."
            if scientific_negative
            else "Run invalid: a frozen replay, terminal-exactness, or certificate sanity gate failed."
        ),
    }

    output = Path(args.output)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
