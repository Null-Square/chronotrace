#!/usr/bin/env python3
# ruff: noqa: E501,I001
"""Run the frozen 32-case pairwise multi-witness confirmation for one heldout seed."""

from __future__ import annotations

import argparse
import gc
import json
import math
from itertools import combinations, permutations
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from pythia_finite_pair_bridge import _configure_portable_numerics, _load_json

from chronotrace.geometry.convex_quadratic import dual_quadratic_hull_certificate, project_quadratic_simplex
from chronotrace.geometry.interactions import measure_ordered_interaction_basis_streaming_exact, ordered_interaction_word_prediction, ordered_probe_count
from chronotrace.geometry.local_order_hierarchy import build_local_order_hierarchy, projected_interaction_linear_objective
from chronotrace.geometry.multi_witness_local_order import solve_local_order_multi_witness_lp
from chronotrace.geometry.projected_interactions import projected_interaction_from_endpoint_delta, projected_word_prediction
from chronotrace.reproducibility import json_sha256, tensor_sha256
from chronotrace.scale import build_scale_worlds_from_codebook
from chronotrace.scale_four import build_four_stage_examples, four_stage_dataset_payload
from chronotrace.scale_runner import execute_plain_sgd_stage, flatten_parameters
from chronotrace.scale_tokens import build_token_codebook, tokenizer_fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation-lock", default="configs/chronotrace_pairwise_multi_witness_confirmation.lock.json")
    parser.add_argument("--methodology-lock", default="configs/chronotrace_pairwise_multi_witness_methodology.lock.json")
    parser.add_argument("--source-k3-protocol", default="configs/pythia_14m_k3_convex_last_stage_diagnostic.lock.json")
    parser.add_argument("--source-k23-protocol", default="configs/pythia_14m_four_stage_k23_pilot.lock.json")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def _dot(torch: Any, left: Any, right: Any) -> float:
    return float(torch.dot(left.reshape(-1), right.reshape(-1)))


def _candidate_witness(torch: Any, target: Any, histories: tuple[tuple[str, ...], ...], basis: Any, *, weight_tolerance: float) -> tuple[Any, Any, float, float]:
    displacements = [ordered_interaction_word_prediction(history, basis, degree=3) - target for history in histories]
    count = len(displacements)
    gram = np.empty((count, count), dtype=np.float64)
    for left in range(count):
        for right in range(left + 1):
            value = _dot(torch, displacements[left], displacements[right])
            gram[left, right] = value
            gram[right, left] = value
    projection = project_quadratic_simplex(gram, np.zeros(count), 0.0, weight_tolerance=weight_tolerance)
    certificate = dual_quadratic_hull_certificate(gram, np.zeros(count), 0.0, projection)
    mixed = torch.zeros_like(target)
    for weight, displacement in zip(projection.weights, displacements, strict=True):
        if weight:
            mixed = mixed + float(weight) * displacement
    norm = float(torch.linalg.vector_norm(mixed))
    if not math.isfinite(norm) or norm <= 0.0:
        raise FloatingPointError("K3 candidate hull produced a zero/non-finite witness residual")
    witness = -mixed / norm
    unit_norm = float(torch.linalg.vector_norm(witness))
    del displacements, mixed
    gc.collect()
    return witness, projection, certificate.lower_bound, unit_norm


def _reconstruct_total_order(stages: tuple[str, ...], relations: tuple[tuple[str, str], ...]) -> str | None:
    if len(relations) != math.comb(len(stages), 2):
        return None
    predecessors = {stage: 0 for stage in stages}
    seen: set[frozenset[str]] = set()
    for before, after in relations:
        pair = frozenset((before, after))
        if before == after or before not in predecessors or after not in predecessors or pair in seen:
            return None
        seen.add(pair)
        predecessors[after] += 1
    if sorted(predecessors.values()) != list(range(len(stages))):
        return None
    return "".join(sorted(stages, key=predecessors.__getitem__))


def _validate_locks(confirmation: dict[str, Any], methodology: dict[str, Any], source_k3: dict[str, Any], source_k23: dict[str, Any], seed: int) -> None:
    if confirmation.get("freeze_status") != "frozen_before_any_heldout_confirmation_output" or confirmation.get("experiment_role") != "confirmatory_heldout_validation":
        raise RuntimeError("confirmation lock status/role drift")
    if confirmation.get("heldout_confirmation_launch_authorized") is not True or confirmation.get("confirmation_codebooks_observed_before_freeze") is not False:
        raise RuntimeError("confirmation launch/freeze policy drift")
    if confirmation.get("no_intermediate_adaptation") is not True:
        raise RuntimeError("confirmation suite must prohibit intermediate adaptation")
    if methodology.get("methodology_version") != confirmation["methodology_version"] or methodology.get("freeze_status") != "frozen_before_any_heldout_confirmation_output":
        raise RuntimeError("methodology lock drift")
    if methodology.get("method_head_commit") != confirmation["method_head_commit"] or int(methodology["method_ci_run"]) != int(confirmation["method_ci_run"]) or methodology.get("method_ci_conclusion") != "success":
        raise RuntimeError("frozen method provenance drift")
    if methodology.get("confirmation_codebooks_observed_before_freeze") is not False or source_k3.get("confirmation_codebooks_observed") is not False or source_k23.get("confirmation_codebooks_observed") is not False:
        raise RuntimeError("a source lock observed confirmation data before freeze")
    if json_sha256(source_k23) != str(source_k3["source_k23_protocol_sha256"]):
        raise RuntimeError("source K23 protocol hash drift")
    if seed not in {int(value) for value in confirmation["heldout_seeds"]} or seed not in {int(value) for value in source_k3["confirmation_codebooks_prohibited"]}:
        raise RuntimeError("seed is not a frozen previously-heldout seed")
    if tuple(confirmation["stages"]) != ("A", "B", "C", "D") or int(confirmation["confirmation_case_count"]) != 32 or int(confirmation["pairwise_decision_count"]) != 192:
        raise RuntimeError("confirmation design drift")


def main() -> None:
    args = parse_args()
    confirmation = _load_json(args.confirmation_lock)
    methodology = _load_json(args.methodology_lock)
    source_k3 = _load_json(args.source_k3_protocol)
    source_k23 = _load_json(args.source_k23_protocol)
    seed = int(args.seed)
    _validate_locks(confirmation, methodology, source_k3, source_k23, seed)

    stages = tuple(str(value) for value in confirmation["stages"])
    target_labels = tuple(str(value) for value in confirmation["target_histories_per_seed"])
    if len(target_labels) != 8 or len(set(target_labels)) != 8 or any(len(label) != 4 or set(label) != set(stages) for label in target_labels):
        raise RuntimeError("confirmation target history drift")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    numerical_fingerprint = _configure_portable_numerics(torch, source_k23, int(args.threads))
    device = torch.device("cpu")
    model_id, revision = str(confirmation["model"]), str(confirmation["revision"])
    learning_rate = float(confirmation["learning_rate"])
    if (model_id, revision, learning_rate) != (str(source_k3["model"]), str(source_k3["revision"]), float(source_k3["learning_rate"])):
        raise RuntimeError("confirmation model settings drift")

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer_fingerprint(tokenizer) != str(source_k3["tokenizer_fingerprint"]):
        raise RuntimeError("confirmation tokenizer fingerprint drift")
    codebook = build_token_codebook(tokenizer, count=int(confirmation["codebook_count_per_kind"]), seed=seed)
    dataset = four_stage_dataset_payload(tokenizer, codebook)
    worlds = build_scale_worlds_from_codebook(codebook)
    examples = {stage: build_four_stage_examples(worlds, stage) for stage in stages}

    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision)
    model.to(device=device, dtype=torch.float64)
    model.config.use_cache = False
    if any(parameter.is_floating_point() and parameter.dtype != torch.float64 for parameter in model.parameters()):
        raise TypeError("could not enforce FP64 confirmation model")
    theta0 = flatten_parameters(torch, model, preserve_parameter_dtype=True)
    if theta0.dtype != torch.float64 or tensor_sha256(theta0) != str(source_k3["base_parameter_sha256_fp64"]) or tensor_sha256(theta0.to(dtype=torch.float32)) != str(source_k3["base_parameter_sha256_fp32"]):
        raise RuntimeError("confirmation base parameter hash/dtype drift")

    stage_seeds = {stage: int(confirmation["stage_randomness"][stage]) for stage in stages}
    if stage_seeds != {stage: int(source_k3["stage_randomness"][stage]) for stage in stages}:
        raise RuntimeError("confirmation stage randomness drift")
    stage_calls = 0

    def make_stage_map(stage: str):
        def run(initial_vector: Any) -> Any:
            nonlocal stage_calls
            torch.manual_seed(stage_seeds[stage])
            endpoint, metrics = execute_plain_sgd_stage(torch, model, tokenizer, examples[stage], learning_rate=learning_rate, updates=1, device=device, initial_vector=initial_vector, preserve_parameter_dtype=True)
            if not metrics.finite:
                raise FloatingPointError(f"non-finite confirmation stage {stage}")
            stage_calls += 1
            return endpoint
        return run

    stage_maps = {stage: make_stage_map(stage) for stage in stages}
    degree4_words = tuple(permutations(stages))
    prefix_cache = TemporaryDirectory(prefix=f"chronotrace-confirm-prefix-{seed}-")
    target_cache = TemporaryDirectory(prefix=f"chronotrace-confirm-target-{seed}-")
    witness_cache = TemporaryDirectory(prefix=f"chronotrace-confirm-witness-{seed}-")
    prefix_dir, target_dir, witness_dir = Path(prefix_cache.name), Path(target_cache.name), Path(witness_cache.name)
    cached_prefixes: set[tuple[str, ...]] = set()

    def cache_prefix(word: tuple[str, ...], endpoint: Any) -> None:
        if len(word) == 3:
            path = prefix_dir / ("".join(word) + ".pt")
            if path.exists():
                raise RuntimeError("duplicate exact K3 prefix cache entry")
            torch.save(endpoint, path)
            cached_prefixes.add(word)

    basis = measure_ordered_interaction_basis_streaming_exact(stage_maps, theta0, max_degree=3, endpoint_observer=cache_prefix)
    if ordered_probe_count(4, 3) != 40 or basis.stage_executions != 40 or stage_calls != 40 or cached_prefixes != {word[:-1] for word in degree4_words}:
        raise RuntimeError("confirmation K3 basis/prefix count drift")
    if any(value.dtype != torch.float64 for value in basis.interactions.values()):
        raise TypeError("confirmation K3 basis contains non-FP64 tensor")

    weight_tol = float(methodology["witness_bank"]["simplex_weight_tolerance"])
    unit_tol = float(methodology["witness_bank"]["unit_norm_tolerance"])
    projection_abs_tol = float(methodology["interaction_measurement"]["projected_reconstruction_abs_tolerance"])
    projection_rel_tol = float(methodology["interaction_measurement"]["projected_reconstruction_rel_tolerance"])
    witness_meta: dict[str, dict[str, dict[str, Any]]] = {}
    projected_tables: dict[str, dict[str, dict[tuple[str, ...], float]]] = {}
    target_minus_base: dict[str, dict[str, float]] = {}
    base_projection: dict[str, dict[str, float]] = {}
    target_hashes: dict[str, str] = {}

    for target_label in target_labels:
        target = theta0
        for stage in target_label:
            target = stage_maps[stage](target)
        target_hashes[target_label] = tensor_sha256(target)
        torch.save(target, target_dir / f"{target_label}.pt")
        witness_meta[target_label], projected_tables[target_label], target_minus_base[target_label], base_projection[target_label] = {}, {}, {}, {}
        for candidate_last in stages:
            histories = tuple(history for history in degree4_words if history[-1] == candidate_last)
            witness, projection, lower_bound, unit_norm = _candidate_witness(torch, target, histories, basis, weight_tolerance=weight_tol)
            if abs(unit_norm - 1.0) > unit_tol or abs(float(np.sum(projection.weights)) - 1.0) > weight_tol or float(np.min(projection.weights)) < -weight_tol:
                raise FloatingPointError("confirmation witness geometry drift")
            torch.save(witness, witness_dir / f"{target_label}-{candidate_last}.pt")
            base_value = _dot(torch, witness, theta0)
            base_projection[target_label][candidate_last] = base_value
            target_minus_base[target_label][candidate_last] = _dot(torch, witness, target) - base_value
            projected_tables[target_label][candidate_last] = {word: _dot(torch, witness, interaction) for word, interaction in basis.interactions.items()}
            witness_meta[target_label][candidate_last] = {"histories": ["".join(history) for history in histories], "weights": projection.weights.tolist(), "support": list(projection.support), "primal_convex_hull_distance": projection.distance, "dual_lower_bound": lower_bound, "unit_norm": unit_norm}
            del witness
            gc.collect()
        del target
        gc.collect()

    expected_freeze_calls = int(confirmation["execution_sharing"]["witness_freeze_stage_executions_per_seed"])
    if stage_calls != expected_freeze_calls or expected_freeze_calls != 72:
        raise RuntimeError("confirmation witness freeze boundary drift")
    del basis
    gc.collect()

    targets = {label: torch.load(target_dir / f"{label}.pt", map_location=device, weights_only=True) for label in target_labels}
    witnesses = {(label, candidate_last): torch.load(witness_dir / f"{label}-{candidate_last}.pt", map_location=device, weights_only=True) for label in target_labels for candidate_last in stages}
    for path in target_dir.iterdir():
        path.unlink()
    for path in witness_dir.iterdir():
        path.unlink()
    target_cache.cleanup()
    witness_cache.cleanup()

    endpoint_projections = {label: {stage: {} for stage in stages} for label in target_labels}
    degree4_interactions = {label: {stage: {} for stage in stages} for label in target_labels}
    direct_errors = {label: {} for label in target_labels}
    active_hash_match = {label: False for label in target_labels}
    projected_residual_max = 0.0

    for word in degree4_words:
        path = prefix_dir / ("".join(word[:-1]) + ".pt")
        if not path.exists():
            raise RuntimeError("missing exact K3 prefix during confirmation active lift")
        initial = torch.load(path, map_location=device, weights_only=True)
        endpoint = stage_maps[word[-1]](initial)
        path.unlink()
        word_label = "".join(word)
        for target_label in target_labels:
            direct_errors[target_label][word_label] = float(torch.linalg.vector_norm(targets[target_label] - endpoint))
            if word_label == target_label:
                active_hash_match[target_label] = tensor_sha256(endpoint) == target_hashes[target_label]
            for candidate_last in stages:
                endpoint_minus_base = _dot(torch, witnesses[(target_label, candidate_last)], endpoint) - base_projection[target_label][candidate_last]
                endpoint_projections[target_label][candidate_last][word_label] = endpoint_minus_base
                interaction4 = projected_interaction_from_endpoint_delta(word, endpoint_minus_base, projected_tables[target_label][candidate_last])
                projected_tables[target_label][candidate_last][word] = interaction4
                degree4_interactions[target_label][candidate_last][word_label] = interaction4
                reconstructed = projected_word_prediction(word, projected_tables[target_label][candidate_last], max_degree=4)
                projected_residual_max = max(projected_residual_max, abs(reconstructed - endpoint_minus_base))
        del endpoint, initial

    prefix_consumed = not any(prefix_dir.iterdir())
    prefix_cache.cleanup()
    if not prefix_consumed or not all(active_hash_match.values()) or not math.isclose(projected_residual_max, 0.0, rel_tol=projection_rel_tol, abs_tol=projection_abs_tol):
        raise RuntimeError("confirmation active-lift/reconstruction validity gate failed")
    expected_stage_calls = int(confirmation["execution_sharing"]["total_stage_executions_per_seed"])
    if stage_calls != expected_stage_calls or expected_stage_calls != 96:
        raise RuntimeError("confirmation stage execution count drift")
    endpoint_scalar_count = sum(len(values) for target_values in endpoint_projections.values() for values in target_values.values())
    interaction_scalar_count = sum(len(values) for target_values in degree4_interactions.values() for values in target_values.values())
    if endpoint_scalar_count != 768 or interaction_scalar_count != 768:
        raise RuntimeError("confirmation retained scalar count drift")

    del witnesses, targets, stage_maps, model, examples, worlds, tokenizer
    gc.collect()

    hierarchy = build_local_order_hierarchy(stages, max_degree=4)
    if hierarchy.dimension != int(methodology["certificate"]["terminal_hierarchy_dimension"]):
        raise RuntimeError("confirmation hierarchy dimension drift")
    certificate_guard = float(confirmation["certificate"]["certificate_guard"])
    elimination_guard = float(confirmation["certificate"]["elimination_guard"])
    cases: dict[str, Any] = {}
    invalid = False
    full_successes = pair_successes = 0
    minimum_margin = float("inf")

    for target_label in target_labels:
        constants = np.empty(4, dtype=np.float64)
        coefficients = np.empty((4, hierarchy.dimension), dtype=np.float64)
        for index, candidate_last in enumerate(stages):
            constant, row = projected_interaction_linear_objective(hierarchy, projected_tables[target_label][candidate_last], target_minus_base_projection=target_minus_base[target_label][candidate_last])
            constants[index], coefficients[index] = constant, row
        position = {stage: index for index, stage in enumerate(target_label)}
        pair_results: dict[str, Any] = {}
        surviving: list[tuple[str, str]] = []
        case_invalid = False
        case_pairs = 0
        for left, right in combinations(stages, 2):
            true_relation = (left, right) if position[left] < position[right] else (right, left)
            wrong_relation = (true_relation[1], true_relation[0])
            wrong = solve_local_order_multi_witness_lp(hierarchy, constants, coefficients, precedences=(wrong_relation,), certificate_guard=certificate_guard)
            true = solve_local_order_multi_witness_lp(hierarchy, constants, coefficients, precedences=(true_relation,), certificate_guard=certificate_guard)
            wrong_histories = ["".join(history) for history in degree4_words if history.index(wrong_relation[0]) < history.index(wrong_relation[1])]
            exact_wrong = min(direct_errors[target_label][history] for history in wrong_histories)
            lower_sound = wrong.euclidean_distance_lower_bound <= exact_wrong + projection_abs_tol
            true_sane = true.euclidean_distance_lower_bound <= elimination_guard
            wrong_certified = wrong.euclidean_distance_lower_bound > elimination_guard
            if not lower_sound or not true_sane:
                case_invalid = True
            if wrong_certified:
                case_pairs += 1
                pair_successes += 1
                surviving.append(true_relation)
                minimum_margin = min(minimum_margin, wrong.euclidean_distance_lower_bound - elimination_guard)
            pair_results[f"{left}{right}"] = {"true_relation": list(true_relation), "wrong_relation": list(wrong_relation), "wrong_class_primal_objective": wrong.primal_objective, "wrong_class_certified_lower_bound": wrong.certified_lower_bound, "wrong_class_euclidean_distance_lower_bound": wrong.euclidean_distance_lower_bound, "wrong_class_direct_exact_euclidean_distance": exact_wrong, "wrong_lower_bound_sound": lower_sound, "wrong_relation_certified_impossible": wrong_certified, "true_class_primal_objective": true.primal_objective, "true_class_euclidean_distance_lower_bound": true.euclidean_distance_lower_bound, "true_relation_sanity_passed": true_sane}
        reconstructed = _reconstruct_total_order(stages, tuple(surviving))
        full_certified = not case_invalid and case_pairs == 6 and reconstructed == target_label
        if full_certified:
            full_successes += 1
        invalid = invalid or case_invalid
        cases[target_label] = {"target_endpoint_sha256": target_hashes[target_label], "witness_k3": witness_meta[target_label], "active_lift_target_hash_match": active_hash_match[target_label], "pairwise": pair_results, "pairwise_wrong_orientation_certificates": case_pairs, "full_history_certified": full_certified, "reconstructed_history": reconstructed, "invalid": case_invalid}

    result = {"confirmation_lock": confirmation, "confirmation_lock_sha256": json_sha256(confirmation), "methodology_lock_sha256": json_sha256(methodology), "source_k3_protocol_sha256": json_sha256(source_k3), "source_k23_protocol_sha256": json_sha256(source_k23), "seed": seed, "codebook_sha256": codebook.sha256, "dataset_sha256": dataset["sha256"], "numerical_fingerprint": numerical_fingerprint, "stage_executions": stage_calls, "witness_freeze_stage_executions": expected_freeze_calls, "projected_reconstruction_residual_max": projected_residual_max, "prefix_cache_consumed": prefix_consumed, "retained_degree4_endpoint_projection_scalars": endpoint_scalar_count, "retained_degree4_interaction_projection_scalars": interaction_scalar_count, "full_k4_model_space_tensors_retained": False, "cases": cases, "full_history_certificate_coverage": full_successes, "pairwise_wrong_orientation_certificate_coverage": pair_successes, "minimum_wrong_orientation_margin_over_guard": None if minimum_margin == float("inf") else minimum_margin, "invalid_seed_job": invalid, "confirmation_codebooks_observed": True, "heldout_confirmation_launch_authorized": True}
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
