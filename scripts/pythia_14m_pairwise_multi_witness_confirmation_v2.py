#!/usr/bin/env python3
# ruff: noqa: E501,I001
"""Run one seed of the frozen label-blind 32-case multi-witness confirmation."""

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
    projected_interaction_linear_objective,
)
from chronotrace.geometry.pairwise_certificate import certify_pairwise_orientation
from chronotrace.geometry.projected_interactions import (
    projected_interaction_from_endpoint_delta,
    projected_word_prediction,
)
from chronotrace.reproducibility import json_sha256, tensor_sha256
from chronotrace.scale import build_scale_worlds_from_codebook
from chronotrace.scale_four import build_four_stage_examples, four_stage_dataset_payload
from chronotrace.scale_runner import execute_plain_sgd_stage, flatten_parameters
from chronotrace.scale_tokens import build_token_codebook, tokenizer_fingerprint


PAIR_LABELS = ("AB", "AC", "AD", "BC", "BD", "CD")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--confirmation-lock",
        default="configs/chronotrace_pairwise_multi_witness_confirmation.lock.json",
    )
    parser.add_argument(
        "--methodology-lock",
        default="configs/chronotrace_pairwise_multi_witness_methodology.lock.json",
    )
    parser.add_argument(
        "--source-k3-protocol",
        default="configs/pythia_14m_k3_convex_last_stage_diagnostic.lock.json",
    )
    parser.add_argument(
        "--source-k23-protocol",
        default="configs/pythia_14m_four_stage_k23_pilot.lock.json",
    )
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def _dot(torch: Any, left: Any, right: Any) -> float:
    return float(torch.dot(left.reshape(-1), right.reshape(-1)))


def _candidate_witness(
    torch: Any,
    target: Any,
    histories: tuple[tuple[str, ...], ...],
    basis: Any,
    *,
    weight_tolerance: float,
) -> tuple[Any, Any, float, float]:
    displacements = [
        ordered_interaction_word_prediction(history, basis, degree=3) - target
        for history in histories
    ]
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
        raise FloatingPointError("K3 candidate hull produced a zero/non-finite witness")
    witness = -mixed / norm
    unit_norm = float(torch.linalg.vector_norm(witness))
    del displacements, mixed
    gc.collect()
    return witness, projection, certificate.lower_bound, unit_norm


def _reconstruct_total_order(
    stages: tuple[str, ...],
    relations: tuple[tuple[str, str], ...],
) -> str | None:
    if len(relations) != math.comb(len(stages), 2):
        return None
    predecessors = {stage: 0 for stage in stages}
    seen: set[frozenset[str]] = set()
    relation_set = set(relations)
    for before, after in relations:
        pair = frozenset((before, after))
        if (
            before == after
            or before not in predecessors
            or after not in predecessors
            or pair in seen
            or (after, before) in relation_set
        ):
            return None
        seen.add(pair)
        predecessors[after] += 1
    if sorted(predecessors.values()) != list(range(len(stages))):
        return None
    ordered = tuple(sorted(stages, key=predecessors.__getitem__))
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            if (left, right) not in relation_set:
                return None
    return "".join(ordered)


def _balanced_targets(targets: tuple[str, ...], stages: tuple[str, ...]) -> bool:
    if len(targets) != 8 or len(set(targets)) != 8:
        return False
    if any(len(target) != 4 or set(target) != set(stages) for target in targets):
        return False
    for position in range(4):
        counts = {stage: 0 for stage in stages}
        for target in targets:
            counts[target[position]] += 1
        if counts != {stage: 2 for stage in stages}:
            return False
    return True


def _validate_locks(
    confirmation: dict[str, Any],
    methodology: dict[str, Any],
    source_k3: dict[str, Any],
    source_k23: dict[str, Any],
    seed: int,
) -> None:
    if confirmation.get("confirmation_version") != (
        "chronotrace-pairwise-multi-witness-confirmation-32-v2"
    ):
        raise RuntimeError("confirmation version drift")
    if confirmation.get("freeze_status") != (
        "frozen_before_any_heldout_confirmation_output_after_label_blind_audit"
    ):
        raise RuntimeError("confirmation lock status drift")
    if confirmation.get("experiment_role") != "confirmatory_heldout_validation":
        raise RuntimeError("confirmation experiment role drift")
    if confirmation.get("heldout_confirmation_launch_authorized") is not True:
        raise RuntimeError("confirmation launch is not authorized")
    if confirmation.get("confirmation_codebooks_observed_before_freeze") is not False:
        raise RuntimeError("confirmation outputs were observed before freeze")
    if confirmation.get("no_intermediate_adaptation") is not True:
        raise RuntimeError("confirmation suite must prohibit intermediate adaptation")
    if methodology.get("methodology_version") != confirmation["methodology_version"]:
        raise RuntimeError("methodology version drift")
    if methodology.get("freeze_status") != (
        "frozen_before_any_heldout_confirmation_output_after_label_blind_audit"
    ):
        raise RuntimeError("methodology lock status drift")
    if methodology.get("method_head_commit") != confirmation["method_head_commit"]:
        raise RuntimeError("frozen method provenance drift")
    if methodology.get("confirmation_codebooks_observed_before_freeze") is not False:
        raise RuntimeError("methodology lock observed heldout data")
    if source_k3.get("confirmation_codebooks_observed") is not False:
        raise RuntimeError("source K3 lock observed heldout data")
    if source_k23.get("confirmation_codebooks_observed") is not False:
        raise RuntimeError("source K23 lock observed heldout data")
    if json_sha256(source_k23) != str(source_k3["source_k23_protocol_sha256"]):
        raise RuntimeError("source K23 protocol hash drift")
    frozen_seeds = {int(value) for value in confirmation["heldout_seeds"]}
    prohibited = {int(value) for value in source_k3["confirmation_codebooks_prohibited"]}
    if seed not in frozen_seeds or seed not in prohibited:
        raise RuntimeError("seed is not one of the frozen previously-heldout seeds")
    if tuple(confirmation["stages"]) != ("A", "B", "C", "D"):
        raise RuntimeError("confirmation stage set drift")
    if int(confirmation["confirmation_case_count"]) != 32:
        raise RuntimeError("confirmation case count drift")
    if int(confirmation["pairwise_decision_count"]) != 192:
        raise RuntimeError("confirmation pairwise decision count drift")
    if int(confirmation["orientation_class_lp_solve_count"]) != 384:
        raise RuntimeError("confirmation orientation-class LP solve count drift")


def _exact_terminal_hull_distance(
    histories: tuple[str, ...],
    precedence: tuple[str, str],
    target_projection: dict[str, float],
    endpoint_projection: dict[str, dict[str, float]],
    witnesses: tuple[str, ...],
) -> tuple[float, float, str]:
    """Independent full-permutation convex-hull benchmark for K=N validation."""
    from scipy.optimize import linprog

    before, after = precedence
    candidates = tuple(
        history
        for history in histories
        if history.index(before) < history.index(after)
    )
    score_matrix = np.asarray(
        [
            [
                float(target_projection[witness])
                - float(endpoint_projection[witness][history])
                for history in candidates
            ]
            for witness in witnesses
        ],
        dtype=np.float64,
    )
    vertex_scores = [
        (float(np.max(np.abs(score_matrix[:, index]))), history)
        for index, history in enumerate(candidates)
    ]
    nearest_vertex_distance, nearest_vertex = min(vertex_scores)
    vertex_count = len(candidates)
    objective = np.zeros(vertex_count + 1, dtype=np.float64)
    objective[-1] = 1.0
    rows: list[np.ndarray] = []
    for row in score_matrix:
        rows.append(np.concatenate((row, np.asarray([-1.0]))))
        rows.append(np.concatenate((-row, np.asarray([-1.0]))))
    equality = np.zeros((1, vertex_count + 1), dtype=np.float64)
    equality[0, :vertex_count] = 1.0
    result = linprog(
        objective,
        A_ub=np.stack(rows),
        b_ub=np.zeros(2 * len(witnesses), dtype=np.float64),
        A_eq=equality,
        b_eq=np.asarray([1.0], dtype=np.float64),
        bounds=[(0.0, None)] * (vertex_count + 1),
        method="highs",
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"terminal witness-hull benchmark failed: {result.message}")
    return float(result.x[-1]), nearest_vertex_distance, nearest_vertex


def _certificate_payload(
    certificate: Any,
    exact_hull: float,
    direct_euclidean_vertex: float,
) -> dict[str, Any]:
    return {
        "primal_objective": float(certificate.primal_objective),
        "certified_lower_bound": float(certificate.certified_lower_bound),
        "euclidean_distance_lower_bound": float(certificate.euclidean_distance_lower_bound),
        "primal_dual_gap": float(certificate.primal_dual_gap),
        "equality_residual_max": float(certificate.equality_residual_max),
        "inequality_residual_max": float(certificate.inequality_residual_max),
        "raw_dual_objective": float(certificate.raw_dual_objective),
        "dual_l1_mass": float(certificate.dual_l1_mass),
        "minimum_reduced_cost": float(certificate.minimum_reduced_cost),
        "t_reduced_cost": float(certificate.t_reduced_cost),
        "subset_residual_correction": float(certificate.subset_residual_correction),
        "exact_terminal_witness_hull_distance": float(exact_hull),
        "direct_exact_euclidean_vertex_class_distance": float(direct_euclidean_vertex),
    }


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
    if not _balanced_targets(target_labels, stages):
        raise RuntimeError("confirmation target history balance drift")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    numerical_fingerprint = _configure_portable_numerics(
        torch,
        source_k23,
        int(args.threads),
    )
    device = torch.device("cpu")
    model_id = str(confirmation["model"])
    revision = str(confirmation["revision"])
    learning_rate = float(confirmation["learning_rate"])
    if (
        model_id,
        revision,
        learning_rate,
    ) != (
        str(source_k3["model"]),
        str(source_k3["revision"]),
        float(source_k3["learning_rate"]),
    ):
        raise RuntimeError("confirmation model settings drift")

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer_fingerprint(tokenizer) != str(source_k3["tokenizer_fingerprint"]):
        raise RuntimeError("confirmation tokenizer fingerprint drift")
    codebook = build_token_codebook(
        tokenizer,
        count=int(confirmation["codebook_count_per_kind"]),
        seed=seed,
    )
    dataset = four_stage_dataset_payload(tokenizer, codebook)
    worlds = build_scale_worlds_from_codebook(codebook)
    examples = {stage: build_four_stage_examples(worlds, stage) for stage in stages}

    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision)
    model.to(device=device, dtype=torch.float64)
    model.config.use_cache = False
    if any(
        parameter.is_floating_point() and parameter.dtype != torch.float64
        for parameter in model.parameters()
    ):
        raise TypeError("could not enforce FP64 confirmation model")
    theta0 = flatten_parameters(torch, model, preserve_parameter_dtype=True)
    if theta0.dtype != torch.float64:
        raise RuntimeError("confirmation base parameter dtype drift")
    if tensor_sha256(theta0) != str(source_k3["base_parameter_sha256_fp64"]):
        raise RuntimeError("confirmation FP64 base parameter hash drift")
    if tensor_sha256(theta0.to(dtype=torch.float32)) != str(
        source_k3["base_parameter_sha256_fp32"]
    ):
        raise RuntimeError("confirmation FP32-projected base parameter hash drift")

    stage_seeds = {
        stage: int(confirmation["stage_randomness"][stage]) for stage in stages
    }
    if stage_seeds != {
        stage: int(source_k3["stage_randomness"][stage]) for stage in stages
    }:
        raise RuntimeError("confirmation stage randomness drift")
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
                raise FloatingPointError(f"non-finite confirmation stage {stage}")
            stage_calls += 1
            return endpoint

        return run

    stage_maps = {stage: make_stage_map(stage) for stage in stages}
    degree4_words = tuple(permutations(stages))
    histories = tuple("".join(word) for word in degree4_words)
    prefix_cache = TemporaryDirectory(prefix=f"chronotrace-confirm-prefix-{seed}-")
    target_cache = TemporaryDirectory(prefix=f"chronotrace-confirm-target-{seed}-")
    witness_cache = TemporaryDirectory(prefix=f"chronotrace-confirm-witness-{seed}-")
    prefix_dir = Path(prefix_cache.name)
    target_dir = Path(target_cache.name)
    witness_dir = Path(witness_cache.name)
    cached_prefixes: set[tuple[str, ...]] = set()

    def cache_prefix(word: tuple[str, ...], endpoint: Any) -> None:
        if len(word) == 3:
            path = prefix_dir / ("".join(word) + ".pt")
            if path.exists():
                raise RuntimeError("duplicate exact K3 prefix cache entry")
            torch.save(endpoint, path)
            cached_prefixes.add(word)

    basis = measure_ordered_interaction_basis_streaming_exact(
        stage_maps,
        theta0,
        max_degree=3,
        endpoint_observer=cache_prefix,
    )
    if ordered_probe_count(4, 3) != 40:
        raise RuntimeError("confirmation K3 probe count drift")
    if basis.stage_executions != 40 or stage_calls != 40:
        raise RuntimeError("confirmation K3 basis execution drift")
    if cached_prefixes != {word[:-1] for word in degree4_words}:
        raise RuntimeError("confirmation exact K3 prefix coverage drift")
    if any(value.dtype != torch.float64 for value in basis.interactions.values()):
        raise TypeError("confirmation K3 basis contains non-FP64 tensor")

    weight_tol = float(methodology["witness_bank"]["simplex_weight_tolerance"])
    unit_tol = float(methodology["witness_bank"]["unit_norm_tolerance"])
    projection_abs_tol = float(
        methodology["interaction_measurement"]["projected_reconstruction_abs_tolerance"]
    )
    projection_rel_tol = float(
        methodology["interaction_measurement"]["projected_reconstruction_rel_tolerance"]
    )
    witness_meta: dict[str, dict[str, dict[str, Any]]] = {}
    projected_tables: dict[str, dict[str, dict[tuple[str, ...], float]]] = {}
    target_minus_base: dict[str, dict[str, float]] = {}
    target_hashes: dict[str, str] = {}

    for target_label in target_labels:
        target = theta0
        for stage in target_label:
            target = stage_maps[stage](target)
        target_hashes[target_label] = tensor_sha256(target)
        torch.save(target, target_dir / f"{target_label}.pt")
        target_delta = target - theta0
        witness_meta[target_label] = {}
        projected_tables[target_label] = {}
        target_minus_base[target_label] = {}
        for candidate_last in stages:
            candidate_histories = tuple(
                history for history in degree4_words if history[-1] == candidate_last
            )
            witness, projection, lower_bound, unit_norm = _candidate_witness(
                torch,
                target,
                candidate_histories,
                basis,
                weight_tolerance=weight_tol,
            )
            if abs(unit_norm - 1.0) > unit_tol:
                raise FloatingPointError("confirmation witness unit norm drift")
            if abs(float(np.sum(projection.weights)) - 1.0) > weight_tol:
                raise FloatingPointError("confirmation witness simplex sum drift")
            if float(np.min(projection.weights)) < -weight_tol:
                raise FloatingPointError("confirmation witness simplex negativity drift")
            torch.save(witness, witness_dir / f"{target_label}-{candidate_last}.pt")
            target_minus_base[target_label][candidate_last] = _dot(
                torch,
                witness,
                target_delta,
            )
            projected_tables[target_label][candidate_last] = {
                word: _dot(torch, witness, interaction)
                for word, interaction in basis.interactions.items()
            }
            witness_meta[target_label][candidate_last] = {
                "histories": ["".join(history) for history in candidate_histories],
                "weights": projection.weights.tolist(),
                "support": list(projection.support),
                "primal_convex_hull_distance": float(projection.distance),
                "dual_lower_bound": float(lower_bound),
                "unit_norm": float(unit_norm),
            }
            del witness
            gc.collect()
        del target_delta, target
        gc.collect()

    expected_freeze_calls = int(
        confirmation["execution_sharing"]["witness_freeze_stage_executions_per_seed"]
    )
    if stage_calls != expected_freeze_calls or expected_freeze_calls != 72:
        raise RuntimeError("confirmation witness freeze boundary drift")
    del basis
    gc.collect()

    targets = {
        label: torch.load(
            target_dir / f"{label}.pt",
            map_location=device,
            weights_only=True,
        )
        for label in target_labels
    }
    witnesses = {
        (label, candidate_last): torch.load(
            witness_dir / f"{label}-{candidate_last}.pt",
            map_location=device,
            weights_only=True,
        )
        for label in target_labels
        for candidate_last in stages
    }
    for path in target_dir.iterdir():
        path.unlink()
    for path in witness_dir.iterdir():
        path.unlink()
    target_cache.cleanup()
    witness_cache.cleanup()

    endpoint_projections = {
        label: {stage: {} for stage in stages} for label in target_labels
    }
    degree4_interactions = {
        label: {stage: {} for stage in stages} for label in target_labels
    }
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
        endpoint_delta = endpoint - theta0
        word_label = "".join(word)
        for target_label in target_labels:
            direct_errors[target_label][word_label] = float(
                torch.linalg.vector_norm(targets[target_label] - endpoint)
            )
            if word_label == target_label:
                active_hash_match[target_label] = (
                    tensor_sha256(endpoint) == target_hashes[target_label]
                )
            for candidate_last in stages:
                endpoint_minus_base = _dot(
                    torch,
                    witnesses[(target_label, candidate_last)],
                    endpoint_delta,
                )
                endpoint_projections[target_label][candidate_last][
                    word_label
                ] = endpoint_minus_base
                interaction4 = projected_interaction_from_endpoint_delta(
                    word,
                    endpoint_minus_base,
                    projected_tables[target_label][candidate_last],
                )
                projected_tables[target_label][candidate_last][word] = interaction4
                degree4_interactions[target_label][candidate_last][
                    word_label
                ] = interaction4
                reconstructed = projected_word_prediction(
                    word,
                    projected_tables[target_label][candidate_last],
                    max_degree=4,
                )
                projected_residual_max = max(
                    projected_residual_max,
                    abs(reconstructed - endpoint_minus_base),
                )
        del endpoint_delta, endpoint, initial
        gc.collect()

    prefix_consumed = not any(prefix_dir.iterdir())
    prefix_cache.cleanup()
    if not prefix_consumed:
        raise RuntimeError("confirmation prefix cache was not fully consumed")
    if not all(active_hash_match.values()):
        raise RuntimeError("confirmation active-lift target replay failed")
    if not math.isclose(
        projected_residual_max,
        0.0,
        rel_tol=projection_rel_tol,
        abs_tol=projection_abs_tol,
    ):
        raise RuntimeError("confirmation projected Mobius reconstruction drift")
    expected_stage_calls = int(
        confirmation["execution_sharing"]["total_stage_executions_per_seed"]
    )
    if stage_calls != expected_stage_calls or expected_stage_calls != 96:
        raise RuntimeError("confirmation stage execution count drift")
    endpoint_scalar_count = sum(
        len(values)
        for target_values in endpoint_projections.values()
        for values in target_values.values()
    )
    interaction_scalar_count = sum(
        len(values)
        for target_values in degree4_interactions.values()
        for values in target_values.values()
    )
    if endpoint_scalar_count != 768:
        raise RuntimeError("confirmation endpoint scalar count drift")
    if interaction_scalar_count != 768:
        raise RuntimeError("confirmation interaction scalar count drift")

    del witnesses, targets, stage_maps, model, examples, worlds, tokenizer
    gc.collect()

    hierarchy = build_local_order_hierarchy(stages, max_degree=4)
    if hierarchy.dimension != int(
        methodology["certificate"]["terminal_hierarchy_dimension"]
    ):
        raise RuntimeError("confirmation hierarchy dimension drift")
    certificate_guard = float(confirmation["certificate"]["certificate_guard"])
    elimination_guard = float(confirmation["certificate"]["elimination_guard"])

    cases: dict[str, Any] = {}
    invalid = False
    full_successes = 0
    pair_successes = 0
    ambiguous_pairs = 0
    contradictory_pairs = 0
    both_excluded_pairs = 0
    minimum_margin = float("inf")
    maximum_terminal_primal_error = 0.0

    for target_label in target_labels:
        constants = np.empty(4, dtype=np.float64)
        coefficients = np.empty((4, hierarchy.dimension), dtype=np.float64)
        for index, candidate_last in enumerate(stages):
            constant, row = projected_interaction_linear_objective(
                hierarchy,
                projected_tables[target_label][candidate_last],
                target_minus_base_projection=target_minus_base[target_label][
                    candidate_last
                ],
            )
            constants[index] = constant
            coefficients[index] = row

        pair_results: dict[str, Any] = {}
        inferred_relations: list[tuple[str, str]] = []
        case_invalid = False
        case_pairs = 0
        case_ambiguous = 0
        case_contradictions = 0
        case_both_excluded = 0

        for left, right in combinations(stages, 2):
            decision = certify_pairwise_orientation(
                hierarchy,
                constants,
                coefficients,
                left,
                right,
                elimination_guard=elimination_guard,
                certificate_guard=certificate_guard,
            )
            left_relation = (left, right)
            right_relation = (right, left)
            left_hull, left_vertex_proxy, left_nearest = _exact_terminal_hull_distance(
                histories,
                left_relation,
                target_minus_base[target_label],
                endpoint_projections[target_label],
                stages,
            )
            right_hull, right_vertex_proxy, right_nearest = _exact_terminal_hull_distance(
                histories,
                right_relation,
                target_minus_base[target_label],
                endpoint_projections[target_label],
                stages,
            )
            left_histories = [
                history
                for history in histories
                if history.index(left) < history.index(right)
            ]
            right_histories = [
                history
                for history in histories
                if history.index(right) < history.index(left)
            ]
            left_euclidean = min(
                direct_errors[target_label][history] for history in left_histories
            )
            right_euclidean = min(
                direct_errors[target_label][history] for history in right_histories
            )

            class_checks = []
            for certificate, exact_hull, exact_euclidean in (
                (decision.left_before_right, left_hull, left_euclidean),
                (decision.right_before_left, right_hull, right_euclidean),
            ):
                primal_error = abs(float(certificate.primal_objective) - exact_hull)
                maximum_terminal_primal_error = max(
                    maximum_terminal_primal_error,
                    primal_error,
                )
                primal_exact = math.isclose(
                    float(certificate.primal_objective),
                    exact_hull,
                    rel_tol=projection_rel_tol,
                    abs_tol=projection_abs_tol,
                )
                lower_hull_sound = (
                    float(certificate.euclidean_distance_lower_bound)
                    <= exact_hull
                    + projection_abs_tol
                    + projection_rel_tol * abs(exact_hull)
                )
                lower_euclidean_sound = (
                    float(certificate.euclidean_distance_lower_bound)
                    <= exact_euclidean
                    + projection_abs_tol
                    + projection_rel_tol * abs(exact_euclidean)
                )
                class_checks.append(
                    (primal_exact, lower_hull_sound, lower_euclidean_sound)
                )
                if not primal_exact or not lower_hull_sound or not lower_euclidean_sound:
                    case_invalid = True

            inferred = decision.inferred_precedence
            if inferred is not None:
                inferred_relations.append(inferred)

            position = {stage: index for index, stage in enumerate(target_label)}
            expected_relation = (
                (left, right)
                if position[left] < position[right]
                else (right, left)
            )
            contradictory = inferred is not None and inferred != expected_relation
            if contradictory:
                case_invalid = True
                case_contradictions += 1
                contradictory_pairs += 1

            if decision.status == "certified":
                if not contradictory:
                    case_pairs += 1
                    pair_successes += 1
                excluded = (
                    decision.left_before_right
                    if decision.left_before_right_impossible
                    else decision.right_before_left
                )
                minimum_margin = min(
                    minimum_margin,
                    float(excluded.euclidean_distance_lower_bound) - elimination_guard,
                )
            elif decision.status == "ambiguous":
                case_ambiguous += 1
                ambiguous_pairs += 1
            else:
                case_invalid = True
                case_both_excluded += 1
                both_excluded_pairs += 1

            pair_results[f"{left}{right}"] = {
                "left_before_right": _certificate_payload(
                    decision.left_before_right,
                    left_hull,
                    left_euclidean,
                ),
                "right_before_left": _certificate_payload(
                    decision.right_before_left,
                    right_hull,
                    right_euclidean,
                ),
                "left_before_right_nearest_vertex_proxy": left_vertex_proxy,
                "right_before_left_nearest_vertex_proxy": right_vertex_proxy,
                "left_before_right_nearest_history": left_nearest,
                "right_before_left_nearest_history": right_nearest,
                "left_before_right_impossible": bool(
                    decision.left_before_right_impossible
                ),
                "right_before_left_impossible": bool(
                    decision.right_before_left_impossible
                ),
                "status": decision.status,
                "inferred_precedence": (
                    list(inferred) if inferred is not None else None
                ),
                "evaluation_expected_precedence": list(expected_relation),
                "evaluation_contradictory": contradictory,
                "terminal_exactness_passed": all(check[0] for check in class_checks),
                "lower_bound_sound_in_witness_geometry": all(
                    check[1] for check in class_checks
                ),
                "lower_bound_sound_against_euclidean_vertices": all(
                    check[2] for check in class_checks
                ),
            }

        reconstructed = _reconstruct_total_order(
            stages,
            tuple(inferred_relations),
        )
        full_certified = (
            not case_invalid
            and case_pairs == 6
            and reconstructed is not None
        )
        correct_full = full_certified and reconstructed == target_label
        if full_certified and not correct_full:
            case_invalid = True
        if correct_full:
            full_successes += 1
        invalid = invalid or case_invalid

        cases[target_label] = {
            "target_endpoint_sha256": target_hashes[target_label],
            "witness_k3": witness_meta[target_label],
            "active_lift_target_hash_match": active_hash_match[target_label],
            "pairwise": pair_results,
            "label_blind_pairwise_orientation_certificates": case_pairs,
            "ambiguous_pair_count": case_ambiguous,
            "contradictory_pair_count": case_contradictions,
            "both_orientations_excluded_count": case_both_excluded,
            "label_blind_reconstructed_history": reconstructed,
            "full_history_certified": full_certified,
            "evaluation_correct_full_history": correct_full,
            "invalid": case_invalid,
        }

    result = {
        "result_version": "chronotrace-pairwise-multi-witness-confirmation-seed-v2",
        "confirmation_lock": confirmation,
        "confirmation_lock_sha256": json_sha256(confirmation),
        "methodology_lock_sha256": json_sha256(methodology),
        "source_k3_protocol_sha256": json_sha256(source_k3),
        "source_k23_protocol_sha256": json_sha256(source_k23),
        "seed": seed,
        "codebook_sha256": codebook.sha256,
        "dataset_sha256": dataset["sha256"],
        "numerical_fingerprint": numerical_fingerprint,
        "stage_executions": stage_calls,
        "witness_freeze_stage_executions": expected_freeze_calls,
        "projected_reconstruction_residual_max": projected_residual_max,
        "maximum_terminal_primal_exactness_error": maximum_terminal_primal_error,
        "prefix_cache_consumed": prefix_consumed,
        "retained_degree4_endpoint_projection_scalars": endpoint_scalar_count,
        "retained_degree4_interaction_projection_scalars": interaction_scalar_count,
        "full_k4_model_space_tensors_retained": False,
        "cases": cases,
        "full_history_certificate_coverage": full_successes,
        "label_blind_pairwise_orientation_certificate_coverage": pair_successes,
        "ambiguous_pair_count": ambiguous_pairs,
        "contradictory_pair_count": contradictory_pairs,
        "both_orientations_excluded_count": both_excluded_pairs,
        "minimum_excluded_orientation_margin_over_guard": (
            None if minimum_margin == float("inf") else minimum_margin
        ),
        "invalid_seed_job": invalid,
        "confirmation_codebooks_observed": True,
        "heldout_confirmation_launch_authorized": True,
    }
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
