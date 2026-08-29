#!/usr/bin/env python3
# ruff: noqa: E501
"""Run the frozen one-history forward-reachable residual smoke on Pythia-14M."""

from __future__ import annotations

import argparse
import gc
import json
import math
from itertools import permutations
from pathlib import Path
from typing import Any

from pythia_finite_pair_bridge import _configure_portable_numerics, _load_json

from chronotrace.geometry.interactions import (
    measure_ordered_interaction_basis_compact,
    ordered_interaction_word_prediction,
    ordered_probe_count,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="configs/pythia_14m_forward_reachable_residual_smoke.lock.json",
    )
    parser.add_argument(
        "--k23-protocol",
        default="configs/pythia_14m_four_stage_k23_pilot.lock.json",
    )
    parser.add_argument(
        "--picard-selection",
        default="configs/pythia_14m_reverse_peeling_smoke.selection.json",
    )
    parser.add_argument(
        "--armijo-selection",
        default="configs/pythia_14m_reverse_peeling_armijo_diagnostic.selection.json",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def _flatten_gradients(torch: Any, model: Any) -> Any:
    parts = []
    for parameter in model.parameters():
        if parameter.grad is None:
            parts.append(torch.zeros_like(parameter.detach()).reshape(-1).cpu())
        else:
            parts.append(parameter.grad.detach().reshape(-1).cpu())
    if not parts:
        raise RuntimeError("model has no parameters while scoring reachable residuals")
    return torch.cat(parts)


def main() -> None:
    args = parse_args()
    protocol = _load_json(args.protocol)
    k23 = _load_json(args.k23_protocol)
    picard = _load_json(args.picard_selection)
    armijo = _load_json(args.armijo_selection)

    if protocol["freeze_status"] != "frozen_before_any_forward_reachable_pythia_output":
        raise ValueError("forward-reachable smoke protocol is not frozen")
    if protocol["experiment_role"] != "non_confirmatory_methodology_smoke":
        raise ValueError("forward-reachable runner is methodology-smoke only")
    if protocol.get("confirmation_codebooks_observed") is not False:
        raise RuntimeError("forward-reachable protocol touched confirmation codebooks")
    if k23.get("confirmation_codebooks_observed") is not False:
        raise RuntimeError("source K2/K3 protocol touched confirmation codebooks")
    if json_sha256(k23) != str(protocol["source_k23_protocol_sha256"]):
        raise RuntimeError("source K2/K3 protocol hash drift")
    if int(picard.get("source_run_id", -1)) != int(protocol["source_failed_picard_run_id"]):
        raise RuntimeError("source Picard smoke identity drift")
    if picard.get("smoke_pass_all") is not False:
        raise RuntimeError("forward-reachable smoke requires the frozen failed Picard smoke")
    if int(armijo.get("source_run_id", -1)) != int(protocol["source_failed_armijo_run_id"]):
        raise RuntimeError("source Armijo diagnostic identity drift")
    if int(armijo.get("source_artifact_id", -1)) != int(protocol["source_failed_armijo_artifact_id"]):
        raise RuntimeError("source Armijo artifact identity drift")
    if armijo.get("solver_viability_pass_all") is not False:
        raise RuntimeError("forward-reachable smoke requires the frozen failed Armijo diagnostic")
    if armijo.get("next_step") != "pivot_to_finite_forward_reachable_residual_scoring_on_the_same_spent_ABCD_history":
        raise RuntimeError("source Armijo adjudication does not authorize this spent-data pivot")

    stages = tuple(str(value) for value in protocol["stages"])
    if stages != ("A", "B", "C", "D"):
        raise ValueError("forward-reachable smoke requires A/B/C/D")
    target_history = tuple(str(protocol["target_history"]))
    if target_history != ("A", "B", "C", "D"):
        raise ValueError("forward-reachable target must remain ABCD")
    if int(protocol["max_measured_interaction_degree"]) != 3:
        raise ValueError("forward-reachable smoke requires degree-three basis")
    if int(protocol["candidate_hypothesis_count"]) != 24:
        raise ValueError("forward-reachable candidate count drift")
    if int(protocol["candidate_gradient_evaluations"]) != 24:
        raise ValueError("forward-reachable gradient budget drift")

    seed = int(protocol["pilot_codebook_seed"])
    prohibited = {int(value) for value in protocol["confirmation_codebooks_prohibited"]}
    if seed in prohibited:
        raise RuntimeError("forward-reachable smoke may not use a held-out codebook")

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
        raise RuntimeError("forward-reachable tokenizer fingerprint drift")
    codebook = build_token_codebook(
        tokenizer,
        count=int(protocol["codebook_count_per_kind"]),
        seed=seed,
    )
    dataset = four_stage_dataset_payload(tokenizer, codebook)
    if codebook.sha256 != str(protocol["pilot_codebook_sha256"]):
        raise RuntimeError("forward-reachable codebook hash drift")
    if dataset["sha256"] != protocol["pilot_dataset_sha256"]:
        raise RuntimeError("forward-reachable dataset hash drift")
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
        raise TypeError("could not enforce FP64 forward-reachable model")
    theta0 = flatten_parameters(torch, model, preserve_parameter_dtype=True)
    if theta0.dtype != torch.float64:
        raise TypeError("forward-reachable base vector is not FP64")
    fp64_hash = tensor_sha256(theta0)
    if fp64_hash != str(protocol["base_parameter_sha256_fp64"]):
        raise RuntimeError("forward-reachable FP64 base hash drift")
    projected_hash = tensor_sha256(theta0.to(dtype=torch.float32))
    if projected_hash != str(protocol["base_parameter_sha256_fp32"]):
        raise RuntimeError("forward-reachable projected FP32 base hash drift")

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
                raise FloatingPointError(f"non-finite forward-reachable stage {stage}")
            stage_calls += 1
            return endpoint

        return run

    stage_maps = {stage: make_stage_map(stage) for stage in stages}
    basis = measure_ordered_interaction_basis_compact(stage_maps, theta0, max_degree=3)
    expected_basis_calls = ordered_probe_count(len(stages), 3)
    if expected_basis_calls != int(protocol["basis_stage_executions"]):
        raise RuntimeError("frozen basis probe count drift")
    if basis.stage_executions != expected_basis_calls:
        raise RuntimeError("observed basis probe count drift")
    if any(value.dtype != torch.float64 for value in basis.interactions.values()):
        raise TypeError("forward-reachable interaction basis contains non-FP64 tensor")

    target = theta0
    true_predecessor = None
    for index, stage in enumerate(target_history):
        if index == len(target_history) - 1:
            true_predecessor = target.clone()
        target = stage_maps[stage](target)
    if true_predecessor is None:
        raise RuntimeError("failed to retain true predecessor diagnostic")
    target_hash = tensor_sha256(target)
    target_replay_exact = target_hash == str(protocol["target_endpoint_sha256"])
    if not target_replay_exact:
        raise RuntimeError("ABCD target endpoint failed exact frozen replay")
    target_scale = max(1.0, float(torch.linalg.vector_norm(target)))

    histories = tuple(permutations(stages))
    k3_errors: dict[str, float] = {}
    for history in histories:
        prediction = ordered_interaction_word_prediction(history, basis, degree=3)
        k3_errors["".join(history)] = float(torch.linalg.vector_norm(prediction - target))
        del prediction
    k3_ranking = sorted(k3_errors, key=lambda history: (k3_errors[history], history))
    k3_prediction = k3_ranking[0]
    k3_best = k3_errors[k3_prediction]
    k3_runner = k3_errors[k3_ranking[1]]
    k3_true = k3_errors["ABCD"]
    k3_margin = k3_runner - k3_best

    frozen_k3 = protocol["frozen_k3_baseline"]
    rel_tol = float(frozen_k3["numeric_match_rel_tol"])
    abs_tol = float(frozen_k3["numeric_match_abs_tol"])
    k3_numeric_match = all(
        (
            math.isclose(k3_best, float(frozen_k3["best_error"]), rel_tol=rel_tol, abs_tol=abs_tol),
            math.isclose(k3_runner, float(frozen_k3["runner_up_error"]), rel_tol=rel_tol, abs_tol=abs_tol),
            math.isclose(k3_margin, float(frozen_k3["margin"]), rel_tol=rel_tol, abs_tol=abs_tol),
            math.isclose(k3_true, float(frozen_k3["true_candidate_error"]), rel_tol=rel_tol, abs_tol=abs_tol),
        )
    )
    k3_baseline_reproduced = k3_prediction == str(frozen_k3["prediction"]) and k3_numeric_match

    reconstructed_true_predecessor = ordered_interaction_word_prediction(("A", "B", "C"), basis, degree=3)
    true_predecessor_reconstruction_error = float(
        torch.linalg.vector_norm(reconstructed_true_predecessor - true_predecessor)
    )
    true_predecessor_reconstruction_relative_error = (
        true_predecessor_reconstruction_error
        / max(1.0, float(torch.linalg.vector_norm(true_predecessor)))
    )
    del reconstructed_true_predecessor

    gradient_calls = 0

    def gradient_at(stage: str, state: Any) -> Any:
        nonlocal gradient_calls
        load_flat_parameters(torch, model, state, device=device)
        torch.manual_seed(stage_seeds[stage])
        model.zero_grad(set_to_none=True)
        model.train()
        output = model(**batches[stage])
        loss_value = float(output.loss.detach().cpu())
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"non-finite forward-reachable loss for stage {stage}")
        output.loss.backward()
        gradient = _flatten_gradients(torch, model)
        if gradient.dtype != torch.float64:
            raise TypeError("forward-reachable gradient is not FP64")
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError(f"non-finite forward-reachable gradient for stage {stage}")
        model.zero_grad(set_to_none=True)
        gradient_calls += 1
        return gradient

    candidate_records: list[dict[str, Any]] = []
    for candidate_last in stages:
        remaining = tuple(stage for stage in stages if stage != candidate_last)
        for predecessor_word in permutations(remaining):
            predecessor = ordered_interaction_word_prediction(predecessor_word, basis, degree=3)
            gradient = gradient_at(candidate_last, predecessor)
            reachable = predecessor - learning_rate * gradient
            residual = float(torch.linalg.vector_norm(reachable - target))
            if not math.isfinite(residual):
                raise FloatingPointError("non-finite forward-reachable candidate residual")
            candidate_records.append(
                {
                    "history": "".join(predecessor_word) + candidate_last,
                    "predecessor": "".join(predecessor_word),
                    "final_stage": candidate_last,
                    "residual": residual,
                    "relative_residual": residual / target_scale,
                }
            )
            del gradient
            del reachable
            del predecessor
            gc.collect()

    if len(candidate_records) != int(protocol["candidate_hypothesis_count"]):
        raise RuntimeError("forward-reachable candidate count drift")
    ranking = sorted(candidate_records, key=lambda item: (float(item["residual"]), str(item["history"])))
    best = ranking[0]
    runner = ranking[1]
    margin = float(runner["residual"]) - float(best["residual"])
    identifiable = margin > 0.0
    selected_history = str(best["history"]) if identifiable else None
    selected_predecessor = str(best["predecessor"]) if identifiable else None
    selected_final_stage = str(best["final_stage"]) if identifiable else None
    target_noise_radius = margin / 2.0 if identifiable else 0.0

    stage_minima = {}
    for stage in stages:
        stage_records = [item for item in candidate_records if item["final_stage"] == stage]
        stage_best = min(stage_records, key=lambda item: (float(item["residual"]), str(item["history"])))
        stage_minima[stage] = {
            "history": stage_best["history"],
            "predecessor": stage_best["predecessor"],
            "residual": stage_best["residual"],
            "relative_residual": stage_best["relative_residual"],
        }

    all_residuals_finite = all(
        math.isfinite(float(item["residual"])) and math.isfinite(float(item["relative_residual"]))
        for item in candidate_records
    )
    relative_zero_threshold = float(protocol["relative_numerical_zero_threshold"])
    expected_stage_calls = expected_basis_calls + len(target_history)
    if stage_calls != expected_stage_calls:
        raise RuntimeError("forward-reachable stage execution count drift")

    checks = {
        "target_endpoint_replay_exact": target_replay_exact,
        "frozen_k3_baseline_reproduced": k3_baseline_reproduced,
        "frozen_k3_baseline_misses_ABCD": k3_prediction != "ABCD",
        "all_candidate_residuals_finite": all_residuals_finite,
        "candidate_gradient_budget_exact": gradient_calls == int(protocol["candidate_gradient_evaluations"]),
        "true_history_selected": selected_history == "ABCD",
        "true_predecessor_selected": selected_predecessor == "ABC",
        "true_final_stage_selected": selected_final_stage == "D",
        "best_relative_residual_is_numerical_zero": float(best["relative_residual"]) <= relative_zero_threshold,
        "positive_global_residual_margin": margin > 0.0,
    }

    result = {
        "status": "complete",
        "claim": "non_confirmatory_one_history_finite_forward_reachable_residual_smoke",
        "protocol_sha256": json_sha256(protocol),
        "source_k23_protocol_sha256": json_sha256(k23),
        "source_picard_selection_sha256": json_sha256(picard),
        "source_armijo_selection_sha256": json_sha256(armijo),
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
        "candidate_gradient_evaluations": gradient_calls,
        "target_history": "ABCD",
        "target_endpoint_sha256": target_hash,
        "target_endpoint_replay_exact": target_replay_exact,
        "target_parameter_norm": float(torch.linalg.vector_norm(target)),
        "true_predecessor_reconstruction_error": true_predecessor_reconstruction_error,
        "true_predecessor_reconstruction_relative_error": true_predecessor_reconstruction_relative_error,
        "k3_baseline": {
            "prediction": k3_prediction,
            "best_error": k3_best,
            "runner_up_error": k3_runner,
            "margin": k3_margin,
            "true_candidate_error": k3_true,
            "frozen_baseline_reproduced": k3_baseline_reproduced,
        },
        "candidate_records": sorted(candidate_records, key=lambda item: str(item["history"])),
        "stage_minima": stage_minima,
        "decision": {
            "selected_history": selected_history,
            "selected_predecessor": selected_predecessor,
            "selected_final_stage": selected_final_stage,
            "best_residual": best["residual"],
            "best_relative_residual": best["relative_residual"],
            "runner_up_history": runner["history"],
            "runner_up_residual": runner["residual"],
            "global_margin": margin,
            "target_only_noise_radius": target_noise_radius,
            "identifiable": identifiable,
            "ranking": [item["history"] for item in ranking],
        },
        "smoke_checks": checks,
        "smoke_pass_all": all(checks.values()),
        "next_step": (
            "freeze_all24_spent_forward_reachable_margin_map"
            if all(checks.values())
            else "inspect_fixed_residual_table_without_retuning"
        ),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "smoke_pass_all": result["smoke_pass_all"],
                "k3_baseline": result["k3_baseline"],
                "decision": result["decision"],
                "true_predecessor_reconstruction_relative_error": result[
                    "true_predecessor_reconstruction_relative_error"
                ],
                "candidate_gradient_evaluations": result["candidate_gradient_evaluations"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
