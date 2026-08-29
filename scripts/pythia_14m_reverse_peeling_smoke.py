#!/usr/bin/env python3
"""Run the frozen one-history Pythia-14M reverse-peeling smoke."""

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
        default="configs/pythia_14m_reverse_peeling_smoke.lock.json",
    )
    parser.add_argument(
        "--k23-protocol",
        default="configs/pythia_14m_four_stage_k23_pilot.lock.json",
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
        raise RuntimeError("model has no parameters while inverting a stage")
    return torch.cat(parts)


def main() -> None:
    args = parse_args()
    protocol = _load_json(args.protocol)
    k23 = _load_json(args.k23_protocol)
    if protocol["freeze_status"] != "frozen_before_any_reverse_peeling_pythia_output":
        raise ValueError("reverse peeling smoke must be frozen before Pythia output")
    if protocol["experiment_role"] != "non_confirmatory_methodology_smoke":
        raise ValueError("reverse peeling runner is methodology-smoke only")
    if protocol.get("confirmation_codebooks_observed") is not False:
        raise RuntimeError("reverse peeling protocol touched confirmation codebooks")
    if k23.get("confirmation_codebooks_observed") is not False:
        raise RuntimeError("source K2/K3 protocol touched confirmation codebooks")
    if json_sha256(k23) != str(protocol["source_k23_protocol_sha256"]):
        raise RuntimeError("source K2/K3 protocol hash drift")

    stages = tuple(str(value) for value in protocol["stages"])
    if stages != ("A", "B", "C", "D"):
        raise ValueError("reverse peeling smoke requires A/B/C/D")
    target_history = tuple(str(protocol["target_history"]))
    if target_history != ("A", "B", "C", "D"):
        raise ValueError("smoke target must remain frozen to ABCD")
    candidate_last_stages = tuple(str(value) for value in protocol["candidate_last_stages"])
    if candidate_last_stages != stages:
        raise ValueError("candidate last-stage order drift")
    if int(protocol["max_measured_interaction_degree"]) != 3:
        raise ValueError("smoke requires K3 interaction basis")

    seed = int(protocol["pilot_codebook_seed"])
    prohibited = {int(value) for value in protocol["confirmation_codebooks_prohibited"]}
    if seed in prohibited:
        raise RuntimeError("smoke may not use a held-out confirmation codebook")

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
        raise RuntimeError("reverse peeling tokenizer fingerprint drift")
    codebook = build_token_codebook(
        tokenizer,
        count=int(protocol["codebook_count_per_kind"]),
        seed=seed,
    )
    dataset = four_stage_dataset_payload(tokenizer, codebook)
    if codebook.sha256 != str(protocol["pilot_codebook_sha256"]):
        raise RuntimeError("reverse peeling codebook hash drift")
    if dataset["sha256"] != protocol["pilot_dataset_sha256"]:
        raise RuntimeError("reverse peeling dataset hash drift")
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
        raise TypeError("could not enforce FP64 reverse peeling model")
    theta0 = flatten_parameters(torch, model, preserve_parameter_dtype=True)
    if theta0.dtype != torch.float64:
        raise TypeError("reverse peeling base vector is not FP64")
    fp64_hash = tensor_sha256(theta0)
    if fp64_hash != str(protocol["base_parameter_sha256_fp64"]):
        raise RuntimeError("reverse peeling FP64 base hash drift")
    projected_hash = tensor_sha256(theta0.to(dtype=torch.float32))
    if projected_hash != str(protocol["base_parameter_sha256_fp32"]):
        raise RuntimeError("reverse peeling projected FP32 base hash drift")

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
                raise FloatingPointError(f"non-finite reverse peeling stage {stage}")
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
        raise TypeError("reverse peeling interaction basis contains non-FP64 tensor")

    target = theta0
    for stage in target_history:
        target = stage_maps[stage](target)
    target_hash = tensor_sha256(target)
    if target_hash != str(protocol["target_endpoint_sha256"]):
        raise RuntimeError("ABCD target endpoint failed exact frozen replay")

    gradient_calls = 0

    def gradient_at(stage: str, state: Any) -> Any:
        nonlocal gradient_calls
        load_flat_parameters(torch, model, state, device=device)
        torch.manual_seed(stage_seeds[stage])
        model.zero_grad(set_to_none=True)
        model.train()
        output = model(**batches[stage])
        loss = float(output.loss.detach().cpu())
        if not math.isfinite(loss):
            raise FloatingPointError(f"non-finite inverse loss for stage {stage}")
        output.loss.backward()
        gradient = _flatten_gradients(torch, model)
        if gradient.dtype != torch.float64:
            raise TypeError("inverse gradient is not FP64")
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError(f"non-finite inverse gradient for stage {stage}")
        gradient_calls += 1
        model.zero_grad(set_to_none=True)
        return gradient

    tolerance = float(protocol["inverse_relative_update_tolerance"])
    max_iterations = int(protocol["inverse_max_iterations"])
    inverse_results: dict[str, Any] = {}

    for candidate_last in candidate_last_stages:
        state = target.clone()
        update_trace: list[float] = []
        converged = False
        for _ in range(1, max_iterations + 1):
            gradient = gradient_at(candidate_last, state)
            updated = target + learning_rate * gradient
            update_norm = float(torch.linalg.vector_norm(updated - state))
            scale = max(1.0, float(torch.linalg.vector_norm(updated)))
            relative_update = update_norm / scale
            if not math.isfinite(relative_update):
                raise FloatingPointError("non-finite inverse relative update")
            update_trace.append(relative_update)
            state = updated
            del gradient
            if relative_update <= tolerance:
                converged = True
                break
        iterations = len(update_trace)

        final_gradient = gradient_at(candidate_last, state)
        reconstructed = state - learning_rate * final_gradient
        target_scale = max(1.0, float(torch.linalg.vector_norm(target)))
        forward_error = float(torch.linalg.vector_norm(reconstructed - target)) / target_scale
        if not math.isfinite(forward_error):
            raise FloatingPointError("non-finite forward-inverse consistency error")
        contraction_ratios = [
            update_trace[index] / update_trace[index - 1]
            for index in range(1, len(update_trace))
            if update_trace[index - 1] > 0.0
        ]

        remaining = tuple(stage for stage in stages if stage != candidate_last)
        words = tuple(permutations(remaining))
        vectors = tuple(
            ordered_interaction_word_prediction(word, basis, degree=3) for word in words
        )
        distances = [float(torch.linalg.vector_norm(state - vector)) for vector in vectors]
        best_index = min(range(len(distances)), key=lambda index: (distances[index], words[index]))
        inverse_results[candidate_last] = {
            "converged": converged,
            "iterations": iterations,
            "final_relative_update": update_trace[-1],
            "relative_update_trace": update_trace,
            "contraction_ratios": contraction_ratios,
            "forward_inverse_relative_error": forward_error,
            "best_predecessor": "".join(words[best_index]),
            "best_predecessor_residual": distances[best_index],
            "predecessor_residuals": {
                "".join(word): distance for word, distance in zip(words, distances, strict=True)
            },
        }
        del final_gradient
        del reconstructed
        del state
        gc.collect()

    ranking = sorted(
        candidate_last_stages,
        key=lambda stage: (
            float(inverse_results[stage]["best_predecessor_residual"]),
            stage,
        ),
    )
    best_stage = ranking[0]
    best_residual = float(inverse_results[best_stage]["best_predecessor_residual"])
    runner_up_residual = float(inverse_results[ranking[1]]["best_predecessor_residual"])
    margin = runner_up_residual - best_residual
    tied = margin == 0.0
    selected_stage = None if tied else best_stage
    selected_predecessor = (
        None if selected_stage is None else inverse_results[selected_stage]["best_predecessor"]
    )

    checks = {
        "target_endpoint_replay_exact": target_hash == str(protocol["target_endpoint_sha256"]),
        "all_candidate_inverses_converged": all(
            bool(inverse_results[stage]["converged"]) for stage in candidate_last_stages
        ),
        "all_forward_inverse_errors_finite": all(
            math.isfinite(float(inverse_results[stage]["forward_inverse_relative_error"]))
            for stage in candidate_last_stages
        ),
        "true_final_stage_selected": selected_stage == "D",
        "true_predecessor_selected": selected_predecessor == "ABC",
        "positive_final_stage_margin": margin > 0.0,
    }

    expected_stage_calls = expected_basis_calls + len(target_history)
    if stage_calls != expected_stage_calls:
        raise RuntimeError("reverse peeling stage execution count drift")

    result = {
        "status": "complete",
        "claim": "non_confirmatory_one_history_reverse_operator_peeling_smoke",
        "protocol_sha256": json_sha256(protocol),
        "source_k23_protocol_sha256": json_sha256(k23),
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
        "inverse_gradient_evaluations": gradient_calls,
        "target_history": "".join(target_history),
        "target_endpoint_sha256": target_hash,
        "inverse_results": inverse_results,
        "decision": {
            "selected_last_stage": selected_stage,
            "selected_predecessor": selected_predecessor,
            "best_residual": best_residual,
            "runner_up_residual": runner_up_residual,
            "margin": margin,
            "identifiable": not tied,
            "ranking": ranking,
        },
        "smoke_checks": checks,
        "smoke_pass_all": all(checks.values()),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
