#!/usr/bin/env python3
"""Run the frozen D-only Armijo inverse diagnostic on spent ABCD history."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from pythia_finite_pair_bridge import _configure_portable_numerics, _load_json

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
        default="configs/pythia_14m_reverse_peeling_armijo_diagnostic.lock.json",
    )
    parser.add_argument(
        "--k23-protocol",
        default="configs/pythia_14m_four_stage_k23_pilot.lock.json",
    )
    parser.add_argument(
        "--smoke-selection",
        default="configs/pythia_14m_reverse_peeling_smoke.selection.json",
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
        raise RuntimeError("model has no parameters while diagnosing inversion")
    return torch.cat(parts)


def main() -> None:
    args = parse_args()
    protocol = _load_json(args.protocol)
    k23 = _load_json(args.k23_protocol)
    smoke = _load_json(args.smoke_selection)

    if protocol["freeze_status"] != "frozen_before_any_armijo_pythia_output":
        raise ValueError("Armijo diagnostic protocol is not frozen")
    if protocol["experiment_role"] != "spent_history_solver_diagnostic":
        raise ValueError("Armijo diagnostic role drift")
    if protocol.get("confirmation_codebooks_observed") is not False:
        raise RuntimeError("Armijo diagnostic touched confirmation codebooks")
    if smoke.get("smoke_pass_all") is not False:
        raise RuntimeError("Armijo diagnostic requires the frozen failed Picard smoke")
    if int(smoke.get("source_run_id", -1)) != int(protocol["source_failed_smoke_run_id"]):
        raise RuntimeError("failed smoke run identity drift")

    history = tuple(str(protocol["known_history"]))
    predecessor_word = tuple(str(protocol["known_predecessor"]))
    diagnosed_stage = str(protocol["diagnosed_stage"])
    if history != ("A", "B", "C", "D"):
        raise ValueError("diagnostic history must remain ABCD")
    if predecessor_word != ("A", "B", "C") or diagnosed_stage != "D":
        raise ValueError("diagnostic must remain true predecessor ABC and final stage D")

    seed = int(protocol["pilot_codebook_seed"])
    prohibited = {int(value) for value in protocol["confirmation_codebooks_prohibited"]}
    if seed in prohibited:
        raise RuntimeError("diagnostic may not use a held-out confirmation codebook")

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
        raise RuntimeError("Armijo diagnostic tokenizer fingerprint drift")
    codebook = build_token_codebook(
        tokenizer,
        count=int(protocol["codebook_count_per_kind"]),
        seed=seed,
    )
    dataset = four_stage_dataset_payload(tokenizer, codebook)
    if codebook.sha256 != str(protocol["pilot_codebook_sha256"]):
        raise RuntimeError("Armijo diagnostic codebook hash drift")
    if dataset["sha256"] != protocol["pilot_dataset_sha256"]:
        raise RuntimeError("Armijo diagnostic dataset hash drift")
    worlds = build_scale_worlds_from_codebook(codebook)
    examples = {
        stage: build_four_stage_examples(worlds, stage) for stage in ("A", "B", "C", "D")
    }
    batch_d = completion_batch(torch, tokenizer, examples["D"], device)

    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision)
    model.to(device=device, dtype=torch.float64)
    model.config.use_cache = False
    if any(
        parameter.is_floating_point() and parameter.dtype != torch.float64
        for parameter in model.parameters()
    ):
        raise TypeError("could not enforce FP64 Armijo diagnostic model")
    theta0 = flatten_parameters(torch, model, preserve_parameter_dtype=True)
    if theta0.dtype != torch.float64:
        raise TypeError("Armijo diagnostic base vector is not FP64")
    fp64_hash = tensor_sha256(theta0)
    if fp64_hash != str(protocol["base_parameter_sha256_fp64"]):
        raise RuntimeError("Armijo diagnostic FP64 base hash drift")
    projected_hash = tensor_sha256(theta0.to(dtype=torch.float32))
    if projected_hash != str(protocol["base_parameter_sha256_fp32"]):
        raise RuntimeError("Armijo diagnostic projected FP32 base hash drift")

    stage_seeds = {
        stage: int(protocol["stage_randomness"][stage]) for stage in ("A", "B", "C", "D")
    }
    stage_calls = 0

    def run_stage(stage: str, initial: Any) -> Any:
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
            initial_vector=initial,
            preserve_parameter_dtype=True,
        )
        if not metrics.finite:
            raise FloatingPointError(f"non-finite Armijo diagnostic stage {stage}")
        stage_calls += 1
        return endpoint

    predecessor = theta0
    for stage in predecessor_word:
        predecessor = run_stage(stage, predecessor)
    predecessor_hash = tensor_sha256(predecessor)
    target = run_stage(diagnosed_stage, predecessor)
    target_hash = tensor_sha256(target)
    target_replay_exact = target_hash == str(protocol["target_endpoint_sha256"])
    if not target_replay_exact:
        raise RuntimeError("Armijo diagnostic ABCD target failed exact frozen replay")

    loss_evaluations = 0
    gradient_evaluations = 0

    def loss_at(state: Any) -> float:
        nonlocal loss_evaluations
        load_flat_parameters(torch, model, state, device=device)
        torch.manual_seed(stage_seeds[diagnosed_stage])
        model.train()
        with torch.no_grad():
            value = float(model(**batch_d).loss.detach().cpu())
        if not math.isfinite(value):
            raise FloatingPointError("Armijo inverse loss became non-finite")
        loss_evaluations += 1
        return value

    def gradient_at(state: Any) -> tuple[float, Any]:
        nonlocal gradient_evaluations
        load_flat_parameters(torch, model, state, device=device)
        torch.manual_seed(stage_seeds[diagnosed_stage])
        model.zero_grad(set_to_none=True)
        model.train()
        output = model(**batch_d)
        loss_value = float(output.loss.detach().cpu())
        if not math.isfinite(loss_value):
            raise FloatingPointError("Armijo inverse gradient loss became non-finite")
        output.loss.backward()
        gradient = _flatten_gradients(torch, model)
        if gradient.dtype != torch.float64 or not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("Armijo inverse gradient is invalid")
        model.zero_grad(set_to_none=True)
        gradient_evaluations += 1
        return loss_value, gradient

    def inverse_potential(state: Any, loss_value: float) -> float:
        displacement = state - target
        value = 0.5 * float(torch.dot(displacement, displacement)) - learning_rate * loss_value
        if not math.isfinite(value):
            raise FloatingPointError("Armijo inverse potential became non-finite")
        return value

    params = protocol["solver_parameters"]
    initial_step = float(params["initial_step"])
    armijo_constant = float(params["armijo_constant"])
    shrink_factor = float(params["shrink_factor"])
    minimum_step = float(params["minimum_step"])
    max_iterations = int(params["max_iterations"])
    stop_ratio = float(params["stop_residual_ratio_to_initial"])

    state = target.clone()
    initial_predecessor_distance = float(torch.linalg.vector_norm(target - predecessor))
    if initial_predecessor_distance <= 0.0 or not math.isfinite(initial_predecessor_distance):
        raise FloatingPointError("invalid initial target-to-predecessor distance")

    loss_value, grad_loss = gradient_at(state)
    residual = state - target - learning_rate * grad_loss
    initial_residual_norm = float(torch.linalg.vector_norm(residual))
    if initial_residual_norm <= 0.0 or not math.isfinite(initial_residual_norm):
        raise FloatingPointError("invalid initial inverse residual")
    objective_value = inverse_potential(state, loss_value)
    objective_trace = [objective_value]
    residual_trace = [initial_residual_norm]
    accepted_step_trace: list[float] = []
    backtracking_count_trace: list[int] = []
    line_search_failed = False
    stop_reached = False
    iterations = 0

    for iteration in range(1, max_iterations + 1):
        residual_norm = residual_trace[-1]
        if residual_norm <= stop_ratio * initial_residual_norm:
            stop_reached = True
            break
        direction = -residual
        directional_derivative = -(residual_norm**2)
        step = initial_step
        backtracks = 0
        accepted = False
        while step >= minimum_step:
            trial = state + step * direction
            trial_loss = loss_at(trial)
            trial_objective = inverse_potential(trial, trial_loss)
            if trial_objective <= (
                objective_value + armijo_constant * step * directional_derivative
            ):
                state = trial
                objective_value = trial_objective
                objective_trace.append(objective_value)
                accepted_step_trace.append(step)
                backtracking_count_trace.append(backtracks)
                accepted = True
                break
            step *= shrink_factor
            backtracks += 1
        if not accepted:
            line_search_failed = True
            iterations = iteration - 1
            break

        loss_value, grad_loss = gradient_at(state)
        residual = state - target - learning_rate * grad_loss
        residual_norm = float(torch.linalg.vector_norm(residual))
        if not math.isfinite(residual_norm):
            raise FloatingPointError("Armijo inverse residual became non-finite")
        residual_trace.append(residual_norm)
        iterations = iteration
    else:
        stop_reached = residual_trace[-1] <= stop_ratio * initial_residual_norm

    final_predecessor_distance = float(torch.linalg.vector_norm(state - predecessor))
    predecessor_distance_ratio = final_predecessor_distance / initial_predecessor_distance
    residual_ratio = residual_trace[-1] / initial_residual_norm

    final_loss, final_gradient = gradient_at(state)
    reconstructed_target = state - learning_rate * final_gradient
    forward_reconstruction_error = float(
        torch.linalg.vector_norm(reconstructed_target - target)
    )
    forward_error_ratio = forward_reconstruction_error / initial_residual_norm
    final_objective = inverse_potential(state, final_loss)
    objective_monotone = all(
        later <= earlier + 1e-12 * max(1.0, abs(earlier), abs(later))
        for earlier, later in zip(objective_trace, objective_trace[1:], strict=False)
    )

    checks = {
        "target_endpoint_replay_exact": target_replay_exact,
        "line_search_never_failed": not line_search_failed,
        "inverse_potential_monotone": objective_monotone,
        "inverse_residual_reduced_100x": residual_ratio <= 0.01,
        "predecessor_distance_reduced_100x": predecessor_distance_ratio <= 0.01,
        "forward_reconstruction_error_within_1pct_initial_residual": forward_error_ratio <= 0.01,
    }

    result = {
        "status": "complete",
        "claim": "spent_history_true_stage_armijo_inverse_solver_diagnostic",
        "protocol_sha256": json_sha256(protocol),
        "failed_smoke_selection_sha256": json_sha256(smoke),
        "confirmation_codebooks_observed": False,
        "model": model_id,
        "revision": revision,
        "precision": "fp64",
        "learning_rate": learning_rate,
        "pilot_codebook_seed": seed,
        "base_parameter_sha256_fp64": fp64_hash,
        "base_parameter_sha256_projected_fp32": projected_hash,
        "numerical_execution_fingerprint_sha256": numerical_fingerprint,
        "known_history": "".join(history),
        "known_predecessor": "".join(predecessor_word),
        "diagnosed_stage": diagnosed_stage,
        "predecessor_sha256": predecessor_hash,
        "target_endpoint_sha256": target_hash,
        "target_endpoint_replay_exact": target_replay_exact,
        "stage_executions": stage_calls,
        "solver": {
            "iterations": iterations,
            "stop_reached": stop_reached,
            "line_search_failed": line_search_failed,
            "initial_residual_norm": initial_residual_norm,
            "final_residual_norm": residual_trace[-1],
            "inverse_residual_ratio": residual_ratio,
            "initial_target_to_predecessor_distance": initial_predecessor_distance,
            "final_distance_to_true_predecessor": final_predecessor_distance,
            "predecessor_distance_ratio": predecessor_distance_ratio,
            "forward_reconstruction_error": forward_reconstruction_error,
            "forward_error_ratio_to_initial_residual": forward_error_ratio,
            "objective_trace": objective_trace,
            "residual_trace": residual_trace,
            "accepted_step_trace": accepted_step_trace,
            "backtracking_count_trace": backtracking_count_trace,
            "final_objective_recomputed": final_objective,
            "loss_evaluations": loss_evaluations,
            "gradient_evaluations": gradient_evaluations,
        },
        "solver_viability_checks": checks,
        "solver_viability_pass_all": all(checks.values()),
        "next_step": (
            "freeze_new_four_candidate_Armijo_smoke"
            if all(checks.values())
            else "pivot_to_forward_reachable_set_or_Jacobian_conditioned_scoring"
        ),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "solver_viability_pass_all": result["solver_viability_pass_all"],
                "solver": result["solver"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
