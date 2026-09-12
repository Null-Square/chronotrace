#!/usr/bin/env python3
"""Run frozen all-24 reverse peeling on the spent Pythia-14M codebook."""

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
        default="configs/pythia_14m_reverse_peeling_all24.lock.json",
    )
    parser.add_argument(
        "--interpretation",
        default="configs/pythia_14m_reverse_peeling_all24_interpretation.lock.json",
    )
    parser.add_argument(
        "--k23-protocol",
        default="configs/pythia_14m_four_stage_k23_pilot.lock.json",
    )
    parser.add_argument(
        "--replay-lock",
        default="configs/pythia_14m_four_stage_k23_pilot_replay.lock.json",
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
        raise RuntimeError("model has no parameters while inverting a stage")
    return torch.cat(parts)


def _classify_result(
    *,
    full_correct: int,
    last_stage_true_positive_margin: int,
    true_inverse_converged: int,
    repairs: int,
    regressions: int,
) -> tuple[str, dict[str, bool]]:
    checks = {
        "true_last_stage_inverse_converges_24_of_24": true_inverse_converged == 24,
        "full_chronology_at_least_18_of_24": full_correct >= 18,
        "full_chronology_beats_direct_k3": full_correct > 3,
        "full_chronology_beats_recency": full_correct > 1,
        "repairs_exceed_regressions": repairs > regressions,
        "true_last_stage_positive_margin_at_least_18_of_24": (
            last_stage_true_positive_margin >= 18
        ),
    }
    if all(checks.values()):
        return "strong", checks
    if 12 <= full_correct <= 17 and repairs > regressions:
        return "promising_not_strong", checks
    if 4 <= full_correct <= 11:
        return "weak", checks
    return "fail", checks


def main() -> None:
    args = parse_args()
    protocol = _load_json(args.protocol)
    interpretation = _load_json(args.interpretation)
    k23 = _load_json(args.k23_protocol)
    replay = _load_json(args.replay_lock)
    smoke = _load_json(args.smoke_selection)

    expected_freeze = "frozen_before_reverse_peeling_smoke_output_launch_blocked_on_smoke_pass"
    if protocol["freeze_status"] != expected_freeze:
        raise ValueError("all-24 reverse peeling protocol is not frozen")
    if interpretation["freeze_status"] != "frozen_before_reverse_peeling_smoke_output":
        raise ValueError("all-24 reverse peeling interpretation is not frozen")
    if protocol.get("confirmation_codebooks_observed") is not False:
        raise RuntimeError("all-24 protocol touched confirmation codebooks")
    if interpretation.get("confirmation_codebooks_observed") is not False:
        raise RuntimeError("all-24 interpretation touched confirmation codebooks")
    if replay.get("confirmation_codebooks_observed") is not False:
        raise RuntimeError("replay lock touched confirmation codebooks")
    if int(smoke.get("source_run_id", -1)) != int(protocol["source_smoke_run_id"]):
        raise RuntimeError("smoke selection run identity mismatch")
    if smoke.get("smoke_pass_all") is not True:
        raise RuntimeError("all-24 launch is prohibited because the smoke did not pass")
    if smoke.get("confirmation_codebooks_observed") is not False:
        raise RuntimeError("smoke selection touched confirmation codebooks")

    stages = tuple(str(value) for value in protocol["stages"])
    if stages != ("A", "B", "C", "D"):
        raise ValueError("all-24 reverse peeling requires A/B/C/D")
    histories = tuple(permutations(stages))
    if len(histories) != 24:
        raise RuntimeError("four-stage history enumeration drift")
    if int(protocol["max_measured_interaction_degree"]) != 3:
        raise ValueError("all-24 reverse peeling requires a K3 basis")

    seed = int(protocol["pilot_codebook_seed"])
    prohibited = {int(value) for value in protocol["confirmation_codebooks_prohibited"]}
    if seed in prohibited:
        raise RuntimeError("all-24 run may not use a held-out confirmation codebook")
    if int(replay["pilot_codebook_seed"]) != seed:
        raise RuntimeError("replay lock codebook seed mismatch")

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
        raise RuntimeError("all-24 tokenizer fingerprint drift")
    codebook = build_token_codebook(
        tokenizer,
        count=int(protocol["codebook_count_per_kind"]),
        seed=seed,
    )
    dataset = four_stage_dataset_payload(tokenizer, codebook)
    if codebook.sha256 != str(protocol["pilot_codebook_sha256"]):
        raise RuntimeError("all-24 codebook hash drift")
    if dataset["sha256"] != protocol["pilot_dataset_sha256"]:
        raise RuntimeError("all-24 dataset hash drift")
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
        raise TypeError("could not enforce FP64 all-24 model")
    theta0 = flatten_parameters(torch, model, preserve_parameter_dtype=True)
    if theta0.dtype != torch.float64:
        raise TypeError("all-24 base vector is not FP64")
    fp64_hash = tensor_sha256(theta0)
    if fp64_hash != str(protocol["base_parameter_sha256_fp64"]):
        raise RuntimeError("all-24 FP64 base hash drift")
    projected_hash = tensor_sha256(theta0.to(dtype=torch.float32))
    if projected_hash != str(protocol["base_parameter_sha256_fp32"]):
        raise RuntimeError("all-24 projected FP32 base hash drift")

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
                raise FloatingPointError(f"non-finite all-24 stage {stage}")
            stage_calls += 1
            return endpoint

        return run

    stage_maps = {stage: make_stage_map(stage) for stage in stages}
    basis = measure_ordered_interaction_basis_compact(stage_maps, theta0, max_degree=3)
    expected_basis_calls = ordered_probe_count(len(stages), 3)
    if expected_basis_calls != int(protocol["basis_stage_executions"]):
        raise RuntimeError("frozen all-24 basis probe count drift")
    if basis.stage_executions != expected_basis_calls:
        raise RuntimeError("observed all-24 basis probe count drift")

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
            raise FloatingPointError(f"non-finite all-24 inverse loss for stage {stage}")
        output.loss.backward()
        gradient = _flatten_gradients(torch, model)
        if gradient.dtype != torch.float64:
            raise TypeError("all-24 inverse gradient is not FP64")
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError(f"non-finite all-24 inverse gradient for stage {stage}")
        gradient_calls += 1
        model.zero_grad(set_to_none=True)
        return gradient

    tolerance = float(protocol["inverse_relative_update_tolerance"])
    max_iterations = int(protocol["inverse_max_iterations"])
    direct_k3_correct = {str(value) for value in protocol["frozen_direct_k3_correct_histories"]}
    replay_hashes = {str(key): str(value) for key, value in replay["target_endpoint_hashes"].items()}
    rows: list[dict[str, Any]] = []

    for history in histories:
        history_name = "".join(history)
        target = theta0
        for stage in history:
            target = stage_maps[stage](target)
        target_hash = tensor_sha256(target)
        if target_hash != replay_hashes[history_name]:
            raise RuntimeError(f"all-24 target replay mismatch for {history_name}")

        candidate_results: dict[str, Any] = {}
        for candidate_last in stages:
            state = target.clone()
            update_trace: list[float] = []
            converged = False
            for _iteration in range(1, max_iterations + 1):
                gradient = gradient_at(candidate_last, state)
                updated = target + learning_rate * gradient
                update_norm = float(torch.linalg.vector_norm(updated - state))
                scale = max(1.0, float(torch.linalg.vector_norm(updated)))
                relative_update = update_norm / scale
                if not math.isfinite(relative_update):
                    raise FloatingPointError("non-finite all-24 inverse relative update")
                update_trace.append(relative_update)
                state = updated
                del gradient
                if relative_update <= tolerance:
                    converged = True
                    break

            result: dict[str, Any] = {
                "converged": converged,
                "iterations": len(update_trace),
                "final_relative_update": update_trace[-1],
                "relative_update_trace": update_trace,
            }
            if converged:
                final_gradient = gradient_at(candidate_last, state)
                reconstructed = state - learning_rate * final_gradient
                target_scale = max(1.0, float(torch.linalg.vector_norm(target)))
                forward_error = float(torch.linalg.vector_norm(reconstructed - target)) / target_scale
                if not math.isfinite(forward_error):
                    raise FloatingPointError("non-finite all-24 forward-inverse error")
                remaining = tuple(stage for stage in stages if stage != candidate_last)
                best_word: tuple[str, ...] | None = None
                best_residual = float("inf")
                predecessor_residuals: dict[str, float] = {}
                for word in permutations(remaining):
                    reference = ordered_interaction_word_prediction(word, basis, degree=3)
                    residual = float(torch.linalg.vector_norm(state - reference))
                    predecessor_residuals["".join(word)] = residual
                    if (residual, word) < (best_residual, best_word or word):
                        best_residual = residual
                        best_word = word
                    del reference
                result.update(
                    {
                        "forward_inverse_relative_error": forward_error,
                        "best_predecessor": "".join(best_word or ()),
                        "best_predecessor_residual": best_residual,
                        "predecessor_residuals": predecessor_residuals,
                    }
                )
                del final_gradient
                del reconstructed
            else:
                result.update(
                    {
                        "forward_inverse_relative_error": None,
                        "best_predecessor": None,
                        "best_predecessor_residual": None,
                        "predecessor_residuals": {},
                    }
                )
            candidate_results[candidate_last] = result
            del state
            gc.collect()

        all_converged = all(bool(candidate_results[stage]["converged"]) for stage in stages)
        selected_last: str | None = None
        selected_predecessor: str | None = None
        margin: float | None = None
        identifiable = False
        ranking: list[str] = []
        if all_converged:
            ranking = sorted(
                stages,
                key=lambda stage: (
                    float(candidate_results[stage]["best_predecessor_residual"]),
                    stage,
                ),
            )
            best = float(candidate_results[ranking[0]]["best_predecessor_residual"])
            runner = float(candidate_results[ranking[1]]["best_predecessor_residual"])
            margin = runner - best
            if margin > 0.0:
                identifiable = True
                selected_last = ranking[0]
                selected_predecessor = str(candidate_results[selected_last]["best_predecessor"])

        predicted_history = (
            None
            if selected_last is None or selected_predecessor is None
            else selected_predecessor + selected_last
        )
        full_correct = predicted_history == history_name
        last_stage_correct = selected_last == history[-1]
        true_stage_positive_margin = last_stage_correct and margin is not None and margin > 0.0
        rows.append(
            {
                "history": history_name,
                "target_endpoint_sha256": target_hash,
                "true_last_stage": history[-1],
                "true_predecessor": "".join(history[:-1]),
                "candidate_results": candidate_results,
                "all_candidate_inverses_converged": all_converged,
                "true_last_stage_inverse_converged": bool(
                    candidate_results[history[-1]]["converged"]
                ),
                "decision": {
                    "identifiable": identifiable,
                    "ranking": ranking,
                    "selected_last_stage": selected_last,
                    "selected_predecessor": selected_predecessor,
                    "predicted_history": predicted_history,
                    "margin": margin,
                },
                "full_history_correct": full_correct,
                "last_stage_correct": last_stage_correct,
                "true_last_stage_positive_margin": true_stage_positive_margin,
                "direct_k3_correct": history_name in direct_k3_correct,
            }
        )
        del target
        gc.collect()

    expected_target_calls = int(protocol["target_stage_executions"])
    if expected_target_calls != 4 * len(histories):
        raise RuntimeError("frozen all-24 target execution count drift")
    if stage_calls != expected_basis_calls + expected_target_calls:
        raise RuntimeError("observed all-24 stage execution count drift")

    full_correct = sum(int(row["full_history_correct"]) for row in rows)
    last_stage_correct = sum(int(row["last_stage_correct"]) for row in rows)
    true_inverse_converged = sum(
        int(row["true_last_stage_inverse_converged"]) for row in rows
    )
    all_candidate_converged = sum(
        int(row["all_candidate_inverses_converged"]) for row in rows
    )
    positive_margin = sum(
        int(row["decision"]["margin"] is not None and row["decision"]["margin"] > 0.0)
        for row in rows
    )
    last_stage_true_positive_margin = sum(
        int(row["true_last_stage_positive_margin"]) for row in rows
    )
    repairs = sum(
        int((not row["direct_k3_correct"]) and row["full_history_correct"])
        for row in rows
    )
    regressions = sum(
        int(row["direct_k3_correct"] and (not row["full_history_correct"]))
        for row in rows
    )
    tier, primary_checks = _classify_result(
        full_correct=full_correct,
        last_stage_true_positive_margin=last_stage_true_positive_margin,
        true_inverse_converged=true_inverse_converged,
        repairs=repairs,
        regressions=regressions,
    )

    result = {
        "status": "complete",
        "claim": "spent_codebook_all24_exact_reverse_peeling_adjudication",
        "protocol_sha256": json_sha256(protocol),
        "interpretation_sha256": json_sha256(interpretation),
        "source_k23_protocol_sha256": json_sha256(k23),
        "replay_lock_sha256": json_sha256(replay),
        "smoke_selection_sha256": json_sha256(smoke),
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
        "target_stage_executions": expected_target_calls,
        "total_stage_executions": stage_calls,
        "inverse_gradient_evaluations": gradient_calls,
        "summary": {
            "full_history_correct_of_24": full_correct,
            "last_stage_correct_of_24": last_stage_correct,
            "true_last_stage_inverse_converged_of_24": true_inverse_converged,
            "all_candidate_inverses_converged_of_24": all_candidate_converged,
            "positive_margin_of_24": positive_margin,
            "true_last_stage_positive_margin_of_24": last_stage_true_positive_margin,
            "k3_wrong_to_peeling_correct_repairs": repairs,
            "k3_correct_to_peeling_wrong_regressions": regressions,
            "interpretation_tier": tier,
            "primary_checks": primary_checks,
            "primary_support_all": all(primary_checks.values()),
        },
        "rows": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
