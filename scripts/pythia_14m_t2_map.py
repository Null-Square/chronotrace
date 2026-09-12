#!/usr/bin/env python3
"""Run one frozen independent-codebook slice of the Pythia-14M T2 map."""

from __future__ import annotations

import argparse
import json
from itertools import permutations
from pathlib import Path
from typing import Any

from pythia_finite_pair_bridge import _STAGE_SEEDS, _configure_portable_numerics, _load_json

from chronotrace.geometry.history import (
    kendall_tau_for_orders,
    pairwise_precedence_accuracy,
    prefix_conditioned_commutator_diagnostic,
    prefix_tail_decision_diagnostic,
)
from chronotrace.geometry.secant import (
    decode_finite_pair_permutation,
    finite_pair_identifiability,
    finite_pair_interactions,
    finite_pair_predicted_endpoint,
    finite_pair_signature,
    finite_pair_symmetric_reference,
)
from chronotrace.reproducibility import json_sha256, tensor_sha256
from chronotrace.scale import (
    build_scale_stage_examples,
    build_scale_worlds_from_codebook,
    scale_dataset_payload,
)
from chronotrace.scale_runner import execute_plain_sgd_stage, flatten_parameters
from chronotrace.scale_tokens import build_token_codebook, tokenizer_fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="configs/pythia_14m_t2.lock.json")
    parser.add_argument("--codebook-seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def _named_tensor_hashes(values: dict[str, Any]) -> dict[str, str]:
    return {name: tensor_sha256(values[name]) for name in sorted(values)}


def _validate_protocol(protocol: dict[str, Any], base_lock: dict[str, Any]) -> None:
    bridge = base_lock["chronology_bridge"]
    if protocol["model"] != bridge["model"]:
        raise ValueError("T2 model differs from frozen base scale model")
    if protocol["revision"] != base_lock["revision"]:
        raise ValueError("T2 revision differs from frozen base scale revision")
    if float(protocol["learning_rate"]) != float(base_lock["chosen_learning_rate"]):
        raise ValueError("T2 learning rate differs from chronology-blind frozen rate")
    if tuple(protocol["stages"]) != ("A", "B", "C"):
        raise ValueError("T2 is frozen to stages A/B/C")
    histories = tuple(str(value) for value in protocol["ground_truth_histories"])
    expected = {"".join(order) for order in permutations(("A", "B", "C"))}
    if set(histories) != expected or len(histories) != 6:
        raise ValueError("T2 must contain all six A/B/C histories exactly once")
    if tuple(int(value) for value in protocol["stage_lengths"]) != (1, 2, 4, 8, 16, 32):
        raise ValueError("T2 stage-length map has drifted")
    if int(protocol["codebook_count_per_kind"]) != 16:
        raise ValueError("T2 codebook size has drifted")


def main() -> None:
    args = parse_args()
    protocol = _load_json(args.protocol)
    base_lock = _load_json(protocol["base_scale_lock"])
    _validate_protocol(protocol, base_lock)

    allowed_seeds = tuple(
        int(value) for value in protocol["codebook_seed_derivation"]["seeds"]
    )
    if int(args.codebook_seed) not in allowed_seeds:
        raise ValueError("codebook seed is not part of the frozen T2 map")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    numerical_fingerprint = _configure_portable_numerics(
        torch,
        base_lock,
        int(args.threads),
    )
    device = torch.device("cpu")
    model_id = str(protocol["model"])
    revision = str(protocol["revision"])
    learning_rate = float(protocol["learning_rate"])
    stages = tuple(str(value) for value in protocol["stages"])
    histories = tuple(str(value) for value in protocol["ground_truth_histories"])
    stage_lengths = tuple(int(value) for value in protocol["stage_lengths"])

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    actual_tokenizer_fingerprint = tokenizer_fingerprint(tokenizer)
    if actual_tokenizer_fingerprint != protocol["tokenizer_fingerprint"]:
        raise RuntimeError("T2 tokenizer fingerprint differs from frozen protocol")

    codebook = build_token_codebook(
        tokenizer,
        count=int(protocol["codebook_count_per_kind"]),
        seed=int(args.codebook_seed),
    )
    if codebook.tokenizer_fingerprint != protocol["tokenizer_fingerprint"]:
        raise RuntimeError("generated T2 codebook has tokenizer drift")
    dataset = scale_dataset_payload(codebook)
    worlds = build_scale_worlds_from_codebook(codebook)
    examples = {stage: build_scale_stage_examples(worlds, stage) for stage in stages}

    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision)
    model.to(device=device, dtype=torch.float32)
    model.config.use_cache = False
    if any(
        parameter.is_floating_point() and parameter.dtype != torch.float32
        for parameter in model.parameters()
    ):
        raise TypeError("T2 requires FP32 model parameters")
    if not all(bool(torch.isfinite(parameter.detach()).all()) for parameter in model.parameters()):
        raise FloatingPointError("base Pythia checkpoint contains non-finite parameters")

    theta0 = flatten_parameters(torch, model)
    base_parameter_sha256 = tensor_sha256(theta0)
    if base_parameter_sha256 != protocol["base_parameter_sha256"]:
        raise RuntimeError("T2 base checkpoint differs from frozen protocol")

    scientific_fingerprint = json_sha256(
        {
            "protocol_version": protocol["protocol_version"],
            "model": model_id,
            "revision": revision,
            "learning_rate": learning_rate,
            "optimizer": protocol["optimizer"],
            "codebook_seed": int(args.codebook_seed),
            "codebook_sha256": codebook.sha256,
            "dataset_sha256": dataset["sha256"],
            "base_parameter_sha256": base_parameter_sha256,
            "stage_lengths": stage_lengths,
        }
    )

    def run_condition(updates: int) -> dict[str, Any]:
        stage_calls = {stage: 0 for stage in stages}

        def stage_map(stage: str):
            def run(vector: Any) -> Any:
                torch.manual_seed(_STAGE_SEEDS[stage])
                stage_calls[stage] += 1
                endpoint, metrics = execute_plain_sgd_stage(
                    torch,
                    model,
                    tokenizer,
                    examples[stage],
                    learning_rate=learning_rate,
                    updates=updates,
                    device=device,
                    initial_vector=vector,
                )
                if not metrics.finite:
                    raise FloatingPointError(
                        f"stage {stage} returned non-finite evidence at {updates} updates"
                    )
                return endpoint

            return run

        stage_maps = {stage: stage_map(stage) for stage in stages}
        deltas, interactions = finite_pair_interactions(stage_maps, theta0)
        reference = finite_pair_symmetric_reference(
            theta0,
            deltas,
            interactions,
            stages=stages,
        )
        identifiability = finite_pair_identifiability(interactions, stages=stages)
        signatures = {
            history: finite_pair_signature(tuple(history), interactions, stages=stages)
            for history in histories
        }
        predicted_endpoints = {
            tuple(history): finite_pair_predicted_endpoint(
                tuple(history),
                reference,
                interactions,
                stages=stages,
            )
            for history in histories
        }

        delta_hashes = _named_tensor_hashes(deltas)
        interaction_hashes = {
            f"{source}->{destination}": tensor_sha256(
                interactions[(destination, source)]
            )
            for source in stages
            for destination in stages
            if source != destination
        }
        signature_hashes = {
            history: tensor_sha256(signatures[history]) for history in histories
        }
        basis_sha256 = json_sha256(
            {
                "deltas": delta_hashes,
                "interactions": interaction_hashes,
                "symmetric_reference": tensor_sha256(reference),
                "candidate_signatures": signature_hashes,
            }
        )

        history_endpoints: dict[tuple[str, ...], Any] = {}
        for history_text in histories:
            endpoint = theta0.clone()
            for stage in history_text:
                endpoint = stage_maps[stage](endpoint)
            history_endpoints[tuple(history_text)] = endpoint

        endpoint_hashes = {
            "".join(history): tensor_sha256(endpoint)
            for history, endpoint in history_endpoints.items()
        }
        history_endpoint_sha256 = json_sha256(endpoint_hashes)

        history_rows: list[dict[str, Any]] = []
        full_order_correct = 0
        first_stage_correct = 0
        total_precedence = 0.0
        total_tau = 0.0
        tail_swap_errors = 0
        max_third_order_norm = 0.0

        for history_text in histories:
            history = tuple(history_text)
            endpoint = history_endpoints[history]
            decoded = decode_finite_pair_permutation(
                endpoint,
                reference,
                interactions,
                stages=stages,
            )
            decoded_text = "".join(decoded.permutation)
            correct = decoded_text == history_text
            first_correct = decoded_text[0] == history_text[0]
            tail_swap = history_text[0] + history_text[2] + history_text[1]
            is_tail_swap_error = (not correct) and decoded_text == tail_swap
            precedence = pairwise_precedence_accuracy(history_text, decoded_text)
            tau = kendall_tau_for_orders(history_text, decoded_text)
            residual = endpoint - predicted_endpoints[history]
            residual_norm = float(torch.linalg.vector_norm(residual))

            full_order_correct += int(correct)
            first_stage_correct += int(first_correct)
            total_precedence += precedence
            total_tau += tau
            tail_swap_errors += int(is_tail_swap_error)
            max_third_order_norm = max(max_third_order_norm, residual_norm)

            history_rows.append(
                {
                    "history": history_text,
                    "decoded": decoded_text,
                    "correct": correct,
                    "first_stage_correct": first_correct,
                    "tail_swap": tail_swap,
                    "tail_swap_only_error": is_tail_swap_error,
                    "pairwise_precedence_accuracy": precedence,
                    "kendall_tau": tau,
                    "third_order_residual_norm": residual_norm,
                    "third_order_residual_sha256": tensor_sha256(residual),
                    "decode_margin": float(decoded.margin),
                }
            )

        prefix_rows: list[dict[str, Any]] = []
        for prefix in stages:
            remaining = tuple(stage for stage in stages if stage != prefix)
            commutator = prefix_conditioned_commutator_diagnostic(
                history_endpoints,
                interactions,
                prefix=(prefix,),
                first=remaining[0],
                second=remaining[1],
            )
            decision = prefix_tail_decision_diagnostic(
                history_endpoints,
                predicted_endpoints,
                prefix=(prefix,),
                first=remaining[0],
                second=remaining[1],
            )
            tail_robustness = (
                decision.alignment_coefficient
                - abs(decision.midpoint_bias_coefficient)
            )
            prefix_rows.append(
                {
                    "prefix": prefix,
                    "tail_pair": "".join(remaining),
                    "base_commutator_norm": commutator.base_norm,
                    "conditioned_commutator_norm": commutator.conditioned_norm,
                    "commutator_drift_norm": commutator.drift_norm,
                    "relative_commutator_drift": commutator.relative_drift,
                    "base_conditioned_cosine": commutator.base_conditioned_cosine,
                    "tail_alignment_coefficient": decision.alignment_coefficient,
                    "tail_midpoint_bias_coefficient": decision.midpoint_bias_coefficient,
                    "tail_robustness": tail_robustness,
                    "forward_tail_boundary_score": decision.forward_boundary_score,
                    "reverse_tail_boundary_score": decision.reverse_boundary_score,
                    "both_tail_orders_recoverable": decision.both_tail_orders_recoverable,
                }
            )

        expected_stage_calls = len(stages) ** 2 + len(histories) * len(stages)
        actual_stage_calls = sum(stage_calls.values())
        if actual_stage_calls != expected_stage_calls:
            raise RuntimeError(
                f"T2 condition used {actual_stage_calls} stage calls, "
                f"expected {expected_stage_calls}"
            )

        error_count = len(histories) - full_order_correct
        return {
            "updates_per_stage": updates,
            "stage_executions": actual_stage_calls,
            "finite_pair_identifiable": bool(identifiability.identifiable),
            "minimum_signature_separation": float(
                identifiability.minimum_signature_separation
            ),
            "finite_pair_basis_sha256": basis_sha256,
            "history_endpoint_sha256": history_endpoint_sha256,
            "aggregate": {
                "full_order_correct": full_order_correct,
                "full_order_total": len(histories),
                "first_stage_correct": first_stage_correct,
                "first_stage_total": len(histories),
                "pairwise_precedence_accuracy": total_precedence / len(histories),
                "mean_kendall_tau": total_tau / len(histories),
                "error_count": error_count,
                "tail_swap_only_error_count": tail_swap_errors,
                "tail_swap_fraction_among_errors": (
                    tail_swap_errors / error_count if error_count else 1.0
                ),
                "maximum_third_order_residual_norm": max_third_order_norm,
                "minimum_tail_robustness": min(
                    row["tail_robustness"] for row in prefix_rows
                ),
                "mean_relative_commutator_drift": sum(
                    row["relative_commutator_drift"] for row in prefix_rows
                )
                / len(prefix_rows),
                "mean_base_conditioned_cosine": sum(
                    row["base_conditioned_cosine"] for row in prefix_rows
                )
                / len(prefix_rows),
            },
            "prefix_conditioning": prefix_rows,
            "histories": history_rows,
        }

    conditions = [run_condition(updates) for updates in stage_lengths]
    result = {
        "status": "complete",
        "claim": "independent_pythia_14m_state_conditioned_interaction_map_slice",
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": json_sha256(protocol),
        "model": model_id,
        "revision": revision,
        "learning_rate": learning_rate,
        "optimizer": protocol["optimizer"],
        "codebook_seed": int(args.codebook_seed),
        "codebook_sha256": codebook.sha256,
        "dataset_sha256": dataset["sha256"],
        "world_count": len(worlds),
        "base_parameter_sha256": base_parameter_sha256,
        "numerical_execution_fingerprint_sha256": numerical_fingerprint,
        "scientific_fingerprint_sha256": scientific_fingerprint,
        "conditions": conditions,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
