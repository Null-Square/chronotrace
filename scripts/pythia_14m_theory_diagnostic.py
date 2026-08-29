#!/usr/bin/env python3
"""Diagnose the frozen portable Pythia-14M chronology failure without retuning it."""

from __future__ import annotations

import argparse
import json
from itertools import permutations
from pathlib import Path
from typing import Any

from pythia_finite_pair_bridge import _STAGE_SEEDS, _configure_portable_numerics, _load_json

from chronotrace.geometry.history import (
    directional_contamination_ratio,
    kendall_tau_for_orders,
    pairwise_precedence_accuracy,
    prefix_conditioned_commutator_diagnostic,
)
from chronotrace.geometry.secant import (
    decode_finite_pair_permutation,
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
from chronotrace.scale_tokens import token_codebook_from_dict, validate_token_codebook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/pythia_scale.lock.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def _named_tensor_hashes(values: dict[str, Any]) -> dict[str, str]:
    return {name: tensor_sha256(values[name]) for name in sorted(values)}


def main() -> None:
    args = parse_args()
    lock = _load_json(args.lock)
    bridge = lock["chronology_bridge"]
    portable = lock["portable_kernel_gate"]
    stages = tuple(str(value) for value in bridge["stages"])
    histories = tuple(str(value) for value in bridge["ground_truth_histories"])
    if stages != ("A", "B", "C"):
        raise ValueError("theory diagnostic is frozen to A/B/C")
    if set(histories) != {"".join(order) for order in permutations(stages)}:
        raise ValueError("theory diagnostic requires all six A/B/C histories")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    numerical_fingerprint = _configure_portable_numerics(torch, lock, int(args.threads))
    model_id = str(bridge["model"])
    revision = str(lock["revision"])
    learning_rate = float(lock["chosen_learning_rate"])
    updates = int(bridge["updates_per_stage"])
    device = torch.device("cpu")

    codebook_payload = json.loads(Path(lock["codebook_path"]).read_text(encoding="utf-8"))
    codebook = token_codebook_from_dict(codebook_payload)
    if codebook.sha256 != lock["codebook_sha256"]:
        raise RuntimeError("frozen codebook hash mismatch")

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    validate_token_codebook(tokenizer, codebook)
    if codebook.tokenizer_fingerprint != lock["tokenizer_fingerprint"]:
        raise RuntimeError("frozen tokenizer fingerprint mismatch")
    if scale_dataset_payload(codebook)["sha256"] != lock["dataset_sha256"]:
        raise RuntimeError("frozen scale dataset hash mismatch")

    worlds = build_scale_worlds_from_codebook(codebook)
    examples = {stage: build_scale_stage_examples(worlds, stage) for stage in stages}

    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision)
    model.to(device=device, dtype=torch.float32)
    model.config.use_cache = False
    theta0 = flatten_parameters(torch, model)
    if tensor_sha256(theta0) != portable["base_parameter_sha256"]:
        raise RuntimeError("diagnostic base checkpoint differs from portable adjudication")

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
                raise FloatingPointError(f"stage {stage} returned non-finite evidence")
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
    signatures = {
        history: finite_pair_signature(tuple(history), interactions, stages=stages)
        for history in histories
    }

    delta_hashes = _named_tensor_hashes(deltas)
    interaction_hashes = {
        f"{source}->{destination}": tensor_sha256(interactions[(destination, source)])
        for source in stages
        for destination in stages
        if source != destination
    }
    signature_hashes = {
        history: tensor_sha256(signatures[history]) for history in histories
    }
    basis_hash = json_sha256(
        {
            "deltas": delta_hashes,
            "interactions": interaction_hashes,
            "symmetric_reference": tensor_sha256(reference),
            "candidate_signatures": signature_hashes,
        }
    )
    if basis_hash != portable["finite_pair_basis_sha256"]:
        raise RuntimeError("diagnostic finite-pair basis differs from portable adjudication")

    history_endpoints: dict[tuple[str, ...], Any] = {}
    for history in histories:
        endpoint = theta0.clone()
        for stage in history:
            endpoint = stage_maps[stage](endpoint)
        history_endpoints[tuple(history)] = endpoint

    endpoint_hashes = {
        "".join(history): tensor_sha256(endpoint)
        for history, endpoint in history_endpoints.items()
    }
    endpoint_bundle_hash = json_sha256(endpoint_hashes)
    if endpoint_bundle_hash != portable["history_endpoint_sha256"]:
        raise RuntimeError("diagnostic history endpoints differ from portable adjudication")

    rows: list[dict[str, Any]] = []
    total_precedence = 0.0
    total_tau = 0.0
    first_stage_correct = 0
    for history_text in histories:
        history = tuple(history_text)
        endpoint = history_endpoints[history]
        true_signature = signatures[history_text]
        predicted_true = finite_pair_predicted_endpoint(
            history,
            reference,
            interactions,
            stages=stages,
        )
        residual = endpoint - predicted_true
        decoded = decode_finite_pair_permutation(
            endpoint,
            reference,
            interactions,
            stages=stages,
        )
        decoded_text = "".join(decoded.permutation)
        precedence = pairwise_precedence_accuracy(history_text, decoded_text)
        tau = kendall_tau_for_orders(history_text, decoded_text)
        total_precedence += precedence
        total_tau += tau
        first_stage_correct += int(decoded_text[0] == history_text[0])

        contamination: list[dict[str, Any]] = []
        for alternative in histories:
            if alternative == history_text:
                continue
            ratio = directional_contamination_ratio(
                residual,
                true_signature,
                signatures[alternative],
            )
            alternative_prediction = reference + signatures[alternative]
            contamination.append(
                {
                    "alternative": alternative,
                    "chi": ratio,
                    "crosses_boundary": ratio >= 1.0,
                    "alternative_error": float(
                        torch.linalg.vector_norm(endpoint - alternative_prediction)
                    ),
                }
            )
        contamination.sort(key=lambda item: item["chi"], reverse=True)

        tail_swap = history_text[0] + history_text[2] + history_text[1]
        tail_row = next(item for item in contamination if item["alternative"] == tail_swap)
        rows.append(
            {
                "history": history_text,
                "decoded": decoded_text,
                "correct": decoded_text == history_text,
                "first_stage_correct": decoded_text[0] == history_text[0],
                "pairwise_precedence_accuracy": precedence,
                "kendall_tau": tau,
                "triple_interaction_norm": float(torch.linalg.vector_norm(residual)),
                "triple_interaction_sha256": tensor_sha256(residual),
                "true_pairwise_error": float(torch.linalg.vector_norm(residual)),
                "tail_swap": tail_swap,
                "tail_swap_chi": tail_row["chi"],
                "tail_swap_crosses_boundary": tail_row["crosses_boundary"],
                "maximum_directional_contamination": contamination[0]["chi"],
                "maximum_contamination_alternative": contamination[0]["alternative"],
                "competitors": contamination,
            }
        )

    prefix_rows: list[dict[str, Any]] = []
    for prefix in stages:
        remaining = tuple(stage for stage in stages if stage != prefix)
        diagnostic = prefix_conditioned_commutator_diagnostic(
            history_endpoints,
            interactions,
            prefix=(prefix,),
            first=remaining[0],
            second=remaining[1],
        )
        forward = (prefix, remaining[0], remaining[1])
        reverse = (prefix, remaining[1], remaining[0])
        forward_residual = history_endpoints[forward] - finite_pair_predicted_endpoint(
            forward,
            reference,
            interactions,
            stages=stages,
        )
        reverse_residual = history_endpoints[reverse] - finite_pair_predicted_endpoint(
            reverse,
            reference,
            interactions,
            stages=stages,
        )
        residual_difference = forward_residual - reverse_residual
        conditioned = history_endpoints[forward] - history_endpoints[reverse]
        base = interactions[(remaining[1], remaining[0])] - interactions[
            (remaining[0], remaining[1])
        ]
        drift = conditioned - base
        identity_error = float(torch.linalg.vector_norm(residual_difference - drift))
        prefix_rows.append(
            {
                "prefix": prefix,
                "tail_pair": "".join(remaining),
                "base_commutator_norm": diagnostic.base_norm,
                "conditioned_commutator_norm": diagnostic.conditioned_norm,
                "commutator_drift_norm": diagnostic.drift_norm,
                "relative_commutator_drift": diagnostic.relative_drift,
                "base_conditioned_cosine": diagnostic.base_conditioned_cosine,
                "triple_residual_difference_identity_error": identity_error,
            }
        )

    result = {
        "status": "diagnostic_complete",
        "claim": "mechanism_diagnostic_only_not_independent_evidence",
        "source_portable_workflow_run_id": portable["workflow_run_id"],
        "model": model_id,
        "revision": revision,
        "learning_rate": learning_rate,
        "updates_per_stage": updates,
        "numerical_execution_fingerprint_sha256": numerical_fingerprint,
        "finite_pair_basis_sha256": basis_hash,
        "history_endpoint_sha256": endpoint_bundle_hash,
        "basis_matches_portable_gate": True,
        "endpoints_match_portable_gate": True,
        "stage_executions": sum(stage_calls.values()),
        "aggregate": {
            "full_order_correct": sum(int(row["correct"]) for row in rows),
            "full_order_total": len(rows),
            "first_stage_correct": first_stage_correct,
            "first_stage_total": len(rows),
            "pairwise_precedence_accuracy": total_precedence / len(rows),
            "mean_kendall_tau": total_tau / len(rows),
        },
        "prefix_conditioning": prefix_rows,
        "histories": rows,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
