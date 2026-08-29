#!/usr/bin/env python3
# ruff: noqa: E501,I001
"""Run the frozen all-24 forward-reachable separation certificate on Pythia-14M."""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import tempfile
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np

from pythia_finite_pair_bridge import _configure_portable_numerics, _load_json

from chronotrace.geometry.interactions import (
    measure_ordered_interaction_basis_compact,
    ordered_interaction_word_prediction,
    ordered_probe_count,
)
from chronotrace.geometry.reachable import (
    certify_reachable_distance_table,
    chunked_l2_distance,
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
        default="configs/pythia_14m_forward_reachable_all24.lock.json",
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
        default="configs/pythia_14m_forward_reachable_residual_smoke.selection.json",
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
        raise RuntimeError("model has no parameters while building reachable codewords")
    return torch.cat(parts)


def _write_reachable_delta(
    np_module: Any,
    path: Path,
    predecessor: Any,
    gradient: Any,
    base: Any,
    *,
    learning_rate: float,
    chunk_size: int,
) -> None:
    count = int(base.numel())
    output = np_module.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np_module.float64,
        shape=(count,),
    )
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        chunk = (
            predecessor[start:stop]
            - learning_rate * gradient[start:stop]
            - base[start:stop]
        )
        output[start:stop] = chunk.detach().cpu().numpy()
    output.flush()
    del output


def _tensor_delta_to_memmap_l2(
    np_module: Any,
    tensor: Any,
    base: Any,
    mapped_delta: Any,
    *,
    chunk_size: int,
) -> float:
    count = int(base.numel())
    total = 0.0
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        observed = (tensor[start:stop] - base[start:stop]).detach().cpu().numpy()
        expected = np_module.asarray(mapped_delta[start:stop], dtype=np_module.float64)
        difference = observed - expected
        total += float(np_module.dot(difference, difference))
    if not math.isfinite(total) or total < 0.0:
        raise FloatingPointError("self-residual accumulation became invalid")
    return math.sqrt(total)


def main() -> None:
    args = parse_args()
    protocol = _load_json(args.protocol)
    k23 = _load_json(args.k23_protocol)
    replay = _load_json(args.replay_lock)
    smoke = _load_json(args.smoke_selection)

    if protocol["freeze_status"] != "frozen_before_any_all24_forward_reachable_output":
        raise ValueError("all-24 forward-reachable protocol is not frozen")
    if protocol["experiment_role"] != "non_confirmatory_spent_codebook_separation_map":
        raise ValueError("all-24 forward-reachable role drift")
    if protocol.get("confirmation_codebooks_observed") is not False:
        raise RuntimeError("all-24 forward-reachable protocol touched confirmation codebooks")
    if k23.get("confirmation_codebooks_observed") is not False:
        raise RuntimeError("source K2/K3 protocol touched confirmation codebooks")
    if replay.get("confirmation_codebooks_observed") is not False:
        raise RuntimeError("source replay lock touched confirmation codebooks")
    if smoke.get("confirmation_codebooks_observed") is not False:
        raise RuntimeError("source forward-reachable smoke touched confirmation codebooks")
    if json_sha256(k23) != str(protocol["source_k23_protocol_sha256"]):
        raise RuntimeError("source K2/K3 protocol hash drift")
    if int(replay.get("source_workflow_run_id", -1)) != int(protocol["source_k23_run_id"]):
        raise RuntimeError("source replay workflow identity drift")
    if int(smoke.get("source_run_id", -1)) != int(protocol["source_smoke_run_id"]):
        raise RuntimeError("source forward-reachable smoke run identity drift")
    if int(smoke.get("source_artifact_id", -1)) != int(protocol["source_smoke_artifact_id"]):
        raise RuntimeError("source forward-reachable smoke artifact identity drift")
    if smoke.get("smoke_pass_all") is not True:
        raise RuntimeError("all-24 map requires the frozen passing one-history smoke")
    if smoke.get("next_step") != "freeze_all24_spent_forward_reachable_margin_map":
        raise RuntimeError("source smoke does not authorize the all-24 spent map")

    stages = tuple(str(value) for value in protocol["stages"])
    if stages != ("A", "B", "C", "D"):
        raise ValueError("all-24 map requires A/B/C/D")
    histories = tuple(str(value) for value in protocol["histories"])
    expected_histories = tuple("".join(value) for value in permutations(stages))
    if histories != expected_histories:
        raise ValueError("all-24 history ordering drift")
    if int(protocol["max_measured_interaction_degree"]) != 3:
        raise ValueError("all-24 map requires degree-three basis")
    if int(protocol["candidate_hypothesis_count"]) != len(histories):
        raise ValueError("candidate hypothesis count drift")
    if int(protocol["candidate_gradient_evaluations"]) != len(histories):
        raise ValueError("candidate gradient budget drift")

    frozen_hashes = {str(key): str(value) for key, value in replay["target_endpoint_hashes"].items()}
    if set(frozen_hashes) != set(histories):
        raise RuntimeError("frozen replay history set drift")

    seed = int(protocol["pilot_codebook_seed"])
    prohibited = {int(value) for value in protocol["confirmation_codebooks_prohibited"]}
    if seed in prohibited:
        raise RuntimeError("all-24 map may not use a held-out confirmation codebook")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    numerical_fingerprint = _configure_portable_numerics(torch, k23, int(args.threads))
    device = torch.device("cpu")
    model_id = str(protocol["model"])
    revision = str(protocol["revision"])
    learning_rate = float(protocol["learning_rate"])
    chunk_size = int(protocol["distance_chunk_size_elements"])

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
        raise RuntimeError("frozen basis probe count drift")
    if basis.stage_executions != expected_basis_calls or stage_calls != expected_basis_calls:
        raise RuntimeError("observed basis probe count drift")
    if any(value.dtype != torch.float64 for value in basis.interactions.values()):
        raise TypeError("all-24 interaction basis contains non-FP64 tensor")

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
            raise FloatingPointError(f"non-finite all-24 loss for stage {stage}")
        output.loss.backward()
        gradient = _flatten_gradients(torch, model)
        if gradient.dtype != torch.float64:
            raise TypeError("all-24 candidate gradient is not FP64")
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError(f"non-finite all-24 gradient for stage {stage}")
        model.zero_grad(set_to_none=True)
        gradient_calls += 1
        return gradient

    with tempfile.TemporaryDirectory(prefix="chronotrace-reachable-all24-") as directory:
        tempdir = Path(directory)
        estimated_bytes = len(histories) * int(theta0.numel()) * 8
        free_before = int(shutil.disk_usage(tempdir).free)
        if free_before < estimated_bytes + 512 * 1024 * 1024:
            raise RuntimeError("insufficient temporary disk for FP64 reachable codewords")

        codeword_paths: dict[str, Path] = {}
        for history in histories:
            predecessor_word = tuple(history[:-1])
            final_stage = history[-1]
            predecessor = ordered_interaction_word_prediction(
                predecessor_word,
                basis,
                degree=3,
            )
            gradient = gradient_at(final_stage, predecessor)
            path = tempdir / f"{history}.npy"
            _write_reachable_delta(
                np,
                path,
                predecessor,
                gradient,
                theta0,
                learning_rate=learning_rate,
                chunk_size=chunk_size,
            )
            codeword_paths[history] = path
            del gradient
            del predecessor
            gc.collect()

        if gradient_calls != int(protocol["candidate_gradient_evaluations"]):
            raise RuntimeError("all-24 candidate gradient evaluation count drift")

        temporary_disk_bytes = sum(path.stat().st_size for path in codeword_paths.values())
        mapped = {
            history: np.load(path, mmap_mode="r")
            for history, path in codeword_paths.items()
        }
        pairwise_distances = {
            history: {other: 0.0 for other in histories}
            for history in histories
        }
        minimum_pair = None
        minimum_pairwise_separation = float("inf")
        for left_index, left in enumerate(histories):
            for right in histories[left_index + 1 :]:
                distance = chunked_l2_distance(
                    mapped[left],
                    mapped[right],
                    chunk_size=chunk_size,
                )
                pairwise_distances[left][right] = distance
                pairwise_distances[right][left] = distance
                if distance < minimum_pairwise_separation:
                    minimum_pairwise_separation = distance
                    minimum_pair = (left, right)

        if minimum_pair is None or not math.isfinite(minimum_pairwise_separation):
            raise RuntimeError("failed to construct finite reachable codeword separation table")

        target_hashes: dict[str, str] = {}
        target_replay_exact: dict[str, bool] = {}
        self_errors: dict[str, float] = {}
        relative_self_errors: dict[str, float] = {}
        target_norms: dict[str, float] = {}
        trie_start_calls = stage_calls

        def visit(prefix: tuple[str, ...], state: Any) -> None:
            for stage in stages:
                if stage in prefix:
                    continue
                endpoint = stage_maps[stage](state)
                child = prefix + (stage,)
                if len(child) == len(stages):
                    history = "".join(child)
                    endpoint_hash = tensor_sha256(endpoint)
                    target_hashes[history] = endpoint_hash
                    target_replay_exact[history] = endpoint_hash == frozen_hashes[history]
                    target_norm = float(torch.linalg.vector_norm(endpoint))
                    target_norms[history] = target_norm
                    error = _tensor_delta_to_memmap_l2(
                        np,
                        endpoint,
                        theta0,
                        mapped[history],
                        chunk_size=chunk_size,
                    )
                    self_errors[history] = error
                    relative_self_errors[history] = error / max(1.0, target_norm)
                else:
                    visit(child, endpoint)
                del endpoint
                gc.collect()

        visit((), theta0)
        target_trie_calls = stage_calls - trie_start_calls
        if target_trie_calls != int(protocol["target_trie_stage_executions"]):
            raise RuntimeError("all-24 target permutation-trie execution count drift")
        if set(target_hashes) != set(histories):
            raise RuntimeError("all-24 target trie did not visit every history")

        certificate = certify_reachable_distance_table(pairwise_distances, self_errors)
        threshold = float(protocol["self_residual_relative_threshold"])
        per_history = {
            history: {
                "target_endpoint_sha256": target_hashes[history],
                "target_replay_exact": target_replay_exact[history],
                "target_parameter_norm": target_norms[history],
                "self_residual": self_errors[history],
                "relative_self_residual": relative_self_errors[history],
                "nearest_competing_codeword": certificate.nearest_neighbor[history],
                "nearest_codeword_separation": certificate.nearest_separation[history],
                "certified_margin": certificate.certified_margin[history],
                "target_noise_radius": certificate.target_noise_radius[history],
                "certified": certificate.certified_margin[history] > 0.0,
            }
            for history in histories
        }

        all_values_finite = all(
            math.isfinite(value)
            for row in pairwise_distances.values()
            for value in row.values()
        ) and all(math.isfinite(value) for value in self_errors.values())
        all_relative_self_small = all(
            value <= threshold for value in relative_self_errors.values()
        )
        all_target_replay_exact = all(target_replay_exact.values())
        pairwise_distinct = minimum_pairwise_separation > 0.0
        certified_count = sum(
            int(certificate.certified_margin[history] > 0.0)
            for history in histories
        )

        checks = {
            "source_smoke_is_frozen_pass": smoke.get("smoke_pass_all") is True,
            "all_target_endpoint_hashes_replay_exact": all_target_replay_exact,
            "basis_stage_execution_count_exact": basis.stage_executions == int(protocol["basis_stage_executions"]),
            "target_trie_stage_execution_count_exact": target_trie_calls == int(protocol["target_trie_stage_executions"]),
            "candidate_gradient_evaluation_count_exact": gradient_calls == int(protocol["candidate_gradient_evaluations"]),
            "all_distance_and_self_residual_values_finite": all_values_finite,
            "all_relative_self_residuals_within_threshold": all_relative_self_small,
            "all_reachable_codewords_pairwise_distinct": pairwise_distinct,
            "all_histories_separation_certified": certificate.all_certified and certified_count == len(histories),
            "minimum_target_noise_radius_positive": certificate.minimum_target_noise_radius > 0.0,
        }

        result = {
            "status": "complete",
            "claim": "non_confirmatory_all24_finite_forward_reachable_codebook_separation_certificate",
            "protocol_sha256": json_sha256(protocol),
            "source_k23_protocol_sha256": json_sha256(k23),
            "source_replay_lock_sha256": json_sha256(replay),
            "source_smoke_selection_sha256": json_sha256(smoke),
            "confirmation_codebooks_observed": False,
            "model": model_id,
            "revision": revision,
            "precision": "fp64",
            "learning_rate": learning_rate,
            "pilot_codebook_seed": seed,
            "base_parameter_sha256_fp64": fp64_hash,
            "base_parameter_sha256_projected_fp32": projected_hash,
            "numerical_execution_fingerprint_sha256": numerical_fingerprint,
            "frozen_baselines": protocol["frozen_baselines"],
            "basis_stage_executions": basis.stage_executions,
            "target_trie_stage_executions": target_trie_calls,
            "total_stage_executions": stage_calls,
            "candidate_gradient_evaluations": gradient_calls,
            "candidate_hypothesis_count": len(histories),
            "distance_chunk_size_elements": chunk_size,
            "temporary_disk_bytes": temporary_disk_bytes,
            "temporary_disk_free_before_bytes": free_before,
            "minimum_pairwise_separation": minimum_pairwise_separation,
            "minimum_pairwise_separation_pair": list(minimum_pair),
            "minimum_certified_margin": certificate.minimum_certified_margin,
            "minimum_target_noise_radius": certificate.minimum_target_noise_radius,
            "maximum_relative_self_residual": max(relative_self_errors.values()),
            "certified_full_order_correct": certified_count,
            "certified_full_order_total": len(histories),
            "per_history": per_history,
            "pairwise_reachable_codeword_distances": pairwise_distances,
            "all24_checks": checks,
            "all24_pass_all": all(checks.values()),
            "next_step": (
                "design_subfactorial_local_transition_decoder_on_spent_data"
                if all(checks.values())
                else "inspect_frozen_self_error_and_codeword_separation_tables_without_retuning"
            ),
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "all24_pass_all": result["all24_pass_all"],
                    "certified_full_order_correct": certified_count,
                    "minimum_pairwise_separation": minimum_pairwise_separation,
                    "minimum_pairwise_separation_pair": list(minimum_pair),
                    "maximum_relative_self_residual": result["maximum_relative_self_residual"],
                    "minimum_certified_margin": certificate.minimum_certified_margin,
                    "minimum_target_noise_radius": certificate.minimum_target_noise_radius,
                    "candidate_gradient_evaluations": gradient_calls,
                    "target_trie_stage_executions": target_trie_calls,
                    "temporary_disk_bytes": temporary_disk_bytes,
                },
                indent=2,
                sort_keys=True,
            )
        )

        for value in mapped.values():
            del value
        gc.collect()


if __name__ == "__main__":
    main()
