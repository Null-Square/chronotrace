#!/usr/bin/env python3
"""Run the frozen Pythia-14M finite-pair chronology bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from importlib.metadata import version as package_version
from itertools import permutations
from pathlib import Path
from typing import Any

from chronotrace.geometry.secant import (
    decode_finite_pair_permutation,
    finite_pair_identifiability,
    finite_pair_interactions,
    finite_pair_predicted_endpoint,
    finite_pair_signature,
    finite_pair_symmetric_reference,
    higher_order_remainder_ratio,
)
from chronotrace.reproducibility import json_sha256, tensor_sha256
from chronotrace.scale import (
    build_scale_stage_examples,
    build_scale_worlds_from_codebook,
    scale_dataset_payload,
)
from chronotrace.scale_runner import completion_batch, execute_plain_sgd_stage, flatten_parameters
from chronotrace.scale_tokens import token_codebook_from_dict, validate_token_codebook

_STAGE_SEEDS = {"A": 1101, "B": 2203, "C": 3307}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/pythia_scale.lock.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _symmetric_interactions(
    interactions: dict[tuple[str, str], Any],
) -> dict[tuple[str, str], Any]:
    """Erase pair orientation while preserving the average finite pair effect."""

    result: dict[tuple[str, str], Any] = {}
    handled: set[frozenset[str]] = set()
    for destination, source in interactions:
        pair = frozenset((destination, source))
        if pair in handled:
            continue
        reverse = (source, destination)
        midpoint = 0.5 * (interactions[(destination, source)] + interactions[reverse])
        result[(destination, source)] = midpoint
        result[reverse] = midpoint.clone()
        handled.add(pair)
    return result


def _cpu_model() -> str:
    path = Path("/proc/cpuinfo")
    if not path.exists():
        return platform.processor() or "unknown"
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lower().startswith("model name") and ":" in line:
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _runtime_fingerprint(torch: Any) -> dict[str, Any]:
    config = torch.__config__.show()
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "torch_version": str(torch.__version__),
        "transformers_version": package_version("transformers"),
        "tokenizers_version": package_version("tokenizers"),
        "huggingface_hub_version": package_version("huggingface-hub"),
        "safetensors_version": package_version("safetensors"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "torch_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
        "torch_config_sha256": hashlib.sha256(config.encode("utf-8")).hexdigest(),
        "omp_num_threads": os.getenv("OMP_NUM_THREADS"),
        "mkl_num_threads": os.getenv("MKL_NUM_THREADS"),
        "openblas_num_threads": os.getenv("OPENBLAS_NUM_THREADS"),
        "pythonhashseed": os.getenv("PYTHONHASHSEED"),
        "runner_os": os.getenv("RUNNER_OS"),
        "runner_arch": os.getenv("RUNNER_ARCH"),
        "github_sha": os.getenv("GITHUB_SHA"),
        "github_run_id": os.getenv("GITHUB_RUN_ID"),
        "github_run_attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
        "replica": os.getenv("CHRONOTRACE_REPLICA"),
    }


def _named_tensor_hashes(values: dict[str, Any]) -> dict[str, str]:
    return {name: tensor_sha256(values[name]) for name in sorted(values)}


def main() -> None:
    args = parse_args()
    lock = _load_json(args.lock)
    bridge = lock["chronology_bridge"]
    stages = tuple(str(value) for value in bridge["stages"])
    histories = tuple(str(value) for value in bridge["ground_truth_histories"])
    if stages != ("A", "B", "C"):
        raise ValueError("scale bridge is frozen to stages A/B/C")
    expected_histories = {"".join(value) for value in permutations(stages)}
    if set(histories) != expected_histories:
        raise ValueError("lock must contain all six A/B/C histories exactly once")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(int(args.threads))
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    model_id = str(bridge["model"])
    revision = str(lock["revision"])
    learning_rate = float(lock["chosen_learning_rate"])
    updates = int(bridge["updates_per_stage"])

    codebook_payload = _load_json(lock["codebook_path"])
    codebook = token_codebook_from_dict(codebook_payload)
    if codebook.sha256 != lock["codebook_sha256"]:
        raise RuntimeError("frozen codebook semantic hash mismatch")

    # Validate the tokenizer and exact training batches before loading model weights.
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    validate_token_codebook(tokenizer, codebook)
    if codebook.tokenizer_fingerprint != lock["tokenizer_fingerprint"]:
        raise RuntimeError("frozen tokenizer fingerprint mismatch")
    dataset = scale_dataset_payload(codebook)
    if dataset["sha256"] != lock["dataset_sha256"]:
        raise RuntimeError("frozen scale dataset hash mismatch")

    worlds = build_scale_worlds_from_codebook(codebook)
    examples = {stage: build_scale_stage_examples(worlds, stage) for stage in stages}
    stage_batch_sha256: dict[str, str] = {}
    for stage in stages:
        batch = completion_batch(torch, tokenizer, examples[stage], device)
        stage_batch_sha256[stage] = json_sha256(_named_tensor_hashes(batch))

    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision)
    model.to(device=device, dtype=torch.float32)
    model.config.use_cache = False
    if any(
        parameter.is_floating_point() and parameter.dtype != torch.float32
        for parameter in model.parameters()
    ):
        raise TypeError("scale bridge requires FP32 model parameters")
    if not all(bool(torch.isfinite(parameter.detach()).all()) for parameter in model.parameters()):
        raise FloatingPointError("base Pythia checkpoint contains non-finite parameters")

    theta0 = flatten_parameters(torch, model)
    parameter_count = int(theta0.numel())
    base_norm = float(torch.linalg.vector_norm(theta0))
    base_parameter_sha256 = tensor_sha256(theta0)
    runtime = _runtime_fingerprint(torch)
    stage_calls = {stage: 0 for stage in stages}

    scientific_fingerprint_sha256 = json_sha256(
        {
            "protocol_version": lock["protocol_version"],
            "model": model_id,
            "revision": revision,
            "learning_rate": learning_rate,
            "updates_per_stage": updates,
            "optimizer": "plain_sgd_no_momentum_no_weight_decay",
            "world_count": len(worlds),
            "tokenizer_fingerprint": codebook.tokenizer_fingerprint,
            "codebook_sha256": codebook.sha256,
            "dataset_sha256": dataset["sha256"],
            "base_parameter_sha256": base_parameter_sha256,
            "stage_batch_sha256": stage_batch_sha256,
        }
    )

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
    basis_calls = sum(stage_calls.values())
    expected_basis_calls = len(stages) ** 2
    if basis_calls != expected_basis_calls:
        raise RuntimeError(
            f"finite-pair basis used {basis_calls} stage calls, expected {expected_basis_calls}"
        )
    symmetric_reference = finite_pair_symmetric_reference(
        theta0,
        deltas,
        interactions,
        stages=stages,
    )
    identifiability = finite_pair_identifiability(interactions, stages=stages)
    if not identifiability.identifiable:
        raise RuntimeError("finite-pair signatures are not identifiable on Pythia-14M")

    delta_sha256 = _named_tensor_hashes(deltas)
    interaction_sha256 = {
        f"{source}->{destination}": tensor_sha256(interactions[(destination, source)])
        for source in stages
        for destination in stages
        if source != destination
    }
    symmetric_reference_sha256 = tensor_sha256(symmetric_reference)
    candidate_signature_sha256 = {
        history: tensor_sha256(
            finite_pair_signature(tuple(history), interactions, stages=stages)
        )
        for history in histories
    }
    finite_pair_basis_sha256 = json_sha256(
        {
            "deltas": delta_sha256,
            "interactions": interaction_sha256,
            "symmetric_reference": symmetric_reference_sha256,
            "candidate_signatures": candidate_signature_sha256,
        }
    )

    ablated_interactions = _symmetric_interactions(interactions)
    ablated_identifiability = finite_pair_identifiability(
        ablated_interactions,
        stages=stages,
    )
    if ablated_identifiability.identifiable:
        raise RuntimeError("orientation ablation unexpectedly retained chronology separation")

    records: list[dict[str, Any]] = []
    correct = 0
    minimum_margin = float("inf")
    maximum_ratio = 0.0
    maximum_remainder = 0.0
    history_endpoint_hashes: dict[str, str] = {}
    calls_before_ground_truth = sum(stage_calls.values())

    for history in histories:
        endpoint = theta0.clone()
        for stage in history:
            endpoint = stage_maps[stage](endpoint)
        endpoint_sha256 = tensor_sha256(endpoint)
        history_endpoint_hashes[history] = endpoint_sha256
        decoded = decode_finite_pair_permutation(
            endpoint,
            symmetric_reference,
            interactions,
            stages=stages,
        )
        predicted_true = finite_pair_predicted_endpoint(
            tuple(history),
            symmetric_reference,
            interactions,
            stages=stages,
        )
        remainder = float(torch.linalg.vector_norm(endpoint - predicted_true))
        ratio = higher_order_remainder_ratio(
            endpoint,
            predicted_true,
            minimum_signature_separation=identifiability.minimum_signature_separation,
        )
        predicted_history = "".join(decoded.permutation)
        is_correct = predicted_history == history
        correct += int(is_correct)
        minimum_margin = min(minimum_margin, decoded.margin)
        maximum_ratio = max(maximum_ratio, ratio)
        maximum_remainder = max(maximum_remainder, remainder)
        records.append(
            {
                "history": history,
                "decoded": predicted_history,
                "correct": is_correct,
                "endpoint_sha256": endpoint_sha256,
                "best_error": decoded.best_error,
                "runner_up_error": decoded.runner_up_error,
                "margin": decoded.margin,
                "higher_order_remainder_norm": remainder,
                "higher_order_ratio": ratio,
                "endpoint_displacement_norm": float(torch.linalg.vector_norm(endpoint - theta0)),
            }
        )

    ground_truth_calls = sum(stage_calls.values()) - calls_before_ground_truth
    expected_ground_truth_calls = len(histories) * len(stages)
    if ground_truth_calls != expected_ground_truth_calls:
        raise RuntimeError("unexpected number of full-history validation stage executions")

    singleton_displacements = {
        stage: float(torch.linalg.vector_norm(delta)) for stage, delta in deltas.items()
    }
    history_endpoint_sha256 = json_sha256(history_endpoint_hashes)
    result = {
        "status": "pending_gate",
        "claim": "pythia_14m_finite_pair_chronology_bridge",
        "protocol_version": lock["protocol_version"],
        "model": model_id,
        "revision": revision,
        "parameter_count": parameter_count,
        "base_parameter_norm": base_norm,
        "learning_rate": learning_rate,
        "updates_per_stage": updates,
        "optimizer": "plain_sgd_no_momentum_no_weight_decay",
        "world_count": len(worlds),
        "tokenizer_fingerprint": codebook.tokenizer_fingerprint,
        "codebook_sha256": codebook.sha256,
        "dataset_sha256": dataset["sha256"],
        "basis_stage_executions": basis_calls,
        "ground_truth_validation_stage_executions": ground_truth_calls,
        "singleton_displacement_norms": singleton_displacements,
        "runtime": runtime,
        "reproducibility": {
            "scientific_fingerprint_sha256": scientific_fingerprint_sha256,
            "base_parameter_sha256": base_parameter_sha256,
            "stage_batch_sha256": stage_batch_sha256,
            "delta_sha256": delta_sha256,
            "interaction_sha256": interaction_sha256,
            "symmetric_reference_sha256": symmetric_reference_sha256,
            "finite_pair_basis_sha256": finite_pair_basis_sha256,
            "candidate_signature_sha256": candidate_signature_sha256,
            "history_endpoint_sha256": history_endpoint_sha256,
            "history_endpoint_hashes": history_endpoint_hashes,
        },
        "finite_pair": {
            "correct": correct,
            "total": len(histories),
            "accuracy": correct / len(histories),
            "minimum_decode_margin": minimum_margin,
            "minimum_signature_separation": identifiability.minimum_signature_separation,
            "maximum_higher_order_remainder_norm": maximum_remainder,
            "maximum_higher_order_ratio": maximum_ratio,
        },
        "orientation_ablation": {
            "identifiable": ablated_identifiability.identifiable,
            "minimum_signature_separation": ablated_identifiability.minimum_signature_separation,
        },
        "histories": records,
    }

    failures: list[str] = []
    if correct != int(bridge["required_correct"]):
        failures.append(f"recovered only {correct}/{len(histories)} histories")
    if minimum_margin <= 0:
        failures.append("decode margin is non-positive")
    if not identifiability.identifiable:
        failures.append("finite-pair signatures are not identifiable")
    if ablated_identifiability.identifiable:
        failures.append("orientation ablation remained identifiable")
    result["status"] = "pass" if not failures else "fail"
    result["gate_failures"] = failures

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise RuntimeError("14M chronology gate failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
