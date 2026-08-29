#!/usr/bin/env python3
"""Adjudicate FP32 versus FP64 one-step chronology geometry on frozen T2b instances."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from pythia_finite_pair_bridge import _STAGE_SEEDS, _configure_portable_numerics, _load_json

from chronotrace.geometry.history import finite_pair_commutator
from chronotrace.geometry.secant import finite_pair_interactions
from chronotrace.reproducibility import json_sha256, tensor_sha256
from chronotrace.scale import build_scale_stage_examples, build_scale_worlds_from_codebook
from chronotrace.scale_runner import execute_plain_sgd_stage, flatten_parameters
from chronotrace.scale_tokens import build_token_codebook, tokenizer_fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="configs/pythia_14m_precision_gate.lock.json",
    )
    parser.add_argument("--codebook-seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def _loglog_slope(rows: list[dict[str, Any]]) -> float:
    points = [
        (math.log(float(row["learning_rate"])), math.log(float(row["mean_commutator_norm"])))
        for row in rows
        if float(row["learning_rate"]) > 0.0 and float(row["mean_commutator_norm"]) > 0.0
    ]
    if len(points) < 2:
        return float("nan")
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((value - mean_x) ** 2 for value in xs)
    if denominator <= 0.0:
        return float("nan")
    numerator = sum(
        (x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)
    )
    return numerator / denominator


def _adjacent_slopes(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    result: list[dict[str, float]] = []
    for left, right in zip(rows, rows[1:], strict=False):
        eta_left = float(left["learning_rate"])
        eta_right = float(right["learning_rate"])
        norm_left = float(left["mean_commutator_norm"])
        norm_right = float(right["mean_commutator_norm"])
        slope = float("nan")
        if min(eta_left, eta_right, norm_left, norm_right) > 0.0:
            slope = math.log(norm_right / norm_left) / math.log(eta_right / eta_left)
        result.append(
            {
                "eta_left": eta_left,
                "eta_right": eta_right,
                "slope": slope,
            }
        )
    return result


def _cosine(torch: Any, left: Any, right: Any) -> float:
    left64 = left.to(dtype=torch.float64)
    right64 = right.to(dtype=torch.float64)
    left_norm = torch.linalg.vector_norm(left64)
    right_norm = torch.linalg.vector_norm(right64)
    if float(left_norm) <= 0.0 or float(right_norm) <= 0.0:
        return float("nan")
    return float(torch.dot(left64.reshape(-1), right64.reshape(-1)) / (left_norm * right_norm))


def main() -> None:
    args = parse_args()
    protocol = _load_json(args.protocol)
    t2b = _load_json(protocol["source_t2b_protocol"])
    base_lock = _load_json(t2b["base_scale_lock"])

    seed = int(args.codebook_seed)
    allowed_seeds = tuple(int(value) for value in protocol["codebook_seeds"])
    if seed not in allowed_seeds:
        raise ValueError("codebook seed is not part of the frozen precision gate")
    if int(protocol["updates_per_stage"]) != 1:
        raise ValueError("precision gate is frozen to one update per stage")
    rates = tuple(float(value) for value in protocol["learning_rates"])
    expected_rates = (1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4)
    if rates != expected_rates:
        raise ValueError("precision-gate learning-rate grid has drifted")
    if tuple(protocol["precisions"]) != ("fp32", "fp64"):
        raise ValueError("precision gate must compare fp32 and fp64")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    numerical_fingerprint = _configure_portable_numerics(torch, base_lock, int(args.threads))
    device = torch.device("cpu")
    model_id = str(protocol["model"])
    revision = str(protocol["revision"])
    stages = tuple(str(value) for value in protocol["stages"])
    if stages != ("A", "B", "C"):
        raise ValueError("precision gate is frozen to A/B/C")

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer_fingerprint(tokenizer) != t2b["tokenizer_fingerprint"]:
        raise RuntimeError("precision-gate tokenizer differs from T2b")

    codebook = build_token_codebook(
        tokenizer,
        count=int(t2b["codebook_count_per_kind"]),
        seed=seed,
    )
    worlds = build_scale_worlds_from_codebook(codebook)
    examples = {stage: build_scale_stage_examples(worlds, stage) for stage in stages}
    expected_base_hash = str(t2b["base_parameter_sha256"])

    precision_rows: dict[str, list[dict[str, Any]]] = {}
    raw_commutators: dict[str, dict[float, dict[str, Any]]] = {}
    base_hashes: dict[str, str] = {}

    for precision in protocol["precisions"]:
        dtype = torch.float32 if precision == "fp32" else torch.float64
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision)
        model.to(device=device, dtype=dtype)
        model.config.use_cache = False
        if any(
            parameter.is_floating_point() and parameter.dtype != dtype
            for parameter in model.parameters()
        ):
            raise TypeError(f"could not enforce {precision} model parameters")

        theta0 = flatten_parameters(
            torch,
            model,
            preserve_parameter_dtype=True,
        )
        if theta0.dtype != dtype:
            raise TypeError(f"native vector dtype drift for {precision}")
        base_hashes[precision] = tensor_sha256(theta0)
        projected_base_hash = tensor_sha256(theta0.to(dtype=torch.float32))
        if projected_base_hash != expected_base_hash:
            raise RuntimeError(f"{precision} base checkpoint differs from frozen T2b weights")

        rows: list[dict[str, Any]] = []
        raw_commutators[precision] = {}
        for learning_rate in rates:
            stage_calls = {stage: 0 for stage in stages}
            singleton_gradient_norms: dict[str, float] = {}

            def stage_map(stage: str):
                def run(vector: Any) -> Any:
                    torch.manual_seed(_STAGE_SEEDS[stage])
                    is_singleton_call = stage_calls[stage] == 0
                    stage_calls[stage] += 1
                    endpoint, metrics = execute_plain_sgd_stage(
                        torch,
                        model,
                        tokenizer,
                        examples[stage],
                        learning_rate=learning_rate,
                        updates=1,
                        device=device,
                        initial_vector=vector,
                        preserve_parameter_dtype=True,
                    )
                    if not metrics.finite:
                        raise FloatingPointError(
                            f"non-finite {precision} stage {stage} at eta={learning_rate}"
                        )
                    if is_singleton_call:
                        singleton_gradient_norms[stage] = metrics.max_gradient_norm
                    return endpoint

                return run

            stage_maps = {stage: stage_map(stage) for stage in stages}
            deltas, interactions = finite_pair_interactions(stage_maps, theta0)
            if sum(stage_calls.values()) != len(stages) ** 2:
                raise RuntimeError("precision gate did not use exactly N^2 pair-basis stage calls")

            pair_vectors: dict[str, Any] = {}
            pair_norms: dict[str, float] = {}
            for first, second in (("A", "B"), ("A", "C"), ("B", "C")):
                name = first + second
                vector = finite_pair_commutator(
                    interactions,
                    first=first,
                    second=second,
                )
                pair_vectors[name] = vector.detach().cpu()
                pair_norms[name] = float(torch.linalg.vector_norm(vector))

            raw_commutators[precision][learning_rate] = pair_vectors
            mean_commutator_norm = sum(pair_norms.values()) / len(pair_norms)
            rows.append(
                {
                    "learning_rate": learning_rate,
                    "model_dtype": str(dtype),
                    "singleton_displacement_norms": {
                        stage: float(torch.linalg.vector_norm(delta))
                        for stage, delta in sorted(deltas.items())
                    },
                    "singleton_gradient_norms": singleton_gradient_norms,
                    "pair_commutator_norms": pair_norms,
                    "mean_commutator_norm": mean_commutator_norm,
                    "mean_commutator_norm_divided_by_eta_squared": (
                        mean_commutator_norm / (learning_rate**2)
                    ),
                }
            )

        precision_rows[precision] = rows

    comparisons: list[dict[str, Any]] = []
    for learning_rate in rates:
        pair_rows: dict[str, Any] = {}
        for pair in ("AB", "AC", "BC"):
            fp32 = raw_commutators["fp32"][learning_rate][pair]
            fp64 = raw_commutators["fp64"][learning_rate][pair]
            norm32 = float(torch.linalg.vector_norm(fp32.to(dtype=torch.float64)))
            norm64 = float(torch.linalg.vector_norm(fp64.to(dtype=torch.float64)))
            relative_error = float(
                torch.linalg.vector_norm(
                    fp32.to(dtype=torch.float64) - fp64.to(dtype=torch.float64)
                )
            ) / norm64 if norm64 > 0.0 else float("inf")
            pair_rows[pair] = {
                "fp32_norm": norm32,
                "fp64_norm": norm64,
                "fp32_over_fp64_norm": norm32 / norm64 if norm64 > 0.0 else float("inf"),
                "fp32_fp64_cosine": _cosine(torch, fp32, fp64),
                "relative_vector_error_to_fp64": relative_error,
            }
        comparisons.append(
            {
                "learning_rate": learning_rate,
                "pairs": pair_rows,
            }
        )

    summary: dict[str, Any] = {}
    for precision, rows in precision_rows.items():
        summary[precision] = {
            "global_loglog_slope": _loglog_slope(rows),
            "adjacent_loglog_slopes": _adjacent_slopes(rows),
            "smallest_three_loglog_slope": _loglog_slope(rows[:3]),
            "smallest_four_loglog_slope": _loglog_slope(rows[:4]),
        }

    result = {
        "status": "complete",
        "claim": "numerical_only_fp32_fp64_one_step_commutator_adjudication",
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": json_sha256(protocol),
        "source_t2b_run_id": protocol["source_t2b_run_id"],
        "codebook_seed": seed,
        "codebook_sha256": codebook.sha256,
        "model": model_id,
        "revision": revision,
        "base_parameter_hashes_by_precision": base_hashes,
        "numerical_execution_fingerprint_sha256": numerical_fingerprint,
        "summary": summary,
        "precision_curves": precision_rows,
        "fp32_fp64_comparisons": comparisons,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
