#!/usr/bin/env python3
"""Aggregate the frozen Pythia-14M FP32/FP64 precision-adjudication slices."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from chronotrace.reproducibility import json_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="configs/pythia_14m_precision_gate.lock.json",
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.mean(finite) if finite else float("nan")


def main() -> None:
    args = parse_args()
    protocol = _load(args.protocol)
    protocol_sha = json_sha256(protocol)
    expected_seeds = tuple(int(value) for value in protocol["codebook_seeds"])
    expected_rates = tuple(float(value) for value in protocol["learning_rates"])

    files = sorted(Path(args.input_dir).glob("precision-*.json"))
    if len(files) != len(expected_seeds):
        raise RuntimeError(
            f"expected {len(expected_seeds)} precision slices, found {len(files)}"
        )

    by_seed: dict[int, dict[str, Any]] = {}
    for path in files:
        payload = _load(path)
        seed = int(payload["codebook_seed"])
        if seed in by_seed or seed not in expected_seeds:
            raise RuntimeError(f"unexpected or duplicate precision seed {seed}")
        if payload["protocol_sha256"] != protocol_sha:
            raise RuntimeError(f"precision protocol hash mismatch for seed {seed}")

        base_check = payload.get("base_cross_precision_check")
        if not isinstance(base_check, dict) or base_check.get("exact_fp32_lift_match") is not True:
            raise RuntimeError(
                f"precision seed {seed} lacks the exact FP32-to-FP64 base replay check"
            )
        fp64_hash = payload["base_parameter_hashes_by_precision"]["fp64"]
        if fp64_hash != base_check.get("fp32_lifted_to_fp64_sha256"):
            raise RuntimeError(
                f"precision seed {seed} did not preserve the exact cross-precision base vector"
            )

        for precision in ("fp32", "fp64"):
            rates = tuple(
                float(row["learning_rate"])
                for row in payload["precision_curves"][precision]
            )
            if rates != expected_rates:
                raise RuntimeError(f"precision rate grid mismatch for seed {seed}")
        by_seed[seed] = payload

    if tuple(sorted(by_seed)) != tuple(sorted(expected_seeds)):
        raise RuntimeError("precision gate did not return exactly the frozen seed set")

    seed_rows: list[dict[str, Any]] = []
    for seed in expected_seeds:
        payload = by_seed[seed]
        comparison_by_rate = {
            float(row["learning_rate"]): row for row in payload["fp32_fp64_comparisons"]
        }
        low_comparisons = [
            comparison_by_rate[rate] for rate in expected_rates if rate <= 1e-6
        ]
        low_cosines = [
            float(pair["fp32_fp64_cosine"])
            for row in low_comparisons
            for pair in row["pairs"].values()
        ]
        low_relative_errors = [
            float(pair["relative_vector_error_to_fp64"])
            for row in low_comparisons
            for pair in row["pairs"].values()
        ]
        seed_rows.append(
            {
                "codebook_seed": seed,
                "codebook_sha256": payload["codebook_sha256"],
                "fp32_global_slope": float(payload["summary"]["fp32"]["global_loglog_slope"]),
                "fp64_global_slope": float(payload["summary"]["fp64"]["global_loglog_slope"]),
                "fp32_smallest_three_slope": float(
                    payload["summary"]["fp32"]["smallest_three_loglog_slope"]
                ),
                "fp64_smallest_three_slope": float(
                    payload["summary"]["fp64"]["smallest_three_loglog_slope"]
                ),
                "fp32_smallest_four_slope": float(
                    payload["summary"]["fp32"]["smallest_four_loglog_slope"]
                ),
                "fp64_smallest_four_slope": float(
                    payload["summary"]["fp64"]["smallest_four_loglog_slope"]
                ),
                "low_rate_mean_fp32_fp64_cosine": _finite_mean(low_cosines),
                "low_rate_mean_relative_vector_error_to_fp64": _finite_mean(
                    low_relative_errors
                ),
            }
        )

    curve_rows: list[dict[str, Any]] = []
    for rate in expected_rates:
        row: dict[str, Any] = {"learning_rate": rate}
        for precision in ("fp32", "fp64"):
            selected = []
            for seed in expected_seeds:
                payload = by_seed[seed]
                match = next(
                    item
                    for item in payload["precision_curves"][precision]
                    if float(item["learning_rate"]) == rate
                )
                selected.append(match)
            row[f"{precision}_mean_commutator_norm"] = statistics.mean(
                float(item["mean_commutator_norm"]) for item in selected
            )
            row[f"{precision}_mean_commutator_over_eta_squared"] = statistics.mean(
                float(item["mean_commutator_norm_divided_by_eta_squared"])
                for item in selected
            )
            singleton_values = [
                float(value)
                for item in selected
                for value in item["singleton_displacement_norms"].values()
            ]
            row[f"{precision}_mean_singleton_displacement"] = statistics.mean(
                singleton_values
            )

        comparisons = []
        for seed in expected_seeds:
            payload = by_seed[seed]
            comparisons.append(
                next(
                    item
                    for item in payload["fp32_fp64_comparisons"]
                    if float(item["learning_rate"]) == rate
                )
            )
        cosines = [
            float(pair["fp32_fp64_cosine"])
            for comparison in comparisons
            for pair in comparison["pairs"].values()
        ]
        relative_errors = [
            float(pair["relative_vector_error_to_fp64"])
            for comparison in comparisons
            for pair in comparison["pairs"].values()
        ]
        norm_ratios = [
            float(pair["fp32_over_fp64_norm"])
            for comparison in comparisons
            for pair in comparison["pairs"].values()
        ]
        row["mean_fp32_fp64_cosine"] = _finite_mean(cosines)
        row["mean_relative_vector_error_to_fp64"] = _finite_mean(relative_errors)
        row["mean_fp32_over_fp64_norm"] = _finite_mean(norm_ratios)
        curve_rows.append(row)

    result = {
        "status": "complete",
        "claim": "aggregate_numerical_only_fp32_fp64_one_step_commutator_adjudication",
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": protocol_sha,
        "source_t2b_run_id": protocol["source_t2b_run_id"],
        "seed_count": len(expected_seeds),
        "seed_summaries": seed_rows,
        "aggregate_curves": curve_rows,
        "interpretation_rule": protocol["pre_registered_interpretation"],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
