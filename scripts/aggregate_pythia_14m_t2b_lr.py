#!/usr/bin/env python3
"""Aggregate the frozen four-seed Pythia-14M T2b learning-rate map."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from chronotrace.reproducibility import json_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="configs/pythia_14m_t2b_lr.lock.json")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator <= 0.0:
        return float("nan")
    numerator = sum(
        (x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)
    )
    return numerator / denominator


def main() -> None:
    args = parse_args()
    protocol = _load(args.protocol)
    expected_protocol_sha = json_sha256(protocol)
    expected_seeds = tuple(int(v) for v in protocol["codebook_seed_derivation"]["seeds"])
    expected_rates = tuple(float(v) for v in protocol["learning_rates"])

    files = sorted(Path(args.input_dir).glob("t2b-*.json"))
    if len(files) != len(expected_seeds):
        raise RuntimeError(
            f"expected {len(expected_seeds)} T2b slices, found {len(files)} in {args.input_dir}"
        )

    by_seed: dict[int, dict[str, Any]] = {}
    for path in files:
        payload = _load(path)
        seed = int(payload["codebook_seed"])
        if seed in by_seed:
            raise RuntimeError(f"duplicate T2b codebook seed {seed}")
        if seed not in expected_seeds:
            raise RuntimeError(f"unexpected T2b codebook seed {seed}")
        if payload["protocol_sha256"] != expected_protocol_sha:
            raise RuntimeError(f"T2b protocol hash mismatch for seed {seed}")
        rates = tuple(float(row["learning_rate"]) for row in payload["conditions"])
        if rates != expected_rates:
            raise RuntimeError(f"T2b learning-rate grid mismatch for seed {seed}")
        by_seed[seed] = payload

    if tuple(sorted(by_seed)) != tuple(sorted(expected_seeds)):
        raise RuntimeError("T2b did not return exactly the frozen seed set")

    seed_summaries: list[dict[str, Any]] = []
    restoration_passes = 0
    robustness_passes = 0
    relative_decay_passes = 0
    total_errors = 0
    total_tail_errors = 0

    low_rate = min(expected_rates)
    high_rate = max(expected_rates)

    for seed in expected_seeds:
        payload = by_seed[seed]
        conditions = payload["conditions"]
        by_rate = {float(row["learning_rate"]): row for row in conditions}
        low = by_rate[low_rate]
        high = by_rate[high_rate]

        restoration_rates = [
            float(row["learning_rate"])
            for row in conditions
            if float(row["learning_rate"]) < high_rate
            and bool(row["finite_pair_identifiable"])
            and int(row["aggregate"]["full_order_correct"]) == 6
        ]
        restoration = bool(restoration_rates)
        robustness_improves = (
            float(low["aggregate"]["minimum_tail_robustness"])
            > float(high["aggregate"]["minimum_tail_robustness"])
        )
        relative_decay = (
            float(low["aggregate"]["third_order_to_minimum_signature_ratio"])
            < float(high["aggregate"]["third_order_to_minimum_signature_ratio"])
        )

        restoration_passes += int(restoration)
        robustness_passes += int(robustness_improves)
        relative_decay_passes += int(relative_decay)

        xs: list[float] = []
        ys: list[float] = []
        condition_rows: list[dict[str, Any]] = []
        for row in conditions:
            rate = float(row["learning_rate"])
            mean_base = float(row["aggregate"]["mean_base_commutator_norm"])
            if rate > 0.0 and mean_base > 0.0:
                xs.append(math.log(rate))
                ys.append(math.log(mean_base))
            errors = int(row["aggregate"]["error_count"])
            tail_errors = int(row["aggregate"]["tail_swap_only_error_count"])
            total_errors += errors
            total_tail_errors += tail_errors
            condition_rows.append(
                {
                    "learning_rate": rate,
                    "finite_pair_identifiable": bool(row["finite_pair_identifiable"]),
                    "full_order_correct": int(row["aggregate"]["full_order_correct"]),
                    "first_stage_correct": int(row["aggregate"]["first_stage_correct"]),
                    "pairwise_precedence_accuracy": float(
                        row["aggregate"]["pairwise_precedence_accuracy"]
                    ),
                    "mean_kendall_tau": float(row["aggregate"]["mean_kendall_tau"]),
                    "error_count": errors,
                    "tail_swap_only_error_count": tail_errors,
                    "minimum_signature_separation": float(row["minimum_signature_separation"]),
                    "mean_base_commutator_norm": mean_base,
                    "mean_conditioned_commutator_norm": float(
                        row["aggregate"]["mean_conditioned_commutator_norm"]
                    ),
                    "maximum_third_order_residual_norm": float(
                        row["aggregate"]["maximum_third_order_residual_norm"]
                    ),
                    "third_order_to_minimum_signature_ratio": float(
                        row["aggregate"]["third_order_to_minimum_signature_ratio"]
                    ),
                    "minimum_tail_robustness": float(
                        row["aggregate"]["minimum_tail_robustness"]
                    ),
                    "mean_relative_commutator_drift": float(
                        row["aggregate"]["mean_relative_commutator_drift"]
                    ),
                    "mean_base_conditioned_cosine": float(
                        row["aggregate"]["mean_base_conditioned_cosine"]
                    ),
                }
            )

        seed_summaries.append(
            {
                "codebook_seed": seed,
                "codebook_sha256": payload["codebook_sha256"],
                "scientific_fingerprint_sha256": payload["scientific_fingerprint_sha256"],
                "base_commutator_loglog_slope_vs_eta": _slope(xs, ys),
                "asymptotic_restoration_observed": restoration,
                "restoration_rates": restoration_rates,
                "low_rate_tail_robustness_greater_than_high_rate": robustness_improves,
                "low_rate_third_order_ratio_smaller_than_high_rate": relative_decay,
                "conditions": condition_rows,
            }
        )

    check_1 = restoration_passes >= 3
    check_2 = robustness_passes >= 3
    check_4 = relative_decay_passes >= 3
    tail_fraction = total_tail_errors / total_errors if total_errors else 1.0

    result = {
        "status": "complete",
        "claim": "preregistered_independent_pythia_14m_t2b_lr_map_aggregate",
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": expected_protocol_sha,
        "seed_count": len(expected_seeds),
        "learning_rates": list(expected_rates),
        "selection_rule": protocol["learning_rate_selection_rule"],
        "pre_registered_checks": {
            "check_1_asymptotic_restoration": {
                "passed": check_1,
                "seeds_passing": restoration_passes,
                "required": 3,
            },
            "check_2_tail_robustness_improves": {
                "passed": check_2,
                "seeds_passing": robustness_passes,
                "required": 3,
            },
            "check_3_pair_scaling": {
                "hard_pass_rule": None,
                "note": (
                    "See per-seed log-log slopes of mean base-commutator norm versus eta; "
                    "local theory predicts approach toward 2 before numerical resolution dominates."
                ),
            },
            "check_4_third_order_relative_decay": {
                "passed": check_4,
                "seeds_passing": relative_decay_passes,
                "required": 3,
            },
            "check_5_error_structure": {
                "all_errors": total_errors,
                "tail_swap_only_errors": total_tail_errors,
                "tail_swap_fraction_among_all_errors": tail_fraction,
            },
        },
        "seed_summaries": seed_summaries,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
