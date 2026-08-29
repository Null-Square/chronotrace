#!/usr/bin/env python3
"""Aggregate the frozen four-seed Pythia-14M T2 interaction map."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from chronotrace.reproducibility import json_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="configs/pythia_14m_t2.lock.json")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    dx = [value - mean_x for value in x]
    dy = [value - mean_y for value in y]
    denominator = math.sqrt(
        sum(value * value for value in dx) * sum(value * value for value in dy)
    )
    if denominator == 0.0:
        return float("nan")
    return sum(left * right for left, right in zip(dx, dy, strict=True)) / denominator


def main() -> None:
    args = parse_args()
    protocol = _load(args.protocol)
    expected_protocol_hash = json_sha256(protocol)
    expected_seeds = tuple(
        int(value) for value in protocol["codebook_seed_derivation"]["seeds"]
    )
    expected_lengths = tuple(int(value) for value in protocol["stage_lengths"])

    files = sorted(Path(args.input_dir).glob("*.json"))
    if not files:
        raise RuntimeError("no T2 slice JSON files found")
    slices = [_load(path) for path in files]
    by_seed: dict[int, dict[str, Any]] = {}
    for payload in slices:
        seed = int(payload["codebook_seed"])
        if seed in by_seed:
            raise RuntimeError(f"duplicate T2 slice for seed {seed}")
        if payload["protocol_sha256"] != expected_protocol_hash:
            raise RuntimeError(f"T2 slice {seed} used a different protocol lock")
        observed_lengths = tuple(
            int(row["updates_per_stage"]) for row in payload["conditions"]
        )
        if observed_lengths != expected_lengths:
            raise RuntimeError(f"T2 slice {seed} has stage-length drift")
        by_seed[seed] = payload

    if set(by_seed) != set(expected_seeds):
        raise RuntimeError(
            f"expected T2 seeds {sorted(expected_seeds)}, got {sorted(by_seed)}"
        )

    seed_summaries: list[dict[str, Any]] = []
    all_errors = 0
    all_tail_swap_errors = 0
    first_failure_tail_checks: list[bool] = []
    first_failure_partial_advantage: list[bool] = []

    for seed in expected_seeds:
        payload = by_seed[seed]
        conditions = payload["conditions"]
        first_failure = next(
            (
                row
                for row in conditions
                if int(row["aggregate"]["full_order_correct"])
                < int(row["aggregate"]["full_order_total"])
            ),
            None,
        )
        seed_error_count = sum(int(row["aggregate"]["error_count"]) for row in conditions)
        seed_tail_errors = sum(
            int(row["aggregate"]["tail_swap_only_error_count"])
            for row in conditions
        )
        all_errors += seed_error_count
        all_tail_swap_errors += seed_tail_errors

        if first_failure is None:
            first_failure_updates = None
            first_failure_tail_robustness = None
            first_failure_tail_check = True
            first_failure_first_stage_advantage = True
        else:
            aggregate = first_failure["aggregate"]
            first_failure_updates = int(first_failure["updates_per_stage"])
            first_failure_tail_robustness = float(
                aggregate["minimum_tail_robustness"]
            )
            first_failure_tail_check = first_failure_tail_robustness <= 0.0
            first_failure_first_stage_advantage = int(
                aggregate["first_stage_correct"]
            ) > int(aggregate["full_order_correct"])

        first_failure_tail_checks.append(first_failure_tail_check)
        first_failure_partial_advantage.append(first_failure_first_stage_advantage)

        log2_lengths = [math.log2(int(row["updates_per_stage"])) for row in conditions]
        mean_cosines = [
            float(row["aggregate"]["mean_base_conditioned_cosine"])
            for row in conditions
        ]
        mean_drifts = [
            float(row["aggregate"]["mean_relative_commutator_drift"])
            for row in conditions
        ]

        seed_summaries.append(
            {
                "codebook_seed": seed,
                "codebook_sha256": payload["codebook_sha256"],
                "dataset_sha256": payload["dataset_sha256"],
                "scientific_fingerprint_sha256": payload[
                    "scientific_fingerprint_sha256"
                ],
                "first_full_order_failure_updates": first_failure_updates,
                "first_failure_minimum_tail_robustness": (
                    first_failure_tail_robustness
                ),
                "first_failure_has_nonpositive_tail_robustness": (
                    first_failure_tail_check
                ),
                "first_failure_first_stage_advantage": (
                    first_failure_first_stage_advantage
                ),
                "error_count_across_map": seed_error_count,
                "tail_swap_only_error_count_across_map": seed_tail_errors,
                "tail_swap_fraction_among_errors": (
                    seed_tail_errors / seed_error_count if seed_error_count else 1.0
                ),
                "pearson_mean_cosine_vs_log2_updates": _pearson(
                    log2_lengths,
                    mean_cosines,
                ),
                "pearson_mean_relative_drift_vs_log2_updates": _pearson(
                    log2_lengths,
                    mean_drifts,
                ),
                "conditions": [
                    {
                        "updates_per_stage": int(row["updates_per_stage"]),
                        **row["aggregate"],
                    }
                    for row in conditions
                ],
            }
        )

    tail_swap_fraction = (
        all_tail_swap_errors / all_errors if all_errors else 1.0
    )
    checks = {
        "check_1_first_failure_tail_mechanism": {
            "passed": all(first_failure_tail_checks),
            "per_seed": first_failure_tail_checks,
            "rule": "every observed first full-order failure has minimum tail robustness <= 0",
        },
        "check_2_error_structure": {
            "passed": tail_swap_fraction >= 0.75,
            "tail_swap_fraction_among_all_errors": tail_swap_fraction,
            "tail_swap_only_errors": all_tail_swap_errors,
            "all_errors": all_errors,
            "threshold": 0.75,
        },
        "check_3_partial_chronology_survives": {
            "passed": sum(first_failure_partial_advantage) >= 3,
            "per_seed": first_failure_partial_advantage,
            "seeds_with_first_stage_advantage": sum(
                first_failure_partial_advantage
            ),
            "required": 3,
        },
        "check_4_interaction_drift_descriptive": {
            "hard_pass_rule": None,
            "note": (
                "Pre-registered as descriptive. See per-seed Pearson relationships "
                "for mean cosine and mean relative drift versus log2 stage length."
            ),
        },
    }

    result = {
        "status": "complete",
        "claim": "preregistered_independent_pythia_14m_t2_map_aggregate",
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": expected_protocol_hash,
        "selection_rule": protocol["selection_rule"],
        "seed_count": len(expected_seeds),
        "stage_lengths": list(expected_lengths),
        "pre_registered_checks": checks,
        "seed_summaries": seed_summaries,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
