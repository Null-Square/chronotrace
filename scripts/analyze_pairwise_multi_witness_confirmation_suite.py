#!/usr/bin/env python3
"""Aggregate the four frozen heldout seed results without adaptive interpretation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from chronotrace.reproducibility import json_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _classification(full_coverage: int, invalid: bool) -> str:
    if invalid:
        return "invalid"
    if full_coverage >= 28:
        return "excellent"
    if full_coverage >= 24:
        return "strong"
    if full_coverage >= 16:
        return "partial"
    return "scientific_negative"


def main() -> None:
    args = parse_args()
    lock = _load(args.lock)
    expected_seeds = tuple(int(value) for value in lock["heldout_seeds"])
    expected_targets = tuple(str(value) for value in lock["target_histories_per_seed"])
    lock_sha = json_sha256(lock)

    results = [_load(path) for path in args.inputs]
    by_seed: dict[int, dict[str, Any]] = {}
    for result in results:
        seed = int(result["seed"])
        if seed in by_seed:
            raise RuntimeError("duplicate confirmation seed result")
        by_seed[seed] = result
    if set(by_seed) != set(expected_seeds) or len(by_seed) != 4:
        raise RuntimeError("confirmation aggregate seed coverage drift")

    full_coverage = 0
    pair_coverage = 0
    minimum_margin: float | None = None
    invalid = False
    per_seed: dict[str, Any] = {}
    all_true_sane = True
    all_wrong_sound = True
    all_active_replays = True

    for seed in expected_seeds:
        result = by_seed[seed]
        if result.get("confirmation_lock_sha256") != lock_sha:
            raise RuntimeError("confirmation lock hash drift across seed result")
        if tuple(result["cases"]) != expected_targets:
            raise RuntimeError("confirmation target order/coverage drift")
        if int(result["stage_executions"]) != 96:
            raise RuntimeError("confirmation seed stage execution drift")
        if int(result["witness_freeze_stage_executions"]) != 72:
            raise RuntimeError("confirmation seed witness-freeze drift")
        if int(result["retained_degree4_endpoint_projection_scalars"]) != 768:
            raise RuntimeError("confirmation endpoint scalar retention drift")
        if int(result["retained_degree4_interaction_projection_scalars"]) != 768:
            raise RuntimeError("confirmation interaction scalar retention drift")
        if result.get("full_k4_model_space_tensors_retained") is not False:
            raise RuntimeError("confirmation result retained full K4 tensors")

        seed_full = int(result["full_history_certificate_coverage"])
        seed_pairs = int(result["pairwise_wrong_orientation_certificate_coverage"])
        full_coverage += seed_full
        pair_coverage += seed_pairs
        seed_invalid = bool(result["invalid_seed_job"])
        invalid = invalid or seed_invalid
        margin = result.get("minimum_wrong_orientation_margin_over_guard")
        if margin is not None:
            value = float(margin)
            minimum_margin = value if minimum_margin is None else min(minimum_margin, value)

        for target in expected_targets:
            case = result["cases"][target]
            all_active_replays = all_active_replays and bool(case["active_lift_target_hash_match"])
            for pair in case["pairwise"].values():
                all_true_sane = all_true_sane and bool(pair["true_relation_sanity_passed"])
                all_wrong_sound = all_wrong_sound and bool(pair["wrong_lower_bound_sound"])
        per_seed[str(seed)] = {
            "codebook_sha256": result["codebook_sha256"],
            "dataset_sha256": result["dataset_sha256"],
            "full_history_certificate_coverage": seed_full,
            "pairwise_wrong_orientation_certificate_coverage": seed_pairs,
            "minimum_wrong_orientation_margin_over_guard": margin,
            "invalid": seed_invalid,
        }

    if not all_true_sane or not all_wrong_sound or not all_active_replays:
        invalid = True
    if full_coverage < 0 or full_coverage > 32 or pair_coverage < 0 or pair_coverage > 192:
        raise RuntimeError("confirmation aggregate coverage is outside frozen bounds")

    classification = _classification(full_coverage, invalid)
    aggregate = {
        "selection_version": "chronotrace-pairwise-multi-witness-confirmation-selection-v1",
        "confirmation_lock_sha256": lock_sha,
        "heldout_seed_count": 4,
        "target_count_per_seed": 8,
        "confirmation_case_count": 32,
        "pairwise_decision_count": 192,
        "full_history_certificate_coverage": full_coverage,
        "pairwise_wrong_orientation_certificate_coverage": pair_coverage,
        "full_history_abstention_count": 32 - full_coverage,
        "minimum_wrong_orientation_margin_over_guard": minimum_margin,
        "all_true_relation_sanity_passed": all_true_sane,
        "all_wrong_lower_bounds_sound": all_wrong_sound,
        "all_target_active_lifts_replayed_exactly": all_active_replays,
        "invalid_suite": invalid,
        "outcome_classification": classification,
        "per_seed": per_seed,
        "method_changed_after_confirmation_started": False,
        "interpretation": (
            "Frozen 32-case confirmation interpreted exactly under the preregistered coverage thresholds; no post-launch method adaptation is applied."
        ),
    }
    Path(args.output).write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
