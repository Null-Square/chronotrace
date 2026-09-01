#!/usr/bin/env python3
"""Aggregate the frozen fresh-seed label-blind confirmation suite."""

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
    if lock.get("confirmation_version") != (
        "chronotrace-pairwise-multi-witness-confirmation-32-v3"
    ):
        raise RuntimeError("confirmation aggregate requires the frozen v3 lock")
    if lock.get("freeze_status") != (
        "frozen_before_any_fresh_v3_confirmation_codebook_generation_or_output"
    ):
        raise RuntimeError("fresh confirmation aggregate lock status drift")
    if lock.get("no_intermediate_adaptation") is not True:
        raise RuntimeError("fresh confirmation must prohibit intermediate adaptation")

    expected_seeds = tuple(int(value) for value in lock["fresh_heldout_seeds"])
    expected_targets = tuple(str(value) for value in lock["target_histories_per_seed"])
    lock_sha = json_sha256(lock)
    results = [_load(path) for path in args.inputs]
    by_seed: dict[int, dict[str, Any]] = {}
    for result in results:
        if result.get("result_version") != (
            "chronotrace-pairwise-multi-witness-confirmation-seed-v2"
        ):
            raise RuntimeError("fresh confirmation seed result engine-version drift")
        seed = int(result["seed"])
        if seed in by_seed:
            raise RuntimeError("duplicate fresh confirmation seed result")
        by_seed[seed] = result
    if set(by_seed) != set(expected_seeds) or len(by_seed) != 4:
        raise RuntimeError("fresh confirmation aggregate seed coverage drift")

    full_coverage = 0
    pair_coverage = 0
    ambiguous_pairs = 0
    contradictory_pairs = 0
    both_excluded_pairs = 0
    minimum_margin: float | None = None
    maximum_projected_residual = 0.0
    maximum_terminal_primal_error = 0.0
    invalid = False
    all_terminal_exact = True
    all_hull_sound = True
    all_euclidean_sound = True
    all_active_replays = True
    per_seed: dict[str, Any] = {}

    for seed in expected_seeds:
        result = by_seed[seed]
        if result.get("confirmation_lock_sha256") != lock_sha:
            raise RuntimeError("fresh confirmation lock hash drift across seed result")
        if (
            set(result["cases"]) != set(expected_targets)
            or len(result["cases"]) != len(expected_targets)
        ):
            raise RuntimeError("fresh confirmation target coverage drift")
        if int(result["stage_executions"]) != 96:
            raise RuntimeError("fresh confirmation seed stage execution drift")
        if int(result["witness_freeze_stage_executions"]) != 72:
            raise RuntimeError("fresh confirmation seed witness-freeze drift")
        if int(result["retained_degree4_endpoint_projection_scalars"]) != 768:
            raise RuntimeError("fresh confirmation endpoint scalar retention drift")
        if int(result["retained_degree4_interaction_projection_scalars"]) != 768:
            raise RuntimeError("fresh confirmation interaction scalar retention drift")
        if result.get("full_k4_model_space_tensors_retained") is not False:
            raise RuntimeError("fresh confirmation retained full K4 tensors")

        seed_full = int(result["full_history_certificate_coverage"])
        seed_pairs = int(result["label_blind_pairwise_orientation_certificate_coverage"])
        seed_ambiguous = int(result["ambiguous_pair_count"])
        seed_contradictory = int(result["contradictory_pair_count"])
        seed_both_excluded = int(result["both_orientations_excluded_count"])
        full_coverage += seed_full
        pair_coverage += seed_pairs
        ambiguous_pairs += seed_ambiguous
        contradictory_pairs += seed_contradictory
        both_excluded_pairs += seed_both_excluded
        seed_invalid = bool(result["invalid_seed_job"])
        invalid = invalid or seed_invalid

        projected_residual = float(result["projected_reconstruction_residual_max"])
        terminal_error = float(result["maximum_terminal_primal_exactness_error"])
        maximum_projected_residual = max(maximum_projected_residual, projected_residual)
        maximum_terminal_primal_error = max(maximum_terminal_primal_error, terminal_error)
        margin = result.get("minimum_excluded_orientation_margin_over_guard")
        if margin is not None:
            value = float(margin)
            minimum_margin = value if minimum_margin is None else min(minimum_margin, value)

        for target in expected_targets:
            case = result["cases"][target]
            all_active_replays = all_active_replays and bool(
                case["active_lift_target_hash_match"]
            )
            for pair_label in ("AB", "AC", "AD", "BC", "BD", "CD"):
                pair = case["pairwise"][pair_label]
                all_terminal_exact = all_terminal_exact and bool(
                    pair["terminal_exactness_passed"]
                )
                all_hull_sound = all_hull_sound and bool(
                    pair["lower_bound_sound_in_witness_geometry"]
                )
                all_euclidean_sound = all_euclidean_sound and bool(
                    pair["lower_bound_sound_against_euclidean_vertices"]
                )

        per_seed[str(seed)] = {
            "codebook_sha256": result["codebook_sha256"],
            "dataset_sha256": result["dataset_sha256"],
            "full_history_certificate_coverage": seed_full,
            "label_blind_pairwise_orientation_certificate_coverage": seed_pairs,
            "ambiguous_pair_count": seed_ambiguous,
            "contradictory_pair_count": seed_contradictory,
            "both_orientations_excluded_count": seed_both_excluded,
            "minimum_excluded_orientation_margin_over_guard": margin,
            "invalid": seed_invalid,
        }

    if contradictory_pairs != 0 or both_excluded_pairs != 0:
        invalid = True
    if (
        not all_terminal_exact
        or not all_hull_sound
        or not all_euclidean_sound
        or not all_active_replays
    ):
        invalid = True
    if full_coverage < 0 or full_coverage > 32:
        raise RuntimeError("fresh confirmation full-history coverage outside frozen bounds")
    if pair_coverage < 0 or pair_coverage > 192:
        raise RuntimeError("fresh confirmation pair coverage outside frozen bounds")
    if pair_coverage + ambiguous_pairs != 192 and not invalid:
        raise RuntimeError("valid fresh confirmation pair accounting does not sum to 192")

    classification = _classification(full_coverage, invalid)
    aggregate = {
        "selection_version": "chronotrace-pairwise-multi-witness-confirmation-selection-v3",
        "confirmation_lock_sha256": lock_sha,
        "fresh_seed_count": 4,
        "target_count_per_seed": 8,
        "confirmation_case_count": 32,
        "pairwise_decision_count": 192,
        "orientation_class_lp_solve_count": 384,
        "full_history_certificate_coverage": full_coverage,
        "label_blind_pairwise_orientation_certificate_coverage": pair_coverage,
        "full_history_abstention_count": 32 - full_coverage,
        "ambiguous_pair_count": ambiguous_pairs,
        "contradictory_pair_count": contradictory_pairs,
        "both_orientations_excluded_count": both_excluded_pairs,
        "minimum_excluded_orientation_margin_over_guard": minimum_margin,
        "maximum_projected_reconstruction_residual": maximum_projected_residual,
        "maximum_terminal_primal_exactness_error": maximum_terminal_primal_error,
        "all_terminal_witness_hull_exactness_passed": all_terminal_exact,
        "all_corrected_bounds_sound_in_witness_geometry": all_hull_sound,
        "all_corrected_bounds_sound_against_euclidean_vertices": all_euclidean_sound,
        "all_target_active_lifts_replayed_exactly": all_active_replays,
        "invalid_suite": invalid,
        "outcome_classification": classification,
        "per_seed": per_seed,
        "method_changed_after_fresh_confirmation_started": False,
        "v1_seed_results_used_in_v3_selection": False,
        "interpretation": (
            "Fresh deterministic-seed label-blind 32-case confirmation interpreted exactly "
            "under the preregistered coverage thresholds; no post-launch adaptation is applied."
        ),
    }
    Path(args.output).write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
