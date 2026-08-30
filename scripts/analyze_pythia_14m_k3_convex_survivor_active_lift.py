#!/usr/bin/env python3
"""Analyze K3 convex pruning against frozen K3 tails and exact active codewords.

This is a zero-new-model-output composition diagnostic. It does not authorize held-out
confirmation and it does not treat a degree-three pruning certificate as a true-history
certificate without an explicit higher-order tail bound.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from chronotrace.geometry.distance_hull import certify_hull_from_pairwise_distances


def _load_json(path: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k3-convex-result", required=True)
    parser.add_argument("--k23-result", required=True)
    parser.add_argument("--all24-result", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--numerical-guard", type=float, default=1e-6)
    return parser.parse_args()


def _require_spent(payload: dict[str, Any], *, name: str) -> None:
    if payload.get("confirmation_codebooks_observed") is not False:
        raise RuntimeError(f"{name} is not confirmed spent-only evidence")


def main() -> None:
    args = parse_args()
    convex = _load_json(args.k3_convex_result)
    k23 = _load_json(args.k23_result)
    all24 = _load_json(args.all24_result)
    guard = float(args.numerical_guard)
    if not math.isfinite(guard) or guard < 0.0:
        raise ValueError("numerical guard must be finite and non-negative")

    for name, payload in (("K3 convex", convex), ("K2/K3", k23), ("all24", all24)):
        _require_spent(payload, name=name)
    if convex.get("heldout_confirmation_launch_authorized") is not False:
        raise RuntimeError("K3 convex result unexpectedly authorizes held-out confirmation")
    if convex.get("status") != "complete" or convex.get("promising_partial_pruning") is not True:
        raise RuntimeError("expected the frozen K3 convex promising-partial result")
    if all24.get("status") != "complete" or all24.get("all24_pass_all") is not True:
        raise RuntimeError("expected the frozen all-24 exact reachable certificate")

    identity_fields = ("model", "revision", "pilot_codebook_seed")
    for field in identity_fields:
        if convex.get(field) != k23.get(field) or convex.get(field) != all24.get(field):
            raise RuntimeError(f"spent evidence identity drift for {field}")

    target = str(convex["target_history"])
    if target not in all24["pairwise_reachable_codeword_distances"]:
        raise RuntimeError("K3 convex target is missing from all-24 codeword matrix")
    target_row = all24["per_history"][target]
    if target_row.get("target_replay_exact") is not True:
        raise RuntimeError("all-24 target did not replay exactly")

    stages = tuple(sorted(convex["candidate_last"]))
    survivors = tuple(str(value) for value in convex["surviving_last_stages"])
    eliminated = tuple(str(value) for value in convex["eliminated_last_stages"])
    if set(survivors) | set(eliminated) != set(stages):
        raise RuntimeError("K3 convex stage partition drift")

    tails: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in k23["histories"]:
        truth = str(row["truth"])
        error = float(row["k3"]["true_candidate_error"])
        if not math.isfinite(error) or error < 0.0:
            raise RuntimeError("non-finite K3 true-candidate error")
        tails[truth[-1]].append((truth, error))

    classwise: dict[str, dict[str, Any]] = {}
    safe_exact_skip = True
    distances = all24["pairwise_reachable_codeword_distances"]
    for stage in stages:
        class_histories = tuple(sorted(history for history in distances if history[-1] == stage))
        hull = certify_hull_from_pairwise_distances(target, class_histories, distances)
        truncated_bound = float(convex["candidate_last"][stage]["dual_witness_lower_bound"])
        tail_radius = max(error for _, error in tails[stage])
        tail_adjusted = max(0.0, truncated_bound - tail_radius)
        tail_exact_eliminated = tail_adjusted > guard
        if stage in eliminated and not tail_exact_eliminated:
            safe_exact_skip = False
        classwise[stage] = {
            "history_count": len(class_histories),
            "truncated_K3_dual_lower_bound": truncated_bound,
            "empirical_degree4_tail_radius_from_spent_exact_targets": tail_radius,
            "tail_adjusted_exact_lower_bound": tail_adjusted,
            "tail_bound_certifies_true_class_impossible": tail_exact_eliminated,
            "exact_active_codeword_hull_primal_distance": hull.projection.distance,
            "exact_active_codeword_hull_dual_lower_bound": hull.certificate.lower_bound,
            "exact_active_codeword_hull_support": [
                class_histories[index] for index in hull.projection.support
            ],
        }

    survivor_histories = tuple(sorted(history for history in distances if history[-1] in survivors))
    ranked_survivors = sorted(
        (float(distances[target][history]), history)
        for history in survivor_histories
    )
    best_distance, best_history = ranked_survivors[0]
    runner_up_distance, runner_up_history = ranked_survivors[1]
    if best_history != target or best_distance != 0.0:
        raise RuntimeError("restricted active-codeword decode did not select the target codeword")

    survivor_last_bounds = {
        stage: classwise[stage]["exact_active_codeword_hull_dual_lower_bound"]
        for stage in survivors
    }
    true_last = target[-1]
    wrong_survivors = tuple(stage for stage in survivors if stage != true_last)
    conditional_last_stage_certified = (
        true_last in survivors
        and all(float(survivor_last_bounds[stage]) > guard for stage in wrong_survivors)
        and float(survivor_last_bounds[true_last]) <= guard
    )

    all_candidate_count = len(distances)
    survivor_candidate_count = len(survivor_histories)
    result = {
        "status": "complete",
        "claim": "posthoc_zero_new_model_output_K3_tail_and_active_hull_composition_analysis",
        "confirmation_codebooks_observed": False,
        "heldout_confirmation_launch_authorized": False,
        "model": convex["model"],
        "revision": convex["revision"],
        "pilot_codebook_seed": convex["pilot_codebook_seed"],
        "target_history": target,
        "K3_eliminated_last_stages": list(eliminated),
        "K3_surviving_last_stages": list(survivors),
        "classwise": classwise,
        "K3_eliminated_classes_safe_to_skip_for_true_history": safe_exact_skip,
        "reason_if_not_safe": (
            None
            if safe_exact_skip
            else "classwise norm tail bounds erase at least one truncated K3 elimination margin"
        ),
        "conditional_survivor_only_active_refinement": {
            "candidate_history_count": survivor_candidate_count,
            "all24_candidate_history_count": all_candidate_count,
            "hypothetical_candidate_reduction_fraction": 1.0 - survivor_candidate_count / all_candidate_count,
            "best_history": best_history,
            "best_codeword_distance": best_distance,
            "runner_up_history": runner_up_history,
            "runner_up_codeword_distance": runner_up_distance,
            "codeword_margin": runner_up_distance - best_distance,
            "last_stage_hull_dual_lower_bounds": survivor_last_bounds,
            "certifies_true_last_stage_conditional_on_survivor_set": conditional_last_stage_certified,
            "unconditional_true_history_speedup_claim_allowed": safe_exact_skip,
        },
        "target_exact_replay_self_residual": float(target_row["self_residual"]),
        "target_all24_certified_margin": float(target_row["certified_margin"]),
        "new_model_stage_executions": 0,
        "new_candidate_gradient_evaluations": 0,
        "scientific_interpretation": (
            "The exact active reachable geometry cleanly resolves the C/D survivor ambiguity, "
            "but the available classwise degree-four norm tail bounds are too loose to promote "
            "the static K3 A/B eliminations into true-history eliminations. Therefore a fresh "
            "certified decoder may not safely skip A/B active checks yet. The next scalability "
            "target is a tighter higher-order tail enclosure or an active-transition-preserving "
            "relaxation that certifies those branches without factorial enumeration."
        ),
        "next_step": "derive_and_falsify_tighter_tail_or_active_transition_class_bounds_on_spent_data",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
