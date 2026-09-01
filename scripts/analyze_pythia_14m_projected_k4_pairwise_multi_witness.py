#!/usr/bin/env python3
"""Post-hoc pairwise chronology certificate from the frozen projected-K4 artifact."""

from __future__ import annotations

import argparse
import json
from itertools import combinations, permutations
from pathlib import Path
from typing import Any

import numpy as np

from chronotrace.geometry.local_order_hierarchy import build_local_order_hierarchy
from chronotrace.geometry.multi_witness_local_order import (
    solve_local_order_multi_witness_lp,
)
from chronotrace.reproducibility import json_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    result = _load(args.result)
    protocol = result["protocol"]
    if result.get("all_diagnostic_checks_passed") is not True:
        raise RuntimeError("pairwise post-hoc analysis requires a valid K4 result")
    if result.get("scientific_negative") is not True:
        raise RuntimeError("pairwise post-hoc analysis is frozen for the single-witness negative")
    if result.get("confirmation_codebooks_observed") is not False:
        raise RuntimeError("completed K4 result touched confirmation codebooks")
    if result.get("heldout_confirmation_launch_authorized") is not False:
        raise RuntimeError("completed K4 result authorizes held-out confirmation")
    if int(protocol["pilot_codebook_seed"]) != 1011473075:
        raise RuntimeError("pairwise post-hoc analysis requires the spent methodology seed")

    stages = tuple(str(stage) for stage in protocol["stages"])
    witnesses = tuple(str(stage) for stage in protocol["witness_classes"])
    target = str(protocol["target_history"])
    if stages != ("A", "B", "C", "D") or witnesses != stages or target != "ABCD":
        raise RuntimeError("pairwise post-hoc analysis requires the frozen ABCD setup")

    hierarchy = build_local_order_hierarchy(stages, max_degree=len(stages))
    constants = np.zeros(len(witnesses), dtype=np.float64)
    coefficients = np.zeros((len(witnesses), hierarchy.dimension), dtype=np.float64)
    endpoint_projections = result["degree4_endpoint_projections"]
    direct_errors = result["degree4_direct_endpoint_errors"]

    for witness_index, witness in enumerate(witnesses):
        target_projection = float(endpoint_projections[witness][target])
        for history in permutations(stages):
            label = "".join(history)
            coefficients[witness_index, hierarchy.coordinate_index[history]] = (
                target_projection - float(endpoint_projections[witness][label])
            )

    guard = float(protocol["numerical_elimination_guard"])
    certificate_guard = 1e-10
    target_position = {stage: index for index, stage in enumerate(target)}
    pairwise: dict[str, Any] = {}
    wrong_bounds: list[float] = []

    for left, right in combinations(stages, 2):
        true_relation = (
            (left, right)
            if target_position[left] < target_position[right]
            else (right, left)
        )
        wrong_relation = (true_relation[1], true_relation[0])
        wrong = solve_local_order_multi_witness_lp(
            hierarchy,
            constants,
            coefficients,
            precedences=(wrong_relation,),
            certificate_guard=certificate_guard,
        )
        true = solve_local_order_multi_witness_lp(
            hierarchy,
            constants,
            coefficients,
            precedences=(true_relation,),
            certificate_guard=certificate_guard,
        )
        wrong_histories = [
            "".join(history)
            for history in permutations(stages)
            if history.index(wrong_relation[0]) < history.index(wrong_relation[1])
        ]
        exact_euclidean = min(float(direct_errors[history]) for history in wrong_histories)
        if wrong.euclidean_distance_lower_bound > exact_euclidean + 1e-10:
            raise RuntimeError("pairwise certificate exceeds exact wrong-class Euclidean distance")
        if true.euclidean_distance_lower_bound > guard:
            raise RuntimeError("true pairwise relation was incorrectly certified impossible")
        certified = wrong.euclidean_distance_lower_bound > guard
        wrong_bounds.append(wrong.euclidean_distance_lower_bound)
        pairwise[f"{left}{right}"] = {
            "true_relation": list(true_relation),
            "wrong_relation": list(wrong_relation),
            "wrong_class_primal_objective": wrong.primal_objective,
            "wrong_class_certified_lower_bound": wrong.certified_lower_bound,
            "wrong_class_euclidean_distance_lower_bound": (
                wrong.euclidean_distance_lower_bound
            ),
            "wrong_class_direct_exact_euclidean_distance": exact_euclidean,
            "wrong_relation_certified_impossible": certified,
            "true_class_primal_objective": true.primal_objective,
            "true_class_euclidean_distance_lower_bound": (
                true.euclidean_distance_lower_bound
            ),
        }

    all_wrong_certified = all(
        item["wrong_relation_certified_impossible"] for item in pairwise.values()
    )
    reconstructed = (
        "".join(sorted(stages, key=lambda stage: target_position[stage]))
        if all_wrong_certified
        else None
    )
    if reconstructed is not None and reconstructed != target:
        raise RuntimeError("pairwise certificates reconstructed an unexpected chronology")

    certificate_form = (
        "proof-safe infinity-norm multi-witness LP over local-order marginals with one "
        "wrong precedence class per unordered stage pair"
    )
    if all_wrong_certified:
        interpretation = (
            "All six wrong pairwise order classes are excluded using only the four "
            "witnesses frozen before K4 output. Their conjunction uniquely identifies "
            "ABCD on the spent target. This is post-hoc methodology development and does "
            "not change the preregistered single-witness negative."
        )
    else:
        interpretation = (
            "The frozen witness bank does not certify every pairwise relation on the "
            "spent target."
        )

    analysis = {
        "analysis_version": "pythia-14m-projected-k4-pairwise-multi-witness-posthoc-v1",
        "analysis_role": "posthoc_spent_methodology_development_not_confirmation",
        "source_result_sha256": json_sha256(result),
        "source_run_id": 33327336769,
        "source_job_id": 99299704409,
        "source_artifact_id": 9736705289,
        "source_preregistered_result": "scientific_negative_single_witness",
        "witness_bank_freeze": "all four witnesses were fixed from K3 before any K4 output",
        "certificate_form": certificate_form,
        "hierarchy_degree": hierarchy.max_degree,
        "hierarchy_dimension": hierarchy.dimension,
        "pairwise": pairwise,
        "all_six_wrong_relations_certified_impossible": all_wrong_certified,
        "minimum_wrong_relation_distance_lower_bound": min(wrong_bounds),
        "frozen_elimination_guard": guard,
        "full_chronology_certified_by_pairwise_exclusion": all_wrong_certified,
        "certified_chronology": reconstructed,
        "new_model_executions": 0,
        "confirmation_codebooks_observed": False,
        "heldout_confirmation_launch_authorized": False,
        "interpretation": interpretation,
    }
    Path(args.output).write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(analysis, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
