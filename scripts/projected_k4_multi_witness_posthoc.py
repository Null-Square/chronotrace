#!/usr/bin/env python3
"""Post-hoc spent-data multi-witness analysis of a completed projected-K4 result."""

from __future__ import annotations

import argparse
import json
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np

from chronotrace.geometry.multi_witness import optimize_l1_witness_combination
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
        raise RuntimeError("post-hoc analysis requires a valid completed K4 diagnostic")
    if result.get("scientific_negative") is not True or result.get("strong_success") is not False:
        raise RuntimeError("post-hoc analyzer is frozen for the preregistered K4 negative")
    if result.get("confirmation_codebooks_observed") is not False:
        raise RuntimeError("completed K4 result touched confirmation codebooks")
    if result.get("heldout_confirmation_launch_authorized") is not False:
        raise RuntimeError("completed K4 result authorizes held-out confirmation")
    if protocol.get("pilot_codebook_seed") != 1011473075:
        raise RuntimeError("post-hoc analyzer requires the spent methodology seed")

    stages = tuple(str(stage) for stage in protocol["stages"])
    target_history = str(protocol["target_history"])
    witness_labels = tuple(str(stage) for stage in protocol["witness_classes"])
    if stages != ("A", "B", "C", "D") or witness_labels != stages:
        raise RuntimeError("post-hoc analyzer requires the frozen four-witness A/B/C/D result")

    endpoint_projections = result["degree4_endpoint_projections"]
    direct_errors = result["degree4_direct_endpoint_errors"]
    guard = float(protocol["numerical_elimination_guard"])
    certificate_guard = 1e-12
    candidate_last: dict[str, Any] = {}

    for final_stage in stages:
        histories = tuple("".join(history) for history in permutations(stages) if history[-1] == final_stage)
        score_matrix = np.asarray(
            [
                [
                    float(endpoint_projections[witness][target_history])
                    - float(endpoint_projections[witness][history])
                    for history in histories
                ]
                for witness in witness_labels
            ],
            dtype=np.float64,
        )
        certificate = optimize_l1_witness_combination(
            score_matrix,
            certificate_guard=certificate_guard,
        )
        exact_euclidean = min(float(direct_errors[history]) for history in histories)
        if certificate.euclidean_distance_lower_bound > exact_euclidean + 1e-10:
            raise RuntimeError("multi-witness bound exceeds direct exact Euclidean class distance")
        candidate_last[final_stage] = {
            "histories": list(histories),
            "witness_coefficients": {
                witness: float(value)
                for witness, value in zip(witness_labels, certificate.coefficients, strict=True)
            },
            "witness_coefficient_l1_norm": certificate.l1_norm,
            "numerical_game_value": certificate.numerical_game_value,
            "minimum_vertex_support": certificate.minimum_vertex_support,
            "certified_lower_bound": certificate.certified_lower_bound,
            "euclidean_distance_lower_bound": certificate.euclidean_distance_lower_bound,
            "direct_exact_euclidean_class_distance": exact_euclidean,
            "posthoc_certified_impossible": certificate.euclidean_distance_lower_bound > guard,
        }

    c_certified = bool(candidate_last["C"]["posthoc_certified_impossible"])
    d_survives = not bool(candidate_last["D"]["posthoc_certified_impossible"])
    posthoc = {
        "analysis_version": "projected-k4-multi-witness-posthoc-v1",
        "analysis_role": "posthoc_spent_methodology_development_not_confirmation",
        "source_result_sha256": json_sha256(result),
        "source_run_id": 33327336769,
        "source_job_id": 99299704409,
        "source_artifact_id": 9736705289,
        "source_preregistered_result": "scientific_negative_single_witness",
        "witness_bank_freeze": "all four unit witnesses were frozen from K3 before any K4 output",
        "certificate_theorem": "for any coefficients alpha with L1 norm at most one, v=sum_j alpha_j u_j has Euclidean norm at most one; therefore class distance is at least min_i <v,y-q_i>. The numerical optimizer only proposes alpha; the reported support bound is independently recomputed after L1 normalization.",
        "candidate_last": candidate_last,
        "survivor_posthoc": {
            "C_certified_impossible": c_certified,
            "D_survives": d_survives,
        },
        "frozen_elimination_guard": guard,
        "confirmation_codebooks_observed": false,
        "heldout_confirmation_launch_authorized": false,
        "interpretation": (
            "The pre-K4-frozen witness bank contains enough directional information to separate C from the target when combined proof-safely, even though the preregistered single C witness fails. This is post-hoc method development on spent data and does not convert the preregistered negative into a success."
            if c_certified and d_survives
            else "The frozen witness bank does not resolve C/D under the post-hoc proof-safe combination."
        ),
    }
    Path(args.output).write_text(json.dumps(posthoc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(posthoc, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
