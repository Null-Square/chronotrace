#!/usr/bin/env python3
"""Hash-verified, post-hoc reanalysis of the four frozen v3 JSON artifacts.

No training, model queries, LP solves, or original-protocol edits are performed.
Stored discrete minima are audited, not recomputed from unavailable tensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from chronotrace.geometry.certificate_postprocess import (
    close_certified_precedences,
    decide_from_class_lower_bounds,
)

STAGES = ("A", "B", "C", "D")
PAIRS = ("AB", "AC", "AD", "BC", "BD", "CD")
ORIENTATIONS = ("left_before_right", "right_before_left")
TARGETS = {"ABCD", "BCDA", "CDAB", "DABC", "DCBA", "ADCB", "BADC", "CBAD"}
EXPECTED_SHA256 = {
    2186192236: "5e3c45a5a15e9aca7d359f0106919771b49b15d9d10d07a9150d6a4c5752610e",
    1368008047: "1ec1370652c86386959e27b0300d9c526cce55401ff6e5a51c043e9611553306",
    92712904: "c4d5e5b16fb2c02f5dddf46f74b25ee487f4032405b45adfe825b5bc63034228",
    1944430236: "69dd387ace07ecf71d6f2f0b73206dc282d635ec36bebe52c7da82165ec3e37e",
}
RESIDUAL_BUDGET = 1e-6  # frozen elimination guard, NOT tuned on this audit
NUMERICAL_GUARD = 1e-10  # sensitivity allowance, NOT an interval-arithmetic proof


def analyze_case(
    pairwise: dict[str, Any], witness_norm_bound: float
) -> dict[str, Any]:
    """Decision-only interface: no generating history or evaluation metadata."""
    if set(pairwise) != set(PAIRS):
        raise ValueError("expected all six unordered pairs")
    direct = []
    for pair in PAIRS:
        relation = pairwise[pair]["inferred_precedence"]
        if relation is not None:
            edge = tuple(relation)
            if edge not in (tuple(pair), tuple(reversed(pair))):
                raise ValueError("stored relation disagrees with its pair key")
            direct.append(edge)
    closure = close_certified_precedences(STAGES, direct)
    output: dict[str, Any] = {"logical_closure": asdict(closure)}
    for metric in ("stored_discrete_witness", "stored_euclidean_vertex"):
        decisions, edges = {}, []
        for pair in PAIRS:
            entry = pairwise[pair]
            if metric == "stored_discrete_witness":
                bounds = [
                    float(entry[f"{orientation}_nearest_vertex_proxy"])
                    / witness_norm_bound
                    for orientation in ORIENTATIONS
                ]
            else:
                bounds = [
                    float(entry[orientation]["direct_exact_euclidean_vertex_class_distance"])
                    for orientation in ORIENTATIONS
                ]
            decision = decide_from_class_lower_bounds(
                *tuple(pair),
                *bounds,
                residual_budget=RESIDUAL_BUDGET,
                numerical_guard=NUMERICAL_GUARD,
            )
            decisions[pair] = {**asdict(decision), "bounds": bounds}
            if decision.inferred_precedence is not None:
                edges.append(decision.inferred_precedence)
        graph = close_certified_precedences(STAGES, edges)
        invalid = any(
            decision["status"] == "invalid_both_excluded" for decision in decisions.values()
        ) or graph.status == "invalid_cycle"
        output[metric] = {
            "decisions": decisions,
            "closure": asdict(graph),
            "invalid": invalid,
        }
    return output


def run_audit(raw_root: Path) -> dict[str, Any]:
    paths = sorted(raw_root.rglob("confirmation-v3-seed-*.json"))
    if len(paths) != 4:
        raise ValueError("expected exactly four original seed JSONs")
    seeds_seen: set[int] = set()
    records = []
    sources = []
    original = Counter()
    for path in paths:
        raw = path.read_bytes()
        data = json.loads(raw)
        seed = int(data["seed"])
        digest = hashlib.sha256(raw).hexdigest()
        if seed in seeds_seen or EXPECTED_SHA256.get(seed) != digest:
            raise ValueError(f"duplicate, unknown, or hash-mismatched seed: {seed}")
        seeds_seen.add(seed)
        if data["invalid_seed_job"] or set(data["cases"]) != TARGETS:
            raise ValueError("invalid source job or target coverage")
        if (data["stage_executions"], data["witness_freeze_stage_executions"]) != (96, 72):
            raise ValueError("frozen execution accounting changed")
        sources.append({"seed": seed, "filename": path.name, "sha256": digest})
        for history, case in data["cases"].items():
            if case["invalid"]:
                raise ValueError("invalid original case")
            norm_bound = max(1.0, *(float(v["unit_norm"]) for v in case["witness_k3"].values()))
            # Select only decision inputs. Labels and evaluation fields never enter analyze_case.
            selected = {
                pair: {
                    "inferred_precedence": entry["inferred_precedence"],
                    **{
                        f"{orientation}_nearest_vertex_proxy": entry[
                            f"{orientation}_nearest_vertex_proxy"
                        ]
                        for orientation in ORIENTATIONS
                    },
                    **{
                        orientation: {
                            "direct_exact_euclidean_vertex_class_distance": entry[orientation][
                                "direct_exact_euclidean_vertex_class_distance"
                            ]
                        }
                        for orientation in ORIENTATIONS
                    },
                }
                for pair, entry in case["pairwise"].items()
            }
            predictions = analyze_case(selected, norm_bound)
            records.append({"seed": seed, "history": history, "predictions": predictions})
            original["full_histories"] += int(case["full_history_certified"])
            original["direct_pairs"] += int(case["label_blind_pairwise_orientation_certificates"])
            original["wrong_pairs"] += int(case["contradictory_pair_count"])
    if seeds_seen != set(EXPECTED_SHA256):
        raise ValueError("incomplete source seed set")
    if dict(original) != {"full_histories": 27, "direct_pairs": 182, "wrong_pairs": 0}:
        raise ValueError("source aggregation no longer matches frozen report")

    # All case predictions have now been fixed. Only this section reads ground truth.
    aggregate: dict[str, Any] = {"original_frozen": dict(original)}
    per_seed: dict[str, Any] = {}
    for metric in ("logical_closure", "stored_discrete_witness", "stored_euclidean_vertex"):
        total = Counter(full_histories=0, entailed_pairs=0, wrong_pairs=0, invalid_cases=0)
        seed_totals: dict[int, Counter] = {}
        for record in records:
            result = record["predictions"][metric]
            graph = result if metric == "logical_closure" else result["closure"]
            invalid = (
                graph["status"] == "invalid_cycle"
                if metric == "logical_closure" else result["invalid"]
            )
            history = record["history"]
            edges = graph["entailed_relations"] if not invalid else []
            wrong = sum(history.index(a) > history.index(b) for a, b in edges)
            unique = graph["unique_order"]
            values = {
                "full_histories": int(not invalid and unique is not None),
                "entailed_pairs": len(edges),
                "wrong_pairs": wrong,
                "invalid_cases": int(invalid),
            }
            if unique is not None and not invalid and tuple(history) != tuple(unique):
                raise RuntimeError("post-hoc decoder returned an incorrect full history")
            total.update(values)
            seed_totals.setdefault(record["seed"], Counter()).update(values)
        aggregate[metric] = dict(total)
        per_seed[metric] = {str(seed): dict(v) for seed, v in seed_totals.items()}
    return {
        "audit_version": "chronotrace-posthoc-closure-discrete-audit-2026-09-12-v1",
        "evidence_role": "posthoc_reanalysis_of_frozen_artifacts_not_new_confirmation",
        "scientific_run_id": 33418210637,
        "scientific_commit": "7107221c16a001a7974ca1b436d9cacd26145fe2",
        "audited_branch_commit": "4e71d4f8257293c250a9fbe10390a976525a7520",
        "residual_budget": RESIDUAL_BUDGET,
        "numerical_guard": NUMERICAL_GUARD,
        "numerical_guard_is_formal_roundoff_proof": False,
        "stored_discrete_minima_independently_recomputed": False,
        "additional_training_stage_calls": 0,
        "additional_lp_solves": 0,
        "source_hashes_verified": sources,
        "cases": 32,
        "pair_decisions": 192,
        "aggregate": aggregate,
        "per_seed": per_seed,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = run_audit(args.raw_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
